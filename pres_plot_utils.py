import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt


def add_all_licks(binary_signals):
    """Combine rewarded (key 1) and early (key 2) first-lick timestamps into 'all_first_licks'."""
    bs_event_meta = binary_signals.metadata[['event']].copy()
    all_lick_t = np.sort(np.concatenate([binary_signals[1].t, binary_signals[2].t]))
    new_idx = int(max(binary_signals.keys())) + 1
    d = {k: binary_signals[k] for k in binary_signals.keys()}
    d[new_idx] = nap.Ts(t=all_lick_t)
    binary_signals = nap.TsGroup(d, time_support=binary_signals.time_support)
    bs_event_meta.loc[new_idx, 'event'] = 'all_first_licks'
    binary_signals.set_info(bs_event_meta)
    return binary_signals


def pool_task_stim(binary_signals, opsin, min_dur=0.5):
    """Return (on_times, off_times) for task-related pulses only (duration > min_dur s).
    Excludes short calibration pulses (~100 ms) that occur before/after the session."""
    bs_meta = binary_signals.metadata

    def _pool(keyword):
        idxs = bs_meta[bs_meta['event'].str.contains(keyword, na=False)].index
        if len(idxs) == 0:
            return np.array([])
        return np.sort(np.concatenate([binary_signals[i].t for i in idxs]))

    on_all = _pool(f'{opsin}_on')
    off_all = _pool(f'{opsin}_off')
    if len(on_all) == 0:
        return np.array([]), np.array([])
    task_on, task_off = [], []
    for t_on in on_all:
        following = off_all[off_all > t_on]
        if len(following) > 0 and (following[0] - t_on) > min_dur:
            task_on.append(t_on)
            task_off.append(following[0])
    return np.sort(np.array(task_on)), np.sort(np.array(task_off))


def compute_ramp_scores(ephys, binary_signals, ch_on, c2_on):
    """Compute per-unit first-lick ramp scores on no-stim trials only.

    Score = mean FR in [-150, -10] ms before lick minus mean FR in [-1000, -10] ms before cue.
    Sets first_lick_ramp_score info on ephys in-place and returns the pd.Series."""
    bs_meta = binary_signals.metadata
    lick_idx = bs_meta[bs_meta['event'] == 'all_first_licks'].index[0]
    cue_idx = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    lick_times = binary_signals[lick_idx].t
    cue_times = binary_signals[cue_idx].t

    valid_pairs = []
    for lt in lick_times:
        preceding = cue_times[cue_times < lt]
        if len(preceding) == 0:
            continue
        ct = preceding[-1]
        stim_contaminated = (
            (len(ch_on) and np.any((ch_on >= ct - 2.0) & (ch_on <= lt))) or
            (len(c2_on) and np.any((c2_on >= ct - 2.0) & (c2_on <= lt)))
        )
        if not stim_contaminated:
            valid_pairs.append((lt, ct))

    ramp_scores = {}
    for unit in ephys.keys():
        spk = np.asarray(ephys[unit].index)
        prelick_rates, baseline_rates = [], []
        for lt, ct in valid_pairs:
            pl_start, pl_end = lt - 0.150, lt - 0.010
            prelick_rates.append(np.sum((spk >= pl_start) & (spk < pl_end)) / (pl_end - pl_start))
            bl_start, bl_end = ct - 1.000, ct - 0.010
            baseline_rates.append(np.sum((spk >= bl_start) & (spk < bl_end)) / (bl_end - bl_start))
        ramp_scores[unit] = float(np.mean(prelick_rates) - np.mean(baseline_rates)) if prelick_rates else np.nan

    scores = pd.Series(ramp_scores, name='first_lick_ramp_score')
    ephys.set_info(first_lick_ramp_score=scores)
    return scores


def categorize_outcome_trials(binary_signals, ch_on, ch_off, c2_on, c2_off):
    """Categorize each outcome-tone event as no_stim, chr2, or chrimson.

    A trial is stim if opto_on falls in [cue-2s, outcome] AND opto_off falls
    within 200 ms before the outcome tone. Returns dict of outcome timestamps per condition."""
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    reward_key = bs_meta[bs_meta['event'] == 'reward_tone_on'].index[0]
    error_key = bs_meta[bs_meta['event'] == 'error_tone_on'].index[0]
    outcome_ts = np.sort(np.concatenate([
        binary_signals[reward_key].t,
        binary_signals[error_key].t,
    ]))

    trial_events = {'no_stim': [], 'chr2': [], 'chrimson': []}
    for ot in outcome_ts:
        prec = cue_ts[cue_ts < ot]
        if len(prec) == 0:
            continue
        ct = prec[-1]
        ch_stim = len(ch_on) > 0 and np.any((ch_on >= ct - 2.0) & (ch_on <= ot))
        c2_stim = len(c2_on) > 0 and np.any((c2_on >= ct - 2.0) & (c2_on <= ot))
        ch_off_confirmed = len(ch_off) > 0 and np.any((ch_off >= ot - 0.2) & (ch_off < ot))
        c2_off_confirmed = len(c2_off) > 0 and np.any((c2_off >= ot - 0.2) & (c2_off < ot))
        if ch_stim and ch_off_confirmed:
            trial_events['chrimson'].append(ot)
        elif c2_stim and c2_off_confirmed:
            trial_events['chr2'].append(ot)
        else:
            trial_events['no_stim'].append(ot)
    return trial_events


def plot_outcome_heatmap(ephys, trial_events, save_path=None, bin_width=0.01,
                         t_start=-4.0, t_end=1.0,
                         baseline_window=(-4.0, -2.0), sort_window=(-0.20, 0.0),
                         clim=(-2, 5)):
    """Compute PSTHs, z-score (baseline from no-stim), sort by pre-outcome activity,
    and plot a 3-panel outcome-aligned heatmap. Returns the matplotlib Figure."""
    bin_edges = np.arange(t_start, t_end + bin_width / 2, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)

    emeta = ephys.metadata.copy()
    tranche_map = {'dSPN': 0, 'iSPN': 1}

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    # compute PSTHs for every unit, then filter uncategorized to top quartile
    all_units = list(ephys.keys())
    u2row_all = {u: i for i, u in enumerate(all_units)}
    n_all = len(all_units)

    def _psth_mat(event_list):
        mat = np.zeros((n_all, n_bins))
        if not event_list:
            return mat
        for ui, u in enumerate(all_units):
            spk = np.asarray(ephys[u].index)
            rows = np.zeros((len(event_list), n_bins))
            for ti, t in enumerate(event_list):
                h, _ = np.histogram(spk - t, bins=bin_edges)
                rows[ti] = h / bin_width
            mat[ui] = rows.mean(axis=0)
        return mat

    psth_all = {k: _psth_mat(v) for k, v in trial_events.items()}

    bl_mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    bl_mu = psth_all['no_stim'][:, bl_mask].mean(axis=1)
    bl_sd = psth_all['no_stim'][:, bl_mask].std(axis=1)
    bl_sd[bl_sd == 0] = 1.0
    zpsth_all = {k: (v - bl_mu[:, None]) / bl_sd[:, None] for k, v in psth_all.items()}

    sort_mask = (bin_centers >= sort_window[0]) & (bin_centers < sort_window[1])
    prelick_z_all = zpsth_all['no_stim'][:, sort_mask].mean(axis=1)

    # uncategorized: keep only top quartile by pre-lick z-score
    labeled_units = [u for u in all_units if _get_classif(u) is not None]
    unlabeled_units = [u for u in all_units if _get_classif(u) is None]
    if unlabeled_units:
        scores = np.array([prelick_z_all[u2row_all[u]] for u in unlabeled_units])
        threshold = np.nanpercentile(scores, 75)
        unlabeled_kept = [u for u in unlabeled_units if prelick_z_all[u2row_all[u]] >= threshold]
    else:
        unlabeled_kept = []

    unit_list = labeled_units + unlabeled_kept
    keep_idx = [u2row_all[u] for u in unit_list]
    n_units = len(unit_list)
    u2row = {u: i for i, u in enumerate(unit_list)}
    zpsth = {k: v[keep_idx] for k, v in zpsth_all.items()}
    prelick_z = prelick_z_all[np.array(keep_idx)]

    def _sort_key(u):
        t = tranche_map.get(_get_classif(u), 2)
        z = prelick_z[u2row[u]]
        return (t, -float(z) if np.isfinite(z) else np.inf)

    sorted_units = sorted(unit_list, key=_sort_key)
    sort_idx = [u2row[u] for u in sorted_units]
    zpsth_s = {k: v[sort_idx] for k, v in zpsth.items()}

    boundaries, prev_t = [], None
    for ri, u in enumerate(sorted_units):
        t = tranche_map.get(_get_classif(u), 2)
        if prev_t is not None and t != prev_t:
            boundaries.append(ri)
        prev_t = t

    tranche_labels = []
    for ii, b in enumerate([0] + boundaries + [n_units]):
        if ii == 0:
            continue
        start = ([0] + boundaries)[ii - 1]
        mid = (start + b) / 2.0
        cl = _get_classif(sorted_units[start])
        name = cl if cl in ('dSPN', 'iSPN') else 'Unlabeled'
        tranche_labels.append((mid, name))

    conditions = [
        ('no_stim', 'No Stim'),
        ('chr2', 'Blue Stim (ChR2)'),
        ('chrimson', 'Orange Stim (ChrimsonR)'),
    ]
    fig_h = max(6, n_units * 0.12 + 2)
    fig, axs = plt.subplots(1, 3, figsize=(18, fig_h), sharey=True)
    extent = [bin_centers[0], bin_centers[-1], n_units - 0.5, -0.5]

    im = None
    for ax, (ck, cl_label) in zip(axs, conditions):
        im = ax.imshow(
            zpsth_s[ck],
            aspect='auto', origin='upper',
            extent=extent,
            cmap='RdBu_r', vmin=clim[0], vmax=clim[1],
        )
        ax.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.9)
        for b in boundaries:
            ax.axhline(b - 0.5, color='white', linewidth=1.2, alpha=0.8)
        ax.set_xlabel('Time to first lick (s)', fontsize=15)
        ax.set_yticks([])
        ax.set_title(f'{cl_label}\n(n={len(trial_events[ck])} trials)', fontsize=15)
        if ax is axs[0]:
            for y, name in tranche_labels:
                ax.text(t_start - 0.05, y, name, va='center', ha='right',
                        fontsize=12, fontweight='bold', clip_on=False, rotation=90)
            ax.set_ylabel('Unit (sorted)', fontsize=15)
            ax.yaxis.set_label_coords(-0.15, 0.5)

    fig.suptitle('Firing rate aligned to outcome tone', fontsize=18)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    cbar_fig, cbar_ax = plt.subplots(figsize=(0.4, 3))
    plt.colorbar(im, cax=cbar_ax, label='Firing rate (z-score)')
    cbar_fig.tight_layout()
    if save_path is not None:
        cbar_path = save_path.parent / (save_path.stem + '_colorbar' + save_path.suffix)
        cbar_fig.savefig(cbar_path, dpi=150, bbox_inches='tight')

    return fig, cbar_fig


def plot_cue_aligned_by_latency(ephys, binary_signals, trial_events,
                                save_path=None, bin_width=0.1,
                                t_start=-1.0, t_end=6.0, n_quintiles=5):
    """Population-average firing rate aligned to start_cue, split into lick-latency quintiles.

    Layout: rows = conditions, cols = tranches (dSPN / iSPN / Unlabeled).
    Lines are coloured by mean lick latency of each quintile. Returns the Figure."""
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    # collect (cue_time, outcome_latency) per trial for each condition
    cond_trials = {}
    for cond, outcome_list in trial_events.items():
        trials = []
        for ot in outcome_list:
            prec = cue_ts[cue_ts < ot]
            if len(prec) == 0:
                continue
            ct = prec[-1]
            trials.append((ct, float(ot - ct)))
        cond_trials[cond] = trials

    # tranche assignment
    emeta = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    tranche_units = {}
    for u in ephys.keys():
        label = _get_classif(u) or 'Unlabeled'
        tranche_units.setdefault(label, []).append(u)
    tranche_order = [t for t in ('dSPN', 'iSPN', 'Unlabeled') if t in tranche_units]

    # pool all trials across conditions, tracking condition label and lick latency
    all_cue_times = np.array([t[0] for cond in cond_trials.values() for t in cond])
    all_cue_conds = [c  for c, trials in cond_trials.items() for _ in trials]
    all_cue_lats  = np.array([t[1] for cond in cond_trials.values() for t in cond])

    # sort by cue time - IntervalSet sorts intervals chronologically, so arrays must match
    _sort = np.argsort(all_cue_times)
    all_cue_times = all_cue_times[_sort]
    all_cue_conds = [all_cue_conds[i] for i in _sort]
    all_cue_lats  = all_cue_lats[_sort]
    n_total = len(all_cue_times)

    # restrict spikes to trial windows and bin once for all trials
    all_ep = nap.IntervalSet(start=all_cue_times + t_start, end=all_cue_times + t_end)
    binned_all = ephys.restrict(all_ep).count(bin_size=bin_width)*bin_width

    # extract per-trial firing-rate arrays: (n_trials, n_bins, n_units)
    trial_fr_list = []
    for s, e in zip(all_ep.start, all_ep.end):
        td = binned_all.restrict(nap.IntervalSet(start=s, end=e))
        trial_fr_list.append(td.values)
    trial_fr = np.array(trial_fr_list)            # (n_trials, n_bins, n_units)

    n_bins = trial_fr.shape[1]
    bin_centers = np.arange(n_bins) * bin_width + t_start + bin_width / 2
    unit_names = list(ephys.keys())
    u2col = {u: i for i, u in enumerate(unit_names)}

    # per-unit per-bin mean and SD across all pooled trials, then z-score everything
    mu_all = trial_fr.mean(axis=0)                # (n_bins, n_units)
    sd_all = trial_fr.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    trial_fr_z = (trial_fr - mu_all) / sd_all     # (n_trials, n_bins, n_units)

    # filter unlabeled to top quartile by mean pre-lick z-score from no_stim trials
    ns_idx = np.array([i for i, c in enumerate(all_cue_conds) if c == 'no_stim'])
    ns_lats = all_cue_lats[ns_idx]
    mean_lat_ns = float(ns_lats.mean()) if len(ns_idx) > 0 else 1.0
    pl_mask = (bin_centers >= mean_lat_ns - 0.2) & (bin_centers < mean_lat_ns)
    unit_prelick_z = trial_fr_z[ns_idx][:, pl_mask, :].mean(axis=(0, 1))  # (n_units,)

    unlabeled = tranche_units.get('Unlabeled', [])
    if unlabeled:
        ul_cols = np.array([u2col[u] for u in unlabeled])
        scores = unit_prelick_z[ul_cols]
        thresh = np.nanpercentile(scores, 75)
        tranche_units['Unlabeled'] = [u for u, s in zip(unlabeled, scores) if s >= thresh]

    # global latency range for consistent colour mapping
    lat_min = float(all_cue_lats.min()) if n_total > 0 else 0.0
    lat_max = float(all_cue_lats.max()) if n_total > 0 else 1.0
    cmap = plt.cm.turbo#plt.cm.plasma
    norm = plt.Normalize(vmin=lat_min, vmax=lat_max)

    conditions = [
        ('no_stim', 'No Stim'),
        ('chr2', 'Blue Stim (ChR2)'),
        ('chrimson', 'Orange Stim (ChrimsonR)'),
    ]
    n_rows = len(tranche_order)
    n_cols = len(conditions)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    for ri, tranche in enumerate(tranche_order):
        t_cols = np.array([u2col[u] for u in tranche_units[tranche]])

        for ci, (ck, cl_label) in enumerate(conditions):
            ax = axs[ri, ci]
            cond_idx = np.array([i for i, c in enumerate(all_cue_conds) if c == ck])

            if len(cond_idx) == 0:
                ax.text(0.5, 0.5, 'no trials', transform=ax.transAxes,
                        ha='center', va='center', color='grey', fontsize=15)
            else:
                lat_arr = all_cue_lats[cond_idx]
                sorted_idx = np.argsort(lat_arr)
                q_assign = np.empty(len(lat_arr), dtype=int)
                q_assign[sorted_idx] = np.arange(len(lat_arr)) * n_quintiles // len(lat_arr)
                mean_lats = [lat_arr[q_assign == q].mean() for q in range(n_quintiles)]

                for q in range(n_quintiles):
                    q_global = cond_idx[q_assign == q]
                    if len(q_global) == 0:
                        continue
                    pop_z = trial_fr_z[q_global][:, :, t_cols].mean(axis=(0, 2))
                    color = cmap(norm(mean_lats[q]))
                    ax.plot(bin_centers, pop_z, color=color, linewidth=1.5)
                    ax.axvline(mean_lats[q], color=color, linestyle='--', linewidth=1.0, alpha=0.7)

            ax.axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.75)
            if ri == 0:
                ax.set_title(cl_label, fontsize=17)
            if ci == 0:
                ax.set_ylabel(f'{tranche}\nFiring rate (z-score)', fontsize=15)
            if ri == n_rows - 1:
                ax.set_xlabel('Time from cue (s)', fontsize=15)

    for ax in axs.ravel():
        ax.label_outer()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.4, 3))
    cbar_fig.colorbar(sm, cax=cbar_ax, label='Mean lick latency (s)')
    cbar_fig.tight_layout()
    if save_path is not None:
        cbar_path = save_path.parent / (save_path.stem + '_colorbar' + save_path.suffix)
        cbar_fig.savefig(cbar_path, dpi=150, bbox_inches='tight')

    return fig, cbar_fig


def plot_outcome_heatmap_nostim(ephys, trial_events, save_path=None, bin_width=0.01,
                                t_start=-4.0, t_end=1.0,
                                baseline_window=(-4.0, -2.0), sort_window=(-0.20, 0.0),
                                clim=(-2, 5)):
    """No-stim-only outcome-aligned heatmap. Layout: 3 rows (dSPN / iSPN / Unlabeled) × 1 col."""
    bin_edges = np.arange(t_start, t_end + bin_width / 2, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)

    emeta = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    all_units = list(ephys.keys())
    n_all = len(all_units)

    ns_events = trial_events.get('no_stim', [])
    psth_ns = np.zeros((n_all, n_bins))
    if ns_events:
        for ui, u in enumerate(all_units):
            spk = np.asarray(ephys[u].index)
            rows = np.zeros((len(ns_events), n_bins))
            for ti, t in enumerate(ns_events):
                h, _ = np.histogram(spk - t, bins=bin_edges)
                rows[ti] = h / bin_width
            psth_ns[ui] = rows.mean(axis=0)

    bl_mask = (bin_centers >= baseline_window[0]) & (bin_centers < baseline_window[1])
    bl_mu = psth_ns[:, bl_mask].mean(axis=1)
    bl_sd = psth_ns[:, bl_mask].std(axis=1)
    bl_sd[bl_sd == 0] = 1.0
    zpsth = (psth_ns - bl_mu[:, None]) / bl_sd[:, None]

    sort_mask = (bin_centers >= sort_window[0]) & (bin_centers < sort_window[1])
    prelick_z = zpsth[:, sort_mask].mean(axis=1)
    u2row = {u: i for i, u in enumerate(all_units)}

    labeled = [u for u in all_units if _get_classif(u) is not None]
    unlabeled = [u for u in all_units if _get_classif(u) is None]
    if unlabeled:
        scores = np.array([prelick_z[u2row[u]] for u in unlabeled])
        thresh = np.nanpercentile(scores, 75)
        unlabeled = [u for u in unlabeled if prelick_z[u2row[u]] >= thresh]

    tranche_groups = {}
    for u in labeled:
        tranche_groups.setdefault(_get_classif(u), []).append(u)
    if unlabeled:
        tranche_groups['Unlabeled'] = unlabeled
    tranche_order = [t for t in ('dSPN', 'iSPN', 'Unlabeled') if t in tranche_groups]

    for label in tranche_order:
        tranche_groups[label].sort(
            key=lambda u: -prelick_z[u2row[u]] if np.isfinite(prelick_z[u2row[u]]) else np.inf
        )

    n_tranches = len(tranche_order)
    row_heights = [max(1.5, len(tranche_groups[t]) * 0.12) for t in tranche_order]
    fig, axs = plt.subplots(n_tranches, 1,
                            figsize=(7, sum(row_heights) + 1.5),
                            gridspec_kw={'height_ratios': row_heights},
                            squeeze=False)

    im = None
    for ri, label in enumerate(tranche_order):
        ax = axs[ri, 0]
        units = tranche_groups[label]
        n_u = len(units)
        idx = [u2row[u] for u in units]
        z = zpsth[idx]
        ext = [bin_centers[0], bin_centers[-1], n_u - 0.5, -0.5]
        im = ax.imshow(z, aspect='auto', origin='upper', extent=ext,
                       cmap='RdBu_r', vmin=clim[0], vmax=clim[1])
        ax.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.set_yticks([])
        ax.set_ylabel(f'{label}  (n={n_u})', fontsize=15)
        if ri == n_tranches - 1:
            ax.set_xlabel('Time to outcome (s)', fontsize=15)

    fig.suptitle(f'Firing rate aligned to outcome - No Stim\n(n={len(ns_events)} trials)', fontsize=18)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    cbar_fig, cbar_ax = plt.subplots(figsize=(0.4, 3))
    plt.colorbar(im, cax=cbar_ax, label='Firing rate (z-score)')
    cbar_fig.tight_layout()
    if save_path is not None:
        cbar_path = save_path.parent / (save_path.stem + '_colorbar' + save_path.suffix)
        cbar_fig.savefig(cbar_path, dpi=150, bbox_inches='tight')

    return fig, cbar_fig


def plot_cue_aligned_nostim(ephys, binary_signals, trial_events,
                            save_path=None, bin_width=0.1,
                            t_start=-1.0, t_end=6.0, n_quintiles=5):
    """No-stim-only cue-aligned population PSTH. Layout: 3 rows (dSPN / iSPN / Unlabeled) × 1 col."""
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    ns_outcomes = trial_events.get('no_stim', [])
    trials = []
    for ot in ns_outcomes:
        prec = cue_ts[cue_ts < ot]
        if len(prec) == 0:
            continue
        ct = prec[-1]
        trials.append((ct, float(ot - ct)))

    emeta = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    tranche_units = {}
    for u in ephys.keys():
        label = _get_classif(u) or 'Unlabeled'
        tranche_units.setdefault(label, []).append(u)
    tranche_order = [t for t in ('dSPN', 'iSPN', 'Unlabeled') if t in tranche_units]

    cue_times = np.array([t[0] for t in trials])
    lats = np.array([t[1] for t in trials])
    sort_order = np.argsort(cue_times)
    cue_times = cue_times[sort_order]
    lats = lats[sort_order]
    n_total = len(cue_times)

    if n_total == 0:
        fig, axs = plt.subplots(len(tranche_order), 1, figsize=(6, 3 * len(tranche_order)), squeeze=False)
        for ax in axs.ravel():
            ax.text(0.5, 0.5, 'no trials', transform=ax.transAxes, ha='center', va='center', color='grey', fontsize=15)
        return fig, None

    all_ep = nap.IntervalSet(start=cue_times + t_start, end=cue_times + t_end)
    binned_all = ephys.restrict(all_ep).count(bin_size=bin_width) * bin_width

    trial_fr_list = []
    for s, e in zip(all_ep.start, all_ep.end):
        td = binned_all.restrict(nap.IntervalSet(start=s, end=e))
        trial_fr_list.append(td.values)
    trial_fr = np.array(trial_fr_list)  # (n_trials, n_bins, n_units)

    n_bins = trial_fr.shape[1]
    bin_centers = np.arange(n_bins) * bin_width + t_start + bin_width / 2
    unit_names = list(ephys.keys())
    u2col = {u: i for i, u in enumerate(unit_names)}

    mu_all = trial_fr.mean(axis=0)
    sd_all = trial_fr.std(axis=0)
    sd_all[sd_all == 0] = 1.0
    trial_fr_z = (trial_fr - mu_all) / sd_all

    # filter unlabeled to top quartile by pre-lick z-score
    mean_lat = float(lats.mean())
    pl_mask = (bin_centers >= mean_lat - 0.2) & (bin_centers < mean_lat)
    unit_prelick_z = trial_fr_z[:, pl_mask, :].mean(axis=(0, 1))
    unlabeled = tranche_units.get('Unlabeled', [])
    if unlabeled:
        ul_cols = np.array([u2col[u] for u in unlabeled])
        scores = unit_prelick_z[ul_cols]
        thresh = np.nanpercentile(scores, 75)
        tranche_units['Unlabeled'] = [u for u, s in zip(unlabeled, scores) if s >= thresh]

    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=float(lats.min()), vmax=float(lats.max()))

    sorted_idx = np.argsort(lats)
    q_assign = np.empty(n_total, dtype=int)
    q_assign[sorted_idx] = np.arange(n_total) * n_quintiles // n_total
    mean_lats = [lats[q_assign == q].mean() for q in range(n_quintiles)]

    n_rows = len(tranche_order)
    fig, axs = plt.subplots(n_rows, 1, figsize=(6, 3.5 * n_rows), sharex=True, sharey=True, squeeze=False)

    for ri, tranche in enumerate(tranche_order):
        ax = axs[ri, 0]
        t_cols = np.array([u2col[u] for u in tranche_units[tranche]])
        for q in range(n_quintiles):
            q_idx = np.where(q_assign == q)[0]
            if len(q_idx) == 0:
                continue
            pop_z = trial_fr_z[q_idx][:, :, t_cols].mean(axis=(0, 2))
            color = cmap(norm(mean_lats[q]))
            ax.plot(bin_centers, pop_z, color=color, linewidth=1.5)
            ax.axvline(mean_lats[q], color=color, linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.75)
        ax.set_ylabel(f'{tranche}\nFiring rate (z-score)', fontsize=15)
        if ri == n_rows - 1:
            ax.set_xlabel('Time from cue (s)', fontsize=15)

    for ax in axs.ravel():
        ax.label_outer()

    fig.suptitle(f'Cue-aligned firing rate - No Stim\n(n={n_total} trials)', fontsize=18)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.4, 3))
    cbar_fig.colorbar(sm, cax=cbar_ax, label='Mean lick latency (s)')
    cbar_fig.tight_layout()
    if save_path is not None:
        cbar_path = save_path.parent / (save_path.stem + '_colorbar' + save_path.suffix)
        cbar_fig.savefig(cbar_path, dpi=150, bbox_inches='tight')

    return fig, cbar_fig


def plot_dspn_isPN_diff(ephys, binary_signals, trial_events,
                        save_path=None, bin_width=0.1,
                        t_start=-1.0, t_end=6.0, n_quintiles=5,
                        t_start_outcome=-8.0, t_end_outcome=1.0):
    """dSPN - iSPN population z-score difference. Produces 4 separate figures:
      1. All conditions, outcome-aligned mean trace per condition  [-8, +1 s]
      2. No-stim only,  outcome-aligned mean trace               [-8, +1 s]
      3. All conditions, cue-aligned, split by lick-latency quintile
      4. No-stim only,  cue-aligned, split by lick-latency quintile
    Returns (fig_allcond_mean, fig_nostim_mean, fig_allcond_lat, fig_nostim_lat, cbar_fig)."""
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    emeta = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    d_units = [u for u in ephys.keys() if _get_classif(u) == 'dSPN']
    i_units = [u for u in ephys.keys() if _get_classif(u) == 'iSPN']
    if not d_units or not i_units:
        print('plot_dspn_isPN_diff: no dSPN or iSPN units found')
        return None, None, None, None, None

    u2col = {u: i for i, u in enumerate(ephys.keys())}
    d_cols = np.array([u2col[u] for u in d_units])
    i_cols = np.array([u2col[u] for u in i_units])


    cond_style = {
        'no_stim':  ('k',          'solid',  'No Stim'),
        'chr2':     ('royalblue',  'dashed', 'ChR2 (blue)'),
        'chrimson': ('darkorange', 'dotted', 'ChrimsonR (orange)'),
    }
    title_base = f'dSPN - iSPN\n(n dSPN={len(d_units)}, n iSPN={len(i_units)})'

    def _bin_trials(times):
        """Bin spike counts for each trial window, return (trial_fr_z, bin_centers, n_bins)."""
        ep = nap.IntervalSet(start=times + t_start_outcome, end=times + t_end_outcome)
        binned = ephys.restrict(ep).count(bin_size=bin_width) * bin_width
        fr_list = []
        for s, e in zip(ep.start, ep.end):
            td = binned.restrict(nap.IntervalSet(start=s, end=e))
            fr_list.append(td.values)
        fr = np.array(fr_list)
        nb = fr.shape[1]
        bc = np.arange(nb) * bin_width + t_start_outcome + bin_width / 2
        mu = fr.mean(axis=0);  sd = fr.std(axis=0);  sd[sd == 0] = 1.0
        return (fr - mu) / sd, bc, nb

    def _diff(fr_z, nb, trial_idx):
        d = fr_z[np.ix_(trial_idx, np.arange(nb), d_cols)].mean(axis=(0, 2))
        i = fr_z[np.ix_(trial_idx, np.arange(nb), i_cols)].mean(axis=(0, 2))
        return d - i

    def _ax_decor(ax, xlabel):
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax.set_ylabel('dSPN - iSPN\n(z-score)', fontsize=15)
        ax.set_xlabel(xlabel, fontsize=15)

    # ── Outcome-aligned data (figs 1 & 2) ────────────────────────────────────
    out_times_list, out_conds_list = [], []
    for ck in ('no_stim', 'chr2', 'chrimson'):
        ots = np.array(trial_events.get(ck, []))
        out_times_list.append(ots)
        out_conds_list.extend([ck] * len(ots))
    all_out_times = np.concatenate(out_times_list)
    sort_ord = np.argsort(all_out_times)
    all_out_times = all_out_times[sort_ord]
    all_out_conds = [out_conds_list[i] for i in sort_ord]

    out_fr_z, out_bc, out_nb = _bin_trials(all_out_times)
    ns_out_idx = np.array([i for i, c in enumerate(all_out_conds) if c == 'no_stim'])

    # ── Figure 1: all conditions, outcome-aligned mean ────────────────────────
    fig1, ax = plt.subplots(figsize=(7, 4))
    for ck, (color, ls, label) in cond_style.items():
        idx = np.array([i for i, c in enumerate(all_out_conds) if c == ck])
        if len(idx) == 0:
            continue
        ax.plot(out_bc, _diff(out_fr_z, out_nb, idx),
                color=color, linestyle=ls, linewidth=1.5, label=label)

    _ax_decor(ax, 'Time from first lick (s)')
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig1.suptitle(f'{title_base}\nAll conditions - first lick-aligned mean', fontsize=17)
    fig1.tight_layout()
    if save_path is not None:
        fig1.savefig(save_path.parent / (save_path.stem + '_allcond_mean' + save_path.suffix),
                     dpi=150, bbox_inches='tight')

    # ── Figure 2: no-stim only, outcome-aligned mean ──────────────────────────
    fig2, ax = plt.subplots(figsize=(7, 4))
    if len(ns_out_idx) > 0:
        ax.plot(out_bc, _diff(out_fr_z, out_nb, ns_out_idx), color='k', linewidth=1.5)

    _ax_decor(ax, 'Time from first lick (s)')
    fig2.suptitle(f'{title_base}\nNo Stim only - first lick-aligned mean\n(n={len(ns_out_idx)} trials)',
                  fontsize=17)
    fig2.tight_layout()
    if save_path is not None:
        fig2.savefig(save_path.parent / (save_path.stem + '_nostim_mean' + save_path.suffix),
                     dpi=150, bbox_inches='tight')

    # ── Cue-aligned data (figs 3 & 4) ────────────────────────────────────────
    cond_trials = {}
    for cond, outcome_list in trial_events.items():
        trials = []
        for ot in outcome_list:
            prec = cue_ts[cue_ts < ot]
            if len(prec) == 0:
                continue
            ct = prec[-1]
            trials.append((ct, float(ot - ct)))
        cond_trials[cond] = trials

    all_cue_times = np.array([t[0] for cond in cond_trials.values() for t in cond])
    all_cue_conds = [c for c, trials in cond_trials.items() for _ in trials]
    all_cue_lats  = np.array([t[1] for cond in cond_trials.values() for t in cond])
    sort_order = np.argsort(all_cue_times)
    all_cue_times = all_cue_times[sort_order]
    all_cue_conds = [all_cue_conds[i] for i in sort_order]
    all_cue_lats  = all_cue_lats[sort_order]

    cue_ep = nap.IntervalSet(start=all_cue_times + t_start, end=all_cue_times + t_end)
    cue_binned = ephys.restrict(cue_ep).count(bin_size=bin_width) * bin_width
    cue_fr_list = []
    for s, e in zip(cue_ep.start, cue_ep.end):
        td = cue_binned.restrict(nap.IntervalSet(start=s, end=e))
        cue_fr_list.append(td.values)
    cue_fr = np.array(cue_fr_list)
    cue_nb = cue_fr.shape[1]
    cue_bc = np.arange(cue_nb) * bin_width + t_start + bin_width / 2
    cue_mu = cue_fr.mean(axis=0);  cue_sd = cue_fr.std(axis=0);  cue_sd[cue_sd == 0] = 1.0
    cue_fr_z = (cue_fr - cue_mu) / cue_sd

    cmap = plt.cm.turbo
    norm = plt.Normalize(vmin=float(all_cue_lats.min()), vmax=float(all_cue_lats.max()))
    ns_cue_idx = np.array([i for i, c in enumerate(all_cue_conds) if c == 'no_stim'])

    def _quintile_lines(ax, trial_idx, lats):
        sorted_idx = np.argsort(lats)
        q_assign = np.empty(len(lats), dtype=int)
        q_assign[sorted_idx] = np.arange(len(lats)) * n_quintiles // len(lats)
        for q in range(n_quintiles):
            q_sub = np.where(q_assign == q)[0]
            if len(q_sub) == 0:
                continue
            mean_lat = lats[q_sub].mean()
            color = cmap(norm(mean_lat))
            ax.plot(cue_bc, _diff(cue_fr_z, cue_nb, trial_idx[q_sub]), color=color, linewidth=1.5)
            ax.axvline(mean_lat, color=color, linestyle='--', linewidth=0.8, alpha=0.6)

    # ── Figure 3: all conditions, cue-aligned, latency-binned ────────────────
    fig3, ax = plt.subplots(figsize=(7, 4))
    for ck, (color, ls, label) in cond_style.items():
        idx = np.array([i for i, c in enumerate(all_cue_conds) if c == ck])
        if len(idx) == 0:
            continue
        lats = all_cue_lats[idx]
        sorted_idx = np.argsort(lats)
        q_assign = np.empty(len(lats), dtype=int)
        q_assign[sorted_idx] = np.arange(len(lats)) * n_quintiles // len(lats)
        for q in range(n_quintiles):
            q_sub = np.where(q_assign == q)[0]
            if len(q_sub) == 0:
                continue
            mean_lat = lats[q_sub].mean()
            tcolor = cmap(norm(mean_lat))
            ax.plot(cue_bc, _diff(cue_fr_z, cue_nb, idx[q_sub]),
                    color=tcolor, linestyle=ls, linewidth=1.2)
            ax.axvline(mean_lat, color=tcolor, linestyle='--', linewidth=0.6, alpha=0.5)

    _ax_decor(ax, 'Time from cue (s)')
    for ck, (color, ls, label) in cond_style.items():
        ax.plot([], [], color='grey', linestyle=ls, linewidth=1.2, label=label)
    ax.legend(fontsize=12, title='Linestyle = condition\nColor = latency quintile', title_fontsize=11,
              loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    fig3.suptitle(f'{title_base}\nAll conditions - cue-aligned, by latency quintile', fontsize=17)
    fig3.tight_layout()
    if save_path is not None:
        fig3.savefig(save_path.parent / (save_path.stem + '_allcond_latency' + save_path.suffix),
                     dpi=150, bbox_inches='tight')

    # ── Figure 4: no-stim only, cue-aligned, latency-binned ──────────────────
    fig4, ax = plt.subplots(figsize=(7, 4))
    if len(ns_cue_idx) > 0:
        _quintile_lines(ax, ns_cue_idx, all_cue_lats[ns_cue_idx])

    _ax_decor(ax, 'Time from cue (s)')
    fig4.suptitle(f'{title_base}\nNo Stim only - cue-aligned, latency quintile\n(n={len(ns_cue_idx)} trials)',
                  fontsize=17)
    fig4.tight_layout()
    if save_path is not None:
        fig4.savefig(save_path.parent / (save_path.stem + '_nostim_latency' + save_path.suffix),
                     dpi=150, bbox_inches='tight')

    # ── Shared colorbar for latency figures ───────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.4, 3))
    cbar_fig.colorbar(sm, cax=cbar_ax, label='Mean lick latency (s)')
    cbar_fig.tight_layout()
    if save_path is not None:
        cbar_fig.savefig(save_path.parent / (save_path.stem + '_latency_colorbar' + save_path.suffix),
                         dpi=150, bbox_inches='tight')

    return fig1, fig2, fig3, fig4, cbar_fig


def plot_trial_raster(binary_signals,
                      ch_on=None, ch_off=None, c2_on=None, c2_off=None,
                      ch_intensities=None, c2_intensities=None,
                      save_path=None, t_start=-1.0, t_end=8.0):
    """Trial raster: one row per start cue, aligned at t=0.

    - Black hash: licks
    - Green dot: reward tone onset
    - Red dot: error tone onset
    - Blue band: chr2 opto stim (alpha scaled by intensity if provided)
    - Orange band: chrimson opto stim (alpha scaled by intensity if provided)
    """
    import matplotlib.patches as mpatches

    bs_meta = binary_signals.metadata

    def _ts(event_name):
        idx = bs_meta[bs_meta['event'] == event_name].index
        return binary_signals[idx[0]].t if len(idx) > 0 else np.array([])

    cue_ts = _ts('start_cues')
    rew_ts = _ts('reward_tone_on')
    err_ts = _ts('error_tone_on')

    # collect all lick-related timestamps
    lick_keys = bs_meta[bs_meta['event'].str.contains('lick', na=False, case=False)].index
    lick_ts = (np.sort(np.concatenate([binary_signals[k].t for k in lick_keys]))
               if len(lick_keys) > 0 else np.array([]))

    ch_on  = np.asarray(ch_on  if ch_on  is not None else [])
    ch_off = np.asarray(ch_off if ch_off is not None else [])
    c2_on  = np.asarray(c2_on  if c2_on  is not None else [])
    c2_off = np.asarray(c2_off if c2_off is not None else [])

    def _alpha_arr(intensities, default=0.3):
        if intensities is None or len(intensities) == 0:
            return np.full(max(len(ch_on), len(c2_on)), default)
        arr = np.asarray(intensities, dtype=float)
        span = arr.max() - arr.min()
        norm = (arr - arr.min()) / span if span > 0 else np.ones_like(arr)
        return 0.12 + 0.38 * norm

    ch_alphas = _alpha_arr(ch_intensities) if len(ch_on) > 0 else np.array([])
    c2_alphas = _alpha_arr(c2_intensities) if len(c2_on) > 0 else np.array([])

    n_trials = len(cue_ts)
    fig_h = max(4, min(n_trials * 0.15 + 1, 40))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    stim_layers = [
        (ch_on, ch_off, (0.1, 0.4, 1.0), ch_alphas),
        (c2_on, c2_off, (1.0, 0.5, 0.0), c2_alphas),
    ]

    for ti, ct in enumerate(cue_ts):
        ws, we = ct + t_start, ct + t_end

        for son, soff, rgb, alphas in stim_layers:
            for si in range(len(son)):
                ton, toff = son[si], soff[si]
                if toff < ws or ton > we:
                    continue
                x0 = max(ton, ws) - ct
                x1 = min(toff, we) - ct
                ax.add_patch(mpatches.Rectangle(
                    (x0, ti - 0.5), x1 - x0, 1.0,
                    facecolor=rgb, alpha=float(alphas[si]) if si < len(alphas) else 0.3,
                    linewidth=0, zorder=1,
                ))

        licks_in = lick_ts[(lick_ts >= ws) & (lick_ts < we)]
        if len(licks_in):
            ax.plot(licks_in - ct, np.full(len(licks_in), ti),
                    '|', color='k', markersize=6, markeredgewidth=1.0, zorder=2)

        rews_in = rew_ts[(rew_ts >= ws) & (rew_ts < we)]
        if len(rews_in):
            ax.plot(rews_in - ct, np.full(len(rews_in), ti),
                    'o', color='limegreen', markersize=4, zorder=3)

        errs_in = err_ts[(err_ts >= ws) & (err_ts < we)]
        if len(errs_in):
            ax.plot(errs_in - ct, np.full(len(errs_in), ti),
                    'o', color='red', markersize=4, zorder=3)

    ax.set_xlim(t_start, t_end)
    ax.set_ylim(n_trials - 0.5, -0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(3.33, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Time from cue (s)', fontsize=15)
    ax.set_ylabel('Trial', fontsize=15)

    legend_handles = [
        plt.Line2D([0], [0], marker='|', color='k', linestyle='none',
                   markersize=8, markeredgewidth=1.5, label='Lick'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                   markersize=6, label='Reward'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=6, label='Early'),
        mpatches.Patch(facecolor=(0.1, 0.4, 1.0), alpha=0.5, label='Blue/ChR2 stim'),
        mpatches.Patch(facecolor=(1.0, 0.5, 0.0), alpha=0.5, label='Orange/ChrimsonR stim'),
    ]
    ax.legend(handles=legend_handles, fontsize=12, ncol=1, framealpha=0.9,
              loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    fig.suptitle(f'Trial raster  (n={n_trials} trials)', fontsize=17)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_lick_latency(binary_signals, trial_events, save_path=None):
    """Bar graph and CDF of cue-to-first-lick latency by condition.

    Bar graph: mean +/- SEM per condition with pairwise t-tests.
    CDF: all three conditions overlaid with pairwise KS tests annotated.
    Returns (fig_bar, fig_cdf)."""
    from scipy import stats

    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    cond_order  = ['no_stim', 'chr2', 'chrimson']
    cond_labels = {'no_stim': 'No Stim', 'chr2': 'ChR2\n(blue)', 'chrimson': 'ChrimsonR\n(orange)'}
    cond_colors = {'no_stim': 'grey', 'chr2': 'royalblue', 'chrimson': 'darkorange'}

    lats = {}
    for ck in cond_order:
        arr = []
        for ot in trial_events.get(ck, []):
            prec = cue_ts[cue_ts < ot]
            if len(prec) > 0:
                arr.append(float(ot - prec[-1]))
        lats[ck] = np.array(arr)

    # ── Bar graph ─────────────────────────────────────────────────────────────
    means = [lats[ck].mean() if len(lats[ck]) > 0 else 0.0 for ck in cond_order]
    sems  = [lats[ck].std(ddof=1) / np.sqrt(len(lats[ck])) if len(lats[ck]) > 1 else 0.0
             for ck in cond_order]

    fig_bar, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(3), means, yerr=sems, capsize=5, width=0.5,
           color=[cond_colors[ck] for ck in cond_order],
           edgecolor='k', linewidth=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([cond_labels[ck] for ck in cond_order], fontsize=15)
    ax.set_ylabel('Latency to\nfirst lick (s)', fontsize=15)

    y_top   = max(m + s for m, s in zip(means, sems))
    y_range = max(means) - min(means) if max(means) != min(means) else 1.0
    for level, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        a, b = lats[cond_order[i]], lats[cond_order[j]]
        if len(a) < 2 or len(b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(a, b)
        p_str = f'p={p_val:.3f}' if p_val >= 0.001 else f'p={p_val:.2e}'
        bh   = y_top + y_range * (0.18 + level * 0.20)
        tick = y_range * 0.04
        ax.plot([i, i, j, j], [bh - tick, bh, bh, bh - tick], color='k', linewidth=0.8)
        ax.text((i + j) / 2, bh + tick * 0.3, p_str, ha='center', va='bottom', fontsize=12)

    fig_bar.suptitle('Latency to first lick\nby condition', fontsize=17)
    fig_bar.tight_layout()
    if save_path is not None:
        fig_bar.savefig(save_path.parent / (save_path.stem + '_bar' + save_path.suffix),
                        dpi=150, bbox_inches='tight')

    # ── CDF ───────────────────────────────────────────────────────────────────
    fig_cdf, ax = plt.subplots(figsize=(6, 4))
    for ck in cond_order:
        if len(lats[ck]) == 0:
            continue
        sl = np.sort(lats[ck])
        ax.plot(sl, np.arange(1, len(sl) + 1) / len(sl),
                color=cond_colors[ck], linewidth=1.5,
                label=f"{cond_labels[ck].replace(chr(10), ' ')} (n={len(sl)})")

    ax.set_xlabel('Latency to\nfirst lick (s)', fontsize=15)
    ax.set_ylabel('Cumulative probability', fontsize=15)
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ks_lines = []
    for a_key, b_key in [('no_stim', 'chr2'), ('no_stim', 'chrimson'), ('chr2', 'chrimson')]:
        a, b = lats[a_key], lats[b_key]
        if len(a) < 2 or len(b) < 2:
            continue
        ks_stat, ks_p = stats.ks_2samp(a, b)
        p_str = f'{ks_p:.3f}' if ks_p >= 0.001 else f'{ks_p:.2e}'
        la = cond_labels[a_key].replace('\n', ' ')
        lb = cond_labels[b_key].replace('\n', ' ')
        ks_lines.append(f'{la} vs {lb}:  KS={ks_stat:.3f}, p={p_str}')

    if ks_lines:
        ax.text(0.97, 0.05, '\n'.join(ks_lines), transform=ax.transAxes,
                ha='right', va='bottom', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    fig_cdf.suptitle('CDF of latency\nto first lick', fontsize=17)
    fig_cdf.tight_layout()
    if save_path is not None:
        fig_cdf.savefig(save_path.parent / (save_path.stem + '_cdf' + save_path.suffix),
                        dpi=150, bbox_inches='tight')

    return fig_bar, fig_cdf


def plot_dspn_isPN_bar(ephys, binary_signals, trial_events,
                       save_path=None, pre_window=(-0.5, -0.01)):
    """Bar graph: dSPN - iSPN mean population firing rate in pre-cue vs pre-first-lick window.

    Computed per no-stim trial; paired t-test across trials."""
    from scipy import stats

    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    emeta = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    d_units = [u for u in ephys.keys() if _get_classif(u) == 'dSPN']
    i_units = [u for u in ephys.keys() if _get_classif(u) == 'iSPN']
    if not d_units or not i_units:
        print('plot_dspn_isPN_bar: no dSPN or iSPN units found')
        return None

    d_spks = [np.asarray(ephys[u].index) for u in d_units]
    i_spks = [np.asarray(ephys[u].index) for u in i_units]
    win_dur = pre_window[1] - pre_window[0]

    def _pop_diff(ref_t):
        t0, t1 = ref_t + pre_window[0], ref_t + pre_window[1]
        d_rates = [np.sum((s >= t0) & (s < t1)) / win_dur for s in d_spks]
        i_rates = [np.sum((s >= t0) & (s < t1)) / win_dur for s in i_spks]
        return float(np.mean(d_rates) - np.mean(i_rates))

    precue_diffs, prelick_diffs = [], []
    for ot in trial_events.get('no_stim', []):
        prec = cue_ts[cue_ts < ot]
        if len(prec) == 0:
            continue
        precue_diffs.append(_pop_diff(prec[-1]))
        prelick_diffs.append(_pop_diff(ot))

    precue_diffs    = np.array(precue_diffs)
    prelick_diffs = np.array(prelick_diffs)
    n = len(precue_diffs)

    t_stat, p_val = stats.ttest_rel(precue_diffs, prelick_diffs)

    means = [precue_diffs.mean(), prelick_diffs.mean()]
    sems  = [precue_diffs.std(ddof=1) / np.sqrt(n),
             prelick_diffs.std(ddof=1) / np.sqrt(n)]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([0, 1], means, yerr=sems, capsize=5, width=0.5,
           color=['steelblue', 'salmon'], edgecolor='k', linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-cue\n(-500 to -10 ms)', 'Pre-first lick\n(-500 to -10 ms)'], fontsize=15)
    ax.set_ylabel('dSPN - iSPN\nfiring rate (Hz)', fontsize=15)

    # significance bracket
    y_tops = [m + s for m, s in zip(means, sems)]
    y_bot  = min(m - s for m, s in zip(means, sems))
    bracket_y = max(y_tops) + (max(y_tops) - y_bot) * 0.12
    tick_h    = (max(y_tops) - y_bot) * 0.04
    ax.plot([0, 0, 1, 1], [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
            color='k', linewidth=1.0)
    p_str = f'p = {p_val:.3f}' if p_val >= 0.001 else f'p = {p_val:.2e}'
    ax.text(0.5, bracket_y + tick_h * 0.5, p_str, ha='center', va='bottom', fontsize=14)

    fig.suptitle(f'dSPN - iSPN population firing rate\nNo Stim trials (n={n},'
                 f' t={t_stat:.2f})', fontsize=17)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def print_opto_timing_check(trial_events, binary_signals, ch_on, ch_off, c2_on, c2_off):
    """Print sanity-check timing: opto_on - cue and opto_off - outcome tone (ms)."""
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    for stim_key, stim_on, stim_off in [('chrimson', ch_on, ch_off), ('chr2', c2_on, c2_off)]:
        events = np.array(trial_events[stim_key])
        if len(events) == 0 or len(stim_on) == 0:
            print(f"{stim_key}: no trials found")
            continue
        on_cue_deltas, off_outcome_deltas = [], []
        for ot in events:
            prev_cues = cue_ts[cue_ts < ot]
            if len(prev_cues) == 0:
                continue
            ct_prev = prev_cues[-1]
            trial_on = stim_on[(stim_on >= ct_prev - 2.0) & (stim_on <= ot)]
            if len(trial_on) == 0:
                continue
            cues_after_on = cue_ts[cue_ts >= trial_on[0]]
            ref_cue = cues_after_on[0] if len(cues_after_on) > 0 else ct_prev
            on_cue_deltas.append((trial_on[0] - ref_cue) * 1000)
            trial_off = stim_off[(stim_off >= trial_on[0]) & (stim_off > ct_prev)]
            if len(trial_off) > 0:
                nearest_off = trial_off[np.argmin(np.abs(trial_off - ot))]
                off_outcome_deltas.append((nearest_off - ot) * 1000)
        if on_cue_deltas:
            d = np.array(on_cue_deltas)
            print(f"{stim_key}  opto_on - cue:      mean={np.mean(d):.1f} ms  std={np.std(d):.1f} ms  n={len(d)}")
        if off_outcome_deltas:
            d = np.array(off_outcome_deltas)
            print(f"{stim_key}  opto_off - outcome: mean={np.mean(d):.1f} ms  std={np.std(d):.1f} ms  n={len(d)}")
