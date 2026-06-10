import numpy as np
import pandas as pd
import pynapple as nap
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import plot_size_utils as psu

def _make_dark_cmap(name, darken=0.95):
    colors = plt.get_cmap(name)(np.linspace(0, 1, 256))
    colors[:, :3] *= darken
    return mcolors.LinearSegmentedColormap.from_list(f'{name}_dark', colors)

_jet_dark   = _make_dark_cmap('jet')
_turbo_dark = _make_dark_cmap('turbo')

plt.rcParams.update({
    'font.size':        8,
    'axes.titlesize':   8,
    'axes.labelsize':   8,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'figure.titlesize': 8,
})


def add_all_licks(binary_signals, min_cue_latency=0.1):
    """Combine rewarded (key 1) and early (key 2) first-lick timestamps into 'all_first_licks'.

    Licks occurring within min_cue_latency seconds of the preceding cue are excluded
    (default 100 ms) — these are reflexive responses that precede any sensory processing."""
    bs_event_meta = binary_signals.metadata[['event']].copy()
    all_lick_t = np.sort(np.concatenate([binary_signals[1].t, binary_signals[2].t]))

    cue_rows = bs_event_meta[bs_event_meta['event'] == 'start_cues']
    if len(cue_rows) > 0:
        cue_t = binary_signals[cue_rows.index[0]].t
        keep = np.ones(len(all_lick_t), dtype=bool)
        for i, lt in enumerate(all_lick_t):
            preceding = cue_t[cue_t <= lt]
            if len(preceding) > 0 and (lt - preceding[-1]) < min_cue_latency:
                keep[i] = False
        all_lick_t = all_lick_t[keep]

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


def id_lick_bout_modulated(ephys, binary_signals, trial_events,
                           pre_window=0.1, n_shuffles=1000, p_threshold=0.05):
    """Test whether each unit significantly increases firing in the pre_window seconds
    before each lick_bout_starts event, relative to a shuffle null distribution.

    Analysis is restricted to in-task periods with opto stimulation excluded via
    get_trial_ephys. Shuffle windows are sampled uniformly from the same valid intervals.
    One-tailed p-value is the fraction of shuffle means >= actual mean.

    Returns a DataFrame with columns:
        unit_id, modulation_stat (actual - shuffle mean, Hz), p_val, significant (p < p_threshold).
    """
    bs_meta = binary_signals.metadata
    bout_rows = bs_meta[bs_meta['event'] == 'lick_bout_starts']
    if len(bout_rows) == 0:
        raise ValueError("No 'lick_bout_starts' event found in binary_signals metadata")

    task_ephys = get_trial_ephys(ephys, binary_signals, trial_events, exclude_opto=True)

    # valid intervals: starts and ends from the restricted time support
    ts = task_ephys.time_support
    iv_starts = np.asarray(ts.start)
    iv_ends = np.asarray(ts.end)
    iv_durs = iv_ends - iv_starts
    # only keep intervals long enough to hold a pre_window
    valid = iv_durs > pre_window
    iv_starts, iv_ends, iv_durs = iv_starts[valid], iv_ends[valid], iv_durs[valid]
    # sample-able range within each interval is [start + pre_window, end]
    iv_sample_durs = iv_ends - (iv_starts + pre_window)
    iv_sample_durs = np.maximum(iv_sample_durs, 0.0)
    cum_dur = np.concatenate([[0.0], np.cumsum(iv_sample_durs)])
    total_sample_dur = cum_dur[-1]

    # filter bout_times to those within a valid interval
    bout_times_all = binary_signals[bout_rows.index[0]].t
    in_task = np.zeros(len(bout_times_all), dtype=bool)
    for s, e in zip(iv_starts, iv_ends):
        in_task |= (bout_times_all >= s) & (bout_times_all <= e)
    bout_times = bout_times_all[in_task]
    n_bouts = len(bout_times)
    if n_bouts == 0:
        raise ValueError("No lick_bout_starts events fall within valid task intervals")

    # draw (n_shuffles * n_bouts) random times from valid intervals
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, total_sample_dur, size=n_shuffles * n_bouts)
    iv_idx = np.searchsorted(cum_dur[1:], u)
    rt_flat = (iv_starts + pre_window)[iv_idx] + (u - cum_dur[iv_idx])

    records = []
    for unit in task_ephys.keys():
        spk = np.sort(np.asarray(task_ephys[unit].index))

        actual_counts = np.searchsorted(spk, bout_times) - np.searchsorted(spk, bout_times - pre_window)
        actual_mean = float(actual_counts.mean()) / pre_window

        shuf_counts = np.searchsorted(spk, rt_flat) - np.searchsorted(spk, rt_flat - pre_window)
        shuf_means = shuf_counts.reshape(n_shuffles, n_bouts).mean(axis=1) / pre_window

        p_val = float(np.mean(shuf_means >= actual_mean))
        modulation_stat = actual_mean - float(shuf_means.mean())

        records.append({
            'unit_id': unit,
            'modulation_stat': modulation_stat,
            'p_val': p_val,
            'significant': p_val < p_threshold,
        })

    return pd.DataFrame(records)


def categorize_outcome_trials(binary_signals, ch_on, ch_off, c2_on, c2_off):
    """Categorize each outcome-tone event as no_stim, chr2, or chrimson.

    A trial is stim if opto_on falls in [cue-2s, outcome] AND opto_off falls
    within 200 ms before the outcome tone. Returns dict of outcome timestamps per condition."""
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    def _find_key(meta, *candidates):
        for name in candidates:
            rows = meta[meta['event'] == name]
            if len(rows) > 0:
                return rows.index[0]
        raise ValueError(f"None of {candidates} found in binary_signals. "
                         f"Available events: {meta['event'].dropna().tolist()}")

    reward_key = _find_key(bs_meta, 'reward_tones', 'reward_tone_on', 'reward_tone')
    error_key = _find_key(bs_meta, 'early_tones', 'early_tone_on', 'error_tone_on', 'error_tones')
    outcome_ts = np.sort(np.concatenate([
        binary_signals[reward_key].t,
        binary_signals[error_key].t
    ]))

    reward_set = set(binary_signals[reward_key].t.tolist())

    trial_events = {'no_stim': [], 'chr2': [], 'chrimson': [],
                    'no_stim_reward': [], 'no_stim_error': []}
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
            if float(ot) in reward_set:
                trial_events['no_stim_reward'].append(ot)
            else:
                trial_events['no_stim_error'].append(ot)
    return trial_events


def merge_sessions(sessions, session_gap=86400.0):
    """Merge a list of (ephys, binary_signals, trial_events) tuples into a single pseudo-session.

    Each session is shifted forward in time by session_gap seconds (default 24 h) relative
    to the previous one so timestamps never overlap. Unit IDs are reassigned sequentially;
    all metadata columns (final_classif, first_lick_ramp_score, …) are preserved along with
    new session_idx and orig_unit_id columns.

    Trials with no lick between cue onset and outcome are excluded from trial_events before
    merging (e.g. spurious rewards delivered without a lick response).

    Returns (merged_ephys, merged_binary_signals, merged_trial_events).
    """
    def _filter_lickless_trials(bs, te):
        """Remove trials that have no lick between the preceding cue and the outcome."""
        bs_meta = bs.metadata
        cue_rows = bs_meta[bs_meta['event'] == 'start_cues']
        if len(cue_rows) == 0:
            return te
        cue_ts = bs[cue_rows.index[0]].t

        # Use raw per-trial lick channels by name so the 100ms cue-latency filter
        # applied to all_first_licks does not incorrectly exclude valid trials.
        raw_lick_rows = bs_meta[bs_meta['event'].isin({'rewarded_first_licks', 'early_first_licks'})]
        if len(raw_lick_rows) == 0:
            # fallback: any channel named *first_licks* (catches naming variations)
            raw_lick_rows = bs_meta[bs_meta['event'].str.contains('first_lick', na=False) &
                                    (bs_meta['event'] != 'all_first_licks')]
        if len(raw_lick_rows) == 0:
            return te
        lick_ts = np.sort(np.concatenate([bs[k].t for k in raw_lick_rows.index]))

        # collect the union of all outcome times across conditions
        all_ots = set()
        for cond in ('no_stim', 'chr2', 'chrimson'):
            all_ots.update(float(t) for t in te.get(cond, []))

        # decide which outcomes are valid (have at least one lick between cue and outcome)
        valid = set()
        for ot in all_ots:
            prec = cue_ts[cue_ts < ot]
            if len(prec) == 0:
                continue
            ct = prec[-1]
            if np.any((lick_ts >= ct) & (lick_ts < ot)):
                valid.add(float(ot))

        n_before = sum(len(v) for k, v in te.items() if k in ('no_stim', 'chr2', 'chrimson'))
        filtered = {k: [t for t in v if float(t) in valid] for k, v in te.items()}
        n_after = sum(len(filtered[k]) for k in ('no_stim', 'chr2', 'chrimson'))
        print(f"  lickless trial filter: kept {n_after} / {n_before} trials")
        return filtered

    # --- compute per-session time offsets so sessions are laid end-to-end ---
    offsets = []
    cursor = 0.0
    for ep, bs, te in sessions:
        ts = ep.time_support
        sess_start = float(np.asarray(ts.start)[0])
        sess_end = float(np.asarray(ts.end)[-1])
        offsets.append(cursor - sess_start)
        cursor += (sess_end - sess_start) + session_gap

    # --- merged time support: one interval per original session interval ---
    all_starts, all_ends = [], []
    for si, (ep, bs, te) in enumerate(sessions):
        off = offsets[si]
        ts = ep.time_support
        for s, e in zip(np.asarray(ts.start), np.asarray(ts.end)):
            all_starts.append(s + off)
            all_ends.append(e + off)
    merged_support = nap.IntervalSet(start=all_starts, end=all_ends)

    # --- merged ephys ---
    unit_dict = {}
    meta_rows = {}
    new_id = 0
    for si, (ep, bs, te) in enumerate(sessions):
        off = offsets[si]
        emeta = ep.metadata
        for orig_id in ep.keys():
            unit_dict[new_id] = nap.Ts(t=np.asarray(ep[orig_id].index) + off)
            row = emeta.loc[orig_id].to_dict() if orig_id in emeta.index else {}
            row['session_idx'] = si
            row['orig_unit_id'] = orig_id
            meta_rows[new_id] = row
            new_id += 1

    merged_ephys = nap.TsGroup(unit_dict, time_support=merged_support)
    _PYNAPPLE_RESERVED = {'rate', 'index'}
    meta_df = pd.DataFrame(meta_rows).T
    meta_df = meta_df.drop(columns=[c for c in _PYNAPPLE_RESERVED if c in meta_df.columns])
    merged_ephys.set_info(meta_df)

    # --- merged binary_signals: pool same-named events across sessions ---
    event_ts = {}
    for si, (ep, bs, te) in enumerate(sessions):
        off = offsets[si]
        bs_meta = bs.metadata
        for key in bs.keys():
            event_name = bs_meta.loc[key, 'event']
            event_ts.setdefault(event_name, []).append(np.asarray(bs[key].t) + off)

    bs_dict = {}
    bs_meta_rows = {}
    for new_key, (event_name, ts_list) in enumerate(event_ts.items()):
        bs_dict[new_key] = nap.Ts(t=np.sort(np.concatenate(ts_list)))
        bs_meta_rows[new_key] = {'event': event_name}

    merged_bs = nap.TsGroup(bs_dict, time_support=merged_support)
    merged_bs.set_info(pd.DataFrame(bs_meta_rows).T)

    # --- merged trial_events: filter lickless trials, offset, and pool per condition ---
    merged_te = {}
    for si, (ep, bs, te) in enumerate(sessions):
        off = offsets[si]
        te = _filter_lickless_trials(bs, te)
        for cond, times in te.items():
            merged_te.setdefault(cond, []).extend((np.asarray(times) + off).tolist())
    for cond in merged_te:
        merged_te[cond] = sorted(merged_te[cond])

    return merged_ephys, merged_bs, merged_te


def get_trial_ephys(ephys, binary_signals, trial_events=None, exclude_opto=True):
    ref_ep = get_task_interval(binary_signals, trial_events, exclude_opto)
    # Intersect with the actual recording time support so that inter-session gaps
    # (present in merged multi-session datasets) are excluded from the reference epoch.
    # Formula: A ∩ B = A.set_diff(A.set_diff(B))
    rec_ep = nap.IntervalSet(
        start=np.asarray(ephys.time_support.start),
        end=np.asarray(ephys.time_support.end),
    )
    ref_ep = ref_ep.set_diff(ref_ep.set_diff(rec_ep))

    return ephys.restrict(ref_ep)

def get_task_interval(binary_signals, trial_events=None, exclude_opto=True):
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t
    all_out = np.sort(np.concatenate([np.array(trial_events[c]) for c in ('no_stim', 'chr2', 'chrimson') if trial_events.get(c)]))
    start = float(cue_ts.min()) - 10.0
    end = float(all_out.max()) + 10.0

    if exclude_opto:
        ons_list, offs_list = [], []
        for opsin in ('chr2', 'chrimson'):
            on_rows = bs_meta[bs_meta['event'].str.contains(f'{opsin}_on', na=False)]
            off_rows = bs_meta[bs_meta['event'].str.contains(f'{opsin}_off', na=False)]
            if len(on_rows) == 0:
                continue
            on = np.sort(np.concatenate([binary_signals[i].t for i in on_rows.index]))
            off = np.sort(np.concatenate([binary_signals[i].t for i in off_rows.index])) if len(off_rows) > 0 else np.array([])
            ons_list.append(on)
            offs_list.append(off)

        if not ons_list:
            ref_ep = nap.IntervalSet(start=start, end=end)
        else:
            ons = np.sort(np.concatenate(ons_list))
            offs = np.sort(np.concatenate(offs_list))
            ons = ons[(ons >= start) & (ons <= end)]
            offs = offs[(offs >= start) & (offs <= end)]
            offs = np.concatenate(([start], offs))
            ons = np.concatenate((ons, [end]))
            ref_ep = nap.IntervalSet(start=offs, end=ons)
    else:
        ref_ep = nap.IntervalSet(start=start, end=end)

    return ref_ep


def compute_unit_z(ephys, binary_signals, trial_events, bin_width, exclude_opto=True):
    """Per-unit z-score parameters (mu, sd) computed over the full session.

    Bins the full session (10 s before first cue → 10 s after last outcome) at
    bin_width resolution. Returns (mu, sd) arrays of shape (n_units,).
    Apply as (fr - mu) / sd to z-score any trial-windowed firing in the same units.

    Change the normalisation here to adjust z-scoring across all plotting functions."""
    ephys = get_trial_ephys(ephys, binary_signals, trial_events, exclude_opto)
    ref_fr = ephys.count(bin_size=bin_width).values * bin_width
    mu = ref_fr.mean(axis=0)
    sd = ref_fr.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd

def plot_outcome_heatmap(ephys, binary_signals, trial_events, save_path=None, bin_width=0.05,
                         t_start=-4.0, t_end=1.0,
                         sort_window=(-0.20, 0.0),
                         clim=(-2, 5), zscore=True, label_units=False):
    """Compute PSTHs per neuron, sort by pre-outcome activity, and plot a 3-panel heatmap.
    zscore=True: z-scored firing rates. zscore=False: raw firing rates (spikes/s).
    Returns the matplotlib Figure."""
    bin_edges = np.arange(t_start, t_end + bin_width / 2, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)

    emeta = ephys.metadata.copy()
    tranche_map = {'dSPN': 0, 'iSPN': 1}

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)
    fsuffix = '' if zscore else '_raw'
    cbar_label = 'Firing rate (z-score)' if zscore else 'Firing rate (spikes/s)'

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
                rows[ti] = h * bin_width
            mat[ui] = rows.mean(axis=0)
        return mat

    psth_all = {k: _psth_mat(trial_events.get(k, [])) for k in ('no_stim', 'chr2', 'chrimson')}
    if zscore:
        psth_z_all = {k: (v - z_mu[:, None]) / z_sd[:, None] for k, v in psth_all.items()}
    else:
        psth_z_all = psth_all

    sort_mask = (bin_centers >= sort_window[0]) & (bin_centers < sort_window[1])
    prelick_z_all = psth_z_all['no_stim'][:, sort_mask].mean(axis=1)

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
    psth_z = {k: v[keep_idx] for k, v in psth_z_all.items()}
    prelick_z = prelick_z_all[np.array(keep_idx)]

    def _sort_key(u):
        t = tranche_map.get(_get_classif(u), 2)
        z = prelick_z[u2row[u]]
        return (t, -float(z) if np.isfinite(z) else np.inf)

    sorted_units = sorted(unit_list, key=_sort_key)
    sort_idx = [u2row[u] for u in sorted_units]
    psth_z_s = {k: v[sort_idx] for k, v in psth_z.items()}

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
    fig, axs = plt.subplots(1, 3, figsize=(3, 1), sharey=True)
    extent = [bin_centers[0], bin_centers[-1], n_units - 0.5, -0.5]

    im = None
    for ax, (ck, cl_label) in zip(axs, conditions):
        im = ax.imshow(
            psth_z_s[ck],
            aspect='auto', origin='upper',
            extent=extent,
            cmap='inferno', vmin=clim[0], vmax=clim[1],
        )
        ax.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.9)
        for b in boundaries:
            ax.axhline(b - 0.5, color='white', linewidth=1.2, alpha=0.8)
        ax.set_xlabel('t(s) to 1st lick')
        if label_units:
            ax.set_yticks(range(n_units))
            ax.set_yticklabels([str(u) for u in sorted_units])
            ax.tick_params(axis='y', length=2, pad=1)
        else:
            ax.set_yticks([])
        if ax is axs[0]:
            for y, name in tranche_labels:
                ax.text(t_start - 0.05, y, name, va='center', ha='right', fontweight='bold', clip_on=False, rotation=90)
            ax.set_ylabel('Unit (sorted)')
            ax.yaxis.set_label_coords(-0.15, 0.5)

    psu.adjust_figure_for_panel_size_hetero(fig, panel_width=1.0, panel_height=n_units * 0.02, t=0.35, hs=0.05)
    fig_w, fig_h = fig.get_size_inches()
    cbar_ax = fig.add_axes([0.6 / fig_w, (fig_h - 0.22) / fig_h,
                            (fig_w - 0.6 - 0.1) / fig_w, 0.06 / fig_h])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label=cbar_label)
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    if save_path is not None:
        txt = ['Firing rate aligned to outcome tone']
        for ck, cl_label in conditions:
            txt.append(f'{cl_label}: n={len(trial_events.get(ck, []))} trials')
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text('\n'.join(txt))
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_cue_aligned_by_latency(ephys, binary_signals, trial_events,
                                save_path=None, bin_width=0.05,
                                t_start=-1.0, t_end=6.0, n_quintiles=5,
                                zscore=True):
    """Population-average firing rate aligned to start_cue, split into lick-latency quintiles.

    Layout: rows = conditions, cols = tranches (dSPN / iSPN / Unlabeled).
    zscore=True: z-scored firing rates. zscore=False: raw firing rates (spikes/s).
    Lines are coloured by mean lick latency of each quintile. Returns the Figure."""
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    # collect (cue_time, outcome_latency) per trial for each condition
    cond_trials = {}
    for cond in ('no_stim', 'chr2', 'chrimson'):
        trials = []
        for ot in trial_events.get(cond, []):
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

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)
        trial_fr_z = (trial_fr - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
    else:
        trial_fr_z = trial_fr
    fsuffix = '' if zscore else '_raw'
    fr_ylabel = 'Firing rate (z-score)' if zscore else 'spks/s'

    # filter unlabeled to top quartile by mean pre-lick score from no_stim trials
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
    cmap = _turbo_dark
    norm = plt.Normalize(vmin=lat_min, vmax=lat_max)

    conditions = [
        ('no_stim', 'No Stim'),
        ('chr2', 'Blue Stim (ChR2)'),
        ('chrimson', 'Orange Stim (ChrimsonR)'),
    ]
    n_rows = len(tranche_order)
    n_cols = len(conditions)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows),
                             sharex=True, sharey=True, squeeze=False)

    for ri, tranche in enumerate(tranche_order):
        t_cols = np.array([u2col[u] for u in tranche_units[tranche]])

        for ci, (ck, cl_label) in enumerate(conditions):
            ax = axs[ri, ci]
            cond_idx = np.array([i for i, c in enumerate(all_cue_conds) if c == ck])

            if len(cond_idx) == 0:
                ax.text(0.5, 0.5, 'no trials', transform=ax.transAxes,
                        ha='center', va='center', color='grey')
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
            if ci == 0:
                ax.set_ylabel(f'{tranche}\n{fr_ylabel}')
            if ri == n_rows - 1:
                ax.set_xlabel('t(s) from cue')

    for ax in axs.ravel():
        ax.label_outer()

    psu.adjust_figure_for_panel_size_auto(fig)
    if save_path is not None:
        txt = ['Cue-aligned firing rate by latency quintile']
        for ck, cl_label in conditions:
            txt.append(f'{cl_label}: n={len(cond_trials.get(ck, []))} trials')
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text('\n'.join(txt))
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.06, 1))
    cbar_fig.colorbar(sm, cax=cbar_ax, label='Mean lick latency (s)')
    if save_path is not None:
        cbar_path = save_path.parent / (save_path.stem + '_colorbar' + fsuffix + '.svg')
        cbar_fig.savefig(cbar_path, bbox_inches='tight')
        cbar_fig.savefig(cbar_path.with_suffix('.png'), dpi=150, bbox_inches='tight')

    return fig, cbar_fig


def plot_outcome_heatmap_nostim(ephys, binary_signals, trial_events, save_path=None, bin_width=0.05,
                                t_start=-4.0, t_end=1.0,
                                sort_window=(-0.20, 0.0),
                                clim=(-2, 5), zscore=True, ms_censored=None):
    """No-stim-only outcome-aligned heatmap. Layout: 3 rows (dSPN / iSPN / Unlabeled) × 1 col.
    zscore=True: z-scored firing rates. zscore=False: raw firing rates (spikes/s).
    ms_censored: if set, removes trials whose outcome tone occurs within this many ms of the cue (i.e. early-lick trials)."""
    bin_edges = np.arange(t_start, t_end + bin_width / 2, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins = len(bin_centers)

    emeta = ephys.metadata.copy()

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)
    fsuffix = '' if zscore else '_raw'
    cbar_label = 'Firing rate (z-score)' if zscore else 'Firing rate (spikes/s)'

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    all_units = list(ephys.keys())
    n_all = len(all_units)

    ns_events = trial_events.get('no_stim', [])
    if ms_censored is not None:
        bs_meta = binary_signals.metadata
        cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
        cue_ts = binary_signals[cue_key].t
        thresh_s = ms_censored / 1000.0
        filtered = []
        for ot in ns_events:
            prec = cue_ts[cue_ts < ot]
            if len(prec) == 0:
                continue
            if float(ot - prec[-1]) >= thresh_s:
                filtered.append(ot)
        ns_events = filtered

    psth_ns = np.zeros((n_all, n_bins))
    if ns_events:
        for ui, u in enumerate(all_units):
            spk = np.asarray(ephys[u].index)
            rows = np.zeros((len(ns_events), n_bins))
            for ti, t in enumerate(ns_events):
                h, _ = np.histogram(spk - t, bins=bin_edges)
                rows[ti] = h * bin_width
            psth_ns[ui] = rows.mean(axis=0)

    if zscore:
        psth_z = (psth_ns - z_mu[:, None]) / z_sd[:, None]
    else:
        psth_z = psth_ns

    sort_mask = (bin_centers >= sort_window[0]) & (bin_centers < sort_window[1])
    prelick_z = psth_z[:, sort_mask].mean(axis=1)
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
    row_heights = [len(tranche_groups[t]) * 0.02 for t in tranche_order]
    fig, axs = plt.subplots(n_tranches, 1,
                            figsize=(1, n_tranches),
                            gridspec_kw={'height_ratios': row_heights},
                            squeeze=False)

    im = None
    tranche_counts = {}
    for ri, label in enumerate(tranche_order):
        ax = axs[ri, 0]
        units = tranche_groups[label]
        n_u = len(units)
        tranche_counts[label] = n_u
        idx = [u2row[u] for u in units]
        z = psth_z[idx]
        ext = [bin_centers[0], bin_centers[-1], n_u - 0.5, -0.5]
        im = ax.imshow(z, aspect='auto', origin='upper', extent=ext,
                       cmap='inferno', vmin=clim[0], vmax=clim[1])
        ax.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.9)
        ax.set_yticks([])
        ax.set_ylabel(label)
        if ri == n_tranches - 1:
            ax.set_xlabel('Time to outcome (s)')

    psu.adjust_figure_for_panel_size_hetero(fig, panel_width=1.0, panel_height=row_heights, t=0.35, hs=0.05)
    fig_w, fig_h = fig.get_size_inches()
    cbar_ax = fig.add_axes([0.6 / fig_w, (fig_h - 0.3) / fig_h,
                            (fig_w - 0.6 - 0.1) / fig_w, 0.05 / fig_h])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label=cbar_label)
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    if save_path is not None:
        txt = [f'Firing rate aligned to outcome - No Stim', f'n={len(ns_events)} trials']
        for label, n_u in tranche_counts.items():
            txt.append(f'{label}: {n_u} cells')
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text('\n'.join(txt))
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_cue_outcome_heatmap(ephys, binary_signals, trial_events, save_path=None, bin_width=0.05,
                              t_start_cue=-1.0, t_end_cue=4.0,
                              t_start_outcome=-4.0, t_end_outcome=1.0,
                              sort_window=(-0.20, 0.0),
                              clim=(-0.5, 4), zscore=True, ms_censored=None,
                              labeled_only=False):
    """No-stim cue- and outcome-aligned heatmap. Left column: cue-aligned. Right column: outcome-aligned.
    Same trials and unit ordering in both columns. Rows: dSPN / iSPN / Unlabeled (unless labeled_only=True).
    Shared y-axis per row. Single shared colorbar at top.
    ms_censored: if set, removes trials whose outcome occurs within this many ms of the cue."""
    cue_edges = np.arange(t_start_cue, t_end_cue + bin_width / 2, bin_width)
    cue_centers = (cue_edges[:-1] + cue_edges[1:]) / 2.0

    out_edges = np.arange(t_start_outcome, t_end_outcome + bin_width / 2, bin_width)
    out_centers = (out_edges[:-1] + out_edges[1:]) / 2.0

    emeta = ephys.metadata.copy()

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)
    fsuffix = '' if zscore else '_raw'
    cbar_label = 'Firing rate (z-score)' if zscore else 'Firing rate (spikes/s)'

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    # Build paired (cue_time, outcome_time) for no_stim trials, applying ms_censored filter
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t
    thresh_s = ms_censored / 1000.0 if ms_censored is not None else None

    pairs = []
    for ot in trial_events.get('no_stim', []):
        prec = cue_ts[cue_ts < ot]
        if len(prec) == 0:
            continue
        ct = prec[-1]
        if thresh_s is not None and float(ot - ct) < thresh_s:
            continue
        pairs.append((float(ct), float(ot)))

    cue_times   = [p[0] for p in pairs]
    outcome_times = [p[1] for p in pairs]

    all_units = list(ephys.keys())
    n_all = len(all_units)
    n_cue_bins = len(cue_centers)
    n_out_bins = len(out_centers)

    psth_cue = np.zeros((n_all, n_cue_bins))
    psth_out = np.zeros((n_all, n_out_bins))
    if pairs:
        for ui, u in enumerate(all_units):
            spk = np.asarray(ephys[u].index)
            rows_c = np.zeros((len(pairs), n_cue_bins))
            rows_o = np.zeros((len(pairs), n_out_bins))
            for ti in range(len(pairs)):
                h, _ = np.histogram(spk - cue_times[ti], bins=cue_edges)
                rows_c[ti] = h * bin_width
                h, _ = np.histogram(spk - outcome_times[ti], bins=out_edges)
                rows_o[ti] = h * bin_width
            psth_cue[ui] = rows_c.mean(axis=0)
            psth_out[ui] = rows_o.mean(axis=0)

    if zscore:
        psth_cue = (psth_cue - z_mu[:, None]) / z_sd[:, None]
        psth_out = (psth_out - z_mu[:, None]) / z_sd[:, None]

    # Sort units by pre-outcome activity (outcome-aligned window)
    sort_mask = (out_centers >= sort_window[0]) & (out_centers < sort_window[1])
    prelick_z = psth_out[:, sort_mask].mean(axis=1)
    u2row = {u: i for i, u in enumerate(all_units)}

    labeled   = [u for u in all_units if _get_classif(u) is not None]
    unlabeled = [] if labeled_only else [u for u in all_units if _get_classif(u) is None]
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

    n_tranches  = len(tranche_order)
    row_heights = [len(tranche_groups[t]) * 0.03 for t in tranche_order]

    fig, axs = plt.subplots(n_tranches, 2,
                            figsize=(2, n_tranches),
                            gridspec_kw={'height_ratios': row_heights},
                            sharey='row',
                            squeeze=False)

    im = None
    tranche_counts = {}
    for ri, label in enumerate(tranche_order):
        units = tranche_groups[label]
        n_u   = len(units)
        tranche_counts[label] = n_u
        idx   = [u2row[u] for u in units]

        z_cue = psth_cue[idx]
        z_out = psth_out[idx]
        ext_c = [cue_centers[0],  cue_centers[-1],  n_u - 0.5, -0.5]
        ext_o = [out_centers[0],  out_centers[-1],  n_u - 0.5, -0.5]

        ax_c = axs[ri, 0]
        ax_c.imshow(z_cue, aspect='auto', origin='upper', extent=ext_c,
                    cmap='inferno', vmin=clim[0], vmax=clim[1])
        ax_c.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.9)
        ax_c.set_yticks([])
        ax_c.set_ylabel(label)
        if ri == n_tranches - 1:
            ax_c.set_xlabel('t(s) from cue')

        ax_o = axs[ri, 1]
        im = ax_o.imshow(z_out, aspect='auto', origin='upper', extent=ext_o,
                         cmap='inferno', vmin=clim[0], vmax=clim[1])
        ax_o.axvline(0, color='white', linestyle='--', linewidth=1.5, alpha=0.9)
        ax_o.set_yticks([])
        if ri == n_tranches - 1:
            ax_o.set_xlabel('Time from outcome (s)')

    psu.adjust_figure_for_panel_size_hetero(fig, panel_width=1.0, panel_height=row_heights, t=0.35, hs=0.05)
    fig_w, fig_h = fig.get_size_inches()
    cbar_ax = fig.add_axes([0.6 / fig_w, (fig_h - 0.3) / fig_h,
                            (fig_w - 0.6 - 0.1) / fig_w, 0.05 / fig_h])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label=cbar_label)
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    if save_path is not None:
        txt = [f'Cue + outcome aligned heatmap - No Stim', f'n={len(pairs)} trials']
        for label, n_u in tranche_counts.items():
            txt.append(f'{label}: {n_u} cells')
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text('\n'.join(txt))
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_spn_heatmap(ephys, binary_signals, trial_events, **kwargs):
    """Like plot_cue_outcome_heatmap but restricted to labeled (dSPN / iSPN) units only."""
    return plot_cue_outcome_heatmap(ephys, binary_signals, trial_events, labeled_only=True, **kwargs)


def plot_cue_aligned_nostim(ephys, binary_signals, trial_events,
                            save_path=None, bin_width=0.05,
                            t_start=-1.0, t_end=7.0, n_quintiles=5,
                            zscore=True, ms_censored=None,
                            labeled_only=False, right_label_rotation=90):
    """No-stim-only cue-aligned population PSTH. Layout: 3 rows (dSPN / iSPN / Unlabeled) × 1 col.
    zscore=True: z-scored firing rates. zscore=False: raw firing rates (spikes/s).
    ms_censored: if set, removes trials whose outcome tone occurs within this many ms of the cue (i.e. early-lick trials)."""
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

    # Remove no-response timeout trials (outcome tone at 7 s = no lick preceded it)
    trials = [(ct, lat) for ct, lat in trials if lat < 6.9]

    if ms_censored is not None:
        thresh_s = ms_censored / 1000.0
        trials = [(ct, lat) for ct, lat in trials if lat >= thresh_s]

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
    tranche_order = [t for t in ('dSPN', 'iSPN', 'Unlabeled') if t in tranche_units and (not labeled_only or t != 'Unlabeled')]

    cue_times = np.array([t[0] for t in trials])
    lats = np.array([t[1] for t in trials])
    sort_order = np.argsort(cue_times)
    cue_times = cue_times[sort_order]
    lats = lats[sort_order]
    n_total = len(cue_times)

    if n_total == 0:
        fig, axs = plt.subplots(len(tranche_order), 1, figsize=(1, len(tranche_order)), squeeze=False)
        for ax in axs.ravel():
            ax.text(0.5, 0.5, 'no trials', transform=ax.transAxes, ha='center', va='center', color='grey')
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

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)
        trial_fr_z = (trial_fr - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
    else:
        trial_fr_z = trial_fr
    fsuffix = '' if zscore else '_raw'
    fr_ylabel = 'Z(FR)' if zscore else 'spks/s'

    # filter unlabeled to top quartile by pre-lick score
    mean_lat = float(lats.mean())
    pl_mask = (bin_centers >= mean_lat - 0.2) & (bin_centers < mean_lat)
    unit_prelick_z = trial_fr_z[:, pl_mask, :].mean(axis=(0, 1))
    unlabeled = tranche_units.get('Unlabeled', [])
    if unlabeled:
        ul_cols = np.array([u2col[u] for u in unlabeled])
        scores = unit_prelick_z[ul_cols]
        thresh = np.nanpercentile(scores, 75)
        tranche_units['Unlabeled'] = [u for u, s in zip(unlabeled, scores) if s >= thresh]

    cmap = _turbo_dark
    norm = plt.Normalize(vmin=float(lats.min()), vmax=float(lats.max()))

    sorted_idx = np.argsort(lats)
    q_assign = np.empty(n_total, dtype=int)
    q_assign[sorted_idx] = np.arange(n_total) * n_quintiles // n_total
    mean_lats = [lats[q_assign == q].mean() for q in range(n_quintiles)]

    n_rows = len(tranche_order)
    fig, axs = plt.subplots(n_rows, 1, figsize=(1, n_rows), sharex=True, sharey=True, squeeze=False)

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
        ax.set_ylabel(fr_ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax_r = ax.twinx()
        ax_r.set_ylabel(tranche, rotation=right_label_rotation,
                        labelpad=8 if right_label_rotation != 90 else 4)
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_visible(False)
        if ri == n_rows - 1:
            ax.set_xlabel('t(s) from cue')

    for ax in axs.ravel():
        ax.label_outer()

    psu.adjust_figure_for_panel_size_auto(fig, t=0.35)
    fig_w, fig_h = fig.get_size_inches()
    cbar_ax = fig.add_axes([0.6 / fig_w, (fig_h - 0.3) / fig_h,
                            (fig_w - 0.6 - 0.1) / fig_w, 0.05 / fig_h])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Mean lick latency (s)')
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    if save_path is not None:
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text(
            f'Cue-aligned firing rate - No Stim\nn={n_total} trials'
        )
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_spn_cue_aligned_nostim(ephys, binary_signals, trial_events,
                                save_path=None, bin_width=0.05,
                                t_start=-1.0, t_end=7.0, n_quintiles=5,
                                zscore=True, ms_censored=None):
    """dSPN / iSPN cue-aligned PSTH + trial-by-trial dSPN−iSPN difference (3rd row).
    No unlabeled units. Labels on right axis, flipped 180°."""
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

    trials = [(ct, lat) for ct, lat in trials if lat < 6.9]
    if ms_censored is not None:
        thresh_s = ms_censored / 1000.0
        trials = [(ct, lat) for ct, lat in trials if lat >= thresh_s]

    emeta = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    tranche_units = {}
    for u in ephys.keys():
        label = _get_classif(u)
        if label is not None:
            tranche_units.setdefault(label, []).append(u)
    tranche_order = [t for t in ('dSPN', 'iSPN') if t in tranche_units]

    cue_times = np.array([t[0] for t in trials])
    lats      = np.array([t[1] for t in trials])
    sort_order = np.argsort(cue_times)
    cue_times  = cue_times[sort_order]
    lats       = lats[sort_order]
    n_total    = len(cue_times)

    if n_total == 0:
        fig, axs = plt.subplots(3, 1, figsize=(1, 3), squeeze=False)
        for ax in axs.ravel():
            ax.text(0.5, 0.5, 'no trials', transform=ax.transAxes, ha='center', va='center', color='grey')
        return fig

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

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)
        trial_fr_z = (trial_fr - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
    else:
        trial_fr_z = trial_fr
    fsuffix  = '' if zscore else '_raw'
    fr_ylabel = 'Z(FR)' if zscore else 'spks/s'

    cmap = _turbo_dark
    norm = plt.Normalize(vmin=float(lats.min()), vmax=float(lats.max()))

    sorted_idx = np.argsort(lats)
    q_assign   = np.empty(n_total, dtype=int)
    q_assign[sorted_idx] = np.arange(n_total) * n_quintiles // n_total
    mean_lats  = [lats[q_assign == q].mean() for q in range(n_quintiles)]

    # trial-by-trial dSPN − iSPN difference
    dspn_cols = np.array([u2col[u] for u in tranche_units.get('dSPN', [])])
    ispn_cols  = np.array([u2col[u] for u in tranche_units.get('iSPN', [])])
    has_diff   = len(dspn_cols) > 0 and len(ispn_cols) > 0
    if has_diff:
        trial_diff = (trial_fr_z[:, :, dspn_cols].mean(axis=2) -
                      trial_fr_z[:, :, ispn_cols].mean(axis=2))  # (n_trials, n_bins)

    n_rows = len(tranche_order) + (1 if has_diff else 0)
    fig, axs = plt.subplots(n_rows, 1, figsize=(1, n_rows), sharex=True, squeeze=False)

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
        ax.axvspan(3.33, 7.0, ymin=0.92, ymax=1.0, color='green', alpha=0.45, linewidth=0)
        ax.axvspan(0,3.33, ymin=0.92, ymax=1.0, color='red', alpha=0.45, linewidth=0)
        ax.set_ylabel(fr_ylabel)
        ax.set_ylim(top=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax_r = ax.twinx()
        ax_r.set_ylabel(tranche, rotation=-90, labelpad=8)
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_visible(False)

    if has_diff:
        ax = axs[n_rows - 1, 0]
        ax = axs[n_rows - 1, 0]
        for q in range(n_quintiles):
            q_idx = np.where(q_assign == q)[0]
            if len(q_idx) == 0:
                continue
            diff_q = trial_diff[q_idx].mean(axis=0)
            color  = cmap(norm(mean_lats[q]))
            ax.plot(bin_centers, diff_q, color=color, linewidth=1.5)
            ax.axvline(mean_lats[q], color=color, linestyle='--', linewidth=1.0, alpha=0.7)
        ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.75)
        ax.axvspan(3.33, 7.0, ymin=0.92, ymax=1.0, color='green', alpha=0.45, linewidth=0)
        ax.axvspan(0,3.33, ymin=0.92, ymax=1.0, color='red', alpha=0.45, linewidth=0)
        ax.set_ylim(bottom=-0.5)
        ax.set_ylabel(fr_ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax_r = ax.twinx()
        ax_r.set_ylabel('dSPN-iSPN', rotation=-90, labelpad=8)
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_visible(False)

    axs[-1, 0].set_xlabel('t(s) from cue')
    axs[0, 0].set_xlim(bin_centers[0], bin_centers[-1])

    from matplotlib.ticker import FuncFormatter as _FuncFormatter
    _fmt = _FuncFormatter(lambda x, _: f'{x:g}')
    for ax in axs.ravel():
        ax.label_outer()
        ax.yaxis.set_major_formatter(_fmt)

    psu.adjust_figure_for_panel_size_auto(fig, t=0.35)
    fig_w, fig_h = fig.get_size_inches()
    cbar_ax = fig.add_axes([0.6 / fig_w, (fig_h - 0.3) / fig_h,
                            (fig_w - 0.6 - 0.1) / fig_w, 0.05 / fig_h])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='Mean lick latency (s)')
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    if save_path is not None:
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text(
            f'SPN cue-aligned firing rate - No Stim\nn={n_total} trials'
        )
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_spn_mean_psth(ephys, binary_signals, trial_events,
                       save_path=None, bin_width=0.05,
                       t_start_cue=-0.5, t_end_cue= 4.0,
                       t_start_outcome=-4.0, t_end_outcome=0.5,
                       zscore=True, ms_censored=None, outcome_range=None):
    """3×2 mean PSTH: left column cue-aligned, right column outcome-aligned.
    Rows: dSPN / iSPN / dSPN−iSPN. Same trials in both columns. No quintile binning."""
    cue_edges   = np.arange(t_start_cue,     t_end_cue     + bin_width / 2, bin_width)
    out_edges   = np.arange(t_start_outcome, t_end_outcome + bin_width / 2, bin_width)
    cue_centers = (cue_edges[:-1] + cue_edges[1:]) / 2.0
    out_centers = (out_edges[:-1] + out_edges[1:]) / 2.0

    emeta = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    # Paired (cue_time, outcome_time, latency) for no_stim trials
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts  = binary_signals[cue_key].t
    thresh_s = ms_censored / 1000.0 if ms_censored is not None else None

    pairs = []
    for ot in trial_events.get('no_stim', []):
        prec = cue_ts[cue_ts < ot]
        if len(prec) == 0:
            continue
        ct  = prec[-1]
        lat = float(ot - ct)
        if lat >= 6.9:
            continue
        if thresh_s is not None and lat < thresh_s:
            continue
        if outcome_range is not None:
            lo, hi = outcome_range[0] / 1000.0, outcome_range[1] / 1000.0
            if lat < lo or lat > hi:
                continue
        pairs.append((float(ct), float(ot), lat))

    n_total = len(pairs)
    if n_total == 0:
        fig, _ = plt.subplots(3, 2, figsize=(2, 3))
        return fig

    cue_times     = np.array([p[0] for p in pairs])
    outcome_times = np.array([p[1] for p in pairs])
    lats          = np.array([p[2] for p in pairs])
    mean_lat      = float(lats.mean())

    all_units  = list(ephys.keys())
    unit_names = all_units
    u2col = {u: i for i, u in enumerate(unit_names)}

    # Compute firing rates for both alignments simultaneously
    n_cue_bins = len(cue_centers)
    n_out_bins = len(out_centers)
    n_units    = len(all_units)
    trial_fr_cue = np.zeros((n_total, n_cue_bins, n_units))
    trial_fr_out = np.zeros((n_total, n_out_bins, n_units))

    for ti in range(n_total):
        spikes = {u: np.asarray(ephys[u].index) for u in all_units}
        for ui, u in enumerate(all_units):
            spk = spikes[u]
            h, _ = np.histogram(spk - cue_times[ti],     bins=cue_edges)
            trial_fr_cue[ti, :, ui] = h * bin_width
            h, _ = np.histogram(spk - outcome_times[ti], bins=out_edges)
            trial_fr_out[ti, :, ui] = h * bin_width

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)
        trial_fr_cue = (trial_fr_cue - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
        trial_fr_out = (trial_fr_out - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
    fsuffix   = '' if zscore else '_raw'
    fr_ylabel = 'Z(FR)' if zscore else 'spks/s'

    tranche_units = {}
    for u in all_units:
        label = _get_classif(u)
        if label is not None:
            tranche_units.setdefault(label, []).append(u)

    dspn_cols = np.array([u2col[u] for u in tranche_units.get('dSPN', [])])
    ispn_cols  = np.array([u2col[u] for u in tranche_units.get('iSPN', [])])
    has_diff   = len(dspn_cols) > 0 and len(ispn_cols) > 0

    # Per-alignment population means and diff
    def _mean_ci(fr, cols):
        per_trial = fr[:, :, cols].mean(axis=2)          # (n_trials, n_bins)
        m   = per_trial.mean(axis=0)
        sem = per_trial.std(axis=0) / np.sqrt(n_total)
        return m, m - 1.96 * sem, m + 1.96 * sem

    def _diff_ci(fr):
        per_trial = (fr[:, :, dspn_cols].mean(axis=2) -
                     fr[:, :, ispn_cols].mean(axis=2))   # (n_trials, n_bins)
        m   = per_trial.mean(axis=0)
        sem = per_trial.std(axis=0) / np.sqrt(n_total)
        return m, m - 1.96 * sem, m + 1.96 * sem

    rows = [
        ('dSPN',      _mean_ci(trial_fr_cue, dspn_cols), _mean_ci(trial_fr_out, dspn_cols)),
        ('iSPN',      _mean_ci(trial_fr_cue, ispn_cols),  _mean_ci(trial_fr_out, ispn_cols)),
    ]
    if has_diff:
        rows.append(('dSPN-iSPN', _diff_ci(trial_fr_cue), _diff_ci(trial_fr_out)))

    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 2, figsize=(2, n_rows),
                            sharex='col', sharey='row', squeeze=False)

    from matplotlib.ticker import FuncFormatter as _FuncFormatter
    _fmt = _FuncFormatter(lambda x, _: f'{x:g}')

    for ri, (label, cue_stat, out_stat) in enumerate(rows):
        is_diff = label == 'dSPN-iSPN'

        for ci, (ax, (y, ci_lo, ci_hi), centers, xlabel) in enumerate([
            (axs[ri, 0], cue_stat, cue_centers, 't(s) from cue'),
            (axs[ri, 1], out_stat, out_centers, 't(s) to 1st lick'),
        ]):
            ax.fill_between(centers, ci_lo, ci_hi, color='grey', alpha=0.3, linewidth=0)
            ax.plot(centers, y, color='k', linewidth=1.5)
            ax.axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.75)
            if ci == 0:
                ax.axvline(3.33, color='green', linestyle='--', linewidth=1)
                ax.axvspan(3.33, 7.0, ymin=0.92, ymax=1.0, color='green', alpha=0.45, linewidth=0)
            if is_diff:
                ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(fr_ylabel)
            ax.yaxis.set_major_formatter(_fmt)
            if ri == n_rows - 1:
                ax.set_xlabel(xlabel)

        # Tranche label on right axis of right column only
        ax_r = axs[ri, 1].twinx()
        ax_r.set_ylabel(label, rotation=-90, labelpad=8)
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_visible(False)

    # Apply y-limits after auto-scaling is complete
    for ri, (label, _, _) in enumerate(rows):
        if label == 'dSPN-iSPN':
            axs[ri, 0].set_ylim(top=0.7, bottom=-0.2)  # sharey='row' propagates to right col
        else:
            axs[ri, 0].set_ylim(top=1.3, bottom=-0.2)

    # Pin x limits to the actual data range — sharex='col' propagates to all rows
    axs[0, 0].set_xlim(cue_centers[0], cue_centers[-1])
    axs[0, 1].set_xlim(out_centers[0], out_centers[-1])

    for ax in axs.ravel():
        ax.label_outer()
        ax.yaxis.set_major_formatter(_fmt)

    psu.adjust_figure_for_panel_size_hetero(fig, panel_width=0.8, panel_height=psu.dPanH)

    if save_path is not None:
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text(
            f'SPN mean PSTH - No Stim\nn={n_total} trials'
        )
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_spn_rew_unrew_psth(ephys, binary_signals, trial_events,
                             save_path=None, bin_width=0.05,
                             t_start_cue=-1.0, t_end_cue=7.0,
                             t_start_outcome=-1.5, t_end_outcome=2.0,
                             zscore=True,
                             rewarded_range_ms=(3333, None),
                             unrewarded_range_ms=(500, 3332)):
    """Like plot_spn_mean_psth but superimposes rewarded (green) and unrewarded (maroon) trials.
    rewarded_range_ms / unrewarded_range_ms: (low_ms, high_ms) for outcome latency; None = no bound."""
    cue_edges   = np.arange(t_start_cue,     t_end_cue     + bin_width / 2, bin_width)
    out_edges   = np.arange(t_start_outcome, t_end_outcome + bin_width / 2, bin_width)
    cue_centers = (cue_edges[:-1] + cue_edges[1:]) / 2.0
    out_centers = (out_edges[:-1] + out_edges[1:]) / 2.0

    emeta = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    def _in_range(lat_s, range_ms):
        lo = range_ms[0] / 1000.0 if range_ms[0] is not None else -np.inf
        hi = range_ms[1] / 1000.0 if range_ms[1] is not None else  np.inf
        return lo <= lat_s <= hi

    bs_meta = binary_signals.metadata
    cue_key  = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts   = binary_signals[cue_key].t

    # Collect all pairs that belong to at least one group
    pairs, rew_mask, unrew_mask = [], [], []
    for ot in trial_events.get('no_stim', []):
        prec = cue_ts[cue_ts < ot]
        if len(prec) == 0:
            continue
        ct  = prec[-1]
        lat = float(ot - ct)
        if lat >= 6.9:
            continue
        is_rew   = _in_range(lat, rewarded_range_ms)
        is_unrew = _in_range(lat, unrewarded_range_ms)
        if not (is_rew or is_unrew):
            continue
        pairs.append((float(ct), float(ot), lat))
        rew_mask.append(is_rew)
        unrew_mask.append(is_unrew)

    n_total    = len(pairs)
    rew_idx    = np.where(rew_mask)[0]
    unrew_idx  = np.where(unrew_mask)[0]

    if n_total == 0:
        fig, _ = plt.subplots(3, 2, figsize=(2, 3))
        return fig

    cue_times     = np.array([p[0] for p in pairs])
    outcome_times = np.array([p[1] for p in pairs])

    all_units  = list(ephys.keys())
    u2col      = {u: i for i, u in enumerate(all_units)}
    n_units    = len(all_units)
    n_cue_bins = len(cue_centers)
    n_out_bins = len(out_centers)

    trial_fr_cue = np.zeros((n_total, n_cue_bins, n_units))
    trial_fr_out = np.zeros((n_total, n_out_bins, n_units))
    for ti in range(n_total):
        for ui, u in enumerate(all_units):
            spk = np.asarray(ephys[u].index)
            h, _ = np.histogram(spk - cue_times[ti],     bins=cue_edges)
            trial_fr_cue[ti, :, ui] = h * bin_width
            h, _ = np.histogram(spk - outcome_times[ti], bins=out_edges)
            trial_fr_out[ti, :, ui] = h * bin_width

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)
        trial_fr_cue = (trial_fr_cue - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
        trial_fr_out = (trial_fr_out - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
    fsuffix   = '' if zscore else '_raw'
    fr_ylabel = 'Z(FR)' if zscore else 'spks/s'

    tranche_units = {}
    for u in all_units:
        label = _get_classif(u)
        if label is not None:
            tranche_units.setdefault(label, []).append(u)

    dspn_cols = np.array([u2col[u] for u in tranche_units.get('dSPN', [])])
    ispn_cols  = np.array([u2col[u] for u in tranche_units.get('iSPN', [])])
    has_diff   = len(dspn_cols) > 0 and len(ispn_cols) > 0

    def _stats(fr, cols, idx):
        per_trial = fr[idx][:, :, cols].mean(axis=2)  # (n, n_bins)
        m   = per_trial.mean(axis=0)
        sem = per_trial.std(axis=0) / np.sqrt(len(idx))
        return m, m - 1.96 * sem, m + 1.96 * sem

    def _diff_stats(fr, idx):
        per_trial = (fr[idx][:, :, dspn_cols].mean(axis=2) -
                     fr[idx][:, :, ispn_cols].mean(axis=2))
        m   = per_trial.mean(axis=0)
        sem = per_trial.std(axis=0) / np.sqrt(len(idx))
        return m, m - 1.96 * sem, m + 1.96 * sem

    conditions = [
        ('rewarded',   rew_idx,   'green',  'Rewarded'),
        ('unrewarded', unrew_idx, 'maroon', 'Unrewarded'),
    ]
    row_labels = ['dSPN', 'iSPN'] + (['dSPN-iSPN'] if has_diff else [])
    n_rows = len(row_labels)

    fig, axs = plt.subplots(n_rows, 2, figsize=(2, n_rows),
                            sharex='col', sharey='row', squeeze=False)

    from matplotlib.ticker import FuncFormatter as _FuncFormatter
    _fmt = _FuncFormatter(lambda x, _: f'{x:g}')

    for ri, row_label in enumerate(row_labels):
        is_diff = row_label == 'dSPN-iSPN'

        for ci, (ax, centers, xlabel) in enumerate([
            (axs[ri, 0], cue_centers,  't(s) from cue'),
            (axs[ri, 1], out_centers,  't(s) to 1st lick'),
        ]):
            fr = trial_fr_cue if ci == 0 else trial_fr_out
            for _, idx, color, _ in conditions:
                if len(idx) == 0:
                    continue
                if is_diff:
                    m, lo, hi = _diff_stats(fr, idx)
                elif row_label == 'dSPN':
                    m, lo, hi = _stats(fr, dspn_cols, idx)
                else:
                    m, lo, hi = _stats(fr, ispn_cols, idx)
                ax.fill_between(centers, lo, hi, color=color, alpha=0.2, linewidth=0)
                ax.plot(centers, m, color=color, linewidth=1.5)

            ax.axvline(0, color='k', linestyle='-', linewidth=1, alpha=0.75)
            if ci == 0:
                ax.axvline(3.33, color='green', linestyle='--', linewidth=1, alpha=0.6)
                ax.axvspan(3.33, 7.0, ymin=0.92, ymax=1.0, color='green', alpha=0.45, linewidth=0)
                ax.axvspan(0.0, 3.33, ymin=0.92, ymax=1.0, color='maroon', alpha=0.45, linewidth=0)
            if is_diff:
                ax.axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(fr_ylabel)
            ax.yaxis.set_major_formatter(_fmt)
            if ri == n_rows - 1:
                ax.set_xlabel(xlabel)

        ax_r = axs[ri, 1].twinx()
        ax_r.set_ylabel(row_label, rotation=-90, labelpad=8)
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_visible(False)

    # y-limits after auto-scaling
    for ri, row_label in enumerate(row_labels):
        if row_label == 'dSPN-iSPN':
            axs[ri, 0].set_ylim(bottom=-0.3, top=0.6)
        else:
            axs[ri, 0].set_ylim(top=1.5)

    axs[0, 0].set_xlim(cue_centers[0], cue_centers[-1])
    axs[0, 1].set_xlim(out_centers[0], out_centers[-1])

    axs[0, 1].text(0.99, 0.98, 'Rewarded',   transform=axs[0, 1].transAxes,
                   color='green',  fontsize=7, va='top', ha='right')
    axs[0, 1].text(0.99, 0.85, 'Unrewarded', transform=axs[0, 1].transAxes,
                   color='maroon', fontsize=7, va='top', ha='right')

    for ax in axs.ravel():
        ax.label_outer()
        ax.yaxis.set_major_formatter(_fmt)

    psu.adjust_figure_for_panel_size_hetero(fig, panel_width=1, panel_height=psu.dPanH)

    if save_path is not None:
        n_rew, n_unrew = len(rew_idx), len(unrew_idx)
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text(
            f'SPN rewarded vs unrewarded PSTH - No Stim\n'
            f'Rewarded n={n_rew}, Unrewarded n={n_unrew}'
        )
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_spn_summary(ephys, binary_signals, trial_events,
                     save_path=None, bin_width=0.05,
                     t_start_cue=-1.5, t_end_cue=7.5,
                     t_start_outcome=-0.75, t_end_outcome=0.5,
                     t_start_bout=-0.75, t_end_bout=0.5,
                     n_quintiles=6, zscore=True,
                     ms_censored=None,
                     min_lat_ms=500):
    """3-column summary per cell type row (dSPN, iSPN, dSPN-iSPN):
      Col 0 — lick-bout-aligned PSTH (lick_bout_starts, opto excluded)
      Col 1 — outcome-aligned PSTH, all no-stim trials with 1st-lick latency >= min_lat_ms
      Col 2 — cue-aligned PSTH split by lick-latency quintile
    sharey='row', sharex='col'. Tranche labels on rightmost column only."""
    from matplotlib.ticker import FuncFormatter as _FuncFormatter

    cue_edges   = np.arange(t_start_cue,     t_end_cue     + bin_width / 2, bin_width)
    out_edges   = np.arange(t_start_outcome, t_end_outcome + bin_width / 2, bin_width)
    bout_edges  = np.arange(t_start_bout,    t_end_bout    + bin_width / 2, bin_width)
    cue_centers  = (cue_edges[:-1]  + cue_edges[1:])  / 2.0
    out_centers  = (out_edges[:-1]  + out_edges[1:])  / 2.0
    bout_centers = (bout_edges[:-1] + bout_edges[1:]) / 2.0

    emeta = ephys.metadata.copy()
    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    bs_meta  = binary_signals.metadata
    cue_key  = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts   = binary_signals[cue_key].t

    thresh_s    = ms_censored / 1000.0 if ms_censored is not None else None
    min_lat_s   = min_lat_ms  / 1000.0

    # Build no-stim trial list
    all_pairs = []
    for ot in trial_events.get('no_stim', []):
        prec = cue_ts[cue_ts < ot]
        if len(prec) == 0:
            continue
        ct  = prec[-1]
        lat = float(ot - ct)
        if lat >= 6.9:
            continue
        all_pairs.append((float(ct), float(ot), lat))

    # Quintile trials
    q_pairs     = [(ct, ot, lat) for ct, ot, lat in all_pairs
                   if thresh_s is None or lat >= thresh_s]
    q_cue_times = np.array([p[0] for p in q_pairs])
    q_lats      = np.array([p[2] for p in q_pairs])
    sort_order  = np.argsort(q_cue_times)
    q_cue_times = q_cue_times[sort_order]
    q_lats      = q_lats[sort_order]
    n_q         = len(q_pairs)

    # Filtered outcome trials (lat >= min_lat_ms)
    filt_pairs    = [(ct, ot, lat) for ct, ot, lat in all_pairs if lat >= min_lat_s]
    filt_out_times = np.array([p[1] for p in filt_pairs])
    n_filt         = len(filt_pairs)

    all_units  = list(ephys.keys())
    u2col      = {u: i for i, u in enumerate(all_units)}
    n_units    = len(all_units)
    n_cue_bins = len(cue_centers)
    n_out_bins = len(out_centers)
    n_bout_bins= len(bout_centers)

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)

    def _hist(ref_times, edges, n_bins):
        n = len(ref_times)
        fr = np.zeros((n, n_bins, n_units))
        for ti, t in enumerate(ref_times):
            for ui, u in enumerate(all_units):
                spk = np.asarray(ephys[u].index)
                h, _ = np.histogram(spk - t, bins=edges)
                fr[ti, :, ui] = h * bin_width
        if zscore:
            fr = (fr - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
        return fr

    q_fr   = _hist(q_cue_times,  cue_edges,  n_cue_bins)
    out_fr = _hist(filt_out_times, out_edges, n_out_bins)

    # Lick-bout PSTH
    bout_rows = bs_meta[bs_meta['event'] == 'lick_bout_starts']
    if not bout_rows.empty:
        task_interval = get_task_interval(binary_signals, trial_events, exclude_opto=True)
        bout_ts_obj   = binary_signals[bout_rows.index[0]].restrict(task_interval)
        bout_times    = np.asarray(bout_ts_obj.index)
        bout_fr       = _hist(bout_times, bout_edges, n_bout_bins)
    else:
        bout_times = np.array([])
        bout_fr    = np.zeros((0, n_bout_bins, n_units))

    sorted_idx = np.argsort(q_lats)
    q_assign   = np.empty(n_q, dtype=int)
    q_assign[sorted_idx] = np.arange(n_q) * n_quintiles // n_q
    mean_lats  = [q_lats[q_assign == q].mean() for q in range(n_quintiles)]

    tranche_units = {}
    for u in all_units:
        lbl = _get_classif(u)
        if lbl is not None:
            tranche_units.setdefault(lbl, []).append(u)
    dspn_cols = np.array([u2col[u] for u in tranche_units.get('dSPN', [])])
    ispn_cols  = np.array([u2col[u] for u in tranche_units.get('iSPN', [])])
    has_diff   = len(dspn_cols) > 0 and len(ispn_cols) > 0

    row_labels = ['dSPN', 'iSPN'] + (['dSPN-iSPN'] if has_diff else [])
    n_rows     = len(row_labels)

    def _stats(fr, cols, idx):
        pt  = fr[idx][:, :, cols].mean(axis=2)
        m   = pt.mean(axis=0)
        sem = pt.std(axis=0) / np.sqrt(max(len(idx), 1))
        return m, m - 1.96 * sem, m + 1.96 * sem

    def _diff_stats(fr, idx):
        pt  = fr[idx][:, :, dspn_cols].mean(axis=2) - fr[idx][:, :, ispn_cols].mean(axis=2)
        m   = pt.mean(axis=0)
        sem = pt.std(axis=0) / np.sqrt(max(len(idx), 1))
        return m, m - 1.96 * sem, m + 1.96 * sem

    qcmap     = _turbo_dark
    qnorm     = plt.Normalize(vmin=float(q_lats.min()), vmax=float(q_lats.max()))
    fr_ylabel = 'Z(FR)' if zscore else 'spks/s'
    fsuffix   = '' if zscore else '_raw'
    row_colors = {'dSPN': 'tab:blue', 'iSPN': 'tab:orange', 'dSPN-iSPN': 'black'}
    _fmt       = _FuncFormatter(lambda x, _: f'{x:g}')

    bout_idx = np.arange(len(bout_times))
    filt_idx = np.arange(n_filt)

    fig, axs = plt.subplots(n_rows, 3, sharex='col', sharey='row', squeeze=False)

    for ri, row_label in enumerate(row_labels):
        is_diff = row_label == 'dSPN-iSPN'
        cols    = dspn_cols if row_label == 'dSPN' else ispn_cols
        color   = row_colors[row_label]

        # --- Col 0: lick-bout PSTH ---
        ax0 = axs[ri, 0]
        if len(bout_idx) > 0:
            m, lo, hi = _diff_stats(bout_fr, bout_idx) if is_diff else _stats(bout_fr, cols, bout_idx)
            ax0.fill_between(bout_centers, lo, hi, color=color, alpha=0.2, linewidth=0)
            ax0.plot(bout_centers, m, color=color, linewidth=1.5)
        if is_diff:
            ax0.axhline(0, color='k', linewidth=0.8, alpha=0.5)
        ax0.axvline(0, color='k', linewidth=1, alpha=0.75)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.set_ylabel(fr_ylabel)

        # --- Col 1: outcome-aligned (all trials lat >= min_lat_ms) ---
        ax1 = axs[ri, 1]
        if len(filt_idx) > 0:
            m, lo, hi = _diff_stats(out_fr, filt_idx) if is_diff else _stats(out_fr, cols, filt_idx)
            ax1.fill_between(out_centers, lo, hi, color=color, alpha=0.2, linewidth=0)
            ax1.plot(out_centers, m, color=color, linewidth=1.5)
        if is_diff:
            ax1.axhline(0, color='k', linewidth=0.8, alpha=0.5)
        ax1.axvline(0, color='k', linewidth=1, alpha=0.75)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # --- Col 2: cue-aligned quintile ---
        ax2 = axs[ri, 2]
        for q in range(n_quintiles):
            qi = np.where(q_assign == q)[0]
            if len(qi) == 0:
                continue
            qcolor = qcmap(qnorm(mean_lats[q]))
            if is_diff:
                y = (q_fr[qi][:, :, dspn_cols].mean(axis=2) -
                     q_fr[qi][:, :, ispn_cols].mean(axis=2)).mean(axis=0)
            else:
                y = q_fr[qi][:, :, cols].mean(axis=(0, 2))
            ax2.axvline(mean_lats[q], color=qcolor, linestyle='--', linewidth=1.0, alpha=0.7)
            ax2.plot(cue_centers, y, color=qcolor, linewidth=1.0, alpha=0.95)
        if is_diff:
            ax2.axhline(0, color='k', linewidth=0.8, alpha=0.5)
        ax2.axvline(0, color='k', linewidth=1, alpha=0.75)
        ax2.axvspan(3.33, 7.0, ymin=0.92, ymax=1.0, color='green', alpha=0.45, linewidth=0)
        ax2.axvspan(0.0,  3.33, ymin=0.92, ymax=1.0, color='red',   alpha=0.45, linewidth=0)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Tranche label on rightmost column
        ax_r = ax2.twinx()
        ax_r.set_ylabel(row_label, rotation=-90, labelpad=8)
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_visible(False)

        if ri == n_rows - 1:
            ax0.set_xlabel('t(s) to lick bout')
            ax1.set_xlabel('t(s) to 1st lick')
            ax2.set_xlabel('t(s) from cue')

    # x-limits
    axs[0, 0].set_xlim(bout_centers[0],  bout_centers[-1])
    axs[0, 1].set_xlim(out_centers[0],   out_centers[-1])
    axs[0, 2].set_xlim(cue_centers[0],   cue_centers[-1])

    for ax in axs.ravel():
        ax.label_outer()
        ax.yaxis.set_major_formatter(_fmt)

    psu.adjust_figure_for_panel_size_hetero(fig, panel_width=0.75, panel_height=0.6, t=0.35, ws=0.2)
    fig_w, fig_h = fig.get_size_inches()

    # Latency colorbar above col 2 (quintile column)
    pos     = axs[0, 2].get_position()
    cbar_ax = fig.add_axes([pos.x0, (fig_h - 0.3) / fig_h, pos.width, 0.05 / fig_h])
    sm = plt.cm.ScalarMappable(cmap=qcmap, norm=qnorm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                        label='Mean t(s) to 1st lick')
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    if save_path is not None:
        (save_path.parent / (save_path.stem + fsuffix + '.txt')).write_text(
            f'SPN summary\nQuintile n={n_q}, Outcome-filtered n={n_filt}, '
            f'Lick-bout n={len(bout_times)}'
        )
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig


def plot_dspn_isPN_diff(ephys, binary_signals, trial_events,
                        save_path=None, bin_width=0.05,
                        t_start=-1.0, t_end=7.0, n_quintiles=5,
                        t_start_outcome=-8.0, t_end_outcome=1.0,
                        zscore=True, ms_censored=None):
    """dSPN - iSPN population difference. Produces 6 separate figures:
      1. All conditions, outcome-aligned mean trace per condition  [-8, +1 s]
      2. No-stim only,  outcome-aligned mean trace               [-8, +1 s]
      3. All conditions, cue-aligned, split by lick-latency quintile
      4. No-stim only,  cue-aligned, split by lick-latency quintile
      5. No-stim rewarded only, outcome-aligned mean trace
      6. No-stim unrewarded only, outcome-aligned mean trace
    zscore=True: activity is z-scored per unit. zscore=False: raw firing rates (spikes/s).
    Returns (fig1, fig2, fig3, fig4, fig5, fig6, cbar_fig)."""
    bs_meta = binary_signals.metadata
    cue_key = bs_meta[bs_meta['event'] == 'start_cues'].index[0]
    cue_ts = binary_signals[cue_key].t

    def _keep_outcome(ot):
        """Return False for timeout trials and (if ms_censored set) early-lick trials."""
        prec = cue_ts[cue_ts < ot]
        if len(prec) == 0:
            return False
        lat = float(ot - prec[-1])
        if lat >= 6.9:
            return False
        if ms_censored is not None and lat < ms_censored / 1000.0:
            return False
        return True

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

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)

    def _bin_trials(times):
        ep = nap.IntervalSet(start=times + t_start_outcome, end=times + t_end_outcome)
        binned = ephys.restrict(ep).count(bin_size=bin_width) * bin_width
        fr_list = []
        for s, e in zip(ep.start, ep.end):
            td = binned.restrict(nap.IntervalSet(start=s, end=e))
            fr_list.append(td.values)
        fr = np.array(fr_list)
        nb = fr.shape[1]
        bc = np.arange(nb) * bin_width + t_start_outcome + bin_width / 2
        if zscore:
            fr = (fr - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
        return fr, bc, nb

    def _diff(fr, nb, trial_idx):
        d = fr[np.ix_(trial_idx, np.arange(nb), d_cols)].mean(axis=2)
        i = fr[np.ix_(trial_idx, np.arange(nb), i_cols)].mean(axis=2)
        return d - i

    ylabel = 'dSPN - iSPN\nz-score' if zscore else 'dSPN - iSPN\nspks/s'
    fsuffix = '' if zscore else '_raw'

    def _ax_decor(ax, xlabel):
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    # ── Outcome-aligned data (figs 1 & 2) ────────────────────────────────────
    out_times_list, out_conds_list = [], []
    for ck in ('no_stim', 'chr2', 'chrimson'):
        ots = np.array([o for o in trial_events.get(ck, []) if _keep_outcome(o)])
        out_times_list.append(ots)
        out_conds_list.extend([ck] * len(ots))
    all_out_times = np.concatenate(out_times_list)
    sort_ord = np.argsort(all_out_times)
    all_out_times = all_out_times[sort_ord]
    all_out_conds = [out_conds_list[i] for i in sort_ord]

    out_fr_z, out_bc, out_nb = _bin_trials(all_out_times)
    ns_out_idx = np.array([i for i, c in enumerate(all_out_conds) if c == 'no_stim'])

    # ── Figure 1: all conditions, outcome-aligned mean ────────────────────────
    fig1, ax = plt.subplots(figsize=(1, 1))
    for ck, (color, ls, label) in cond_style.items():
        idx = np.array([i for i, c in enumerate(all_out_conds) if c == ck])
        if len(idx) == 0:
            continue
        diff = _diff(out_fr_z, out_nb, idx)
        mean = diff.mean(axis=0)
        ci95 = 1.96 * diff.std(axis=0) / np.sqrt(len(idx))
        ax.plot(out_bc, mean, color=color, linestyle=ls, linewidth=1.5, label=label)
        ax.fill_between(out_bc, mean - ci95, mean + ci95, color=color, alpha=0.2)

    _ax_decor(ax, 'Time to 1st lick (s)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    psu.adjust_figure_for_panel_size_auto(fig1)
    if save_path is not None:
        (save_path.parent / (save_path.stem + '_allcond_mean' + fsuffix + '.txt')).write_text(
            f'{title_base}\nAll conditions - 1st lick-aligned mean'
        )
        fig1.savefig(save_path.parent / (save_path.stem + '_allcond_mean' + fsuffix + '.svg'), bbox_inches='tight')
        fig1.savefig(save_path.parent / (save_path.stem + '_allcond_mean' + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    # ── Figure 2: no-stim only, outcome-aligned mean ──────────────────────────
    fig2, ax = plt.subplots(figsize=(1, 1))
    if len(ns_out_idx) > 0:
        diff = _diff(out_fr_z, out_nb, ns_out_idx)
        mean = diff.mean(axis=0)
        ci95 = 1.96 * diff.std(axis=0) / np.sqrt(len(ns_out_idx))
        ax.plot(out_bc, mean, color='k', linewidth=1.5)
        ax.fill_between(out_bc, mean - ci95, mean + ci95, color='k', alpha=0.2)

    _ax_decor(ax, 't(s) to 1st lick')
    psu.adjust_figure_for_panel_size_auto(fig2)
    if save_path is not None:
        (save_path.parent / (save_path.stem + '_nostim_mean' + fsuffix + '.txt')).write_text(
            f'{title_base}\nNo Stim only - 1st lick-aligned mean\nn={len(ns_out_idx)} trials'
        )
        fig2.savefig(save_path.parent / (save_path.stem + '_nostim_mean' + fsuffix + '.svg'), bbox_inches='tight')
        fig2.savefig(save_path.parent / (save_path.stem + '_nostim_mean' + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    # ── Split no-stim trials into rewarded vs unrewarded ─────────────────────
    # Use trial_events keys set at categorization time to avoid float cross-comparison
    rew_ts = set(float(t) for t in trial_events.get('no_stim_reward', []))
    err_ts = set(float(t) for t in trial_events.get('no_stim_error', []))
    ns_rew_idx = np.array([i for i in ns_out_idx if float(all_out_times[i]) in rew_ts])
    ns_err_idx = np.array([i for i in ns_out_idx if float(all_out_times[i]) in err_ts])

    # ── Figure 5: no-stim rewarded, outcome-aligned mean ─────────────────────
    fig5, ax = plt.subplots(figsize=(1, 1))
    if len(ns_rew_idx) > 0:
        diff = _diff(out_fr_z, out_nb, ns_rew_idx)
        mean = diff.mean(axis=0)
        ci95 = 1.96 * diff.std(axis=0) / np.sqrt(len(ns_rew_idx))
        ax.plot(out_bc, mean, color='k', linewidth=1.5)
        ax.fill_between(out_bc, mean - ci95, mean + ci95, color='k', alpha=0.2)

    _ax_decor(ax, 't(s) to 1st lick')
    psu.adjust_figure_for_panel_size_auto(fig5)
    if save_path is not None:
        (save_path.parent / (save_path.stem + '_nostim_rewarded' + fsuffix + '.txt')).write_text(
            f'{title_base}\nNo Stim, Rewarded - 1st lick-aligned mean\nn={len(ns_rew_idx)} trials'
        )
        fig5.savefig(save_path.parent / (save_path.stem + '_nostim_rewarded' + fsuffix + '.svg'), bbox_inches='tight')
        fig5.savefig(save_path.parent / (save_path.stem + '_nostim_rewarded' + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    # ── Figure 6: no-stim unrewarded, outcome-aligned mean ───────────────────
    fig6, ax = plt.subplots(figsize=(1, 1))
    if len(ns_err_idx) > 0:
        diff = _diff(out_fr_z, out_nb, ns_err_idx)
        mean = diff.mean(axis=0)
        ci95 = 1.96 * diff.std(axis=0) / np.sqrt(len(ns_err_idx))
        ax.plot(out_bc, mean, color='k', linewidth=1.5)
        ax.fill_between(out_bc, mean - ci95, mean + ci95, color='k', alpha=0.2)

    _ax_decor(ax, 't(s) to 1st lick')
    psu.adjust_figure_for_panel_size_auto(fig6)
    if save_path is not None:
        (save_path.parent / (save_path.stem + '_nostim_unrewarded' + fsuffix + '.txt')).write_text(
            f'{title_base}\nNo Stim, Unrewarded - 1st lick-aligned mean\nn={len(ns_err_idx)} trials'
        )
        fig6.savefig(save_path.parent / (save_path.stem + '_nostim_unrewarded' + fsuffix + '.svg'), bbox_inches='tight')
        fig6.savefig(save_path.parent / (save_path.stem + '_nostim_unrewarded' + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    # ── Cue-aligned data (figs 3 & 4) ────────────────────────────────────────
    cond_trials = {}
    for cond in ('no_stim', 'chr2', 'chrimson'):
        trials = []
        for ot in trial_events.get(cond, []):
            if not _keep_outcome(ot):
                continue
            prec = cue_ts[cue_ts < ot]
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
    if zscore:
        cue_fr_z = (cue_fr - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]
    else:
        cue_fr_z = cue_fr

    cmap = _turbo_dark
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
            diff = _diff(cue_fr_z, cue_nb, trial_idx[q_sub])
            mean = diff.mean(axis=0)
            ci95 = 1.96 * diff.std(axis=0) / np.sqrt(len(q_sub))
            ax.plot(cue_bc, mean, color=color, linewidth=1.5)
            ax.fill_between(cue_bc, mean - ci95, mean + ci95, color=color, alpha=0.2)
            ax.axvline(mean_lat, color=color, linestyle='--', linewidth=0.8, alpha=0.6)

    # ── Figure 3: all conditions, cue-aligned, latency-binned ────────────────
    fig3, ax = plt.subplots(figsize=(1, 1))
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
            diff = _diff(cue_fr_z, cue_nb, idx[q_sub])
            mean = diff.mean(axis=0)
            ci95 = 1.96 * diff.std(axis=0) / np.sqrt(len(q_sub))
            ax.plot(cue_bc, mean, color=tcolor, linestyle=ls, linewidth=1.2)
            ax.fill_between(cue_bc, mean - ci95, mean + ci95, color=tcolor, alpha=0.15)
            ax.axvline(mean_lat, color=tcolor, linestyle='--', linewidth=0.6, alpha=0.5)

    _ax_decor(ax, 't(s) from cue')
    ax.set_ylim(-1, 1)
    for ck, (color, ls, label) in cond_style.items():
        ax.plot([], [], color='grey', linestyle=ls, linewidth=1.2, label=label)
    ax.legend(title='Linestyle = condition\nColor = latency quintile', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    psu.adjust_figure_for_panel_size_auto(fig3)
    if save_path is not None:
        (save_path.parent / (save_path.stem + '_allcond_latency' + fsuffix + '.txt')).write_text(
            f'{title_base}\nAll conditions - cue-aligned, by latency quintile'
        )
        fig3.savefig(save_path.parent / (save_path.stem + '_allcond_latency' + fsuffix + '.svg'), bbox_inches='tight')
        fig3.savefig(save_path.parent / (save_path.stem + '_allcond_latency' + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    # ── Figure 4: no-stim only, cue-aligned, latency-binned ──────────────────
    fig4, ax = plt.subplots(figsize=(1, 1))
    if len(ns_cue_idx) > 0:
        _quintile_lines(ax, ns_cue_idx, all_cue_lats[ns_cue_idx])

    _ax_decor(ax, 't(s) from cue')
    ax.set_ylim(-1, 1)
    psu.adjust_figure_for_panel_size_auto(fig4)
    if save_path is not None:
        (save_path.parent / (save_path.stem + '_nostim_latency' + fsuffix + '.txt')).write_text(
            f'{title_base}\nNo Stim only - cue-aligned, latency quintile\nn={len(ns_cue_idx)} trials'
        )
        fig4.savefig(save_path.parent / (save_path.stem + '_nostim_latency' + fsuffix + '.svg'), bbox_inches='tight')
        fig4.savefig(save_path.parent / (save_path.stem + '_nostim_latency' + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    # ── Shared colorbar for latency figures ───────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.06, 1))
    cbar_fig.colorbar(sm, cax=cbar_ax, label='Mean lick latency (s)')
    if save_path is not None:
        cbar_fig.savefig(save_path.parent / (save_path.stem + '_latency_colorbar' + fsuffix + '.svg'), bbox_inches='tight')
        cbar_fig.savefig(save_path.parent / (save_path.stem + '_latency_colorbar' + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig1, fig2, fig3, fig4, fig5, fig6, cbar_fig


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
    rew_ts = _ts('reward_tones')
    err_ts = _ts('early_tones')

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
    fig, ax = plt.subplots(figsize=(1, 1))

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
    ax.set_xlabel('t(s) from cue')
    ax.set_ylabel('Trial')

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
    ax.legend(handles=legend_handles, ncol=1, framealpha=0.9,
              loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    psu.adjust_figure_for_panel_size_auto(fig)
    if save_path is not None:
        save_path.with_suffix('.txt').write_text(f'Trial raster\nn={n_trials} trials')
        fig.savefig(save_path.with_suffix('.svg'), bbox_inches='tight')
        fig.savefig(save_path.with_suffix('.png'), dpi=150, bbox_inches='tight')

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

    fig_bar, ax = plt.subplots(figsize=(1, 1))
    ax.bar(range(3), means, yerr=sems, capsize=5, width=0.5,
           color=[cond_colors[ck] for ck in cond_order],
           edgecolor='k', linewidth=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([cond_labels[ck] for ck in cond_order])
    ax.set_ylabel('t(s) to\n1st lick')

    y_top   = max(m + s for m, s in zip(means, sems))
    y_range = max(means) - min(means) if max(means) != min(means) else 1.0
    bar_stats_lines = ['Latency to 1st lick by condition']
    for ck in cond_order:
        bar_stats_lines.append(f'{cond_labels[ck].replace(chr(10), " ")}: mean={lats[ck].mean():.3f}s, n={len(lats[ck])}')
    for level, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        a, b = lats[cond_order[i]], lats[cond_order[j]]
        if len(a) < 2 or len(b) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(a, b)
        p_str = f'p={p_val:.3f}' if p_val >= 0.001 else f'p={p_val:.2e}'
        bh   = y_top + y_range * (0.18 + level * 0.20)
        tick = y_range * 0.04
        ax.plot([i, i, j, j], [bh - tick, bh, bh, bh - tick], color='k', linewidth=0.8)
        ax.text((i + j) / 2, bh + tick * 0.3, p_str, ha='center', va='bottom')
        la = cond_labels[cond_order[i]].replace('\n', ' ')
        lb = cond_labels[cond_order[j]].replace('\n', ' ')
        bar_stats_lines.append(f'{la} vs {lb}: t={t_stat:.3f}, {p_str}')

    psu.adjust_figure_for_panel_size_auto(fig_bar)
    if save_path is not None:
        (save_path.parent / (save_path.stem + '_bar' + '.txt')).write_text('\n'.join(bar_stats_lines))
        fig_bar.savefig(save_path.parent / (save_path.stem + '_bar' + '.svg'), bbox_inches='tight')
        fig_bar.savefig(save_path.parent / (save_path.stem + '_bar' + '.png'), dpi=150, bbox_inches='tight')

    # ── CDF ───────────────────────────────────────────────────────────────────
    fig_cdf, ax = plt.subplots(figsize=(1, 1))
    for ck in cond_order:
        if len(lats[ck]) == 0:
            continue
        sl = np.sort(lats[ck])
        ax.plot(sl, np.arange(1, len(sl) + 1) / len(sl),
                color=cond_colors[ck], linewidth=1.5,
                label=f"{cond_labels[ck].replace(chr(10), ' ')} (n={len(sl)})")

    ax.set_xlabel('Latency to\n1st lick (s)')
    ax.set_ylabel('Cumulative probability')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

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
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    psu.adjust_figure_for_panel_size_auto(fig_cdf)
    if save_path is not None:
        cdf_txt = ['CDF of t(s) to 1st lick'] + ks_lines
        (save_path.parent / (save_path.stem + '_cdf' + '.txt')).write_text('\n'.join(cdf_txt))
        fig_cdf.savefig(save_path.parent / (save_path.stem + '_cdf' + '.svg'), bbox_inches='tight')
        fig_cdf.savefig(save_path.parent / (save_path.stem + '_cdf' + '.png'), dpi=150, bbox_inches='tight')

    return fig_bar, fig_cdf


def plot_dspn_isPN_bar(ephys, binary_signals, trial_events,
                       save_path=None, bin_width=0.05,
                       pre_window=(-0.2, 0.0),
                       ms_censored=None):
    """Bar: dSPN − iSPN z-scored activity in pre-cue vs pre-outcome window, no-stim only.

    Outputs three figures: all no-stim, rewarded only, early (error) only.
    pre_window: (offset_start, offset_end) in seconds relative to reference event.
    ms_censored: removes trials with outcome within this many ms of cue. Timeout trials
    (outcome >= 6.9 s after cue) are always excluded."""
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
        return None, None, None

    u2col = {u: i for i, u in enumerate(ephys.keys())}
    d_cols = np.array([u2col[u] for u in d_units])
    i_cols = np.array([u2col[u] for u in i_units])

    z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)

    def _zscore_diff(ref_t):
        """Mean z-scored dSPN − iSPN activity in pre_window relative to ref_t."""
        t0, t1 = ref_t + pre_window[0], ref_t + pre_window[1]
        ep = nap.IntervalSet(start=t0, end=t1)
        binned = ephys.restrict(ep).count(bin_size=bin_width) * bin_width
        if len(binned.values) == 0:
            return np.nan
        z_val = (binned.values.mean(axis=0) - z_mu) / z_sd
        return float(z_val[d_cols].mean() - z_val[i_cols].mean())

    def _build_diffs(outcome_times):
        precue_diffs, preout_diffs = [], []
        for ot in outcome_times:
            prec = cue_ts[cue_ts < ot]
            if len(prec) == 0:
                continue
            lat = float(ot - prec[-1])
            if lat >= 6.9:
                continue
            if ms_censored is not None and lat < ms_censored / 1000.0:
                continue
            ct = prec[-1]
            d_pre = _zscore_diff(ct)
            d_out = _zscore_diff(ot)
            if not (np.isnan(d_pre) or np.isnan(d_out)):
                precue_diffs.append(d_pre)
                preout_diffs.append(d_out)
        return np.array(precue_diffs), np.array(preout_diffs)

    lo_ms, hi_ms = int(pre_window[0] * 1000), int(pre_window[1] * 1000)

    def _swarm_overlay(ax, x_center, values):
        """Overlay beeswarm dots at x_center for the given values."""
        try:
            import seaborn as sns
            sns.swarmplot(x=np.full(len(values), x_center), y=values,
                          ax=ax, color='k', size=2.0, alpha=0.45, zorder=5,
                          native_scale=True, warn_thresh=1.0)
        except Exception:
            rng = np.random.default_rng(42)
            jitter = rng.uniform(-0.15, 0.15, size=len(values))
            ax.scatter(x_center + jitter, values,
                       color='k', s=8, alpha=0.4, zorder=5, linewidths=0)

    def _make_bar_fig(diffs, title):
        """Single bar: mean(pre-lick − pre-cue) ± SEM, one-sample t-test vs 0."""
        n = len(diffs)
        fig, ax = plt.subplots(figsize=(1, 1))
        if n < 2:
            ax.text(0.5, 0.5, f'too few trials (n={n})', transform=ax.transAxes,
                    ha='center', va='center', color='grey', fontsize=8)
            psu.adjust_figure_for_panel_size_auto(fig, panel_width=0.8, panel_height=0.8)
            return fig, f'{title}\ntoo few trials (n={n})'

        t_stat, p_val = stats.ttest_1samp(diffs, 0)
        mean_d = diffs.mean()
        sem_d  = diffs.std(ddof=1) / np.sqrt(n)

        ax.bar([0], [mean_d], yerr=[sem_d], capsize=3, width=0.6,
               color='steelblue', edgecolor='k', linewidth=0.5,
               alpha=0.9, zorder=2)
        _swarm_overlay(ax, 0, diffs)
        ax.axhline(0, color='k', linewidth=0.8)
        ax.set_xticks([0])
        ax.set_xticklabels([f'pre-lick\n− pre-cue\n({lo_ms} to {hi_ms} ms)'],
                           fontsize=8)
        ax.set_ylabel('Δ dSPN-iSPN\nZ(FR)', fontsize=8)
        ax.tick_params(labelsize=8)

        span  = max(diffs.max() - diffs.min(), abs(mean_d) * 2, 1e-6)
        ann_y = max(diffs.max(), mean_d + sem_d) + span * 0.08
        tick_h = span * 0.03
        ax.plot([-0.15, 0.15], [ann_y, ann_y], color='k', linewidth=1.0)
        stars  = ('****' if p_val < 0.0001 else '***' if p_val < 0.001 else
                  '**'   if p_val < 0.01   else '*'   if p_val < 0.05  else 'ns')
        text_y = ann_y + tick_h * 0.5
        y_bot  = min(diffs.min(), mean_d - sem_d) - span * 0.05
        ax.set_ylim(y_bot, text_y + tick_h)
        ax.text(0, text_y, stars, ha='center', va='bottom', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        psu.adjust_figure_for_panel_size_auto(fig, panel_width=0.8, panel_height=0.8)
        p_str = f'p={p_val:.3f}' if p_val >= 0.001 else f'p={p_val:.2e}'
        return fig, f'{title}\nn={n}, t={t_stat:.2f}, {p_str} ({stars})'

    title_base = f'dSPN-iSPN  (ndSPN={len(d_units)}, niSPN={len(i_units)})'

    pre_c,   pre_o   = _build_diffs(trial_events.get('no_stim',        []))
    pre_c_r, pre_o_r = _build_diffs(trial_events.get('no_stim_reward', []))
    pre_c_e, pre_o_e = _build_diffs(trial_events.get('no_stim_error',  []))

    fig_all, txt_all = _make_bar_fig(pre_o   - pre_c,   f'{title_base}\nNo Stim — all')
    fig_rew, txt_rew = _make_bar_fig(pre_o_r - pre_c_r, f'{title_base}\nNo Stim — rewarded')
    fig_err, txt_err = _make_bar_fig(pre_o_e - pre_c_e, f'{title_base}\nNo Stim — early (error)')

    if save_path is not None:
        stem = save_path.stem
        for fig, tag, txt in [(fig_all, '_all', txt_all), (fig_rew, '_rewarded', txt_rew), (fig_err, '_error', txt_err)]:
            (save_path.parent / (stem + tag + '.txt')).write_text(txt)
            fig.savefig(save_path.parent / (stem + tag + '.svg'), bbox_inches='tight')
            fig.savefig(save_path.parent / (stem + tag + '.png'), dpi=150, bbox_inches='tight')

    return fig_all, fig_rew, fig_err


def plot_dspn_ispn_lick_bout_bar(ephys, binary_signals, trial_events,
                                  save_path=None, bin_width=0.05,
                                  pre_bout_window=(-0.2, 0.0),
                                  baseline_window=(-1.2, -1.0)):
    """Bar: dSPN − iSPN z-scored activity in pre_bout_window vs baseline_window,
    both measured relative to each lick_bout_start event.

    Bouts are restricted to the task period with opto excluded.
    Paired t-test across bouts. Returns (fig, summary_str).
    """
    from scipy import stats

    bs_meta = binary_signals.metadata
    emeta   = ephys.metadata.copy()

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    d_units = [u for u in ephys.keys() if _get_classif(u) == 'dSPN']
    i_units = [u for u in ephys.keys() if _get_classif(u) == 'iSPN']
    if not d_units or not i_units:
        print('plot_dspn_ispn_lick_bout_bar: no dSPN or iSPN units found')
        return None, None

    u2col  = {u: i for i, u in enumerate(ephys.keys())}
    d_cols = np.array([u2col[u] for u in d_units])
    i_cols = np.array([u2col[u] for u in i_units])

    z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)

    def _zscore_diff(ref_t, window):
        t0, t1 = ref_t + window[0], ref_t + window[1]
        ep     = nap.IntervalSet(start=t0, end=t1)
        binned = ephys.restrict(ep).count(bin_size=bin_width) * bin_width
        if len(binned.values) == 0:
            return np.nan
        z_val = (binned.values.mean(axis=0) - z_mu) / z_sd
        return float(z_val[d_cols].mean() - z_val[i_cols].mean())

    # Lick-bout events restricted to task interval (opto excluded)
    bout_rows = bs_meta[bs_meta['event'] == 'lick_bout_starts']
    if bout_rows.empty:
        print('plot_dspn_ispn_lick_bout_bar: no lick_bout_starts event found')
        return None, None

    task_interval = get_task_interval(binary_signals, trial_events, exclude_opto=True)
    bout_ts       = binary_signals[bout_rows.index[0]].restrict(task_interval)
    bout_times    = np.asarray(bout_ts.index)

    pre_bout_diffs, baseline_diffs = [], []
    for bt in bout_times:
        d_pre  = _zscore_diff(bt, pre_bout_window)
        d_base = _zscore_diff(bt, baseline_window)
        if not (np.isnan(d_pre) or np.isnan(d_base)):
            pre_bout_diffs.append(d_pre)
            baseline_diffs.append(d_base)

    diffs  = np.array(pre_bout_diffs) - np.array(baseline_diffs)
    n      = len(diffs)
    title  = (f'dSPN-iSPN lick-bout  '
              f'(ndSPN={len(d_units)}, niSPN={len(i_units)})')

    fig, ax = plt.subplots(figsize=(1, 1))
    if n < 2:
        ax.text(0.5, 0.5, f'too few bouts (n={n})', transform=ax.transAxes,
                ha='center', va='center', color='grey', fontsize=8)
        psu.adjust_figure_for_panel_size_auto(fig, panel_width=0.8, panel_height=0.8)
        return fig, f'{title}\ntoo few bouts (n={n})'

    t_stat, p_val = stats.ttest_1samp(diffs, 0)
    mean_d = diffs.mean()
    sem_d  = diffs.std(ddof=1) / np.sqrt(n)

    ax.bar([0], [mean_d], yerr=[sem_d], capsize=3, width=0.6,
           color='steelblue', edgecolor='k', linewidth=0.5,
           alpha=0.9, zorder=2)

    try:
        import seaborn as sns
        sns.swarmplot(x=np.zeros(n), y=diffs, ax=ax,
                      color='k', size=2.0, alpha=0.45, zorder=5,
                      native_scale=True, warn_thresh=1.0)
    except Exception:
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.15, 0.15, size=n)
        ax.scatter(jitter, diffs, color='k', s=8, alpha=0.4, zorder=5, linewidths=0)

    ax.axhline(0, color='k', linewidth=0.8)

    pre_lo = int(pre_bout_window[0] * 1000)
    pre_hi = int(pre_bout_window[1] * 1000)
    ax.set_xticks([0])
    ax.set_xticklabels([f'pre-bout − baseline\n({pre_lo} to {pre_hi} ms)'], fontsize=8)
    ax.set_ylabel('Δ dSPN-iSPN\nZ(FR)', fontsize=8)
    ax.tick_params(labelsize=8)

    span   = max(diffs.max() - diffs.min(), abs(mean_d) * 2, 1e-6)
    ann_y  = max(diffs.max(), mean_d + sem_d) + span * 0.08
    tick_h = span * 0.03
    ax.plot([-0.15, 0.15], [ann_y, ann_y], color='k', linewidth=1.0)
    stars  = ('****' if p_val < 0.0001 else '***' if p_val < 0.001 else
              '**'   if p_val < 0.01   else '*'   if p_val < 0.05  else 'ns')
    text_y = ann_y + tick_h * 0.5
    y_bot  = min(diffs.min(), mean_d - sem_d) - span * 0.05
    ax.set_ylim(y_bot, text_y + tick_h)
    ax.text(0, text_y, stars, ha='center', va='bottom', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    psu.adjust_figure_for_panel_size_auto(fig, panel_width=0.8, panel_height=0.8)

    p_str   = f'p={p_val:.3f}' if p_val >= 0.001 else f'p={p_val:.2e}'
    summary = f'{title}\nn={n}, t={t_stat:.2f}, {p_str} ({stars})'

    if save_path is not None:
        save_path = Path(save_path)
        (save_path.parent / (save_path.stem + '.txt')).write_text(summary)
        fig.savefig(save_path.parent / (save_path.stem + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + '.png'), dpi=150, bbox_inches='tight')

    return fig, summary


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


def plot_spn_lick_bout_psth(ephys, binary_signals, trial_events, save_path=None,
                             bin_width=0.05, t_start=-2.0, t_end=1.0, zscore=True):
    """Population PSTH aligned to lick_bout_starts, split by dSPN / iSPN.

    Style matches the middle column of plot_spn_summary: mean ± 95% CI per
    cell-type row, with a dSPN-iSPN difference row when both are present.
    Restricted to the behavioural testing period with opto excluded.
    Saves as SVG + PNG when save_path is provided (stem used for filenames).
    Returns the matplotlib Figure.
    """
    from matplotlib.ticker import FuncFormatter as _FuncFormatter

    bs_meta = binary_signals.metadata
    bout_rows = bs_meta[bs_meta['event'] == 'lick_bout_starts']
    if len(bout_rows) == 0:
        print("No 'lick_bout_starts' event found in binary_signals metadata; skipping.")
        return None

    task_interval = get_task_interval(binary_signals, trial_events, exclude_opto=True)
    bout_ts = binary_signals[bout_rows.index[0]].restrict(task_interval)
    if len(bout_ts) == 0:
        print("No lick_bout_starts events within task interval; skipping.")
        return None

    bout_times = np.asarray(bout_ts.index)
    bin_edges   = np.arange(t_start, t_end + bin_width / 2, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n_bins  = len(bin_centers)
    n_bouts = len(bout_times)

    emeta     = ephys.metadata.copy()
    all_units = list(ephys.keys())
    n_units   = len(all_units)
    u2col     = {u: i for i, u in enumerate(all_units)}

    if zscore:
        z_mu, z_sd = compute_unit_z(ephys, binary_signals, trial_events, bin_width)

    fr = np.zeros((n_bouts, n_bins, n_units))
    for ti, bt in enumerate(bout_times):
        for ui, u in enumerate(all_units):
            spk = np.asarray(ephys[u].index)
            h, _ = np.histogram(spk - bt, bins=bin_edges)
            fr[ti, :, ui] = h * bin_width
    if zscore:
        fr = (fr - z_mu[np.newaxis, np.newaxis, :]) / z_sd[np.newaxis, np.newaxis, :]

    def _get_classif(u):
        if 'final_classif' in emeta.columns and u in emeta.index:
            c = emeta.loc[u, 'final_classif']
            if c in ('dSPN', 'iSPN'):
                return c
        return None

    tranche_units = {}
    for u in all_units:
        lbl = _get_classif(u)
        if lbl is not None:
            tranche_units.setdefault(lbl, []).append(u)

    dspn_cols = np.array([u2col[u] for u in tranche_units.get('dSPN', [])])
    ispn_cols  = np.array([u2col[u] for u in tranche_units.get('iSPN', [])])
    has_diff   = len(dspn_cols) > 0 and len(ispn_cols) > 0

    row_labels = ['dSPN', 'iSPN'] + (['dSPN-iSPN'] if has_diff else [])
    n_rows     = len(row_labels)

    def _stats(cols):
        pt  = fr[:, :, cols].mean(axis=2)
        m   = pt.mean(axis=0)
        sem = pt.std(axis=0) / np.sqrt(max(len(pt), 1))
        return m, m - 1.96 * sem, m + 1.96 * sem

    def _diff_stats():
        pt  = fr[:, :, dspn_cols].mean(axis=2) - fr[:, :, ispn_cols].mean(axis=2)
        m   = pt.mean(axis=0)
        sem = pt.std(axis=0) / np.sqrt(max(len(pt), 1))
        return m, m - 1.96 * sem, m + 1.96 * sem

    _fmt      = _FuncFormatter(lambda x, _: f'{x:g}')
    fr_ylabel = 'Z(FR)' if zscore else 'spks/s'
    fsuffix   = '' if zscore else '_raw'
    colors    = {'dSPN': 'green', 'iSPN': 'maroon', 'dSPN-iSPN': 'black'}

    fig, axs = plt.subplots(n_rows, 1, squeeze=False)

    for ri, row_label in enumerate(row_labels):
        ax       = axs[ri, 0]
        is_diff  = row_label == 'dSPN-iSPN'
        color    = colors[row_label]

        m, lo, hi = _diff_stats() if is_diff else _stats(
            dspn_cols if row_label == 'dSPN' else ispn_cols)

        ax.fill_between(bin_centers, lo, hi, color=color, alpha=0.2, linewidth=0)
        ax.plot(bin_centers, m, color=color, linewidth=1.5)
        if is_diff:
            ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
        ax.axvline(0, color='k', linewidth=1, alpha=0.75)
        ax.set_xlim(bin_centers[0], bin_centers[-1])
        ax.set_ylabel(fr_ylabel, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_major_formatter(_fmt)
        ax.tick_params(labelsize=8)
        if ri < 2:
            ax.set_ylim([0.0,0.6])
            ax.set_yticks([0, 0.5])
        else:
            ax.set_ylim(-0.2, 0.2)
            ax.set_yticks([-0.1, 0.0, 0.1])

        ax_r = ax.twinx()
        ax_r.set_ylabel(row_label, rotation=-90, labelpad=8, fontsize=8)
        ax_r.set_yticks([])
        for spine in ax_r.spines.values():
            spine.set_visible(False)

        if ri == n_rows - 1:
            ax.set_xlabel('t(s) to lick bout start', fontsize=8)
        else:
            ax.tick_params(labelbottom=False)

    psu.adjust_figure_for_panel_size_hetero(fig, panel_width=0.8, panel_height=0.6, t=0.35)

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.svg'), bbox_inches='tight')
        fig.savefig(save_path.parent / (save_path.stem + fsuffix + '.png'), dpi=150, bbox_inches='tight')

    return fig
