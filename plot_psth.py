import pynapple as nap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use('Qt5Agg')
import re
#joblib
from joblib import Parallel, delayed

def plot_psth(timestamps, tref, unitID, event_ID, save_folder=None, minmax=(-1, 1), off_event=None):

    peth = nap.compute_perievent(
        timestamps=timestamps,
        tref=tref,
        minmax=minmax,
        time_unit="s")

    #if off_event is a ts or array
    if off_event is not None and hasattr(off_event, 't'):
        off_events = off_event.t-tref.t
        off_event = np.median(off_events)

    spikes = peth.to_tsd()
    #check that spikes is not empty
    if len(spikes) == 0:
        print(f"No spikes found for Unit {unitID} around event {event_ID}")
        return

    fig, axs = plt.subplots(2,1, figsize=(10, 6), sharex=True)

    ax = axs[0]
    ax.plot(np.mean(peth.count(0.01), 1) / 0.01, linewidth=2, color="black")
    ax.set_ylabel("Rate (spikes/sec, 0.01s bins)")

    ax = axs[1]
    #get the number of trials
    ntrials = max(peth) - min(peth)
    ms = -1/1000 * ntrials  + 3
    ax.plot(spikes, "|", markersize=ms, color="black", mew=1)
    ax.set_ylabel("trial/event #")
    ax.set_xlabel("time from event (s)")

    for i in range(2):
        ax = axs[i]
        ax.set_xlim(minmax)
        ax.axvline(0.0)
        if off_event is not None:
            ax.axvline(off_event, color='black', linestyle='--')
    #add a suptitle
    suptitle = f"Unit {unitID} PSTH around event {event_ID}"
    fig.suptitle(suptitle)
    if save_folder is not None:
        fig.savefig(save_folder / f"Unit_{unitID}_PSTH_event_{event_ID}.png")
        plt.close(fig)
    else:
        plt.show()

def plot_on_events_psth(ephys_data, sigs, pynapple_folder):
    sigs_metadata = sigs.metadata
    on_events_idx = sigs_metadata[sigs_metadata['event'].str.contains('_on')].index.tolist()

    neurons_idx = ephys_data.keys()

    psth_folder = pynapple_folder / "psth_plots"
    psth_folder.mkdir(exist_ok=True)
    for event in on_events_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder
            ) for neuron in neurons_idx)

    cue_events_idx = sigs_metadata[sigs_metadata['event'].str.contains('_cues')].index.tolist()
    for event in cue_events_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder,
                minmax=(-2, 2)
            ) for neuron in neurons_idx)

    first_licks_idx = sigs_metadata[sigs_metadata['event'].str.contains('_first_licks')].index.tolist()
    for event in first_licks_idx:
        tref = sigs[event]
        event_name = sigs_metadata.loc[event, 'event']
        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                save_folder=psth_folder,
                minmax=(-4, 1)
            ) for neuron in neurons_idx)

    opto_events = sigs_metadata[sigs_metadata['event'].str.contains('chr')]
    opto_off_events_idx = opto_events[opto_events['event'].str.contains('_off')].index.tolist()
    #only get opto events that end with '_on'
    opto_on_events_idx = opto_events[opto_events['event'].str.contains('_on')].index.tolist()
    for i, event in enumerate(opto_on_events_idx):
        tref = sigs[event]
        off_event = sigs[opto_off_events_idx[i]]
        event_name = sigs_metadata.loc[event, 'event']

        parallel_ = Parallel(n_jobs=-1, verbose=5)
        parallel_(
            delayed(plot_psth)(
                ephys_data[neuron],
                tref,
                neuron,
                event_name,
                off_event=off_event,
                save_folder=psth_folder,
                minmax=(-0.5,0.5)
            ) for neuron in neurons_idx)


def _extract_power(event_name):
    """Extract the trailing integer from an event name like 'chrimson_on_40'."""
    m = re.search(r"(\d+)$", str(event_name))
    return int(m.group(1)) if m else None


def _draw_opto_psth_on_axes(timestamps, opsin, power_trefs, power_off_events,
                            ax_psth, ax_raster, minmax=(-0.5, 0.5)):
    """
    Draw a combined PSTH + raster for one unit / one opsin onto the provided axes.

    Parameters
    ----------
    timestamps : nap.Ts
        Spike timestamps for a single unit.
    opsin : str
        Opsin name (used only for the column title).
    power_trefs : list of (power, tref)
        Sorted by power.
    power_off_events : list of (power, nap.Ts | None)
        Matching off-event timestamps.
    ax_psth : matplotlib.axes.Axes
        Axes for the PSTH (top).
    ax_raster : matplotlib.axes.Axes
        Axes for the raster (bottom).
    minmax : tuple
        Pre/post window in seconds.
    """
    if len(power_trefs) == 0:
        ax_psth.set_title(opsin, fontsize=15)
        ax_psth.text(0.5, 0.5, 'no events', transform=ax_psth.transAxes,
                     ha='center', va='center', fontsize=14, color='grey')
        return

    cmap = plt.cm.viridis
    n_powers = len(power_trefs)
    colors = [cmap(i / max(n_powers - 1, 1)) for i in range(n_powers)]

    # --- compute peri-event for each power level ---
    all_peths = []
    all_spike_tsds = []
    trial_counts = []
    off_medians = []

    for (power, tref), (_, off_ts) in zip(power_trefs, power_off_events):
        peth = nap.compute_perievent(timestamps=timestamps, tref=tref,
                                     minmax=minmax, time_unit="s")
        spikes_tsd = peth.to_tsd()
        all_peths.append(peth)
        all_spike_tsds.append(spikes_tsd)
        n_trials = int(max(peth) - min(peth)) + 1 if len(spikes_tsd) > 0 else len(tref)
        trial_counts.append(n_trials)

        if off_ts is not None and hasattr(off_ts, 't') and len(off_ts.t) > 0:
            off_rel = off_ts.t - tref.t
            off_medians.append(float(np.median(off_rel)))
        else:
            off_medians.append(None)

    total_trials = sum(trial_counts)
    if total_trials == 0:
        ax_psth.set_title(opsin, fontsize=15)
        return

    # ---- PSTH ----
    bin_size = 0.005
    all_rates = []
    for i, (power, _) in enumerate(power_trefs):
        rate = np.mean(all_peths[i].count(bin_size), axis=1) / bin_size
        all_rates.append(rate)
        ax_psth.plot(rate, linewidth=0.75, alpha=0.75, color=colors[i],
                     label=f"{power} µW")
    # grand average
    if len(all_rates) > 0:
        grand_avg = np.mean(
            np.column_stack([r.values if hasattr(r, 'values') else r for r in all_rates]),
            axis=1)
        ref = all_rates[0]
        if hasattr(ref, 'index'):
            ax_psth.plot(ref.index, grand_avg, linewidth=2, color='black', label='mean')
        else:
            ax_psth.plot(grand_avg, linewidth=2, color='black', label='mean')
    ax_psth.set_ylabel("Rate (spikes/s)", fontsize=15)
    ax_psth.legend(fontsize=9, loc='lower left', bbox_to_anchor=(1.01, 0), borderaxespad=0,
                   ncol=1)
    ax_psth.set_title(opsin, fontsize=15)

    # ---- Raster ----
    trial_offset = 0
    for i, (power, _) in enumerate(power_trefs):
        spikes_tsd = all_spike_tsds[i]
        if len(spikes_tsd) == 0:
            trial_offset += trial_counts[i]
            continue
        spike_times = spikes_tsd.index
        trial_ids = spikes_tsd.values + trial_offset - min(all_peths[i])
        ms = max(-total_trials / 1000 * 1 + 3, 0.5)
        ax_raster.plot(spike_times, trial_ids, "|",
                       markersize=ms, color=colors[i], mew=0.8)
        trial_offset += trial_counts[i]

    cum = 0
    for i in range(n_powers - 1):
        cum += trial_counts[i]
        ax_raster.axhline(cum, color='grey', linewidth=0.5, linestyle='-')

    ax_raster.set_ylabel("Trial #", fontsize=15)
    ax_raster.set_xlabel("Time from stimulus onset (s)", fontsize=15)

    # vertical lines
    for ax in (ax_psth, ax_raster):
        ax.set_xlim(minmax)
        ax.axvline(0.0, color='black', linewidth=0.8)
        for i, off_med in enumerate(off_medians):
            if off_med is not None:
                ax.axvline(off_med, color=colors[i], linestyle='--',
                           linewidth=0.6, alpha=0.7)


_OPSIN_WF_COLORS = {'chr2': 'steelblue', 'chrimson': 'darkorange'}


def plot_opto_psth_single_unit(timestamps, opsin_data, unitID,
                               waveform_data=None, unit_label=None,
                               save_folder=None, minmax=(-0.5, 0.5)):
    """
    Plot chrimson and chr2 PSTH + raster side-by-side for one unit.
    When waveform_data is provided a third row of waveform plots is added
    below the raster (peak channel, spikes during opto-on intervals only).

    Parameters
    ----------
    timestamps : nap.Ts
        Spike timestamps for the unit.
    opsin_data : dict
        ``{opsin_name: (power_trefs, power_off_events)}`` for each opsin.
    unitID : int or str
        Unit identifier.
    waveform_data : dict or None
        ``{opsin_name: ndarray (n_spikes, n_samples) or None}``
        Pre-selected waveforms (peak channel) per opsin.
    save_folder : Path or None
    minmax : tuple
    """
    opsins = list(opsin_data.keys())
    n_opsins = len(opsins)
    if n_opsins == 0:
        return

    has_wf = waveform_data is not None

    if has_wf:
        fig = plt.figure(figsize=(8 * n_opsins, 10), layout='constrained')
        gs = gridspec.GridSpec(3, n_opsins, height_ratios=[1, 2.5, 2], figure=fig)
        axs = np.empty((3, n_opsins), dtype=object)
        for col in range(n_opsins):
            axs[0, col] = fig.add_subplot(gs[0, col])
            axs[1, col] = fig.add_subplot(gs[1, col], sharex=axs[0, col])
            axs[2, col] = fig.add_subplot(gs[2, col])
        for col in range(1, n_opsins):
            axs[0, col].sharey(axs[0, 0])
            axs[1, col].sharey(axs[1, 0])
            axs[2, col].sharey(axs[2, 0])
            axs[2, col].sharex(axs[2, 0])
    else:
        fig, axs = plt.subplots(2, n_opsins, figsize=(8 * n_opsins, 7),
                                sharex=True, sharey='row',
                                gridspec_kw={'height_ratios': [1, 2.5]})
        if n_opsins == 1:
            axs = axs[:, np.newaxis]

    for col, opsin in enumerate(opsins):
        power_trefs, power_off_events = opsin_data[opsin]
        _draw_opto_psth_on_axes(timestamps, opsin, power_trefs, power_off_events,
                                ax_psth=axs[0, col], ax_raster=axs[1, col],
                                minmax=minmax)

    if has_wf:
        for col, opsin in enumerate(opsins):
            ax = axs[2, col]
            wfs = waveform_data.get(opsin)
            color = _OPSIN_WF_COLORS.get(opsin, 'grey')
            if wfs is not None and len(wfs) > 0:
                for wf in wfs:
                    ax.plot(wf, color=color, alpha=0.2, linewidth=0.5)
                ax.plot(wfs.mean(axis=0), color='black', linewidth=2,
                        label=f'mean (n={len(wfs)})')
                ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
            else:
                ax.text(0.5, 0.5, 'no spikes in interval',
                        transform=ax.transAxes, ha='center', va='center', color='grey')
            ax.set_xlabel('Sample', fontsize=15)
            if col == 0:
                ax.set_ylabel('Amplitude (µV)', fontsize=15)

    label_str = f"  [{unit_label}]" if unit_label is not None else ""
    fig.suptitle(f"Unit {unitID}{label_str}  -  opto PSTH (all powers)", fontsize=18)
    if not has_wf:
        fig.tight_layout()

    if save_folder is not None:
        fig.savefig(save_folder / f"Unit_{unitID}_opto_all_powers_psth.png", dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _collect_opsin_events(bs_metadata, bs, opsin):
    """Return (power_trefs, power_off_events) for *opsin*, or None if no events."""
    on_mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
    on_events_df = bs_metadata[on_mask].copy()
    if on_events_df.empty:
        return None
    on_events_df['power'] = on_events_df['event'].apply(_extract_power)
    on_events_df = on_events_df.dropna(subset=['power'])
    on_events_df = on_events_df.sort_values('power')

    off_mask = bs_metadata['event'].str.contains(f'{opsin}_off', na=False)
    off_events_df = bs_metadata[off_mask].copy()
    off_events_df['power'] = off_events_df['event'].apply(_extract_power)

    power_trefs = []
    power_off_events = []
    for _, row in on_events_df.iterrows():
        power = int(row['power'])
        tref = bs[row.name]
        power_trefs.append((power, tref))

        off_row = off_events_df[off_events_df['power'] == power]
        off_ts = bs[off_row.index[0]] if not off_row.empty else None
        power_off_events.append((power, off_ts))

    return power_trefs, power_off_events


def _get_opto_on_off_times(bs, opsin):
    """Return (on_times, off_times) arrays for all opto pulses of the given opsin."""
    metadata = bs.metadata
    events = metadata[metadata['event'].str.contains(opsin, case=False, na=False)]
    events = events[~events['event'].str.contains('trace', na=False)]
    on_rows = events[events['event'].str.contains('_on', na=False)]
    off_rows = events[events['event'].str.contains('_off', na=False)]

    def _collect(rows):
        times = []
        for idx in rows.index:
            ts = bs[idx]
            times.extend(ts.t.tolist() if hasattr(ts, 't') else ts.index.tolist())
        return np.array(sorted(times))

    return _collect(on_rows), _collect(off_rows)


def _precompute_opto_waveforms(analyzer, bs, opsins):
    """
    For each unit, extract peak-channel waveforms that fall during opto-on
    intervals per opsin.

    Returns dict: {unit_id: {opsin: ndarray (n_spikes, n_samples) or None}}
    """
    waveforms_ext = analyzer.get_extension('waveforms')
    rsp_indices = analyzer.get_extension('random_spikes').get_data()
    spk_vec = analyzer.sorting.to_spike_vector()
    fs = analyzer.sampling_frequency
    unit_ids = analyzer.unit_ids

    opsin_intervals = {opsin: _get_opto_on_off_times(bs, opsin) for opsin in opsins}

    all_wf_data = {}
    for unit_id in unit_ids:
        waveforms = waveforms_ext.get_waveforms_one_unit(unit_id)  # (n_rsp, n_samples, n_ch)
        peak_ch = int(np.argmax(np.max(np.abs(waveforms.mean(axis=0)), axis=0)))

        unit_idx = list(unit_ids).index(unit_id)
        in_rsp = spk_vec['unit_index'][rsp_indices] == unit_idx
        times_s = spk_vec['sample_index'][rsp_indices[in_rsp]] / fs

        unit_wf = {}
        for opsin in opsins:
            on_times, off_times = opsin_intervals[opsin]
            if len(on_times) == 0:
                unit_wf[opsin] = None
                continue
            mask = np.zeros(len(times_s), dtype=bool)
            for on, off in zip(on_times, off_times):
                mask |= (times_s >= on) & (times_s <= off)
            selected = waveforms[mask, :, peak_ch]
            unit_wf[opsin] = selected if len(selected) > 0 else None

        all_wf_data[unit_id] = unit_wf

    return all_wf_data


def plot_all_opto_psth(ephys_data, bs, pynapple_folder, minmax=(-0.5, 0.5), analyzer=None):
    """
    For each unit, plot chrimson and chr2 PSTH + raster side-by-side in one
    figure, with raster rows ordered by ascending power.

    Parameters
    ----------
    ephys_data : nap.TsGroup-like
        Spike data keyed by unit id.
    bs : nap.TsGroup-like
        Binary-signals object with ``.metadata`` DataFrame containing an
        ``'event'`` column (e.g. ``'chrimson_on_40'``, ``'chr2_off_20'``).
    pynapple_folder : Path
        Base folder; figures are saved under ``opto_psth_plots/``.
    minmax : tuple
        Pre/post window in seconds (default ±0.5 s).
    """
    save_folder = pynapple_folder / "opto_psth_plots"
    save_folder.mkdir(exist_ok=True)

    bs_metadata = bs.metadata
    neurons_idx = ephys_data.keys()

    # collect event data for each opsin once (shared across all units)
    opsin_data = {}
    for opsin in ('chrimson', 'chr2'):
        result = _collect_opsin_events(bs_metadata, bs, opsin)
        if result is not None:
            opsin_data[opsin] = result

    if len(opsin_data) == 0:
        print("No chrimson or chr2 events found in bs.metadata; skipping opto PSTH.")
        return

    all_wf_data = None
    if analyzer is not None:
        print("Pre-computing opto waveforms...")
        all_wf_data = _precompute_opto_waveforms(analyzer, bs, list(opsin_data.keys()))

    metadata = ephys_data.metadata

    ks_col = next((c for c in ('KSLabel', 'ks_label') if c in metadata.columns), None)

    def _unit_label(neuron):
        parts = []
        if ks_col is not None:
            parts.append(f"KS:{metadata.loc[neuron, ks_col]}")
        return '  '.join(parts) if parts else None

    Parallel(n_jobs=-1, verbose=5)(
        delayed(plot_opto_psth_single_unit)(
            timestamps=ephys_data[neuron],
            opsin_data=opsin_data,
            unitID=neuron,
            waveform_data=all_wf_data.get(neuron) if all_wf_data is not None else None,
            unit_label=_unit_label(neuron),
            save_folder=save_folder,
            minmax=minmax
        ) for neuron in neurons_idx
    )


def test(base_folder=None):
    if base_folder is None:
        base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
    #    base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")

    pynapple_folder = base_folder / "pynapple"

    ephys_file = pynapple_folder / "spikes.npz"
    ephys_data = nap.load_file(ephys_file)
    sigs_file = pynapple_folder / "binary_signals.npz"
    sigs = nap.load_file(sigs_file)

    plot_on_events_psth(ephys_data, sigs, pynapple_folder)

def plot_all():
    base_folders = [
        Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250822"),
        Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905"),
    ]
    for base_folder in base_folders:
        test(base_folder=base_folder)

    #get sigs_metadata rows where binary_event_name contains 'on'

