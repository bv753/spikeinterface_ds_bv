import pynapple as nap
from pathlib import Path
import parse_nidq as pni
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

def plot_corellograms(ephys_data, bs, opto_events_idx, opto_tagging_folder):
    power_levels = opto_events_idx['power'].unique().tolist()
    correlograms_folder = opto_tagging_folder / "correlograms"
    correlograms_folder.mkdir(exist_ok=True)
    correlograms = []

    # iterate through rows of opto_events_idx
    for index, row in opto_events_idx.iterrows():
        print(index)
        event_corr = nap.compute_eventcorrelogram(
            group=ephys_data, event=bs[index], binsize=0.1, windowsize=0.5
        )
        correlograms.append(event_corr)

    for n, neuron in enumerate(ephys_data.keys()):
        #make a subplot 5 panels, 1 for each power level, and in each panel plot the correlogram for that power level,
        fig, axs = plt.subplots(1, len(power_levels), figsize=(20, 5), sharey=True)
        for p, power in enumerate(power_levels):
            ax = axs[p]
            #get indices of correlograms where power level matches
            chr2_indices = opto_events_idx[
                (opto_events_idx['power'] == power) &
                (opto_events_idx['event'].str.contains('chr2'))
                ].index.tolist()
            chrimson_indices = opto_events_idx[
                (opto_events_idx['power'] == power) &
                (opto_events_idx['event'].str.contains('chrimson'))
                ].index.tolist()
            #get the correlograms for those indices
            #plot chr2 in shades of blue and chrimson in shades of red
            for idx in chrimson_indices:
                event_corr = correlograms[idx]
                ax.plot(
                    event_corr.index,
                    event_corr[neuron].values,
                    color='red',#chrimson_cmap(p),
                    label='chrimson' if idx == chrimson_indices[0] else ""
                    )
            for idx in chr2_indices:
                event_corr = correlograms[idx]
                ax.plot(
                    event_corr.index,
                    event_corr[neuron].values,
                    color='blue',#chr2_cmap(p),
                    label='ChR2' if idx == chr2_indices[0] else ""
                    )
            ax.set_title(f'Power Level: {power}')
            ax.set_xlabel('Time (s)')
        #save
        fig.suptitle(f'Neuron {neuron} Opto Tagging Correlograms')
        axs[0].set_ylabel('correlation')
        axs[-1].legend()
        plt.tight_layout()
        plt.savefig(correlograms_folder / f'neuron_{neuron}_opto_tagging_correlograms.png')
        plt.close(fig)


def count_spikes_in_intervals(spike_times, starts, ends):
    counts = 0
    j = 0
    n = spike_times.shape[0]
    for i in range(starts.shape[0]):
        start = starts[i]
        end = ends[i]
        # Move spike index pointer until spike_times[j] >= start
        while j < n and spike_times[j] < start:
            j += 1
        k = j
        while k < n and spike_times[k] < end:
            counts += 1
            k += 1
    return counts

# ------------------------------------------------------------
# Utility #3: Circular shift spike train
# ------------------------------------------------------------
#@njit(fastmath=True, parallel=True)

def get_auc(tuning_curve, thresh):
    y = tuning_curve.values#[tuning_curve.values > thresh]
    y = y[1:]
    auc = np.sum(y)
    return auc

def shuff_aucs_diff(tuning_curve, thresh):
    curve1 = tuning_curve[0].values
    curve2 = tuning_curve[1].values
    mask = np.random.rand(len(curve1)) < 0.5

    # Use the mask to swap efficiently
    a_swapped = np.where(mask, curve2, curve1)
    b_swapped = np.where(mask, curve1, curve2)
    #compute aucs
    auc1 = np.sum(a_swapped[a_swapped > thresh][1:])
    auc2 = np.sum(b_swapped[b_swapped > thresh][1:])
    #compute difference
    auc_diff = auc1 - auc2
    return auc_diff


def get_tuning_curves(neuron_data, opsins, opsin_traces, power_bins, fs_array):
    tuning_curves = []
    for o, opsin in enumerate(opsins):
        fs = float(fs_array[o])
        trace = opsin_traces[o]
        tuning_curve = nap.compute_tuning_curves(
            data=neuron_data,
            features=trace,
            bins=[power_bins],
            fs=fs)
        #replace nan with 0 in tuning_curve
        tuning_curve = tuning_curve.fillna(0)
        tuning_curves.append(tuning_curve)

    return tuning_curves


def get_firing_rate_percentiles(spikes, n_bins=30, n_iter=10000, bin_size=0.01, conf_level=0.95):
    #verify that spikes is a Ts object
    if not isinstance(spikes, nap.Ts):
        raise ValueError("spikes must be a nap.Ts object")

    sumspikes = spikes.count(bin_size)
    #get percentiles of firing rates
    #bootstrap the percentiles by sampling n_bins with size bin_size 10000 times

    def subsample_rate(counts):
        #pick n_bins random indices
        sample = np.random.choice(counts, size=n_bins, replace=False)
        return np.sum(sample) / (n_bins * bin_size)

    samples = [subsample_rate(sumspikes.values) for _ in range(n_iter)]
    #get 2.5 and 97.5 percentiles
    samples = np.array(samples)
    low_pctl = np.percentile(samples, (1 - conf_level) / 2 * 100)
    high_pctl = np.percentile(samples, (1 + conf_level) / 2 * 100)
    #alternative method:
    percentile_thresholds = [low_pctl, high_pctl]

    return percentile_thresholds


def plot_tuning_curves(neuron, opsins, tuning_curves, pctl_thrsh, auc_diff_pctls, tuning_curves_folder):
    #for tunovth, make any value of tuning curves less than auc_diff_percentiles[1] equal to 0
    tunovth = [tuning_curve.copy() for tuning_curve in tuning_curves]
    for o in range(len(opsins)):
        tunovth[o].values[tunovth[o].values < pctl_thrsh[0]] = 0
        tunovth[o].values[0] = 0  #set the 0 power level to 0
    cusum = [tuning_curve.cumsum() for tuning_curve in tunovth]
    cumulative_difference = cusum[0] - cusum[1]

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax = axs[0]
    for o, opsin in enumerate(opsins):
        line = tuning_curves[o].plot.line(x='0', add_legend=False, ax=ax)[0]
        line.set_label(opsin)
        if opsin == 'chrimson':
            color = 'red'
        elif opsin == 'chr2':
            color = 'blue'
        else:
            print(f"Unknown opsin: {opsin}, defaulting to black")
            color = 'black'
        line.set_color(color)
        line.set_label(opsin)
    #add horizontal lines for pctl_thrsh
    tiles = ['null 95th %-tile', '99th %-tile']
    linestyles = ['--', '-.']
    for i, thrsh in enumerate(pctl_thrsh):
        ax.axhline(thrsh, color='green',
                   linestyle=linestyles[i],
                   label=tiles[i],
                   alpha=0.7)
    ax.set_xlabel('')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.legend()
    ax.set_title('')

    ax = axs[1]

    colors = ['blue', 'red']
    for o, opsin in enumerate(opsins):
        line = cusum[o].plot.line(x='0', add_legend=False, ax=ax)[0]
        line.set_color(colors[o])
        line.set_label(f'Cumulative {opsin}')

    # plot the 5th and 95th percentiles as horizontal lines
    low_thrsh, high_thrsh = auc_diff_pctls
    ax.axhline(low_thrsh, color='orange', linestyle='--', label='shuff 5th Percentile', alpha=0.7)
    ax.axhline(high_thrsh, color='orange', linestyle='-.', label='shuff 95th Percentile', alpha=0.7)


    #now plot the cumulative difference
    line = cumulative_difference.plot.line(x='0', add_legend=False, ax=ax)[0]
    line.set_color('gray')
    line.set_label('Cumulative Difference')
    #set a dashed horizontal line at y=0
    ax.axhline(0, color='gray', linestyle='--')
    ax.legend()
    ax.set_ylabel('Firing Rate Difference (Hz)')
    ax.set_xlabel(r'Light Intensity ($\mu$W)')
    #remove any vestigal title
    ax.set_title('')

    plt.suptitle(f'Neuron {neuron} Opto Tagging Tuning Curves')
    plt.tight_layout()
    plt.savefig(tuning_curves_folder / f'neuron_{neuron}_opto_tagging_tuning_curves.png')
    plt.close()

def get_opto_epochs(bs, opsin):
    bs_metadata = bs.metadata
    events_idx = bs_metadata[bs_metadata['event'].str.contains(opsin)]
    events_idx = events_idx[~events_idx['event'].str.contains('trace')]
    events_on_idx = events_idx[events_idx['event'].str.contains('_on')]
    off_events_idx = events_idx[events_idx['event'].str.contains('_off')]

    on_events = bs[events_on_idx.index].to_tsd().t  # add 10 ms to account for delay
    off_events = bs[off_events_idx.index].to_tsd().t  # subtract 10 ms to account for delay

    prestim_start = on_events - 1.01
    prestim_end = on_events - 0.01

    opto_on_events = on_events + 0.005
    opto_off_events = off_events - 0.005
    opto_duration = np.round(np.mean(off_events - on_events),3)

    # concat iti_starts and opto_on_events
    opto_on_events = np.concatenate((opto_on_events, prestim_start))
    opto_off_events = np.concatenate((opto_off_events, prestim_end))
    # sort the timestamps
    sorted_indices = np.argsort(opto_on_events)
    opto_on_events = opto_on_events[sorted_indices]
    opto_off_events = opto_off_events[sorted_indices]

    opto_on_events_ts = nap.Ts(opto_on_events, time_units='s')
    opto_off_events_ts = nap.Ts(opto_off_events, time_units='s')
    opto_epochs = nap.IntervalSet(start=opto_on_events_ts.t, end=opto_off_events_ts.t)

    #double check that duration is not zero
    if opto_duration == 0:
        raise ValueError("Opto duration is zero, check opto events")
    return opto_epochs, opto_duration

def get_opto_trace(bs, opsin):
    bs_metadata = bs.metadata
    trace_events_df = bs_metadata[bs_metadata['event'].str.contains('trace')]
    idx = trace_events_df[trace_events_df['event'].str.contains(opsin)].index.tolist()
    trace = bs[idx[0]]
    return trace

def get_powers_and_bins(trace):
    powers = np.unique(trace.values)
    # get midpoints between power levels for binning
    mids = (powers[:-1] + powers[1:]) / 2
    lower = powers[0] - (powers[1] - powers[0]) / 2
    upper = powers[-1] + (powers[-1] - powers[-2]) / 2
    power_bins = np.concatenate(([lower], mids, [upper]))
    return powers, power_bins

def get_sampling_rate(trace):
    fs = int(1 / np.mean(trace.time_diff().values))
    return fs

def get_percentile_thresholds(ephys_data, avg_opto_duration):
    percentile_thresholds95 = Parallel(n_jobs=-1)(delayed(get_firing_rate_percentiles)(
        ephys_data[neuron], n_bins=30, n_iter=10000, bin_size=avg_opto_duration, conf_level=0.95) for neuron in tqdm(ephys_data.keys()))
    percentile_thresholds95 = np.array(percentile_thresholds95)[:,1]

    percentile_thresholds99 = Parallel(n_jobs=-1)(delayed(get_firing_rate_percentiles)(
        ephys_data[neuron], n_bins=30, n_iter=10000, bin_size=avg_opto_duration, conf_level=0.99) for neuron in tqdm(ephys_data.keys()))
    percentile_thresholds99 = np.array(percentile_thresholds99)[:,1]

    return percentile_thresholds95, percentile_thresholds99

def get_auc_diff_percentiles(ephys_data, tun, percentile_thresholds95, n_iter=10000):
    auc_diff_percentiles = []
    for n, neuron in enumerate(ephys_data.keys()):
        diffs = []
        for _ in range(n_iter):
            diffs.append(shuff_aucs_diff(tun[n], percentile_thresholds95[n]))
        #get the 5th and 95th percentiles
        diffs = np.array(diffs)
        low_pctl = np.percentile(diffs, 2.5)
        high_pctl = np.percentile(diffs, 97.5)
        auc_diff_percentiles.append((low_pctl, high_pctl))
    return auc_diff_percentiles

def add_tuning_curve_metadata(ephys_data, tuning_curves, opsins, percentile_thresholds95, percentile_thresholds99, aucs, auc_diff_percentiles):
    #get the neurons that have any part of their tuning curve above the 95th percentile
    metadata = ephys_data.metadata
    significant_95 = []
    significant_99 = []
    for o, opsin in enumerate(opsins):
        tc = tuning_curves[o]
        significant95 = tc.values > percentile_thresholds95[:][:, np.newaxis]
        significant95 = np.any(significant95[:,1:], axis=1)
        significant_95.append(significant95)
        significant99 = tc.values > percentile_thresholds99[:][:, np.newaxis]
        significant99 = np.any(significant99[:,1:], axis=1)
        significant_99.append(significant99)
        metadata[f'significant_{opsin}_95'] = significant95
        metadata[f'significant_{opsin}_99'] = significant99
        metadata[f'auc_{opsin}'] = aucs[o]

    metadata['auc_diff'] = metadata['auc_chr2'] - metadata['auc_chrimson']
    metadata['auc_diff_5th_percentile'] = [auc_diff_percentiles[n][0] for n in range(len(ephys_data.keys()))]
    metadata['auc_diff_95th_percentile'] = [auc_diff_percentiles[n][1] for n in range(len(ephys_data.keys()))]
    metadata['significant_auc_diff'] = (metadata['auc_diff'] <= metadata['auc_diff_5th_percentile']) | (metadata['auc_diff'] >= metadata['auc_diff_95th_percentile'])

    metadata['iSPN'] = (metadata['significant_chrimson_95'] & (metadata['auc_chr2'] < metadata['auc_chrimson']) & metadata['significant_auc_diff'])
    metadata['dSPN'] = (metadata['significant_chr2_95'] & (metadata['auc_chrimson'] < metadata['auc_chr2']) & metadata['significant_auc_diff'])

    #remove rate from metadata if it exists
    if 'rate' in metadata.columns:
        metadata = metadata.drop(columns=['rate'])
    return metadata


def add_ci_to_tuning_curve(tuning_curve, spikes, feature, epochs=None, n_bootstrap=10, ci=95):
    """
    Add bootstrapped confidence intervals to an existing tuning curve xarray.

    Parameters:
    -----------
    tuning_curve : xarray.DataArray
        Pre-computed tuning curve from nap.compute_tuning_curves
    spikes : nap.TsGroup
        Spike times for multiple units
    feature : nap.Tsd
        Feature trace (e.g., opto power)
    epochs : nap.IntervalSet or None
        Epochs to restrict analysis to
    n_bootstrap : int
        Number of bootstrap iterations per bin
    ci : float
        Confidence interval percentage

    Returns:
    --------
    tuning_curve : xarray.DataArray with added 'lower_ci' and 'upper_ci' coordinates
    """

    # Extract bin edges and other attributes
    bin_edges = tuning_curve.bin_edges[0]
    occupancy = tuning_curve.occupancy
    n_bins = len(bin_edges) - 1
    n_units = len(spikes)

    # Restrict feature to epochs if provided
    if epochs is not None:
        feature = feature.restrict(epochs)

    # Initialize CI arrays
    lower_bound = np.zeros((n_units, n_bins))
    upper_bound = np.zeros((n_units, n_bins))

    # For each unit
    for unit_idx, unit_id in enumerate(spikes.keys()):
        spike_times = spikes[unit_id].index

        # Restrict spikes to epochs if provided
        if epochs is not None:
            spike_times_epoch = spikes[unit_id].restrict(epochs).index
        else:
            spike_times_epoch = spike_times

        # For each bin
        for bin_idx in range(n_bins):
            # Find feature values in this bin
            bin_mask = (feature.values >= bin_edges[bin_idx]) & (feature.values < bin_edges[bin_idx + 1])

            if not np.any(bin_mask):
                lower_bound[unit_idx, bin_idx] = 0
                upper_bound[unit_idx, bin_idx] = 0
                continue

            # Get times when feature was in this bin
            bin_times = feature.index[bin_mask]

            # Total time in bin (in seconds)
            total_time = occupancy[bin_idx] / tuning_curve.fs

            if total_time == 0:
                lower_bound[unit_idx, bin_idx] = 0
                upper_bound[unit_idx, bin_idx] = 0
                continue

            # Count spikes in each feature timestamp within this bin
            dt = 1 / tuning_curve.fs / 2

            spike_counts = []
            for t in bin_times:
                n_spikes = np.sum((spike_times_epoch >= t - dt) & (spike_times_epoch < t + dt))
                spike_counts.append(n_spikes)

            spike_counts = np.array(spike_counts)

            # Bootstrap the firing rate
            bootstrap_rates = []
            for _ in range(n_bootstrap):
                resampled_counts = np.random.choice(spike_counts, size=len(spike_counts), replace=True)
                rate = np.sum(resampled_counts) / total_time
                bootstrap_rates.append(rate)

            # Calculate confidence intervals
            alpha = (100 - ci) / 2
            lower_bound[unit_idx, bin_idx] = np.percentile(bootstrap_rates, alpha)
            upper_bound[unit_idx, bin_idx] = np.percentile(bootstrap_rates, 100 - alpha)

    # Add CIs as new coordinates to the xarray
    tuning_curve = tuning_curve.assign_coords({
        'lower_ci': (tuning_curve.dims, lower_bound),
        'upper_ci': (tuning_curve.dims, upper_bound)
    })

    return tuning_curve

def plot_all_tuning_curves(ephys_data, bs, save_folder=None, overwrite=True, njobs=-1):
    # tuning curve approach
    # get indices of bs metadata containing the word 'trace'
    if save_folder is not None:
        opto_tagging_folder = save_folder / "opto_tagging"
        opto_tagging_folder.mkdir(exist_ok=True)
        tuning_curves_folder = opto_tagging_folder / "tuning_curves"
        tuning_curves_folder.mkdir(exist_ok=True)
    else:
        tuning_curves_folder = None


    opsins = ['chr2', 'chrimson']
    opto_durations = []
    tuning_curves = []
    ciLs = []
    ciHs = []
    n_bootstrap = 1000
    ci = 95
    for opsin in opsins:
        opto_epochs, duration = get_opto_epochs(bs, opsin)
        trace = get_opto_trace(bs, opsin)
        powers, bins = get_powers_and_bins(trace)
        fs = get_sampling_rate(trace)

        tuning_curve = nap.compute_tuning_curves(
            data=ephys_data,
            features=trace,
            bins=[bins],
            epochs=opto_epochs,
            fs=fs)

        #tuning_curve = add_ci_to_tuning_curve(tuning_curve,
        #    ephys_data, trace,
        #    epochs=opto_epochs, n_bootstrap=n_bootstrap, ci=ci
        #)

        tuning_curves.append(tuning_curve)
        opto_durations.append(duration)


    percentile_thresholds95, percentile_thresholds99 = get_percentile_thresholds(ephys_data, np.mean(opto_durations))

    aucs = []
    for o, opsin in enumerate(opsins):
        auc = [get_auc(tc, percentile_thresholds95[i]) for i, tc in enumerate(tuning_curves[o])]
        aucs.append(auc)
        print(f"Computed tuning curves and AUCs for opsin: {opsin}")

    #reverse packing of tuning_curves to have list of tuning curves per neuron
    tun = [*zip(*tuning_curves)]

    auc_diff_percentiles = get_auc_diff_percentiles(ephys_data, tun, percentile_thresholds95, n_iter=10000)

    metadata = add_tuning_curve_metadata(ephys_data, tuning_curves, opsins, percentile_thresholds95, percentile_thresholds99, aucs, auc_diff_percentiles)

    #combine percentile thresholds into a list of tuples
    percentile_thresholds = [(percentile_thresholds95[n], percentile_thresholds99[n]) for n in range(len(ephys_data.keys()))]
    Parallel(n_jobs=njobs)(delayed(plot_tuning_curves)(
        neuron, opsins, tun[n], percentile_thresholds[n], auc_diff_percentiles[n], tuning_curves_folder) for n, neuron in tqdm(enumerate(ephys_data.keys())))

    if overwrite:
        ephys_file = save_folder / "spikes.npz"
        ephys_data.set_info(metadata=metadata)
        ephys_data.save(str(ephys_file))

    return metadata

def test():
    #base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
    base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250905")
    #base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")
    pynapple_folder = base_folder / "pynapple"
    bs = pni.get_binary_signals(base_folder, overwrite=False)
    ephys_file = pynapple_folder / "spikes.npz"
    ephys_data = nap.load_file(str(ephys_file))
    #edtest = nap.load_file(str(ephys_file))
    opto_tagging_folder = pynapple_folder / "opto_tagging"
    opto_tagging_folder.mkdir(exist_ok=True)

    bs_metadata = bs.metadata
    # get bs metadata row indices where binary_event_name contains 'chrimson_on' or 'chr2_on'
    opto_events_idx = bs_metadata[bs_metadata['event'].str.contains('chrimson|chr2')]
    opto_events_on_idx = opto_events_idx[opto_events_idx['event'].str.contains('_on')]
    # 'event' column entries have a number at the end, extract that number and convert to int
    opto_events_on_idx['power'] = opto_events_on_idx['event'].str.extract(r'(\d+)$')[0].astype(int).tolist()


    metadata = plot_all_tuning_curves(ephys_data, bs, opto_events_idx, opto_tagging_folder)
    #remove the 'rate' column from metadata if it exists
    if 'rate' in metadata.columns:
        metadata = metadata.drop(columns=['rate'])
    ephys_data.set_info(metadata = metadata)
    ephys_data.save(str(ephys_file))


#TODO: plot and compare waveform of evoked spikes and other spikes

#opto_events_idx = opto_events_idx.reset_index(drop=True)

#chr2_events_idx = opto_events_idx[opto_events_idx['event'].str.contains('chr2_on')]
#chrimson_events_idx = opto_events_idx[opto_events_idx['event'].str.contains('chrimson_on')]
#chrimson_events_idx = chrimson_events_idx.index.tolist()


def test_stim_firing(ephys_data, bs, opsins=None, post_window_ms=(1.0, 31.0),
                     baseline_duration_s=1, baseline_gap_ms=1.0,
                     baseline_bin_ms=10.0, alpha=0.05, save_folder=None):
    """
    For each unit and each opsin, test whether post-stimulus firing
    (1–31 ms after each opto-on event) is significantly elevated compared
    to a pre-stimulus baseline using the Mann–Whitney U test, and plot a
    bar graph of baseline vs evoked firing rate for every unit.

    Baseline: for a given opsin, collect **all** on-event times across every
    power level.  For each on-event, take the window from −*baseline_duration_s*
    to −*baseline_gap_ms* ms before onset, divide into non-overlapping
    *baseline_bin_ms*-ms bins, and count spikes in each.  This shared baseline
    distribution is used for all power-level comparisons of that opsin.

    Post-stimulus: the spike count in the [post_window_ms[0], post_window_ms[1]]
    ms window after each opto-on event (one count per trial per power).

    All counts are converted to firing rates (Hz) by dividing by bin width.

    Parameters
    ----------
    ephys_data : nap.TsGroup-like
        Spike trains keyed by unit id.
    bs : object
        Binary-signals object with ``.metadata`` and ``bs[idx]`` access.
    opsins : list of str or None
        Opsin prefixes to test (default ``['chrimson', 'chr2']``).
    post_window_ms : tuple of float
        (start, end) of the post-stimulus counting window in ms (default 1–31).
    baseline_duration_s : float
        Seconds before stimulus onset used for baseline (default 5).
    baseline_gap_ms : float
        Dead-zone between baseline end and stimulus onset in ms (default 1).
    baseline_bin_ms : float
        Width of each baseline bin in ms (default 30, matching post window).
    alpha : float
        Significance threshold (default 0.05).
    save_folder : Path or None
        If provided, a bar-graph PNG is saved per unit into
        ``save_folder / stim_firing_plots``.

    Returns
    -------
    results_df : pandas.DataFrame
        One row per unit × opsin with columns:
        ``['unit', 'opsin', 'power', 'n_trials', 'rate_post_hz',
          'rate_baseline_hz', 'U_stat', 'p_value', 'significant']``
    summary_df : pandas.DataFrame
        One row per unit.  Contains two sets of columns:

        *MWU comparisons* (pooled-trial post vs baseline):
        ``mwu_stat/pval_chrimson_vs_baseline``,
        ``mwu_stat/pval_chr2_vs_baseline``,
        ``mwu_stat/pval_chrimson_vs_chr2``,
        ``mwu_stat/pval_chrimson_baseline_vs_chr2_baseline``,
        ``tagged``.

        *Spearman dose-response* (rate ~ power, including power=0 baseline):
        ``spearman_rho_chrimson``, ``spearman_pval_chrimson`` (one-sided, H1: ρ>0),
        ``spearman_rho_chr2``,     ``spearman_pval_chr2``,
        ``delta_rho`` (ρ_chrimson − ρ_chr2),
        ``fisher_z_stat``, ``fisher_z_pval`` (two-sided, tests whether the
        two dose-response slopes are significantly different).

        *Hierarchical cell-type label*:
        ``dominant_opsin``, ``cell_type`` —
        'iSPN' (chrimson-specific), 'dSPN' (chr2-specific),
        'muSPN' (both significant but indistinguishable), NaN (not tagged).
    """
    import re
    import pandas as pd
    from scipy.stats import mannwhitneyu, spearmanr, norm as _norm
    from pathlib import Path

    if opsins is None:
        opsins = ['chrimson', 'chr2']

    bs_metadata = bs.metadata
    post_start_s = post_window_ms[0] / 1000.0
    post_end_s = post_window_ms[1] / 1000.0
    post_bin_s = post_end_s - post_start_s
    baseline_bin_s = baseline_bin_ms / 1000.0
    baseline_gap_s = baseline_gap_ms / 1000.0

    def _extr_power(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else None

    # ---- set up save folder ----
    if save_folder is not None:
        plot_folder = Path(save_folder) / "stim_firing_plots"
        plot_folder.mkdir(parents=True, exist_ok=True)
    else:
        plot_folder = None

    records = []
    # raw_rates[(unit, opsin)] = {'post', 'baseline', 'trial_powers', 'trial_baseline_means'}
    raw_rates = {}

    # Pre-collect per-opsin onset times AND their associated power levels
    opsin_info = {}
    for opsin in opsins:
        on_mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
        on_events_df = bs_metadata[on_mask].copy()
        if on_events_df.empty:
            continue
        on_events_df['power'] = on_events_df['event'].apply(_extr_power)
        on_events_df = on_events_df.dropna(subset=['power'])

        all_onsets = []
        all_powers_per_onset = []
        for idx in on_events_df.index:
            power = on_events_df.loc[idx, 'power']
            ts_obj = bs[idx]
            if hasattr(ts_obj, 't'):
                onsets_for_event = ts_obj.t.tolist()
            elif hasattr(ts_obj, 'index'):
                onsets_for_event = ts_obj.index.tolist()
            else:
                onsets_for_event = []
            all_onsets.extend(onsets_for_event)
            all_powers_per_onset.extend([float(power)] * len(onsets_for_event))

        if len(all_onsets) == 0:
            continue
        sort_idx = np.argsort(all_onsets)
        opsin_info[opsin] = {
            'all_onsets': np.array(all_onsets)[sort_idx],
            'all_powers': np.array(all_powers_per_onset)[sort_idx],
        }

    # ---- main loop: per unit ----
    units = list(ephys_data.keys())
    for unit in units:
        spike_times = np.asarray(ephys_data[unit].index)

        for opsin, info in opsin_info.items():
            all_onsets = info['all_onsets']
            all_powers = info['all_powers']

            # --- shared baseline: bins in [-baseline_duration_s, -baseline_gap_ms] ---
            n_bins_per_trial = int(baseline_duration_s / baseline_bin_s)
            baseline_counts = []
            for onset in all_onsets:
                bl_end = onset - baseline_gap_s
                bl_start = bl_end - baseline_duration_s
                for b in range(n_bins_per_trial):
                    bin_lo = bl_start + b * baseline_bin_s
                    bin_hi = bin_lo + baseline_bin_s
                    cnt = int(np.sum(
                        (spike_times >= bin_lo) & (spike_times < bin_hi)
                    ))
                    baseline_counts.append(cnt)
            baseline_counts = np.array(baseline_counts)
            baseline_rates = baseline_counts / baseline_bin_s  # Hz (all bins)

            # Per-trial mean baseline rate (one per onset, for Spearman at power=0)
            trial_baseline_means = np.array([
                int(np.sum((spike_times >= (onset - baseline_gap_s - baseline_duration_s))
                           & (spike_times < (onset - baseline_gap_s))))
                / baseline_duration_s
                for onset in all_onsets
            ], dtype=float)

            # --- post-stimulus counts (one per trial) ---
            post_counts = np.empty(len(all_onsets), dtype=int)
            for i, onset in enumerate(all_onsets):
                win_start = onset + post_start_s
                win_end = onset + post_end_s
                post_counts[i] = int(np.sum(
                    (spike_times >= win_start) & (spike_times < win_end)
                ))
            post_rates = post_counts / post_bin_s  # Hz

            raw_rates[(unit, opsin)] = {
                'post':                 post_rates,
                'baseline':             baseline_rates,
                'trial_powers':         all_powers,
                'trial_post_rates':     post_rates,
                'trial_baseline_means': trial_baseline_means,
            }

            # --- Mann-Whitney U (one-sided: post > baseline) ---
            if len(post_rates) > 0 and len(baseline_rates) > 0:
                try:
                    u_stat, p_val = mannwhitneyu(
                        post_rates, baseline_rates, alternative='greater'
                    )
                except ValueError:
                    u_stat, p_val = np.nan, 1.0
            else:
                u_stat, p_val = np.nan, 1.0

            records.append({
                'unit': unit,
                'opsin': opsin,
                'power': 'all',
                'n_trials': len(all_onsets),
                'rate_post_hz': float(np.mean(post_rates)),
                'rate_baseline_hz': float(np.mean(baseline_rates)),
                'U_stat': u_stat,
                'p_value': p_val,
                'significant': p_val < alpha,
            })

    results_df = pd.DataFrame(records)

    # ---- Spearman dose-response helpers ----
    def _spearman_one_sided(powers, rates):
        """Spearman ρ(power, rate), one-sided p-value testing ρ > 0."""
        if len(powers) < 4:
            return np.nan, np.nan
        rho, p_two = spearmanr(powers, rates)
        p_one = float(p_two / 2) if rho > 0 else float(1.0 - p_two / 2)
        return float(rho), p_one

    def _fisher_z_test(rho1, n1, rho2, n2):
        """Two-sided Fisher z-test comparing two Spearman correlations."""
        if np.isnan(rho1) or np.isnan(rho2) or n1 < 4 or n2 < 4:
            return np.nan, np.nan
        z1 = np.arctanh(np.clip(rho1, -0.9999, 0.9999))
        z2 = np.arctanh(np.clip(rho2, -0.9999, 0.9999))
        se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
        z_diff = (z1 - z2) / se
        p_two = float(2.0 * _norm.sf(abs(z_diff)))
        return float(z_diff), p_two

    # ---- per-unit summary ----
    def _mwu(a, b, alternative='two-sided'):
        if len(a) >= 2 and len(b) >= 2:
            try:
                res = mannwhitneyu(a, b, alternative=alternative)
                return float(res.statistic), float(res.pvalue)
            except ValueError:
                pass
        return np.nan, np.nan

    summary_records = []
    for unit in units:
        ch  = raw_rates.get((unit, 'chrimson'), {})
        c2  = raw_rates.get((unit, 'chr2'),     {})
        ch_post = ch.get('post',     np.array([]))
        ch_bl   = ch.get('baseline', np.array([]))
        c2_post = c2.get('post',     np.array([]))
        c2_bl   = c2.get('baseline', np.array([]))

        # Existing MWU comparisons
        u_ch_bl, p_ch_bl = _mwu(ch_post, ch_bl,  alternative='greater')
        u_c2_bl, p_c2_bl = _mwu(c2_post, c2_bl,  alternative='greater')
        u_ch_c2, p_ch_c2 = _mwu(ch_post, c2_post, alternative='two-sided')
        u_bl_bl, p_bl_bl = _mwu(ch_bl,   c2_bl,   alternative='two-sided')

        opsin_vs_bl_sig = (
            (not np.isnan(p_ch_bl) and p_ch_bl < alpha) or
            (not np.isnan(p_c2_bl) and p_c2_bl < alpha)
        )
        tagged = bool(opsin_vs_bl_sig and (not np.isnan(p_ch_c2) and p_ch_c2 < alpha))

        # Spearman dose-response: combine stim trials (power > 0) with
        # per-trial mean baseline (power = 0) to anchor the curve.
        ch_powers = np.concatenate([np.zeros(len(ch.get('trial_baseline_means', []))),
                                    ch.get('trial_powers', np.array([]))])
        ch_rates  = np.concatenate([ch.get('trial_baseline_means', np.array([])),
                                    ch.get('trial_post_rates',     np.array([]))])
        c2_powers = np.concatenate([np.zeros(len(c2.get('trial_baseline_means', []))),
                                    c2.get('trial_powers', np.array([]))])
        c2_rates  = np.concatenate([c2.get('trial_baseline_means', np.array([])),
                                    c2.get('trial_post_rates',     np.array([]))])

        rho_ch, p_rho_ch = _spearman_one_sided(ch_powers, ch_rates)
        rho_c2, p_rho_c2 = _spearman_one_sided(c2_powers, c2_rates)

        delta_rho = (float(rho_ch) - float(rho_c2)
                     if not (np.isnan(rho_ch) or np.isnan(rho_c2)) else np.nan)
        fisher_z, fisher_pval = _fisher_z_test(rho_ch, len(ch_powers),
                                                rho_c2, len(c2_powers))

        # Hierarchical cell-type label
        ch_sig     = (not np.isnan(p_rho_ch)  and p_rho_ch  < alpha)
        c2_sig     = (not np.isnan(p_rho_c2)  and p_rho_c2  < alpha)
        fisher_sig = (not np.isnan(fisher_pval) and fisher_pval < alpha)

        if ch_sig and not c2_sig:
            cell_type = 'iSPN'
        elif c2_sig and not ch_sig:
            cell_type = 'dSPN'
        elif ch_sig and c2_sig:
            if fisher_sig:
                cell_type = 'iSPN' if (not np.isnan(delta_rho) and delta_rho > 0) else 'dSPN'
            else:
                cell_type = 'muSPN'
        else:
            cell_type = np.nan

        dominant_opsin = None
        if fisher_sig and not np.isnan(delta_rho):
            dominant_opsin = 'chrimson' if delta_rho > 0 else 'chr2'

        summary_records.append({
            'unit': unit,
            # MWU comparisons (kept for backwards compatibility)
            'mwu_stat_chrimson_vs_baseline':               u_ch_bl,
            'mwu_pval_chrimson_vs_baseline':               p_ch_bl,
            'mwu_stat_chr2_vs_baseline':                   u_c2_bl,
            'mwu_pval_chr2_vs_baseline':                   p_c2_bl,
            'mwu_stat_chrimson_vs_chr2':                   u_ch_c2,
            'mwu_pval_chrimson_vs_chr2':                   p_ch_c2,
            'mwu_stat_chrimson_baseline_vs_chr2_baseline': u_bl_bl,
            'mwu_pval_chrimson_baseline_vs_chr2_baseline': p_bl_bl,
            'tagged': tagged,
            # Spearman dose-response
            'spearman_rho_chrimson':  rho_ch,
            'spearman_pval_chrimson': p_rho_ch,
            'spearman_rho_chr2':      rho_c2,
            'spearman_pval_chr2':     p_rho_c2,
            'delta_rho':              delta_rho,
            'fisher_z_stat':          fisher_z,
            'fisher_z_pval':          fisher_pval,
            'dominant_opsin':         dominant_opsin,
            'cell_type':              cell_type,
        })

    summary_df = pd.DataFrame(summary_records)

    # ---- bar-graph per unit ----
    if len(results_df) > 0:
        _plot_stim_firing_bars(results_df, plot_folder, alpha)

    return results_df, summary_df


def _plot_stim_firing_bars(results_df, plot_folder, alpha=0.05):
    """
    For each unit, draw a grouped bar chart: baseline rate vs post-stim rate
    for every opsin × power combination.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of ``test_stim_firing``.
    plot_folder : Path or None
        Where to save PNGs.  If None, ``plt.show()`` is used.
    alpha : float
        Significance level for annotating bars with an asterisk.
    """
    import matplotlib.pyplot as plt

    units = results_df['unit'].unique()
    for unit in units:
        udf = results_df[results_df['unit'] == unit].copy()
        udf = udf.sort_values(['opsin', 'power'])

        labels = [f"{row['opsin']}\n{row['power']} µW" for _, row in udf.iterrows()]
        baseline_vals = udf['rate_baseline_hz'].values
        post_vals = udf['rate_post_hz'].values
        sig_flags = udf['significant'].values

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
        bars_bl = ax.bar(x - width / 2, baseline_vals, width, label='Baseline', color='grey', alpha=0.7)
        bars_po = ax.bar(x + width / 2, post_vals, width, label='Post-stim (1–31 ms)', color='steelblue')

        # annotate significant comparisons
        y_max = max(np.max(baseline_vals), np.max(post_vals)) if len(baseline_vals) > 0 else 1
        for i, sig in enumerate(sig_flags):
            if sig:
                bar_top = max(baseline_vals[i], post_vals[i])
                ax.text(x[i], bar_top + y_max * 0.02, '*', ha='center', va='bottom',
                        fontsize=14, fontweight='bold', color='red')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Firing rate (Hz)')
        ax.set_title(f'Unit {unit} – Baseline vs Evoked Firing Rate')
        ax.legend()
        fig.tight_layout()

        if plot_folder is not None:
            fig.savefig(plot_folder / f'Unit_{unit}_stim_firing_bar.png', dpi=150)
            plt.close(fig)
        else:
            plt.show()

def plot_stim_firing_bars(ephys_data, bs, save_folder=None, alpha=0.05,
                          post_window_ms=(1.0, 31.0),
                          baseline_duration_s=1.0, baseline_gap_ms=1.0,
                          baseline_bin_ms=10.0):
    """Per-unit bar chart: baseline vs per-power evoked firing rate.

    One subplot per opsin, each 0.8 × 0.8 in.  Orange bars = chrimson,
    blue bars = chr2 (shade varies by power level).  Grey bar = pooled
    baseline.  Error bars are 95 % CI of the mean.  All text 8 pt.
    Saves SVG + PNG under save_folder/stim_firing_plots_v2/.
    """
    import re
    import plot_size_utils as psu

    opsins = ['chrimson', 'chr2']
    _CMAPS  = {'chrimson': plt.cm.Oranges, 'chr2': plt.cm.Blues}
    _LABELS = {'chrimson': 'amber', 'chr2': 'blue'}

    post_start_s   = post_window_ms[0] / 1000.0
    post_end_s     = post_window_ms[1] / 1000.0
    post_bin_s     = post_end_s - post_start_s
    baseline_bin_s = baseline_bin_ms  / 1000.0
    baseline_gap_s = baseline_gap_ms  / 1000.0

    def _extr_power(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else None

    def _ci95(arr):
        arr = np.asarray(arr, dtype=float)
        if len(arr) < 2:
            return 0.0
        return 1.96 * arr.std() / np.sqrt(len(arr))

    bs_metadata = bs.metadata

    # collect per-opsin onset times grouped by power
    opsin_info = {}
    for opsin in opsins:
        on_mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
        on_df = bs_metadata[on_mask].copy()
        if on_df.empty:
            continue
        on_df['power'] = on_df['event'].apply(_extr_power)
        on_df = on_df.dropna(subset=['power'])

        power_groups = {}
        all_onsets, all_powers = [], []
        for idx in on_df.index:
            pw = int(on_df.loc[idx, 'power'])
            ts_obj = bs[idx]
            onsets = ts_obj.t.tolist() if hasattr(ts_obj, 't') else ts_obj.index.tolist()
            power_groups.setdefault(pw, []).extend(onsets)
            all_onsets.extend(onsets)
            all_powers.extend([float(pw)] * len(onsets))

        if not all_onsets:
            continue
        sort_idx = np.argsort(all_onsets)
        opsin_info[opsin] = {
            'all_onsets':   np.array(all_onsets)[sort_idx],
            'power_groups': {k: np.array(v) for k, v in power_groups.items()},
        }

    if not opsin_info:
        print("No opsin events found; skipping plot_stim_firing_bars.")
        return

    if save_folder is not None:
        out_folder = Path(save_folder) / 'stim_firing_plots_v2'
        out_folder.mkdir(parents=True, exist_ok=True)
    else:
        out_folder = None

    units = list(ephys_data.keys())
    present_opsins = [op for op in opsins if op in opsin_info]
    n_cols = len(present_opsins)

    for unit in units:
        spike_times = np.asarray(ephys_data[unit].index)

        fig, axs = plt.subplots(1, n_cols, squeeze=False)

        for ci, opsin in enumerate(present_opsins):
            info          = opsin_info[opsin]
            all_onsets    = info['all_onsets']
            power_groups  = info['power_groups']
            sorted_powers = sorted(power_groups)
            n_pw          = max(len(sorted_powers) - 1, 1)
            cmap          = _CMAPS[opsin]
            ax            = axs[0, ci]

            # pooled baseline for this opsin
            n_bins = int(baseline_duration_s / baseline_bin_s)
            bl_rates = []
            for onset in all_onsets:
                bl_end   = onset - baseline_gap_s
                bl_start = bl_end - baseline_duration_s
                for b in range(n_bins):
                    lo = bl_start + b * baseline_bin_s
                    hi = lo + baseline_bin_s
                    bl_rates.append(
                        int(np.sum((spike_times >= lo) & (spike_times < hi)))
                        / baseline_bin_s
                    )
            bl_rates = np.array(bl_rates)
            bl_mean  = float(bl_rates.mean()) if len(bl_rates) else 0.0
            bl_ci    = _ci95(bl_rates)

            xs      = [0]
            means   = [bl_mean]
            cis     = [bl_ci]
            colors  = ['grey']
            xlabels = ['BL']

            for pi, pw in enumerate(sorted_powers):
                onsets_pw  = power_groups[pw]
                post_rates = np.array([
                    int(np.sum((spike_times >= (on + post_start_s)) &
                               (spike_times <  (on + post_end_s))))
                    / post_bin_s
                    for on in onsets_pw
                ], dtype=float)

                xs.append(pi + 1)
                means.append(float(post_rates.mean()) if len(post_rates) else 0.0)
                cis.append(_ci95(post_rates))
                colors.append(cmap(0.45 + 0.55 * pi / n_pw))
                xlabels.append(f'{pw} µW')

            xs    = np.array(xs)
            means = np.array(means)
            cis   = np.array(cis)

            ax.bar(xs, means, color=colors, width=0.7)
            ax.errorbar(xs, means, yerr=cis,
                        fmt='none', color='black', linewidth=0.8,
                        capsize=2, capthick=0.8)

            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=8, rotation=45, ha='right')
            ax.set_ylabel('Firing rate (Hz)', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_title(_LABELS[opsin], fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        psu.adjust_figure_for_panel_size_hetero(fig, panel_width=0.8, panel_height=0.8)

        if out_folder is not None:
            fig.savefig(out_folder / f'Unit_{unit}_stim_firing_bar.svg', bbox_inches='tight')
            fig.savefig(out_folder / f'Unit_{unit}_stim_firing_bar.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def plot_time_to_first_spike_distribution(ephys_data, bs, save_folder=None, max_latency_ms=100, alpha=0.05):
    """
    Plot the distribution of time to first spike (TTFS) relative to optogenetic stimulus onset
    for each neuron and opsin. The TTFS is calculated as the latency between the stimulus
    onset and the first spike after the onset.

    Parameters
    ----------
    ephys_data : nap.TsGroup-like
        Spike trains keyed by unit id.
    bs : object
        Binary-signals object with ``.metadata`` and ``bs[idx]`` access.
    save_folder : Path or None
        If provided, a summary PNG is saved to ``save_folder / ttfs_summary.png``.
    max_latency_ms : float
        Maximum latency for considering a spike (default 100 ms).
    alpha : float
        Significance threshold used for ``cell_type`` labelling (default 0.05).

    Returns
    -------
    df : pandas.DataFrame
        Per-trial TTFS data with columns:
        ``['unit', 'opsin', 'power', 'onset_s', 'ttfs_ms', 'prestim_ttfs_ms', 'responded']``.
        Trials where no spike occurred within ``max_latency_ms`` are censored
        (``responded=False``, ``ttfs_ms=max_latency_ms``).
    cox_df : pandas.DataFrame
        Per-unit Cox Proportional Hazards results with columns:
        ``['unit', 'n_obs', 'n_events', 'concordance_index',
           'coef_power_chrimson', 'hr_power_chrimson', 'p_power_chrimson',
           'coef_power_chr2',     'hr_power_chr2',     'p_power_chr2',
           'lrt_stat', 'lrt_pval',
           'delta_coef', 'dominant_opsin', 'cell_type']``.
        ``hr`` is the hazard ratio exp(coef); higher means spikes earlier
        per unit increase in power.  Two nested models are compared:
        M1 (separate slopes per opsin) vs M0 (shared slope); the LRT
        p-value tests whether the two dose-response slopes are
        significantly different.  ``delta_coef`` = coef_chrimson −
        coef_chr2 (positive = chrimson steeper).  ``dominant_opsin`` is
        set only when LRT is significant.  ``cell_type`` labels neurons
        hierarchically: 'iSPN' (chrimson-specific), 'dSPN' (chr2-
        specific), 'muSPN' (both significant but indistinguishable), or
        NaN (not tagged).  All numeric columns are NaN if fitting fails.
    ttfs_folder : Path
        Path to the folder containing saved unit figures.
    """
    import re
    import pandas as pd
    from pathlib import Path

    if save_folder is None:
        save_folder = Path.cwd()
    else:
        save_folder = Path(save_folder)
    ttfs_folder = save_folder / 'opto_tagging' / 'ttfs'
    ttfs_folder.mkdir(parents=True, exist_ok=True)

    bs_metadata = bs.metadata

    records = []
    # for each opsin, find all on events and extract spike times in the post-stimulus window
    for opsin in ['chrimson', 'chr2']:
        # find rows corresponding to opsin on events
        mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
        events_idx = bs_metadata[mask]
        if events_idx.shape[0] == 0:
            continue

        # extract power from the event string (expects integer at the end e.g. 'chr2_on_20')
        def _extract_power(s):
            m = re.search(r"(\d+)$", str(s))
            return int(m.group(1)) if m else np.nan

        events_idx = events_idx.copy()
        events_idx['power'] = events_idx['event'].apply(_extract_power)

        for idx, row in events_idx.iterrows():
            ts_obj = bs[idx]
            onsets = ts_obj.t if hasattr(ts_obj, 't') else ts_obj.index
            power = row['power']
            for onset in onsets:
                for unit in ephys_data.keys():
                    spike_times = ephys_data[unit].index
                    # relative spike times (s)
                    rel = spike_times - onset
                    max_lat_s = float(max_latency_ms) / 1000.0
                    mask_spikes = (rel > 0) & (rel <= max_lat_s)
                    if np.any(mask_spikes):
                        latency_s = np.min(rel[mask_spikes])
                        ttfs_ms = float(latency_s * 1000.0)
                        responded = True
                    else:
                        ttfs_ms = float(max_latency_ms)
                        responded = False
                        
                    # Pre-stimulus control: first spike in the 100 ms window before onset.
                    # Trials with no spike are right-censored at max_latency_ms.
                    rel_null = spike_times - (onset - float(max_latency_ms) / 1000.0)
                    mask_null = (rel_null > 0) & (rel_null <= float(max_latency_ms) / 1000.0)
                    prestim_responded = bool(np.any(mask_null))
                    if prestim_responded:
                        prestim_ttfs_ms = float(np.min(rel_null[mask_null]) * 1000.0)
                    else:
                        prestim_ttfs_ms = float(max_latency_ms)

                    records.append({
                        'unit': unit,
                        'opsin': opsin,
                        'power': power,
                        'onset_s': float(onset),
                        'ttfs_ms': ttfs_ms,
                        'responded': responded,
                        'prestim_ttfs_ms': prestim_ttfs_ms,
                        'prestim_responded': prestim_responded,
                    })

    df = pd.DataFrame(records)

    # ---- Cox Proportional Hazards per unit ----
    # Two nested models compared by likelihood ratio test (LRT):
    #
    #   M1 (full):  separate power slopes per opsin
    #               power_chrimson = power × I(chrimson)
    #               power_chr2     = power × I(chr2)
    #               — orthogonal predictors, always identifiable
    #
    #   M0 (null):  single shared power slope
    #               power (same coefficient for both opsins)
    #
    # LRT stat = 2 × (loglik_M1 − loglik_M0) ~ χ²(df=1)
    # Tests whether the two dose-response slopes are significantly different.
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from scipy.stats import chi2 as _chi2

    cox_records = []
    units = sorted(df['unit'].dropna().unique())
    for unit in units:
        unit_df = df[df['unit'] == unit].copy()
        unit_df['opsin_encoded'] = (unit_df['opsin'] == 'chrimson').astype(int)
        unit_df['event'] = unit_df['responded'].astype(int)
        unit_df['power_chrimson'] = unit_df['power'] * unit_df['opsin_encoded']
        unit_df['power_chr2']     = unit_df['power'] * (1 - unit_df['opsin_encoded'])
        stim_df = unit_df[['ttfs_ms', 'event', 'power',
                            'power_chrimson', 'power_chr2']].dropna()

        # Pre-stim rows at power=0 — both per-opsin slopes are 0 here,
        # so they anchor the baseline without affecting slope estimation.
        prestim_df = pd.DataFrame({
            'ttfs_ms':        unit_df['prestim_ttfs_ms'].values,
            'event':          unit_df['prestim_responded'].astype(int).values,
            'power':          0.0,
            'power_chrimson': 0.0,
            'power_chr2':     0.0,
        }).dropna()
        fit_df = pd.concat([stim_df, prestim_df], ignore_index=True)

        n_obs    = len(fit_df)
        n_events = int(fit_df['event'].sum())

        coef_ch = hr_ch = p_ch = np.nan
        coef_c2 = hr_c2 = p_c2 = np.nan
        lrt_stat = lrt_pval = np.nan
        ci = np.nan

        if n_events >= 2:
            try:
                # Full model — separate slopes
                cph_full = CoxPHFitter(penalizer=0.1)
                cph_full.fit(fit_df[['ttfs_ms', 'event', 'power_chrimson', 'power_chr2']],
                             duration_col='ttfs_ms', event_col='event')
                s = cph_full.summary
                if 'power_chrimson' in s.index:
                    coef_ch = float(s.loc['power_chrimson', 'coef'])
                    hr_ch   = float(s.loc['power_chrimson', 'exp(coef)'])
                    p_ch    = float(s.loc['power_chrimson', 'p'])
                if 'power_chr2' in s.index:
                    coef_c2 = float(s.loc['power_chr2', 'coef'])
                    hr_c2   = float(s.loc['power_chr2', 'exp(coef)'])
                    p_c2    = float(s.loc['power_chr2', 'p'])
                ci      = float(cph_full.concordance_index_)
                ll_full = float(cph_full.log_likelihood_)

                # Null model — shared slope
                cph_null = CoxPHFitter(penalizer=0.1)
                cph_null.fit(fit_df[['ttfs_ms', 'event', 'power']],
                             duration_col='ttfs_ms', event_col='event')
                ll_null  = float(cph_null.log_likelihood_)

                lrt_stat = float(max(0.0, 2.0 * (ll_full - ll_null)))
                lrt_pval = float(_chi2.sf(lrt_stat, df=1))
            except Exception:
                pass

        cox_records.append({
            'unit': unit,
            'n_obs': n_obs,
            'n_events': n_events,
            'concordance_index': ci,
            'coef_power_chrimson': coef_ch,
            'hr_power_chrimson':   hr_ch,
            'p_power_chrimson':    p_ch,
            'coef_power_chr2':     coef_c2,
            'hr_power_chr2':       hr_c2,
            'p_power_chr2':        p_c2,
            'lrt_stat':            lrt_stat,
            'lrt_pval':            lrt_pval,
        })

    cox_df = pd.DataFrame(cox_records)

    # ---- derived summary columns ----
    cox_df['delta_coef'] = cox_df['coef_power_chrimson'] - cox_df['coef_power_chr2']

    ch_sig  = cox_df['p_power_chrimson'] < alpha
    c2_sig  = cox_df['p_power_chr2']     < alpha
    lrt_sig = cox_df['lrt_pval']         < alpha

    # dominant_opsin: which opsin has the steeper slope, but only reported
    # when the LRT confirms the difference is significant.
    cox_df['dominant_opsin'] = np.where(
        lrt_sig,
        np.where(cox_df['delta_coef'] > 0, 'chrimson', 'chr2'),
        None,
    )

    # cell_type: hierarchical labelling
    #   1. one opsin significant, other not → label by the significant one
    #   2. both significant + LRT significant → label by stronger slope
    #   3. both significant + LRT not significant → 'muSPN' (ambiguous)
    #   4. neither significant → NaN (not tagged)
    def _cell_type(row):
        ch  = row['p_power_chrimson'] < alpha
        c2  = row['p_power_chr2']     < alpha
        lrt = row['lrt_pval']         < alpha
        if ch and not c2:
            return 'iSPN'
        if c2 and not ch:
            return 'dSPN'
        if ch and c2:
            if lrt:
                return 'iSPN' if row['delta_coef'] > 0 else 'dSPN'
            return 'muSPN'
        return np.nan

    cox_df['cell_type'] = cox_df.apply(_cell_type, axis=1)

    # ---- per-unit TTFS plots (Kaplan-Meier cumulative incidence) ----
    # Censored trials (no spike) contribute to the risk set but do not pull
    # the curve to 1, giving an unbiased estimate of spiking probability.
    opsins = ['chrimson', 'chr2']
    all_powers = sorted(df['power'].dropna().unique())
    n_opsins = len(opsins)
    n_powers = max(len(all_powers), 1)

    cox_lookup = cox_df.set_index('unit') if len(cox_df) > 0 else None

    for unit in units:
        fig, axs = plt.subplots(nrows=n_powers, ncols=n_opsins,
                                figsize=(5 * n_opsins, 3 * n_powers),
                                sharex=True, sharey=True)
        if n_powers == 1 and n_opsins == 1:
            axs = np.array([[axs]])
        elif n_powers == 1:
            axs = axs[np.newaxis, :]
        elif n_opsins == 1:
            axs = axs[:, np.newaxis]

        unit_df = df[df['unit'] == unit]

        # Pre-stim KM fit (pooled across all opsins/powers for this unit)
        prestim_dur = unit_df['prestim_ttfs_ms'].values.astype(float)
        prestim_evt = unit_df['prestim_responded'].values.astype(bool)
        kmf_null = None
        if len(prestim_dur) > 0:
            kmf_null = KaplanMeierFitter()
            kmf_null.fit(prestim_dur, event_observed=prestim_evt)

        for col, op in enumerate(opsins):
            op_df = unit_df[unit_df['opsin'] == op]
            for row_idx, power in enumerate(all_powers):
                ax = axs[row_idx, col]
                power_df = op_df[op_df['power'] == power]
                dur = power_df['ttfs_ms'].values.astype(float)
                evt = power_df['responded'].values.astype(bool)
                pow_label = str(int(power)) if power == int(power) else str(power)

                if kmf_null is not None:
                    t_null = kmf_null.timeline
                    ci_null = 1.0 - kmf_null.survival_function_.values.flatten()
                    ax.step(t_null, ci_null, color='gray', linewidth=2,
                            label='Pre-stim', alpha=0.8, where='post')

                if len(dur) > 0:
                    kmf_op = KaplanMeierFitter()
                    kmf_op.fit(dur, event_observed=evt)
                    t_op = kmf_op.timeline
                    ci_op = 1.0 - kmf_op.survival_function_.values.flatten()
                    ax.step(t_op, ci_op, color='C0', linewidth=2,
                            label=op, alpha=0.8, where='post')
                    ax.set_title(f'{op} @ {pow_label} \u00b5W', fontsize=8)
                else:
                    ax.set_title(f'{op} @ {pow_label} \u00b5W (No Data)', fontsize=8)

                ax.set_xlim(0, max_latency_ms)
                ax.set_ylim(0, 1)
                if row_idx == n_powers - 1:
                    ax.set_xlabel('Time to first spike (ms)')
                if col == 0:
                    ax.set_ylabel('Cumulative incidence')
                if kmf_null is not None or len(dur) > 0:
                    ax.legend(fontsize=8)

        # annotate suptitle with Cox results for this unit
        cox_note = ''
        if cox_lookup is not None and unit in cox_lookup.index:
            crow = cox_lookup.loc[unit]
            p_ch   = float(crow['p_power_chrimson'])
            hr_ch  = float(crow['hr_power_chrimson'])
            p_c2   = float(crow['p_power_chr2'])
            hr_c2  = float(crow['hr_power_chr2'])
            p_lrt  = float(crow['lrt_pval'])
            parts = []
            if not np.isnan(p_ch):
                parts.append(f'chrimson: HR={hr_ch:.2f}, p={p_ch:.3g}')
            if not np.isnan(p_c2):
                parts.append(f'chr2: HR={hr_c2:.2f}, p={p_c2:.3g}')
            if not np.isnan(p_lrt):
                parts.append(f'LRT p={p_lrt:.3g}')
            if parts:
                cox_note = '\nCox PH \u2014 ' + '  |  '.join(parts)

        fig.suptitle(f'Unit {unit} \u2014 TTFS Cumulative Incidence{cox_note}', fontsize=9)
        plt.tight_layout()
        plt.savefig(ttfs_folder / f'Unit_{unit}_ttfs.png')
        plt.close(fig)

    return df, cox_df, ttfs_folder


def plot_ttfs_cdf_single(ephys_data, bs, save_folder=None, max_latency_ms=100):
    """Single-panel CDF variant of plot_time_to_first_spike_distribution.

    All per-power CDF curves are drawn on one 0.8 × 0.8 in axes per unit.
    Chrimson curves are shades of orange, chr2 curves are shades of blue
    (darker = higher power). The pooled pre-stimulus shuffle is a single
    grey step curve. No legend. All text 8 pt.
    Saves SVG + PNG under save_folder/opto_tagging/ttfs_single/.
    """
    import re
    import pandas as pd
    from lifelines import KaplanMeierFitter
    import plot_size_utils as psu

    if save_folder is None:
        save_folder = Path.cwd()
    else:
        save_folder = Path(save_folder)
    out_folder = save_folder / 'opto_tagging' / 'ttfs_single'
    out_folder.mkdir(parents=True, exist_ok=True)

    bs_metadata = bs.metadata
    records = []

    def _extract_power(s):
        m = re.search(r"(\d+)$", str(s))
        return int(m.group(1)) if m else np.nan

    for opsin in ['chrimson', 'chr2']:
        mask = bs_metadata['event'].str.contains(f'{opsin}_on', na=False)
        events_idx = bs_metadata[mask].copy()
        if events_idx.empty:
            continue
        events_idx['power'] = events_idx['event'].apply(_extract_power)
        max_lat_s = float(max_latency_ms) / 1000.0

        for idx, row in events_idx.iterrows():
            ts_obj = bs[idx]
            onsets = ts_obj.t if hasattr(ts_obj, 't') else ts_obj.index
            power = row['power']
            for onset in onsets:
                for unit in ephys_data.keys():
                    spike_times = ephys_data[unit].index
                    rel = spike_times - onset
                    mask_post = (rel > 0) & (rel <= max_lat_s)
                    responded = bool(np.any(mask_post))
                    ttfs_ms = float(np.min(rel[mask_post]) * 1000.0) if responded else float(max_latency_ms)

                    rel_null = spike_times - (onset - max_lat_s)
                    mask_null = (rel_null > 0) & (rel_null <= max_lat_s)
                    prestim_responded = bool(np.any(mask_null))
                    prestim_ttfs_ms = (float(np.min(rel_null[mask_null]) * 1000.0)
                                       if prestim_responded else float(max_latency_ms))

                    records.append({
                        'unit': unit, 'opsin': opsin, 'power': power,
                        'ttfs_ms': ttfs_ms, 'responded': responded,
                        'prestim_ttfs_ms': prestim_ttfs_ms,
                        'prestim_responded': prestim_responded,
                    })

    df = pd.DataFrame(records)
    if df.empty:
        print("No TTFS data found; skipping.")
        return

    _CMAPS  = {'chrimson': plt.cm.Oranges, 'chr2': plt.cm.Blues}

    for unit in sorted(df['unit'].dropna().unique()):
        unit_df = df[df['unit'] == unit]

        fig, ax = plt.subplots(1, 1)

        # pre-stim shuffle — pooled across all opsins/powers, one grey curve
        prestim_dur = unit_df['prestim_ttfs_ms'].values.astype(float)
        prestim_evt = unit_df['prestim_responded'].values.astype(bool)
        if len(prestim_dur) > 0:
            kmf_null = KaplanMeierFitter()
            kmf_null.fit(prestim_dur, event_observed=prestim_evt)
            ci_null = 1.0 - kmf_null.survival_function_.values.flatten()
            ax.step(kmf_null.timeline, ci_null,
                    color='grey', linewidth=1.0, alpha=0.8, where='post')

        # per-opsin × per-power curves
        for opsin in ['chrimson', 'chr2']:
            op_df = unit_df[unit_df['opsin'] == opsin]
            if op_df.empty:
                continue
            cmap = _CMAPS[opsin]
            op_powers = sorted(op_df['power'].dropna().unique())
            n_op = max(len(op_powers) - 1, 1)
            for pi, power in enumerate(op_powers):
                dur = op_df.loc[op_df['power'] == power, 'ttfs_ms'].values.astype(float)
                evt = op_df.loc[op_df['power'] == power, 'responded'].values.astype(bool)
                if len(dur) == 0:
                    continue
                color = cmap(0.45 + 0.55 * pi / n_op)
                kmf = KaplanMeierFitter()
                kmf.fit(dur, event_observed=evt)
                ci_op = 1.0 - kmf.survival_function_.values.flatten()
                ax.step(kmf.timeline, ci_op,
                        color=color, linewidth=1.0, alpha=0.9, where='post')

        ax.set_xlim(0, max_latency_ms)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time to first spike (ms)', fontsize=8)
        ax.set_ylabel('Cumulative incidence', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        psu.adjust_figure_for_panel_size_auto(fig, panel_width=0.8, panel_height=0.8)

        fig.savefig(out_folder / f'Unit_{unit}_ttfs_single.svg', bbox_inches='tight')
        fig.savefig(out_folder / f'Unit_{unit}_ttfs_single.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
