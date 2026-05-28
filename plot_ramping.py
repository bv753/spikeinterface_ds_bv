import spikeinterface as si
import pynapple as nap
from pathlib import Path
import matplotlib.pyplot as plt
import pres_plot_utils as pu

base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511")
pynapple_folder = base_folder / "pynapple"
ephys_file = pynapple_folder / "sing_tagged_units.npz"
binary_signals = pynapple_folder / "binary_signals.npz"

ephys = nap.load_file(ephys_file)
binary_signals = nap.load_file(binary_signals)

binary_signals = pu.add_all_licks(binary_signals)

ch_on, ch_off = pu.pool_task_stim(binary_signals, 'chrimson')
c2_on, c2_off = pu.pool_task_stim(binary_signals, 'chr2')

pu.compute_ramp_scores(ephys, binary_signals, ch_on, c2_on)

trial_events = pu.categorize_outcome_trials(binary_signals, ch_on, ch_off, c2_on, c2_off)

pu.plot_outcome_heatmap(
    ephys, trial_events,
    save_path=pynapple_folder / 'first_lick_heatmap.png',
)

pu.plot_outcome_heatmap_nostim(
    ephys, trial_events,
    save_path=pynapple_folder / 'first_lick_heatmap_nostim.png',
)

pu.plot_cue_aligned_by_latency(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'cue_aligned_by_latency.png',
)

pu.plot_cue_aligned_nostim(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'cue_aligned_nostim.png',
)

pu.plot_dspn_isPN_diff(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'dspn_ispn_diff.png',
)

pu.plot_trial_raster(
    binary_signals,
    ch_on=ch_on, ch_off=ch_off, c2_on=c2_on, c2_off=c2_off,
    save_path=pynapple_folder / 'trial_raster.png',
)

pu.plot_lick_latency(
    binary_signals, trial_events,
    save_path=pynapple_folder / 'lick_latency.png',
)

pu.plot_dspn_isPN_bar(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'dspn_ispn_bar.png',
)

pu.print_opto_timing_check(trial_events, binary_signals, ch_on, ch_off, c2_on, c2_off)

plt.show()






