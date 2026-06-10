import spikeinterface as si
import pynapple as nap
from pathlib import Path
import matplotlib.pyplot as plt
import pres_plot_utils as pu

#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_050526")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260506")
base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511")
pynapple_folder = base_folder / "pynapple"
ephys_file = pynapple_folder / "sing_tagged_units.npz"
binary_signals = pynapple_folder / "binary_signals.npz"

ephys = nap.load_file(ephys_file)
binary_signals = nap.load_file(binary_signals)

binary_signals = pu.add_all_licks(binary_signals)

ch_on, ch_off = pu.pool_task_stim(binary_signals, 'chrimson')
c2_on, c2_off = pu.pool_task_stim(binary_signals, 'chr2')

scores = pu.compute_ramp_scores(ephys, binary_signals, ch_on, c2_on)

trial_events = pu.categorize_outcome_trials(binary_signals, ch_on, ch_off, c2_on, c2_off)

lick_mod = pu.id_lick_bout_modulated(ephys, binary_signals, trial_events, p_threshold=0.01)
sig_units = lick_mod.loc[lick_mod['significant'], 'unit_id'].tolist()
ephys = ephys[sig_units]
emd = ephys.metadata
#sort by ramp score
emd = emd.assign(ramp_score=scores)
emd = emd.sort_values('ramp_score', ascending=False)

print(f"Lick-bout modulated units: {len(sig_units)} / {len(lick_mod)}")

pu.plot_outcome_heatmap(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'first_lick_heatmap.png',
    label_units=True
)
pu.plot_outcome_heatmap(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'first_lick_heatmap.png',
    zscore=False, clim=(0,None)
)

pu.plot_outcome_heatmap_nostim(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'first_lick_heatmap_nostim.png'
)

pu.plot_outcome_heatmap_nostim(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'first_lick_heatmap_nostim.png', clim=(0,None),
    zscore=False,
)

pu.plot_cue_aligned_by_latency(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'cue_aligned_by_latency.png',
)
pu.plot_cue_aligned_by_latency(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'cue_aligned_by_latency.png',
    zscore=False,
)

pu.plot_cue_aligned_nostim(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'cue_aligned_nostim.png',
)
pu.plot_cue_aligned_nostim(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'cue_aligned_nostim.png',
    zscore=False,
)

pu.plot_dspn_isPN_diff(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'dspn_ispn_diff.png',
)
pu.plot_dspn_isPN_diff(
    ephys, binary_signals, trial_events,
    save_path=pynapple_folder / 'dspn_ispn_diff.png',
    zscore=False,
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






