import spikeinterface as si
import pynapple as nap
from pathlib import Path
import matplotlib.pyplot as plt
import pres_plot_utils as pu

subj_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2")
save_folder = subj_folder / "compendium"
save_folder.mkdir(exist_ok=True)

base_folders = [
    Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_050526"),
    Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260506"),
    Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511"),
]

pynapple_folders = [f / "pynapple" for f in base_folders]

# --- per-session processing ---
sessions = []
for pynapple_folder in pynapple_folders:
    ep = nap.load_file(pynapple_folder / "sing_tagged_units.npz")
    bs = nap.load_file(pynapple_folder / "binary_signals.npz")
    bs = pu.add_all_licks(bs)

    ch_on, ch_off = pu.pool_task_stim(bs, 'chrimson')
    c2_on, c2_off = pu.pool_task_stim(bs, 'chr2')
    pu.compute_ramp_scores(ep, bs, ch_on, c2_on)

    te = pu.categorize_outcome_trials(bs, ch_on, ch_off, c2_on, c2_off)

    lick_mod = pu.id_lick_bout_modulated(ep, bs, te)
    sig_units = lick_mod.loc[lick_mod['significant'], 'unit_id'].tolist()
    ep = ep[sig_units]
    print(f"{pynapple_folder.parent.name}: {len(sig_units)} / {len(lick_mod)} lick-bout modulated units")

    sessions.append((ep, bs, te))

# --- merge all sessions into one pseudo-session ---
ephys, binary_signals, trial_events = pu.merge_sessions(sessions)
print(f"Total merged units: {len(ephys.keys())}")

# --- plots ---
pu.plot_spn_lick_bout_psth(ephys, binary_signals, trial_events, save_folder /
                            "lick_bout_psth")

pu.plot_spn_summary(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'spn_summary.png',
    ms_censored=500,
    n_quintiles=7,
)

pu.plot_outcome_heatmap_nostim(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'first_lick_heatmap_nostim.png',
    clim=(None,None)
)

pu.plot_cue_outcome_heatmap(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'cue_outcome_heatmap.png',
    ms_censored=50, clim=(-0.25, 4),
)

pu.plot_cue_outcome_heatmap(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'cue_outcome_heatmap.png',
    ms_censored=50, clim=(0, 0.1),
    zscore=False
)

pu.plot_spn_heatmap(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'spn_heatmap.png',
    ms_censored=500, clim=(-0.25, 2.5)
)

pu.plot_outcome_heatmap_nostim(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'first_lick_heatmap_nostim.png',
    clim=(0, None), zscore=False,
)

pu.plot_cue_aligned_nostim(
    ephys, binary_signals, trial_events, n_quintiles=10,
    save_path=save_folder / 'cue_aligned_nostim.png', ms_censored=50
)

pu.plot_spn_cue_aligned_nostim(
    ephys, binary_signals, trial_events, n_quintiles=6,
    save_path=save_folder / 'spn_cue_aligned_nostim.png', ms_censored=1000
)

pu.plot_spn_mean_psth(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'spn_mean_psth_rewarded.png', ms_censored=3333
)

pu.plot_spn_mean_psth(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'spn_mean_psth_unrewarded.png', outcome_range = [500,3332]
)

pu.plot_spn_rew_unrew_psth(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'spn_rew_unrew_psth.png',
    rewarded_range_ms=(3333, None),
    unrewarded_range_ms=(100, 3332),
)



pu.plot_dspn_isPN_diff(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'dspn_ispn_diff.png',
    n_quintiles=10, ms_censored=1000
)

fig, summary = pu.plot_dspn_ispn_lick_bout_bar(
    ephys, binary_signals, trial_events,
    save_path=save_folder / "dspn_ispn_lick_bout_bar"
)

pu.plot_dspn_isPN_bar(
    ephys, binary_signals, trial_events,
    save_path=save_folder / 'dspn_ispn_bar.svg',
    ms_censored=500,
)

plt.show()
