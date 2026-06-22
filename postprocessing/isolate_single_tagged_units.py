import spikeinterface as si
import pynapple as nap
from pathlib import Path
from postprocessing import parse_opto_tagging as pot

#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_050526")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260506")
base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511")
pynapple_folder = base_folder / "pynapple"
ephys_file = pynapple_folder / "spikes.npz"
binary_signals = pynapple_folder / "binary_signals.npz"

ephys = nap.load_file(ephys_file)
binary_signals = nap.load_file(binary_signals)
import pandas as pd

kst_df, cox_df, figpath = pot.plot_time_to_first_spike_distribution(ephys, binary_signals, save_folder=pynapple_folder)
mwu_df, mwu_summary_df = pot.test_stim_firing(ephys, binary_signals, save_folder=pynapple_folder)

ephys_metadata = ephys.metadata.copy()
# -- merge per-unit classifications into ephys_metadata --
ephys_metadata = ephys_metadata.join(
    cox_df.set_index('unit')['cell_type'].rename('ttfs_classif'),
    how='left'
)
ephys_metadata = ephys_metadata.join(
    mwu_summary_df.set_index('unit')['cell_type'].rename('firing_classif'),
    how='left'
)

# -- load manually curated good units if present --
manual_good_path = pynapple_folder / "manual_good_units.txt"
if manual_good_path.exists():
    import re
    manual_good_ids = [int(x) for x in re.findall(r'\d+', manual_good_path.read_text())]
else:
    manual_good_ids = []
manual_ok = ephys_metadata.index.isin(manual_good_ids)
print(f"manual_good_units.txt: loaded {len(manual_good_ids)} IDs: {manual_good_ids}")
print(f"  matched {manual_ok.sum()} units in ephys_metadata (index dtype: {ephys_metadata.index.dtype})")

# -- final_classif: ks_label good/empty OR manually curated, AND both methods agree on iSPN or dSPN --
ks_col = next((c for c in ('KSLabel', 'ks_label') if c in ephys_metadata.columns), None)
if ks_col is not None:
    ks_ok = ephys_metadata[ks_col].isin(['good']) | ephys_metadata[ks_col].isna() | (ephys_metadata[ks_col] == '') | manual_ok
else:
    ks_ok = pd.Series(True, index=ephys_metadata.index)

spn_types = {'iSPN', 'dSPN'}
both_agree = ephys_metadata['ttfs_classif'] == ephys_metadata['firing_classif']
both_spn   = ephys_metadata['ttfs_classif'].isin(spn_types)

ephys_metadata['final_classif'] = ephys_metadata['ttfs_classif'].where(
    ks_ok & both_agree & both_spn
)



# units where both methods agree on iSPN/dSPN AND quality is good
best_tagged = ephys_metadata[ephys_metadata['final_classif'].notna()].copy()

# units where both methods agree on iSPN/dSPN BUT were excluded by quality label
# (possible false negatives — tagged neurons flagged as noise by KiloSort)
poss_fn = ephys_metadata[both_agree & both_spn & ~ks_ok].copy()

# units where quality is good, both classifications are non-null, but they disagree
both_labeled = ephys_metadata['ttfs_classif'].notna() & ephys_metadata['firing_classif'].notna()
dis_df = ephys_metadata[ks_ok & both_labeled & ~both_agree].copy()

# disagreeing good-quality units: use whichever classification is not 'muSPN',
# preferring ttfs_classif; fall back to 'iSPN' only if both are 'muSPN'.
for uid in dis_df.index:
    ttfs = ephys_metadata.loc[uid, 'ttfs_classif']
    firing = ephys_metadata.loc[uid, 'firing_classif']
    if pd.notna(ttfs) and ttfs != 'muSPN':
        ephys_metadata.loc[uid, 'final_classif'] = ttfs
    elif pd.notna(firing) and firing != 'muSPN':
        ephys_metadata.loc[uid, 'final_classif'] = firing
    else:
        ephys_metadata.loc[uid, 'final_classif'] = 'muSPN'

# manually labeled units that still lack a final_classif: assign from whichever
# method gave an SPN type (ttfs first, then firing), falling back to 'iSPN'
unclassified_manual = manual_ok & ephys_metadata['final_classif'].isna()
for uid in ephys_metadata.index[unclassified_manual]:
    ttfs = ephys_metadata.loc[uid, 'ttfs_classif']
    firing = ephys_metadata.loc[uid, 'firing_classif']
    if pd.notna(ttfs) and ttfs in spn_types:
        ephys_metadata.loc[uid, 'final_classif'] = ttfs
    elif pd.notna(firing) and firing in spn_types:
        ephys_metadata.loc[uid, 'final_classif'] = firing
    else:
        ephys_metadata.loc[uid, 'final_classif'] = 'muSPN'

best_tagged = ephys_metadata[ephys_metadata['final_classif'].notna()].copy()

# all good-quality units regardless of classification (KS label or manual override)
ks_good = (ephys_metadata[ks_col] == 'good') if ks_col is not None else pd.Series(True, index=ephys_metadata.index)
singU_metadata = ephys_metadata[ks_good | manual_ok].copy()

# subset TsGroup to good units, attach enriched metadata, and save
sing_units = ephys[singU_metadata.index.tolist()]
new_cols = ['ttfs_classif', 'firing_classif', 'final_classif']
sing_units.set_info(singU_metadata[[c for c in new_cols if c in singU_metadata.columns]])
sing_units.save(str(pynapple_folder / 'sing_tagged_units'))