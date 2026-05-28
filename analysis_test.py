import pynapple as nap
from pathlib import Path
import parse_nidq as pni
import plot_psth as pp
import parse_opto_tagging as pot
import get_ramp_modulation as grm

def run_analysis(base_folder, dinmap=None, ainmap=None, optoidx=None, overwrite=False):
        pynapple_folder = base_folder / "pynapple"
        bs = pni.get_binary_signals(base_folder, dinmap=dinmap, ainmap=ainmap, optoidx=optoidx, overwrite=overwrite, plot=True)
        ephys_file = pynapple_folder / "spikes.npz"
        ephys_data = nap.load_file(ephys_file)

        pot.plot_all_tuning_curves(ephys_data, bs, pynapple_folder, overwrite=overwrite)
        pp.plot_on_events_psth(ephys_data, bs, pynapple_folder)
        grm.get_ramp_modulation(base_folder)

# pathlist = [#Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209"),
#             #Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251210"),#]
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211"),
#             Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251212"),
pathlist=[Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213")]
#pathlist = [Path(r"C:\Users\assad\Documents\recording_files\DS23\DS23_20260211")]
for base_folder in pathlist:
    run_analysis(base_folder, overwrite=True)
