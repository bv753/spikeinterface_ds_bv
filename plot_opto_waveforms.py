import spikeinterface as si
import si_pipeline as sip
import plot_psth as pp
from pathlib import Path


#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_050526")
    #r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260506")
base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511")
pynapple_folder = base_folder / "pynapple" #Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511\pynapple")

ephys_data, bin_sigs, _ = sip.get_pynapple_data(base_folder, overwrite=False)
analyzer = si.load_sorting_analyzer(folder=base_folder / 'analyzer_clean')

pp.plot_all_opto_psth_alt(ephys_data, bin_sigs, pynapple_folder,
  analyzer=analyzer, minmax=(-0.1, 0.2))

pp.plot_all_opto_psth(ephys_data, bin_sigs, pynapple_folder, analyzer=analyzer)
