import si_pipeline as sip
from pathlib import Path

pathlist = [Path(r"C:\Users\assad\Documents\recording_files\DS2\DS2_20260511")]

base_folder = pathlist[0]

#sip.plot_time_to_first_spike(base_folder)


bad_chans = [44, 142, 154, 181, 198, 199, 228, 383]  # add 11?
for folder in pathlist:
    sip.run_pipeline(folder, bad_chans=bad_chans)