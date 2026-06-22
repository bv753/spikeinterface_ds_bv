import matplotlib.pyplot as plt
import spikeinterface.full as si
from pathlib import Path
import matplotlib
matplotlib.use('Qt5Agg')
#make a dataframe from temp_metrics
import pandas as pd
import numpy as np


#base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117")
#base_folder =  Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251118")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211")
base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213")

analyzer_clean = si.load_sorting_analyzer(folder=base_folder / 'analyzer_clean')
spikeglx_folder = list(base_folder.glob('*_g0'))[0]
