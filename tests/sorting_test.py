import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import spikeinterface.core as sc
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spp

from pathlib import Path
#base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117")
#base_folder =  Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251118")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211")
base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213")

#get the folders inside base_folder that end with '_g0'
spikeglx_folder = list(base_folder.glob('*_g0'))[0]

stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
raw_rec = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
raw_rec.get_probe().to_dataframe()

filter_freqs = [300, 420, 480]
recnotch = raw_rec
for freq in filter_freqs:
    recnotch = spp.notch_filter(recnotch, freq=freq, q=300)

#rechpf = spp.highpass_filter(recnotch)
recbpf = spp.bandpass_filter(recnotch, freq_min=300, freq_max=4000)
recfs = spp.phase_shift(raw_rec)

bad_channel_ids, channel_labels = si.detect_bad_channels(recfs)
dead_channel_bool = np.array(channel_labels == 'dead')
dead_channel_bool[[44,142,154,181,198,199,228,383]] = True  # add manually some dead channels
dead_channel_ids = recfs.channel_ids[dead_channel_bool]
rec = recfs.remove_channels(dead_channel_ids)

rec_CAR = si.common_reference(rec, operator="average", reference="global")

job_kwargs = dict(n_jobs=-1, chunk_duration='1s', progress_bar=True)

rec_CAR.save(folder=base_folder / 'preprocess', format='binary', **job_kwargs)
rec = sc.load(base_folder / 'preprocess')

paramsKS4 = si.get_default_sorter_params('kilosort4')
paramsKS4['n_jobs'] = -1
paramsKS4['do_CAR'] = False

sorting = si.run_sorter('kilosort4', rec, folder=base_folder / 'kilosort4_output' ,
                        verbose=True, **paramsKS4)
