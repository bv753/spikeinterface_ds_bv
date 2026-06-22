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
base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213")

#get the folders inside base_folder that end with '_g0'
spikeglx_folder = list(base_folder.glob('*_g0'))[0]

stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
raw_rec = si.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
raw_rec.get_probe().to_dataframe()


#fig, ax = plt.subplots(figsize=(15, 10))
#si.plot_probe_map(raw_rec, ax=ax, with_channel_ids=True)
# plt.show()

rechpf = spp.highpass_filter(raw_rec, freq_min=300)
recbpf = spp.bandpass_filter(raw_rec, freq_min=300, freq_max=3000)

bad_channel_ids, channel_labels = si.detect_bad_channels(rechpf)
dead_channel_bool = np.array(channel_labels == 'dead')
dead_channel_bool[[44,142,154,181,198,199,228,383]] = True  # add manually some dead channels
dead_channel_ids = rechpf.channel_ids[dead_channel_bool]
rechpf = rechpf.remove_channels(dead_channel_ids)
raw_rec = raw_rec.remove_channels(dead_channel_ids)
recbpf = recbpf.remove_channels(dead_channel_ids)


recCAR_hpf = si.phase_shift(rechpf)
recCAR_hpf = si.common_reference(recCAR_hpf, operator="average", reference="global")

recCAR_bpf = si.phase_shift(recbpf)
recCAR_bpf = si.common_reference(recCAR_bpf, operator="average", reference="global")

filter_freqs = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660]
recnotch = raw_rec
rechpfnotch = rechpf
for freq in filter_freqs:
    recnotch = spp.notch_filter(recnotch, freq=freq, q=300)

rechpfnotch = spp.highpass_filter(recnotch, freq_min=300)
chids = raw_rec.channel_ids
#get every 3d channel id

import scipy

start = 10000000
end = start+1000000
trace = raw_rec.get_traces(channel_ids=chids)[start:end,:]
recnotch_trace = recnotch.get_traces(channel_ids=chids, start_frame=start, end_frame=end)
rechpf_trace = rechpf.get_traces(channel_ids=chids, start_frame=start, end_frame=end)
recbpf_trace = recbpf.get_traces(channel_ids=chids, start_frame=start, end_frame=end)
rechpfnotch_trace = rechpfnotch.get_traces(channel_ids=chids, start_frame=start, end_frame=end)
recCAR_hpf_trace = recCAR_hpf.get_traces(channel_ids=chids, start_frame=start, end_frame=end)
recCAR_bpf_trace = recCAR_bpf.get_traces(channel_ids=chids, start_frame=start, end_frame=end)

sf = raw_rec.get_sampling_frequency()
nps = int(sf//1)
f_raw, p_raw = scipy.signal.welch(np.swapaxes(trace, 0,1), fs=sf, nperseg=nps)
f_notch, p_notch = scipy.signal.welch(np.swapaxes(recnotch_trace,0,1), fs=sf, nperseg=nps)
f_hpf, p_hpf = scipy.signal.welch(np.swapaxes(rechpf_trace,0,1), fs=sf, nperseg=nps)
f_bpf, p_bpf = scipy.signal.welch(np.swapaxes(recbpf.get_traces(channel_ids=chids, start_frame=start, end_frame=end),0,1), fs=sf, nperseg=nps)
f_notchbp, p_notchbp = scipy.signal.welch(np.swapaxes(rechpfnotch_trace,0,1), fs=sf, nperseg=nps)
f_car_hpf, p_car_hpf = scipy.signal.welch(np.swapaxes(recCAR_hpf_trace,0,1), fs=sf, nperseg=nps)
f_car_bpf, p_car_bpf = scipy.signal.welch(np.swapaxes(recCAR_bpf_trace,0,1), fs=sf, nperseg=nps)

#average p_raw
p_raw_avg = np.mean(p_raw, axis=0)
p_notch_avg = np.mean(p_notch, axis=0)
p_hpf_avg = np.mean(p_hpf, axis=0)
p_bpf_avg = np.mean(p_bpf, axis=0)
p_notchbp_avg = np.mean(p_notchbp, axis=0)
p_car_hpf_avg = np.mean(p_car_hpf, axis=0)
p_car_bpf_avg = np.mean(p_car_bpf, axis=0)
#get the peak frequency in p_raw_avg
peak_freq = f_raw[np.argmax(p_raw_avg)]
peak_CAR_hpf_freq = f_car_hpf[np.argmax(p_car_hpf_avg)]
peak_CAR_bpf_freq = f_car_bpf[np.argmax(p_car_bpf_avg)]
peak_hpf_freq = f_hpf[np.argmax(p_hpf_avg)]
peak_bpf_freq = f_bpf[np.argmax(p_bpf_avg)]

fig, ax = plt.subplots()
ax.semilogy(f_raw, p_raw_avg, label='raw')
#ax.semilogy(f_hpf, p_hpf_avg, label='highpass')
#ax.semilogy(f_bpf, p_bpf_avg, label='bandpass')
#ax.semilogy(f_notch, p_notch_avg, label='notch')
#ax.semilogy(f_notchbp, p_notchbp_avg, label='highpass+notch')
ax.semilogy(f_car_hpf, p_car_hpf_avg, label='highpass+CAR')
ax.semilogy(f_car_bpf, p_car_bpf_avg, label='bandpass+CAR')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Power/Frequency [V**2/Hz]')
ax.legend()
#plot a vertical line at 60 Hz
#limit x axis to 0-200 Hz
ax.set_xlim(0, 6000)
#log scale the y axis