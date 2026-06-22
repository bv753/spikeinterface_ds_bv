import matplotlib.pyplot as plt
import matplotlib
import spikeinterface.full as si
import spikeinterface.core as sc
from pathlib import Path
matplotlib.use('Qt5Agg')
from spikeinterface.curation import remove_duplicated_spikes
from tests import curation_test as ct

#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")
#spikeglx_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824\DS13_20250824_g0")
#base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
#spikeglx_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\DS13_20250822\\DS13_20250822_g0")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")
#spikeglx_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905\DS13_20250905_g0")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117")
#spikeglx_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117\DS1_20251117_g0")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251118")
#spikeglx_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117\DS1_20251118_g0")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213")

#get the folders inside base_folder that end with '_g0'
spikeglx_folder = list(base_folder.glob('*_g0'))[0]

rec = sc.load(base_folder / 'preprocess')
sorting = si.read_kilosort(folder_path=base_folder / 'kilosort4_output' / 'sorter_output')
clean_sorting = remove_duplicated_spikes(sorting, censored_period_ms=0.1)
job_kwargs = dict(n_jobs=22, chunk_duration='1s', progress_bar=True)

my_protocol = {
    'preprocessing': {
        'bandpass_filter': {},
        'common_reference': {'operator': 'average'},
        'detect_and_remove_bad_channels': {},
    },
    'sorting': {
        'sorter_name': 'kilosort4',
        'verbose': False,
        'snippet_T2': 15,
        'remove_existing_folder': True,
        'progress_bar': False
    },
    'postprocessing': {
        'random_spikes': {},
        'waveforms': {},
        'principal_components': {},
        'quality_metrics': {},
        'noise_levels': {},
        'templates': {},
        'unit_locations': {'method': 'center_of_mass'},
        'spike_amplitudes': {},
        'spike_locations': {},
        'isi_histograms': {},
        'template_similarity': {},
        'template_metrics': {},
        'correlograms': {},
    },
}

analyzer = si.create_sorting_analyzer(clean_sorting, rec, sparse=True, format="memory", n_jobs=22)
analyzer.compute(my_protocol['postprocessing'])
analyzer.save_as(folder=base_folder / "analyzer", format="binary_folder")

isi_hist = analyzer.extensions['isi_histograms'].data
isi = isi_hist['isi_histograms']
bins = isi_hist['bins']
#for each index of dim 0 in isi, get the index of the largest value
maxbin = []
for unhist in range(isi.shape[0]):
    max_index = isi[unhist].argmax()
    maxbinlow = bins[max_index]
    maxbinhigh = bins[max_index + 1]
    maxbinmed = (maxbinlow + maxbinhigh) / 2
    maxbin.append(maxbinmed)

si.export_report(analyzer, base_folder / 'report_uncurated', format='png')


ct.curation(base_folder)
