import matplotlib
matplotlib.use('Qt5Agg')
import spikeinterface.full as si
from spikeinterface.curation import remove_redundant_units
from spikeinterface.curation import auto_merge_units
from pathlib import Path
#from spikeinterface_gui import run_mainwindow
from spikeinterface.exporters import to_pynapple_tsgroup

#base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
#spikeglx_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250822\DS13_20250822_g0")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")
#spikeglx_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905\DS13_20250905_g0")
#base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")
#spikeglx_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824\DS13_20250824_g0")
#base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117")
#spikeglx_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117\DS1_20251117_g0")

def curation(base_folder):
    analyzer = si.load_sorting_analyzer(folder=base_folder / "analyzer")

    clean_sorting = remove_redundant_units(
        analyzer,
        duplicate_threshold=0.9,
        remove_strategy="minimum_shift",
    )

    analyzer_clean = analyzer.select_units(clean_sorting.unit_ids)

    template_diff_thresh = [0.05, 0.15, 0.25]
    presets = ["x_contaminations"] * len(template_diff_thresh)
    steps_params = [
        {"template_similarity": {"template_diff_thresh": i}}
        for i in template_diff_thresh
    ]
    job_kwargs = dict(n_jobs=22, chunk_duration='1s', progress_bar=True)
    analyzer_merged = auto_merge_units(
        analyzer_clean,
        presets=presets,
        steps_params=steps_params,
        recursive=True
    )

    #si.plot_sorting_summary(sorting_analyzer=analyzer, curation=True, backend='spikeinterface_gui')

    metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'amplitude_cutoff','amplitude_median', 'firing_range']

    metrics = si.compute_quality_metrics(analyzer_merged, metric_names=metric_names)

    min_firing_rate = 0.5
    min_amplitude_thresh = -1000
    max_amplitude_thresh = -15
    isi_violations_ratio_thresh = 5
    presence_ratio_thresh = 0.4
    snr_thresh = 0.5

    our_query = (f"(amplitude_median > {min_amplitude_thresh}) "
                 f"& (amplitude_median < {max_amplitude_thresh}) "
                 f"& (firing_rate > {min_firing_rate}) "
                 f"& (isi_violations_ratio < {isi_violations_ratio_thresh}) "
                 f" & (presence_ratio > {presence_ratio_thresh}) "
                 f" & (snr > {snr_thresh}) ")
    print(our_query)

    keep_units = metrics.query(our_query)
    keep_unit_ids = keep_units.index.values
    keep_unit_ids

    analyzer_clean = analyzer_merged.select_units(keep_unit_ids, folder=base_folder / 'analyzer_clean', format='binary_folder')

    si.export_report(analyzer_clean, base_folder / 'report_clean', format='png')

    analyzer_clean = si.load_sorting_analyzer(base_folder / 'analyzer_clean')

    my_tsgroup = to_pynapple_tsgroup(analyzer_clean,
        attach_unit_metadata=True)

    # Note: can add metadata using e.g.
    # my_tsgroup.set_info({'brain_region': ['MEC', 'MEC', ...]})
    pynapple_folder = base_folder / "pynapple"
    pynapple_folder.mkdir(exist_ok=True, parents=True)
    my_tsgroup.save(pynapple_folder / "spikes.npz")

    #run_mainwindow(analyzer_clean,curation=True)

    analyzer_clean = si.load_sorting_analyzer(base_folder / 'analyzer_clean')

    my_tsgroup = to_pynapple_tsgroup(analyzer_clean,
        attach_unit_metadata=True)

    # Note: can add metadata using e.g.
    # my_tsgroup.set_info({'brain_region': ['MEC', 'MEC', ...]})

    pynapple_folder.mkdir(exist_ok=True, parents=True)
    pynapple_path = pynapple_folder / "spikes.npz"
    my_tsgroup.save(str(pynapple_path))


