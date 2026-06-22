###
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd
import spikeinterface.full as si
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

def curate_units(analyzer_clean):
    #analyzer_clean = si.load_sorting_analyzer(folder=analyzer_path)
    #base folder is one level up from analyzer_path
    #base_folder = analyzer_path.parent
    #spikeglx_folder = list(base_folder.glob('*_g0'))[0]

    qual_metrics = analyzer_clean.extensions['quality_metrics'].data
    temp_metrics = analyzer_clean.extensions['template_metrics'].data
    qual_metrics_df = pd.DataFrame(qual_metrics['metrics'])
    temp_metrics_df = pd.DataFrame(temp_metrics['metrics'])
    # concatenate template and quality metrics dataframes
    unified_df = pd.concat([qual_metrics_df, temp_metrics_df], axis=1)
    unified_df['isi_violations_pct'] = unified_df['isi_violations_count'] / unified_df['num_spikes']
    unified_df['rp_violations_pct'] = unified_df['rp_violations'] / unified_df['num_spikes']

    # remove neurons with nan in exp_decay (guaranteed dogshit)
    nan_exp_decay_index = unified_df[unified_df['exp_decay'].isna()].index
    unified_df = unified_df.drop(index=nan_exp_decay_index)
    nan_l_ratio_index = unified_df[unified_df['l_ratio'].isna()].index
    unified_df = unified_df.drop(index=nan_l_ratio_index)
    unified_df = unified_df[unified_df['num_spikes'] > 1000]
    unified_df['sd_ratio_distance'] = np.abs(np.log(unified_df['sd_ratio']))

    # get index of those neurons
    positive_neurons = unified_df[(unified_df['main_peak_to_trough_ratio'] <= -1)
                                  | (unified_df['recovery_slope'] > 0)]


    unified_df = unified_df.drop(index=positive_neurons.index.tolist())

    good_df = unified_df[
        ((unified_df['num_negative_peaks'] <= 2) &
         (unified_df['num_positive_peaks'] <= 1))
        | ((unified_df['num_negative_peaks'] <= 1) &
           (unified_df['num_positive_peaks'] <= 2))]
    good_df = good_df[
        (good_df['isi_violations_ratio'] <= 1.5) &
        (good_df['presence_ratio'] >= 0.95) &
        (good_df['num_spikes'] >= 500) &
        (good_df['snr'] >= 1.0)]

    dogshit_df = unified_df[
        (unified_df['isi_violations_ratio'] > 5) |
        (unified_df['presence_ratio'] < 0.9) |
        (unified_df['snr'] < 0.75) |
        (unified_df['num_negative_peaks'] > 4) |
        (unified_df['num_positive_peaks'] > 4) |
        (unified_df['peak_to_valley'] > 0.002) |
        (unified_df['half_width'] > 0.002)]

    good_index = good_df.index
    dogshit_index = dogshit_df.index

    best_df = unified_df.loc[good_index]
    dogshit_df = unified_df.loc[dogshit_index]
    # any indices in dogshit_df should be removed from best_df
    best_df = best_df.drop(index=dogshit_df.index, errors='ignore')

    rest_df = unified_df.drop(index=best_df.index)
    rest_df = rest_df.drop(index=dogshit_df.index)



    varcols = ['half_width', 'exp_decay',
               'isi_violations_ratio', 'isi_violations_pct', 'rp_violations_pct',
               'presence_ratio', 'snr', 'sync_spike_2', 'sync_spike_4', 'sync_spike_8', 'l_ratio', 'sd_ratio_distance']
    X = pd.concat([best_df, dogshit_df], axis=0)
    X = X[varcols]
    # identify if any columns have nan values
    nan_cols = X.columns[X.isna().any()].tolist()

    y = np.concatenate([
        np.ones(len(best_df)),
        np.zeros(len(dogshit_df))
    ])

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",  # or "l1" for feature selection
            solver="liblinear",  # needed for L1
            class_weight="balanced"
        ))
    ])

    clf.fit(X, y)

    rest_test = rest_df[varcols]
    nan_cols = rest_test.columns[rest_test.isna().any()].tolist()
    proba = clf.predict_proba(rest_test)[:, 1]
    rest_df["p_good"] = proba
    rest_df["label"] = (proba > 0.5).astype(int)
    best_test = best_df[varcols]
    best_proba = clf.predict_proba(best_test)[:, 1]
    best_df["p_good"] = best_proba
    best_df["label"] = (best_proba > 0.5).astype(int)

    good_rest_df = rest_df[rest_df["label"] == 1]
    best_df = pd.concat([best_df, good_rest_df], axis=0)

    good_positive_neurons=positive_neurons[
        ((positive_neurons['num_positive_peaks'] <= 2) & (positive_neurons['num_negative_peaks'] <=1))
        | ((positive_neurons['num_negative_peaks'] <= 2) & (positive_neurons['num_positive_peaks'] <=1))
    ]
    good_positive_neurons = good_positive_neurons[
        (good_positive_neurons['isi_violations_ratio'] <= 1)]
    best_df = pd.concat([best_df, good_positive_neurons], axis=0)

    # order by index
    best_df = best_df.sort_index()
    discarded_df = rest_df[rest_df["label"] == 0]
    return best_df.index.tolist()

def test():
    # base_folder = Path("C:\\Users\\assad\\Documents\\analysis_files\\DS13\\DS13_20250822")
    # base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250905")
    # base_folder = Path(r"C:\Users\assad\Documents\analysis_files\DS13\DS13_20250824")
    # base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251117")
    # base_folder =  Path(r"C:\Users\assad\Documents\recording_files\DS1\DS1_20251118")
    # base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251209")
    # base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251211")
    base_folder = Path(r"C:\Users\assad\Documents\recording_files\DS21\DS21_20251213")
    analyzer_path = base_folder / 'analyzer_clean'
    curate_units(analyzer_path)