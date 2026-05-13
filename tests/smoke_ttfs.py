import numpy as np
import pandas as pd
from pathlib import Path

# Create minimal fake objects that match the interface used by parse_opto_tagging
class FakeSignal:
    def __init__(self, t_array):
        self._t = np.array(t_array)
    def to_tsd(self):
        class TSD:
            def __init__(self, t):
                self.t = t
        return TSD(self._t)

class FakeBS:
    def __init__(self, metadata, signals):
        self.metadata = metadata
        self._signals = signals
    def __getitem__(self, idx):
        return self._signals[idx]

class FakeSpikeTrain:
    def __init__(self, times):
        # nap.Ts exposes .index as numpy array of times
        self.index = np.array(times)

# Build metadata with non-sequential indices to reproduce the original bug
meta = pd.DataFrame({
    'event': ['chr2_on_10', 'chr2_on_20', 'chr2_trace']
}, index=[101, 305, 999])

# Create signals mapping keyed by the metadata index values
signals = {
    101: FakeSignal([1.0]),
    305: FakeSignal([2.0]),
    999: FakeSignal([0.0])
}

bs = FakeBS(meta, signals)

# Create fake ephys_data with two units
ephys_data = {
    'u1': FakeSpikeTrain([0.9, 1.005, 1.02, 2.1]),
    'u2': FakeSpikeTrain([1.5, 2.005, 2.01])
}

# Now import the function under test
from parse_opto_tagging import plot_time_to_first_spike_distribution

out_folder = Path('.') / 'tests' / 'smoke_output'
out_folder.mkdir(parents=True, exist_ok=True)

df, figpath = plot_time_to_first_spike_distribution(ephys_data, bs, save_folder=out_folder, opsins=['chr2'], post_window_ms=50.0, min_latency_ms=0.2)
print('DF head:')
print(df.head())
print('Saved figure at:', figpath)

