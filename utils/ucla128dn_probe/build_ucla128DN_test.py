import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from probeinterface import Probe
from probeinterface.plotting import plot_probe

HERE = Path('utils') / 'ucla128dn_probe'


def load_matlab_matrix(path):
    """Parse a numeric matrix from a simple .m file (strips % comments, extracts [...] block)."""
    text = Path(path).read_text()
    text = re.sub(r'%[^\n]*', '', text)           # strip comments
    match = re.search(r'\[(.*?)]', text, re.S)
    if not match:
        raise ValueError(f"No matrix found in {path}")
    rows = [
        list(map(float, row.split()))
        for row in match.group(1).splitlines()
        if row.strip()
    ]
    return np.array(rows)


# probe_128DN.m: [probe_ch, x, y, z, shaft]  (x/z in microns; y always 0)
probewiring   = load_matlab_matrix(HERE / 'probe_128DN.m')

# wiring_128chPCB.m: [probe_ch, pcb_top, pcb_bottom]
pcbwiring     = load_matlab_matrix(HERE / 'wiring_128chPCB.m')

# Intan_128ch_headstage.m: [pcb_ch, amp_ch]
headstagewiring = load_matlab_matrix(HERE / 'Intan_128ch_headstage.m')

# Build lookup tables
pcb_from_probe = {int(r[0]): int(r[1]) for r in pcbwiring}   # probe_ch -> pcb (top)
amp_from_pcb   = {int(r[0]): int(r[1]) for r in headstagewiring}  # pcb -> amp (1-indexed)

# Chain: probe_ch -> pcb_top -> amp_ch; convert to 0-indexed for Intan
device_channel_indices = np.array([
    amp_from_pcb[pcb_from_probe[int(row[0])]] - 1
    for row in probewiring
])

positions = np.column_stack([probewiring[:, 1], -probewiring[:, 3]])  # x and -z (tip at min y)
shank_ids = probewiring[:, 4].astype(int).astype(str)    # shaft 1-4

probe = Probe(ndim=2, si_units='um')
probe.set_contacts(positions=positions, shank_ids=shank_ids)
probe.set_device_channel_indices(device_channel_indices)
probe.create_auto_shape(probe_type='tip')
probe.name = 'UCLA_128DN'

plot_probe(probe, with_contact_id=True)
plt.tight_layout()
plt.show()

