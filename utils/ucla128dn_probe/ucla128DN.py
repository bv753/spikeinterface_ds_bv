import re
import numpy as np
from pathlib import Path
from probeinterface import Probe

HERE = Path('utils') / 'ucla128dn_probe'


def _load_matlab_matrix(path):
    text = Path(path).read_text()
    text = re.sub(r'%[^\n]*', '', text)
    match = re.search(r'\[(.*?)]', text, re.S)
    if not match:
        raise ValueError(f"No matrix found in {path}")
    rows = [
        list(map(float, row.split()))
        for row in match.group(1).splitlines()
        if row.strip()
    ]
    return np.array(rows)


def get_probe():
    probewiring    = _load_matlab_matrix(HERE / 'probe_128DN.m')
    pcbwiring      = _load_matlab_matrix(HERE / 'wiring_128chPCB.m')
    headstagewiring = _load_matlab_matrix(HERE / 'Intan_128ch_headstage.m')

    pcb_from_probe = {int(r[0]): int(r[1]) for r in pcbwiring}
    amp_from_pcb   = {int(r[0]): int(r[1]) for r in headstagewiring}

    device_channel_indices = np.array([
        amp_from_pcb[pcb_from_probe[int(row[0])]] - 1
        for row in probewiring
    ])

    positions = np.column_stack([probewiring[:, 1], -probewiring[:, 3]])
    shank_ids = probewiring[:, 4].astype(int).astype(str)

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=positions, shank_ids=shank_ids)
    probe.set_device_channel_indices(device_channel_indices)
    probe.create_auto_shape(probe_type='tip')
    probe.name = 'UCLA_128DN'
    return probe
