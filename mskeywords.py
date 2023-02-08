import sys
import os
import numpy as np

import pyralysis
import pyralysis.io


def pointing(file_ms):

    reader = pyralysis.io.DaskMS(input_name=file_ms)
    dataset = reader.read(calculate_psf=False)
    print("done reading")

    field_dataset = dataset.field.dataset
    ref_dir = field_dataset[0].REFERENCE_DIR.compute()
    phase_dir = field_dataset[0].PHASE_DIR.compute()
    ref_dirs = []
    phase_dirs = []
    for i, row in enumerate(field_dataset):
        ref_dir=(180./np.pi)*row.REFERENCE_DIR.compute().to_numpy()
        ref_dir=np.squeeze(ref_dir)
        if (ref_dir[0] <0):
            ref_dir[0] += 360.
            
        phase_dir=(180./np.pi)*row.PHASE_DIR.compute().to_numpy()
        phase_dir=np.squeeze(phase_dir)
        if (phase_dir[0]<0.):
            phase_dir[0]+=360.
        ref_dirs.append(ref_dir)
        phase_dirs.append(phase_dir)

    return ref_dirs, phase_dirs
