import sys
import os
import numpy as np
# import re
# from astropy.io import fits
# from astropy.units import Quantity

import pyralysis
import pyralysis.io
# from pyralysis.transformers.weighting_schemes import Robust
from pyralysis.units import lambdas_equivalencies
import astropy.units as un
import dask.array as da

from pyralysis.units import array_unit_conversion


def apply_gain_shift(file_ms,
                     file_ms_output='output_dask.ms',
                     alpha_R=1.,
                     Shift=False,
                     file_ms_ref=False):

    # file_ms_ref : reference ms for pointing

    print("applying shift with alpha_R = ", alpha_R," Shift = ", Shift)
    print("file_ms :", file_ms)
    print("file_ms_output :", file_ms_output)
    print(
        "building output ms structure by copying from filen_ms to file_ms_output"
    )

    os.system("rm -rf " + file_ms_output)
    os.system("rsync -a " + file_ms + "/  " + file_ms_output + "/")

    reader = pyralysis.io.DaskMS(input_name=file_ms)
    dataset = reader.read()

    field_dataset = dataset.field.dataset

    delta_x = Shift[0] * np.pi / (180. * 3600.)
    delta_y = Shift[1] * np.pi / (180. * 3600.)

    for ms in dataset.ms_list:  # loops over spws
        uvw = ms.visibilities.uvw.data
        spw_id = ms.spw_id
        pol_id = ms.polarization_id
        ncorrs = dataset.polarization.ncorrs[pol_id]
        nchans = dataset.spws.nchans[spw_id]

        uvw_broadcast = da.tile(uvw, nchans).reshape((len(uvw), nchans, 3))

        chans = dataset.spws.dataset[spw_id].CHAN_FREQ.data.squeeze(
            axis=0).compute() * un.Hz

        chans_broadcast = chans[np.newaxis, :, np.newaxis]

        uvw_lambdas = uvw_broadcast / chans_broadcast.to(un.m, un.spectral())

        # uvw_lambdas = array_unit_conversion(
        #    array=uvw_broadcast,
        #    unit=un.lambdas,
        #    equivalencies=lambdas_equivalencies(restfreq=chans_broadcast))

        uvw_lambdas = da.map_blocks(lambda x: x.value,
                                    uvw_lambdas,
                                    dtype=np.float64)

        if Shift:
            print("applying gain and shift")
            uus = uvw_lambdas[:, :, 0]
            vvs = uvw_lambdas[:, :, 1]
            eulerphase = alpha_R * da.exp(
                2j * np.pi *
                (uus * delta_x + vvs * delta_y)).astype(np.complex64)
            ms.visibilities.data *= eulerphase[:, :, np.newaxis]
        else:
            print("applying gain")
            ms.visibilities.data *= alpha_R

    if file_ms_output:
        print("PUNCH OUPUT MS")
        if file_ms_ref:
            print(
                "paste pointing center from reference vis file into output vis file"
            )
            print("loading reference ms")

            ref_reader = pyralysis.io.DaskMS(input_name=file_ms_ref)
            ref_dataset = ref_reader.read()
            field_dataset = ref_dataset.field.dataset

            if len(field_dataset) == len(dataset.field.dataset):
                for i, row in enumerate(dataset.field.dataset):
                    row['REFERENCE_DIR'] = field_dataset[i].REFERENCE_DIR
                    row['PHASE_DIR'] = field_dataset[i].PHASE_DIR
            else:
                for i, row in enumerate(dataset.field.dataset):
                    row['REFERENCE_DIR'] = field_dataset[0].REFERENCE_DIR
                    row['PHASE_DIR'] = field_dataset[0].PHASE_DIR

            # Write FIELD TABLE
            print(" Write FIELD TABLE ")
            reader.write_xarray_ds(dataset=dataset.field.dataset,
                                   ms_name=file_ms_output,
                                   table_name="FIELD")
            # Write MAIN TABLE
            print(" Write MAIN TABLE ")
            reader.write(dataset=dataset,
                         ms_name=file_ms_output,
                         columns="DATA")

    return


# #full LBs:
# alpha_R=0.7663912035737
# delta_x=-0.011864883679078701
# delta_y=-0.018108213291175686
#
#
# #>250klambda
# alpha_R=0.7574830128410711
# delta_x=-0.012395319330140228
# delta_y=-0.018187380096834835
#
#
#
# file_ms='PDS70_SB16_cont.ms.selfcal'
# file_ms_output='PDS70_SB16_cont_selfcal_aligned.ms'
# file_ms_ref='PDS70_cont.ms'
# os.system("rm -rf "+file_ms_output)
# os.system("rsync -va "+file_ms+"/  "+file_ms_output+"/")
#
# SBs = apply_gain_shift(file_ms,file_ms_output=file_ms_output,alpha_R=alpha_R,Shift=[delta_x,delta_y],file_ms_ref=file_ms_ref)
# #SBs = apply_gain_shift(file_ms,file_ms_output=file_ms_output,alpha_R=alpha_R,file_ms_ref=file_ms_ref)
#
#
