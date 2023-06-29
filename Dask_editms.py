import sys
import os
import numpy as np
from pprint import pprint
# import re
# from astropy.io import fits
# from astropy.units import Quantity

import dask.multiprocessing

#dask.config.set(scheduler=dask.multiprocessing.get)

import pyralysis
import pyralysis.io
# from pyralysis.transformers.weighting_schemes import Robust
from pyralysis.units import lambdas_equivalencies
import astropy.units as un
import dask.array as da

from pyralysis.units import array_unit_conversion


def apply_gain_shift(*args, **kwargs):
    apply(*args, **kwargs)


def apply(
        file_ms,
        file_ms_output='output_dask.ms',
        alpha_R=None,  # 1.
        addPS=None,  # {'x0':0.,'y0':0.,'F':0},
        Shift=None,
        # datacolumn='CORRECTED_DATA',  # DATA
        # datacolumns_output='CORRECTED_DATA',  # DATA
        file_ms_ref=False,
        Verbose=False):

    # file_ms_ref : reference ms for pointing
    # Shift: apply shift, pass shift alpha , dec in arcsec
    # addPS: add PS, pass position in arcsec offset from phase center, and flux in Jy

    print("applying shift with alpha_R = ", alpha_R, " Shift = ", Shift)
    print("file_ms :", file_ms)
    print("file_ms_output :", file_ms_output)
    print(
        "building output ms structure by copying from filen_ms to file_ms_output"
    )
    print("adding point sources", addPS)

    os.system("rm -rf " + file_ms_output)
    #os.system("rsync -a " + file_ms + "/  " + file_ms_output + "/")

    reader = pyralysis.io.DaskMS(input_name=file_ms)
    input_dataset = reader.calculate_psf=False)
    print("done reading")

    field_dataset = input_dataset.field.dataset

    if Shift is not None:
        delta_x = Shift[0] * np.pi / (180. * 3600.)
        delta_y = Shift[1] * np.pi / (180. * 3600.)
        print("will apply shifts ", delta_x, delta_y)

    for ims, ms in enumerate(input_dataset.ms_list):
        print("looping over partioned ms", ims)  # spwid/field
        column_keys = ms.visibilities.dataset.data_vars.keys()
        if Verbose:
            print("column_keys", column_keys)

        uvw = ms.visibilities.uvw.data
        spw_id = ms.spw_id
        pol_id = ms.polarization_id
        ncorrs = input_dataset.polarization.ncorrs[pol_id]
        nchans = input_dataset.spws.nchans[spw_id]
        print("spw_id", spw_id, "nchans", nchans)

        uvw_broadcast = da.tile(uvw, nchans).reshape((len(uvw), nchans, 3))
        #print("broadcasted uvw values to all channels")

        #print("dask .compute on channel frequencies")
        chans = input_dataset.spws.dataset[spw_id].CHAN_FREQ.data.squeeze(
            axis=0).compute() * un.Hz
        #print("done dask .compute")

        chans_broadcast = chans[np.newaxis, :, np.newaxis]
        #print("broadcasted channels to same dimmensions as uvw")
        uvw_lambdas = uvw_broadcast / chans_broadcast.to(un.m, un.spectral())

        # uvw_lambdas = array_unit_conversion(
        #    array=uvw_broadcast,
        #    unit=un.lambdas,
        #    equivalencies=lambdas_equivalencies(restfreq=chans_broadcast))

        uvw_lambdas = da.map_blocks(lambda x: x.value,
                                    uvw_lambdas,
                                    dtype=np.float64)

        msdatacolumns = []
        for acolumn in column_keys:
            if ("DATA" in acolumn) or ("CORRECTED" in acolumn) or ("MODEL"
                                                                   in acolumn):
                msdatacolumns.append(acolumn)

        if Shift is not None:
            print("applying gain and shift")
            uus = uvw_lambdas[:, :, 0]
            vvs = uvw_lambdas[:, :, 1]
            eulerphase = alpha_R * da.exp(
                2j * np.pi *
                (uus * delta_x + vvs * delta_y)).astype(np.complex64)
            # for acolumn in column_keys:
            for acolumn in msdatacolumns:
                #if "DATA" in acolumn:
                print("shifting column ", acolumn)
                ms.visibilities.dataset[acolumn] *= eulerphase[:, :,
                                                               np.newaxis]

            # if "CORRECTED_DATA" in column_keys:
            #     ms.visibilities.corrected *= eulerphase[:, :, np.newa            #        msdatacolumns.append(acolumn)
            # if "DATA" in column_keys:
            #     ms.visibilities.data *= eulerphase[:, :, np.newaxis]
            # if "MODEL_DATA" in column_keys:
            #     ms.visibilities.model *= eulerphase[:, :, np.newaxis]
            #
        elif alpha_R is not None:
            print("applying gain")
            for acolumn in msdatacolumns:
                print("shifting column ", acolumn)
                ms.visibilities.dataset[acolumn] *= alpha_R

        if addPS is not None:
            for iPS, aPS in enumerate(addPS):
                x0 = aPS['x0'] * np.pi / (180. * 3600.)
                y0 = aPS['y0'] * np.pi / (180. * 3600.)
                Flux = aPS['F']
                print("adding PS: x0 ", x0, " y0 ", y0, "F", Flux)
                uus = uvw_lambdas[:, :, 0]
                vvs = uvw_lambdas[:, :, 1]
                VisPS = Flux * da.exp(
                    2j * np.pi * (uus * x0 + vvs * y0)).astype(np.complex64)
                for acolumn in msdatacolumns:
                    ms.visibilities.dataset[acolumn] += VisPS[:, :, np.newaxis]

    if not os.path.isdir(file_ms_output):
        os.system("rsync -a " + file_ms + "/  " + file_ms_output + "/")

    print("PUNCH OUPUT MS")
    if file_ms_ref:
        print(
            "paste pointing center from reference vis file into output vis file"
        )
        print("loading reference ms")

        ref_reader = pyralysis.io.DaskMS(input_name=file_ms_ref)
        ref_dataset = ref_reader.read(calculate_psf=False)
        field_dataset = ref_dataset.field.dataset

        #if len(field_dataset) == len(input_dataset.field.dataset):
        #    print("ANCHOR ")
        #    input_dataset.field.dataset = field_dataset
        #    print("uncomment above")
        #
        #    ## print("field_dataset[0].REFERENCE_DIR",field_dataset[0].REFERENCE_DIR.compute())
        #    ## print("field_dataset[0].PHASE_DIR",field_dataset[0].PHASE_DIR.compute())
        #else:
        #print("field_dataset", field_dataset)
        #pprint(field_dataset)
        #print("field_dataset.REFERENCE_DIR", field_dataset.REFERENCE_DIR.compute())
        #print("field_dataset.PHASE_DIR", field_dataset.PHASE_DIR.compute())
        #pprint(field_dataset.REFERENCE_DIR)
        #for i, row in enumerate(input_dataset.field.dataset):
        #print("row", row)
        #pprint(row)
        #print("input_dataset.field.dataset",input_dataset.field.dataset)
        #print("input_dataset.field.dataset.REFERENCE_DIR",input_dataset.field.dataset.REFERENCE_DIR.compute())
        #print("input_dataset.field.dataset.PHASE_DIR",input_dataset.field.dataset.PHASE_DIR.compute())
        input_dataset.field.dataset.REFERENCE_DIR[:] = field_dataset.REFERENCE_DIR[0]
        input_dataset.field.dataset.PHASE_DIR[:] = field_dataset.PHASE_DIR[0]


        # Write FIELD TABLE
        print("Write FIELD TABLE ")
        #print("Changed REFERENCE_DIR", dataset.field.dataset[0].REFERENCE_DIR.compute())
        #print("Changed PHASE_DIR", dataset.field.dataset[0].PHASE_DIR.compute())
        reader.write_xarray_ds(dataset=input_dataset.field.dataset,
                               ms_name=file_ms_output,
                               columns=[
                                   'REFERENCE_DIR', 'PHASE_DIR',
                                   'PhaseDir_Ref', 'RefDir_Ref'
                               ],
                               table_name="FIELD")
    # Write MAIN TABLE
    print("Write MAIN TABLE ", msdatacolumns)
    reader.write(dataset=input_dataset,
                 ms_name=file_ms_output,
                 columns=msdatacolumns)

    #X-check pointing

    check_reader = pyralysis.io.DaskMS(input_name=file_ms_output)
    check_dataset = check_reader.read(calculate_psf=False)
    field_dataset = check_dataset.field.dataset
    # for i, row in enumerate(field_dataset):
    #print("output REFERENCE_DIR", field_dataset.REFERENCE_DIR.compute())
    #print("output PHASE_DIR", field_dataset.PHASE_DIR.compute())

    return


