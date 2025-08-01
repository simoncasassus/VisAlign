import os
import sys
from pyralysis.estimators import Degridding, BilinearInterpolation, NearestNeighbor
from pyralysis.transformers import ModelVisibilities
from pyralysis.io.fits import FITS
from pyralysis.models import PowerLawIntensityModel
from pyralysis.io import DaskMS
import dask.array as da
from astropy import units as un
#from astropy.io import fits
import numpy as np
from iminuit import Minuit


def mchi2(alpha, ref_freq, mss):

    chi2 = 0
    nsum = 0
    for ms in mss.ms_list:
        #column_keys = ms.visibilities.dataset.data_vars.keys()
        #print("column_keys", column_keys)
        #ms.visibilities.dataset['MODEL_DATA'].data = ms.visibilities.dataset['DATA'].data - ms.visibilities.dataset['MODEL_DATA'].data
        weights = ms.visibilities.weight.data
        Vobs = ms.visibilities.dataset['DATA'].data
        Vmodel = ms.visibilities.dataset['MODEL_DATA'].data

        flags = ms.visibilities.flag.data
        weights_broadcast = weights[:, np.newaxis, :]
        weights_broadcast *= ~flags

        spw_id = ms.spw_id
        nchans = mss.spws.nchans[spw_id]
        chans = mss.spws.dataset[spw_id].CHAN_FREQ.data.squeeze(
            axis=0).compute() * un.Hz
        freqs = chans.value
        freqs_broadcast = freqs[np.newaxis, :, np.newaxis]

        nsum += da.sum((weights_broadcast > 0))
        chi2_arr = weights_broadcast * (
            da.abs(Vobs - Vmodel * (freqs_broadcast / ref_freq)**alpha))**2
        chi2 += da.sum(chi2_arr)

    return chi2.compute(), nsum.compute()


def compute_specindex(mss, ref_freq=345E9):

    alpha = 0.
    achi2, nsum = mchi2(alpha, ref_freq, mss)
    print("achi2 init", achi2, " nsum ", nsum)

    alpha = 2.
    achi2, nsum = mchi2(alpha, ref_freq, mss)
    print("achi2 init", achi2, " nsum ", nsum)

    print("setting up Minuit")
    f = lambda alpha: mchi2(alpha, ref_freq, mss)[0]

    m = Minuit(f, alpha=0.)

    m.errordef = Minuit.LEAST_SQUARES
    m.print_level = 1
    m.tol = 1e-4

    m.errors['alpha'] = 1E-3

    m.limits['alpha'] = [-5, 6]

    print("start Minuit.migrad")
    m.migrad()
    m.hesse()

    DoMinos = True
    if DoMinos:
        print("start Minuit.minos")
        m.minos()

    #print(m.get_param_states())
    print("m.params", m.params)
    print("m.errors", m.errors)
    params = m.params
    print("Best fit:")
    for iparam, aparam in enumerate(params):
        aparam_name = aparam.name
        aparam_value = aparam.value
        print(aparam_name, aparam_value)

    alpha = params[0].value
    achi2, nsum = mchi2(alpha, ref_freq, mss)
    print("achi2 best", achi2, " nsum ", nsum)

    return



def main():
    file_modelimage = '/home/simon/PDS70snow/IB17_revisit/red/output_snow_inaligned/_ap0.fits'
    fits_io = FITS(input_name=file_modelimage)

    image = fits_io.read()

    ref_freq = image.data.attrs['CRVAL3']
    #ref_freq = fits.open(file_modelimage)[0].header['CRVAL3']

    intensity_model = PowerLawIntensityModel(reference_frequency=ref_freq,
                                             spectral_index=0.0)

    datadir = '/home/simon/PDS70snow/IB17_revisit/red/PSfit/ap0_residuals_alignedconcat/'

    sourcems = 'PDS70_IB17_inalign_ap0__2-Dec-2017_statwt.ms'
    sourcems = '/home/simon/PDS70snow/IB17_revisit/red/PDS70_IB17_inalign_cont_aligned_toLB19_snow_ap0.ms.selfcal'
    #output_tag = "output_specindex"

    asourcems = datadir + sourcems
    asourcems = sourcems 
    print("processing ", asourcems)

    ds = DaskMS(input_name=asourcems, chunks={'row': 100000, 'chan': 30})
    mss = ds.read(filter_flag_column=False, calculate_psf=False)
    print("done reading")
    mss.field.center_phase_dir = image.phase_center

    mv = BilinearInterpolation(input_data=mss,
                               image=image,
                               hermitian_symmetry=False,
                               padding_factor=1.,
                               intensity_model=intensity_model)

    mv.transform()
    print("done interpolation")

    #for ms in mss.ms_list:
    #    column_keys = ms.visibilities.dataset.data_vars.keys()
    #    #print("column_keys", column_keys)
    #    #ms.visibilities.dataset['MODEL_DATA'].data = ms.visibilities.dataset['DATA'].data - ms.visibilities.dataset['MODEL_DATA'].data
    #    obs_data = ms.visibilities.dataset['DATA'].data
    #    model_data = ms.visibilities.dataset['MODEL_DATA'].data

    compute_specindex(mss, ref_freq=ref_freq)


if __name__ == "__main__":
    main()
