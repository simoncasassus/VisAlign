import sys
import astropy.units as u

include_path = '/home/simon/common/python/include/'
sys.path.append(include_path)

import Pyra_alignms






file_visSBs = 'PDS70_SB16_cont_chi2_casarestore.ms.selfcal.statwt.4xcorr'
file_visLBs = 'PDS70_cont_copy_chi2_casarestore_2inf.ms.selfcal.statwt.4xcorr'


dx = 0.003 * u.arcsec  #LBs
imsize = 2048

Pyra_alignms.xcorr(
    file_visSBs,
    file_visLBs,
    dx,
    imsize,
    Grid=False,
    Grid_LBs=False,
    DoMinos=False,
    kernel_w_L = 17,
    kernel_w_S = 27,
    wprof_factor = 8.,
    #uvrange=[0.25E6,0.75E6],
    uvrange=False,
    DefaultUvrange=True,
    outputdir='output_xcorr_SB16_defaultuvrange_Wfactor3/')
