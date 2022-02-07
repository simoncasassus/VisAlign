import sys
import numpy as np
include_path = '/home/simon/common/python/include/'
sys.path.append(include_path)

import Dask_editms

file_ms = 'PDS70_SB16_cont_chi2_casarestore.ms.selfcal.statwt'
file_ms_output = 'PDS70_SB16_cont_chi2_casarestore.ms.selfcal.statwt.aligned'
file_ms_ref = 'PDS70_cont_copy_chi2_casarestore_2inf.ms.selfcal.statwt'
# load best fit params from xcorr:

bestfit = np.load('output_xcorr_SB16_defaultuvrange/bestfit_xcorr_wshift.npy')
alpha_R = bestfit[0]
delta_x = bestfit[1]
delta_y = bestfit[2]

SBs = Dask_editms.apply_gain_shift(file_ms,
                                   file_ms_output=file_ms_output,
                                   alpha_R=alpha_R,
                                   Shift=[delta_x, delta_y],
                                   file_ms_ref=file_ms_ref)
