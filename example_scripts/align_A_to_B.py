import sys
import os
import numpy as np

include_path = '/home/simon/common/python/include/'
include_path = '/home/simon/gitcommon/VisAlign/'
sys.path.append(include_path)

import Dask_editms

file_ms = '/home/simon/HD135344B/red/SB21/DoAr44_SB21_cont_OOframe_p0_p1_ap0_ap1.ms.selfcal'
file_ms_output = 'HD135344B_SB21_aligned.ms'
file_ms_ref = '/home/simon/HD135344B/data/calibrated_cont_self_LB19.ms'

os.system("rm -rf "+file_ms_output)

bestfit = np.load('output_align_A_to_B//bestfit_xcorr_wshift.npy')
alpha_R = bestfit[0]
delta_x = bestfit[1]
delta_y = bestfit[2]

# alpha_R = 1.

SBs = Dask_editms.apply_gain_shift(file_ms,
                                   file_ms_output=file_ms_output,
                                   alpha_R=alpha_R,
                                   datacolumn='DATA',
                                   datacolumns_output='DATA',
                                   Shift=[delta_x, delta_y],
                                   file_ms_ref=file_ms_ref)
