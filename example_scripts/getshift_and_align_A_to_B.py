import os
import sys
import astropy.units as u
import numpy as np

HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/VisAlign/'
sys.path.append(include_path)

print(sys.path)
#import VisAlign.AlignMS as AlignMS
import AlignMS
import Dask_editms
import mskeywords
import pprint

file_visBset = 'MWC758_LB17_B6_606.ms'  #REFERENCE
file_visAset = '/home/simon/MWC758_B6/red/SB17/MWC758_B6_SB_snow_p0_p1_p2.ms.selfcal'

outputdir = 'output_align_SB17_LB17/'

file_ms_output = 'MWC758_B6_SB17_snow_aligned_toLB17.ms'

"""
phase centers could be different - set CoarseAlign to True if this might be the case
"""
file_ms_output_coarsealign = 'MWC758_B6_SB17_snow_coarsealigned_toLB17.ms'
CoarseAlign = True

"""
set canvas for gridding
"""
dx = 0.005 * u.arcsec  # pixel size for the finer resolution dataset
imsize = 2048  # should be large enough so that 1/(dx * imsize) is comparable to the shortest baselines.


######################################################################


def select1stfield(direction):
    direction = np.squeeze(direction)
    print("direction.shape", direction.shape)
    if len(direction.shape) > 1:
        direction = direction[0]
    return direction


ref_dirs_A, phase_dirs_A = mskeywords.pointing(file_visAset)
ref_dirs_A = select1stfield(ref_dirs_A)
phase_dirs_A = select1stfield(phase_dirs_A)

print("file_visAset ref_dirs", pprint.pformat(ref_dirs_A))
print("file_visAset phase_dirs", pprint.pformat(phase_dirs_A))

ref_dirs_B, phase_dirs_B = mskeywords.pointing(file_visBset)
ref_dirs_B = select1stfield(ref_dirs_B)
phase_dirs_B = select1stfield(phase_dirs_B)

#ref_dirs_B = np.squeeze(ref_dirs_B)
#phase_dirs_B = np.squeeze(phase_dirs_B)
print("file_visBset ref_dirs", pprint.pformat(ref_dirs_B))
print("file_visBset phase_dirs", pprint.pformat(phase_dirs_B))

delta_x_0 = -1 * 3600. * (phase_dirs_B[0] - phase_dirs_A[0]) / np.cos(
    phase_dirs_B[1] * np.pi / 180.)
delta_y_0 = -1 * (phase_dirs_B[1] - phase_dirs_A[1]) * 3600.

print("coarse shift: delta_x", delta_x_0)
print("coarse shift: delta_y", delta_y_0)


if CoarseAlign:
    file_ms = file_visAset
    file_ms_ref = file_visBset
    alpha_R = 1.
    SBs = Dask_editms.apply_gain_shift(
        file_ms,
        file_ms_output=file_ms_output_coarsealign,
        alpha_R=alpha_R,
        #datacolumn='DATA',
        #datacolumns_output='DATA',
        Shift=[delta_x_0, delta_y_0],
        file_ms_ref=file_ms_ref)

    file_visAset = file_ms_output_coarsealign

AlignMS.xcorr(
    file_visAset,
    file_visBset,  # reference 
    dx,
    imsize,
    DoMinos=True,
    Grid=True,
    Grid_Bset=True,
    GridScheme='Pyra',  # 'tclean'
    Fix_alpha_R=False,
    range_delta_x=(-.5, .5),
    range_delta_y=(-.5, .5),
    outputdir=outputdir)


"""
now apply the shift
"""


file_ms = file_visAset
file_ms_ref = file_visBset

bestfit = np.load(outputdir + 'bestfit_xcorr_wshift.npy')
alpha_R = bestfit[0]
delta_x = bestfit[1]
delta_y = bestfit[2]

# alpha_R = 1.

SBs = Dask_editms.apply_gain_shift(file_ms,
                                   file_ms_output=file_ms_output,
                                   alpha_R=alpha_R,
                                   Shift=[delta_x, delta_y],
                                   file_ms_ref=file_ms_ref)
