import os
import sys
import astropy.units as u

HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/VisAlign/'
sys.path.append(include_path)

print(sys.path)
#import VisAlign.AlignMS as AlignMS
import AlignMS

file_visBset = '/home/simon/HD135344B/data/calibrated_cont_self_LB19.ms'  #REFERENCE
file_visAset = '/home/simon/HD135344B/red/SB21/DoAr44_SB21_cont_OOframe_p0_p1_ap0_ap1.ms.selfcal'

dx = 0.005 * u.arcsec  #Bset
imsize = 4096

AlignMS.xcorr(
    file_visAset,
    file_visBset,  # reference 
    dx,
    imsize,
    DoMinos=False,
    Grid=True, 
    Grid_Bset=True,
    GridScheme='Pyra',   # 'tclean'
    FilterWeights=False,
    #kernel_w_Bset=17,
    #kernel_w_Aset=27,
    #uvrange=[0.25E6,1.E6], recommended uvrange:  90643.71368279879 231645.04607826355
    #uvrange=[0.1E6, 0.307E6],
    #uvrange=False,
    #DefaultUvrange=False,
    outputdir='output_align_A_to_B/')
