import os
import sys
import numpy as np
import astropy.units as u
from astropy.io import fits

HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)

#HOME = os.environ.get('HOME')
#include_path = HOME + '/gitcommon/VisAlign/'
#sys.path.append(include_path)

import VisAlign.Pyra_grid as Pyra_grid
import VisAlign.Tclean_grid as Tclean_grid

def punch_vis(im, duvalue, fileout, CRPIX1=1., CRPIX2=1.):
    print("punching ", fileout)
    nx, ny = im.shape
    hdu = fits.PrimaryHDU()
    hdu.data = im
    hdr = hdu.header
    CRPIX1 = (nx + 1.) / 2.
    CRPIX2 = (ny + 1.) / 2.
    hdr['CRPIX1'] = CRPIX1
    hdr['CRVAL1'] = 0.
    hdr['CDELT1'] = -duvalue
    hdr['CRPIX2'] = CRPIX2
    hdr['CRVAL2'] = 0.
    hdr['CDELT2'] = duvalue
    hdr['BUNIT'] = 'Jy'
    hdu.header = hdr
    hdu.writeto(fileout, overwrite=True)

outputdir = 'gridding_trials_hermit_4/'
outputdir = 'gridding_trials_hermit_5/'
#outputdir = 'gridding_trials_residuals_nohermit/'
os.system('mkdir ' + outputdir)

file_visSBs = './data/PDS70_IB17_cont_copy_chi2_casarestore_2inf.ms.selfcal.statwt.4xcorr'
file_visSBs = './data/PDS70_IB17_cont_copy_chi2_casarestore_2inf.ms.selfcal.statwt'
#file_visSBs = './data/output_OOselfcal__IB17_usermask_chi2_newcasarestore_2inf/_ph0_residuals.data.ms'
dx = 0.005 * u.arcsec  # if None uses theoretical formula
imsize = 2048 # 8192
#imsize = 4096
du = (1 / (imsize * dx.to(u.rad).value))
duvalue = du

GridScheme = 'None'

GridScheme = 'Pyra'
if (GridScheme == 'Pyra'):
    file_dirty = outputdir + 'dirty_' + os.path.basename(
        file_visSBs) + '_Pyra.fits'
    file_beam = outputdir + 'psf_' + os.path.basename(
        file_visSBs) + '_Pyra.fits'
    file_gridded_vis = outputdir + 'gridded_visibilities_nat_Pyra.npy'
    file_gridded_weights = outputdir + 'gridded_weights_nat_Pyra.npy'
    from pyralysis.units import lambdas_equivalencies
    dx, gridded_visibilities_nat, gridded_weights_nat = Pyra_grid.gridvis(
        file_visSBs,
        imsize=imsize,
        hermitian_symmetry=True,
        wantdirtymap=file_dirty,
        wantpsf=file_beam,
        dx=dx)
    du = (1 / (imsize * dx)).to(u.lambdas,
                                equivalencies=lambdas_equivalencies())
    np.save(file_gridded_vis, gridded_visibilities_nat)
    np.save(file_gridded_weights, gridded_weights_nat)
    print("sky image pixels: ", dx.to(u.arcsec))


    du = (1 / (imsize * dx.to(u.rad).value))
    duvalue = du
    gridded_visibilities_nat = np.load(file_gridded_vis)
    gridded_weights_nat = np.load(file_gridded_weights)
    V_pyragridder = gridded_visibilities_nat[ :, :]
    V_pyragridderR = gridded_visibilities_nat[ :, :].real
    V_pyragridderI = gridded_visibilities_nat[ :, :].imag
    w_pyragridder = gridded_weights_nat[ :, :]
    Vamp_pyragridder = np.sqrt(V_pyragridderI**2 + V_pyragridderR**2)
    punch_vis(V_pyragridderR, du, outputdir + 'V_pyragridderR.fits')
    punch_vis(V_pyragridderI, du, outputdir + 'V_pyragridderI.fits')
    punch_vis(Vamp_pyragridder, du, outputdir + 'Vamp_pyragridder.fits')
    punch_vis(w_pyragridder, du, outputdir + 'w_pyragridder.fits')

    
GridScheme = 'tclean'
if (GridScheme == 'tclean'):
    file_gridded_vis = outputdir + 'gridded_visibilities_nat_tcleangridder.npy'
    file_gridded_weights = outputdir + 'gridded_weights_nat_tcleangridder.npy'
    runtclean = True
    if runtclean:
        Tclean_grid.gendirty(file_visSBs,
                             imsize=imsize,
                             tcleanimagename='dirty_tclean_' +
                             os.path.basename(file_visSBs),
                             dx=dx,
                             outputdir=outputdir)

    dx, gridded_visibilities_nat, gridded_weights_nat = Tclean_grid.getfft(
        file_dirty=outputdir + 'dirty_tclean_' +
        os.path.basename(file_visSBs) + '.image.fits',
        file_psf=outputdir + 'dirty_tclean_' + os.path.basename(file_visSBs) +
        '.psf.fits',
        rmsnoise=3.6E-5,  # Jy/beam noise
        DoApod=True,
        outputdir=outputdir,
        FWHM_apod=3.)
    np.save(file_gridded_vis, gridded_visibilities_nat)
    np.save(file_gridded_weights, gridded_weights_nat)

    V_tcleangridder = gridded_visibilities_nat
    V_tcleangridderR = gridded_visibilities_nat.real
    V_tcleangridderI = gridded_visibilities_nat.imag
    w_tcleangridder = gridded_weights_nat
    Vamp_tcleangridder = np.sqrt(V_tcleangridderI**2 + V_tcleangridderR**2)
    punch_vis(V_tcleangridderR, du, outputdir + 'V_tcleangridderR.fits')
    punch_vis(V_tcleangridderI, du, outputdir + 'V_tcleangridderI.fits')
    punch_vis(Vamp_tcleangridder, du, outputdir + 'Vamp_tcleangridder.fits')
    punch_vis(w_tcleangridder, du, outputdir + 'w_tcleangridder.fits')

    
# now compare tclean gridder on Pyra dirty and Pyra beam
GridScheme = 'tclean'
if (GridScheme == 'tclean'):
    file_gridded_vis = outputdir + 'pyragridded_visibilities_nat_tcleangridder.npy'
    file_gridded_weights = outputdir + 'pyragridded_weights_nat_tcleangridder.npy'
    file_dirty = outputdir + 'dirty_' + os.path.basename(
        file_visSBs) + '_Pyra.fits'
    file_beam = outputdir + 'psf_' + os.path.basename(
        file_visSBs) + '_Pyra.fits'
    dx, gridded_visibilities_nat, gridded_weights_nat = Tclean_grid.getfft(
        file_dirty=file_dirty,
        file_psf=file_beam,
        rmsnoise=3.6E-5,  # Jy/beam noise
        DoApod=True,
        outputdir=outputdir,
        FWHM_apod=3.)
    np.save(file_gridded_vis, gridded_visibilities_nat)
    np.save(file_gridded_weights, gridded_weights_nat)

    V_cross = gridded_visibilities_nat
    V_crossR = gridded_visibilities_nat.real
    V_crossI = gridded_visibilities_nat.imag
    w_cross = gridded_weights_nat
    Vamp_cross = np.sqrt(V_crossI**2 + V_crossR**2)
    punch_vis(V_crossR, du, outputdir + 'V_crossR.fits')
    punch_vis(V_crossI, du, outputdir + 'V_crossI.fits')
    punch_vis(Vamp_cross, du, outputdir + 'Vamp_cross.fits')
    punch_vis(w_cross, du, outputdir + 'w_cross.fits')

