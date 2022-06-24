import os
import sys
import astropy.units as u
from astropy.units import Quantity
import numpy as np
import re
from astropy.io import fits
from copy import deepcopy
from scipy.signal import medfilt2d

load_path_4scripts = os.environ['HOME'] + '/gitcommon/VisAlign/'

HOME = os.environ.get('HOME')
include_path = HOME + '/common/python/include'
sys.path.append(include_path)

from ImUtils.Cube2Im import slice0

from astropy.wcs import WCS


def apodize(hdu, FWHM=12., NMAX=16384):  # 8192
    hduapod = deepcopy(hdu)
    #hdr=hdu[0].header
    #hdrapod=hduapod[0].header
    #size_in = hdr['NAXIS2'] * hdr['CDELT2'] * 3600.
    #size_ideal = 4. * FWHM
    #NIDEAL = size_ideal / (hdr['CDELT2'] * 3600.)
    #NAPOD = NIDEAL
    #if NAPOD > NMAX:
    #    NAPOD = NMAX
    #else:
    #    NAPOD=2**(np.rint(np.log2(NIDEAL)))
    #print("NAXIS2 = ", hdr['NAXIS2'], "NAPOD = ",NAPOD)

    im = hduapod.data
    hdr = hduapod.header
    nx = hdr['NAXIS1']
    ny = hdr['NAXIS2']
    ivec = np.arange(0, nx)
    jvec = np.arange(0, ny)
    iis, jjs = np.meshgrid(ivec, jvec)
    i0 = (float(nx) - 1) / 2.
    j0 = (float(ny) - 1) / 2.
    xxs = (iis - i0) * hdr['CDELT2'] * 3600.
    yys = (jjs - j0) * hdr['CDELT2'] * 3600.
    rrs = np.sqrt(xxs**2 + yys**2)
    sigma = FWHM / (2. * np.sqrt(2. * np.log(2)))
    atten = np.exp(-0.5 * rrs**2 / sigma**2)
    im = im * atten
    hduapod.data = im
    return hduapod


def punch_vis(im, du, fileout, CRPIX1=1., CRPIX2=1.):
    print("punching ", fileout)
    nx, ny = im.shape
    hdu = fits.PrimaryHDU()
    hdu.data = im
    hdr = hdu.header
    CRPIX1 = (nx + 1.) / 2.
    CRPIX2 = (ny + 1.) / 2.
    hdr['CRPIX1'] = CRPIX1
    hdr['CRVAL1'] = 0.
    hdr['CDELT1'] = -du
    hdr['CRPIX2'] = CRPIX2
    hdr['CRVAL2'] = 0.
    hdr['CDELT2'] = du
    hdr['BUNIT'] = 'Jy'
    hdu.header = hdr
    hdu.writeto(fileout, overwrite=True)


def gridvis(
        file_ms,
        imsize=2048,
        rmsnoise=3.6E-5,  # Jy/beam noise
        hermitian_symmetry=False,
        dx=None,
        outputdir='',
        FWHM_apod=3.,  # arcsec
        DoApod=True,
        tcleanimagename=False):

    gendirty(file_ms,
             imsize=imsize,
             tcleanimagename='dirty_tclean_' + os.path.basename(file_ms),
             dx=dx,
             outputdir=outputdir)

    dx, gridded_visibilities_nat, gridded_weights_nat = getfft(
        file_dirty=outputdir + 'dirty_tclean_' + os.path.basename(file_ms) +
        '.image.fits',
        file_psf=outputdir + 'dirty_tclean_' + os.path.basename(file_ms) +
        '.psf.fits',
        rmsnoise=rmsnoise,  # Jy/beam noise
        DoApod=DoApod,
        outputdir=outputdir,
        FWHM_apod=FWHM_apod)

    dx = Quantity(dx, 'deg')
    return dx, gridded_visibilities_nat, gridded_weights_nat.real


#def gridvis0(file_ms,
#            imsize=2048,
#            rmsnoise = 3.6E-5,  # Jy/beam noise
#            hermitian_symmetry=False,
#            dx=None,
#            outputdir='',
#            FWHM_apod = 3.,  # arcsec
#            tcleanimagename=False):
#
#    print("processing: ", file_ms)
#
#    cellsize = str(dx.to(u.arcsec).value) + 'arcsec'
#
#    imagename = outputdir + tcleanimagename
#    casacommand = '/usr/local/casareleases/casa-6.4.0-16/bin/casa --log2term --nogui -c ' + load_path_4scripts + 'tclean_gendirty.py' + ' ' + file_ms + ' ' + cellsize + ' ' + str(
#        imsize) + ' ' + str(imsize) + ' ' + imagename
#    print("casacommand", casacommand)
#    os.system(casacommand)
#
#    DoApod = True
#    DoSpike = False
#
#    file_in = imagename + '.psf.fits'
#    hdupsf = slice0(file_in)
#    hdrpsf = hdupsf.header
#    nx = hdrpsf['NAXIS1']
#    ny = hdrpsf['NAXIS2']
#    if DoSpike:
#        psf = hdupsf.data
#        psf *= 0
#        psf[int(hdrpsf['CRPIX1']) - 1, int(hdrpsf['CRPIX2']) - 1] = 1.
#    elif DoApod:
#        hdupsfapod = apodize(hdupsf, FWHM=FWHM_apod)
#        psf = hdupsfapod.data
#        fileout = re.sub('.fits', '.apod.fits', os.path.basename(file_in))
#        hdupsfapod.writeto(outputdir+fileout, overwrite=True)
#    else:
#        psf = hdupsf.data
#
#    file_in = imagename + '.image.fits'
#    hdudirty = slice0(file_in)
#    hdrdirty = hdudirty.header
#    if DoApod:
#        hdudirtyapod = apodize(hdudirty, FWHM=FWHM_apod)
#        fileout = re.sub('.fits', '.apod.fits', os.path.basename(file_in))
#        hdudirtyapod.writeto(outputdir+fileout, overwrite=True)
#        dirty = hdudirtyapod.data
#    else:
#        dirty = hdudirty.data
#
#    dirtyprep = np.fft.fftshift(dirty)
#    VIB = np.fft.ifft2(dirtyprep)
#    VIB = np.fft.ifftshift(VIB)
#    #VIB /= (nx * ny) << not required because of internal ifft norm
#
#    norm_Wk = 1 / rmsnoise**2  # Sum_k W_k
#    VIB *= norm_Wk
#
#
#    psfprep = np.fft.fftshift(psf)
#    wVIB = np.fft.ifft2(psfprep)
#    wVIB = np.fft.ifftshift(wVIB)
#    #wVIB /= (nx * ny)  << not required because of internal ifft norm
#    wVIB *= norm_Wk
#
#    VIB = VIB / wVIB
#
#
#    dx = hdrpsf['CDELT2']
#    dx_rad = dx * np.pi / (180.)
#    imsize = hdrpsf['NAXIS1']
#
#    du = (1 / (imsize * dx_rad))
#
#
#    dx = Quantity(dx, 'deg')
#    return dx, VIB, wVIB.real


def gendirty(file_ms,
             imsize=2048,
             dx=None,
             outputdir='',
             tcleanimagename=False):

    cellsize = str(dx.to(u.arcsec).value) + 'arcsec'

    imagename = outputdir + tcleanimagename
    casacommand = 'casa --log2term --nogui -c ' + load_path_4scripts + 'tclean_gendirty.py' + ' ' + file_ms + ' ' + cellsize + ' ' + str(
        imsize) + ' ' + str(imsize) + ' ' + imagename
    print("casacommand", casacommand)
    os.system(casacommand)


def getfft(
        file_dirty,
        file_psf,
        rmsnoise=3.6E-5,  # Jy/beam noise
        hermitian_symmetry=False,
        outputdir='',
        DoApod=True,
        DoSpike=False,  # just a xcheck on fft
        FWHM_apod=3.):

    hdupsf = slice0(file_psf)
    hdrpsf = hdupsf.header
    nx = hdrpsf['NAXIS1']
    ny = hdrpsf['NAXIS2']
    if DoSpike:
        psf = hdupsf.data
        psf *= 0
        psf[int(hdrpsf['CRPIX1']) - 1, int(hdrpsf['CRPIX2']) - 1] = 1.
    elif DoApod:
        hdupsfapod = apodize(hdupsf, FWHM=FWHM_apod)
        psf = hdupsfapod.data
        fileout = re.sub('.fits', '.apod.fits', os.path.basename(file_psf))
        hdupsfapod.writeto(outputdir + fileout, overwrite=True)
    else:
        psf = hdupsf.data

    hdudirty = slice0(file_dirty)
    hdrdirty = hdudirty.header
    if DoApod:
        hdudirtyapod = apodize(hdudirty, FWHM=FWHM_apod)
        fileout = re.sub('.fits', '.apod.fits', os.path.basename(file_dirty))
        hdudirtyapod.writeto(outputdir + fileout, overwrite=True)
        dirty = hdudirtyapod.data
    else:
        dirty = hdudirty.data

    dirtyprep = np.fft.fftshift(dirty)
    VIB = np.fft.ifft2(dirtyprep)
    VIB = np.fft.ifftshift(VIB)
    #VIB /= (nx * ny) << not required because of internal ifft norm

    norm_Wk = 1 / rmsnoise**2  # Sum_k W_k
    VIB *= norm_Wk

    psfprep = np.fft.fftshift(psf)
    wVIB = np.fft.ifft2(psfprep)
    wVIB = np.fft.ifftshift(wVIB)
    #wVIB /= (nx * ny)  << not required because of internal ifft norm
    wVIB *= norm_Wk

    VIB = VIB / wVIB

    dx = hdrpsf['CDELT2']
    dx_rad = dx * np.pi / (180.)
    imsize = hdrpsf['NAXIS1']

    du = (1 / (imsize * dx_rad))

    dx = Quantity(dx, 'deg')
    return dx, VIB, wVIB.real
