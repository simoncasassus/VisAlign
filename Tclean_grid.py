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


def apodize(hdu, FWHM=12.):
    hduapod = deepcopy(hdu)
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


def gridvis(file_ms,
            imsize=2048,
            hermitian_symmetry=False,
            dx=None,
            outputdir='',
            tcleanimagename=False):

    print("processing: ", file_ms)

    cellsize = str(dx.to(u.arcsec).value) + 'arcsec'

    imagename = outputdir + tcleanimagename
    casacommand = '/usr/local/casareleases/casa-6.4.0-16/bin/casa --log2term --nogui -c ' + load_path_4scripts + 'tclean_gendirty.py' + ' ' + file_ms + ' ' + cellsize + ' ' + str(
        imsize) + ' ' + str(imsize) + ' ' + imagename
    print("casacommand", casacommand)
    os.system(casacommand)

    FWHM_apod = 3.  # arcsec
    DoApod = True
    DoSpike = False

    file_in = imagename + '.psf.fits'
    hdupsf = slice0(file_in)
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
        fileout = re.sub('.fits', '.apod.fits', os.path.basename(file_in))
        hdupsfapod.writeto(fileout, overwrite=True)
    else:
        psf = hdupsf.data

    file_in = imagename + '.image.fits'
    rmsnoise = 3.6E-5  # Jy/beam noise
    hdudirty = slice0(file_in)
    hdrdirty = hdudirty.header
    if DoApod:
        hdudirtyapod = apodize(hdudirty, FWHM=FWHM_apod)
        fileout = re.sub('.fits', '.apod.fits', os.path.basename(file_in))
        hdudirtyapod.writeto(fileout, overwrite=True)
        dirty = hdudirtyapod.data
    else:
        dirty = hdudirty.data

    dirtyprep = np.fft.ifftshift(dirty)
    VIB = np.fft.fft2(dirtyprep)
    VIB = np.fft.fftshift(VIB)
    VIB /= (nx * ny)

    norm_Wk = 1 / rmsnoise**2  # Sum_k W_k
    VIB *= norm_Wk

    #VIB = VIB.transpose()

    psfprep = np.fft.ifftshift(psf)
    wVIB = np.fft.fft2(psfprep)
    wVIB = np.fft.fftshift(wVIB)
    wVIB /= (nx * ny)
    wVIB *= norm_Wk

    VIB = VIB / wVIB

    #wVIB = wVIB.transpose()

    dx = hdrpsf['CDELT2']
    dx_rad = dx * np.pi / (180.)
    imsize = hdrpsf['NAXIS1']

    du = (1 / (imsize * dx_rad))

    #punch_vis(VIB.real, du, 'VIB_R.fits')
    #punch_vis(VIB.imag, du, 'VIB_I.fits')
    #punch_vis(wVIB.real, du, 'wVIB_R.fits')
    #punch_vis(wVIB.imag, du, 'wVIB_I.fits')
    #punch_vis(np.sqrt(wVIB.imag**2 + wVIB.real**2), du, 'wVIB_amp.fits')

    ##filters
    #
    #wVIB_R = wVIB.real
    #kernel_w_S = 5
    #min_w = 100.
    #wmedian = medfilt2d(wVIB_R, kernel_size=kernel_w_S)
    #mask = ((wVIB_R < wmedian / 2.) | (wVIB_R < min_w))
    #
    #DoUvrange = True
    #if DoUvrange:
    #    #recommended uvrange:  167858.72904223335 1376441.5781463135
    #    uvrange = [167858.729, 1376441.578]
    #    ivec = np.arange(0, nx)
    #    jvec = np.arange(0, ny)
    #    iis, jjs = np.meshgrid(ivec, jvec)
    #    i0 = (float(nx) - 1) / 2.
    #    j0 = (float(ny) - 1) / 2.
    #    xxs = (iis - i0) * -du
    #    yys = (jjs - j0) * du
    #    uvradss = np.sqrt(xxs**2 + yys**2)
    #    maskUvrads = ((uvradss < uvrange[0]) | (uvradss > uvrange[1]))
    #    VIB[maskUvrads] = 0
    #    wVIB[maskUvrads] = 0
    #    #mask = mask & maskUvrads
    #
    #VIB[mask] = 0
    #wVIB[mask] = 0.
    #
    #punch_vis(VIB.real, du, 'VIB_R_wfilt.fits')
    #punch_vis(VIB.imag, du, 'VIB_I_wfilt.fits')
    #punch_vis(np.sqrt(VIB.imag**2 + VIB.real**2), du, 'VIB_amp_wfilt.fits')
    #punch_vis(wVIB.real, du, 'wVIB_R_wfilt.fits')
    #punch_vis(wVIB.imag, du, 'wVIB_I_wfilt.fits')
    #punch_vis(np.sqrt(wVIB.imag**2 + wVIB.real**2), du, 'wVIB_amp_wfilt.fits')

    dx = Quantity(dx, 'deg')
    return dx, VIB, wVIB.real
