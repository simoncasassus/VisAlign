import sys
import astropy.units as u
from scipy import ndimage
import scipy as sp
import numpy as np
from astropy.units import Quantity
import re
from iminuit import Minuit
from astropy.io import fits
import os

#HOME = os.environ.get('HOME')
#include_path = HOME + '/gitcommon/VisAlign/'
#sys.path.append(include_path)


def cartesian2polar(outcoords, inputshape, origin, fieldscale=1.):
    rindex, thetaindex = outcoords
    x0, y0 = origin
    theta = thetaindex * 2 * np.pi / (inputshape[0] - 1)
    y = rindex * np.cos(theta) / fieldscale
    x = rindex * np.sin(theta) / fieldscale
    ix = -x + x0
    iy = y + y0
    return (iy, ix)


def polarexpand(im):
    (ny, nx) = im.shape
    im_polar = sp.ndimage.geometric_transform(im,
                                              cartesian2polar,
                                              order=1,
                                              output_shape=(im.shape),
                                              extra_keywords={
                                                  'inputshape':
                                                  im.shape,
                                                  'fieldscale':
                                                  1.,
                                                  'origin':
                                                  (((nx + 1) / 2) - 1,
                                                   ((ny + 1) / 2) - 1)
                                              })
    return im_polar


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


def shiftvis(V_Aset, uus, vvs, alpha_R, delta_x, delta_y):
    argphase = 2. * np.pi * (uus * (delta_x * np.pi / (180. * 3600.)) + vvs *
                             (delta_y * np.pi / (180. * 3600.)))
    # eulerphase = np.cos(argphase)+1j*np.sin(argphase)
    eulerphase = np.exp(1j * argphase)

    V_Bset_m = alpha_R * V_Aset * eulerphase
    return V_Bset_m


def chi2(V_Aset, V_Bset, w, uus, vvs, alpha_R, delta_x, delta_y):
    V_Bset_m = shiftvis(V_Aset, uus, vvs, alpha_R, delta_x, delta_y)
    diff = V_Bset - V_Bset_m
    squarediff = (diff.real**2) + (diff.imag**2)
    retval = np.sum(w * squarediff)
    if np.isnan(retval):
        print("chi2 is NaN")
        retval = np.inf
    return retval


def chi2DERRS(V_Aset, V_Bset, varA, varB, uus, vvs, alpha_R, delta_x, delta_y):
    V_Bset_m = shiftvis(V_Aset, uus, vvs, alpha_R, delta_x, delta_y)
    diff = V_Bset - V_Bset_m
    squarediff = (diff.real**2) + (diff.imag**2)
    weights = 1. / (varB + alpha_R**2 * varA)
    retval = np.sum(squarediff * weights)
    if np.isnan(retval):
        print("chi2 is NaN")
        retval = np.inf
    return retval


def xcorr(
        file_visAset,
        file_visBset,  # reference visibility dataset
        dx,
        imsize,
        GridScheme='Pyra',  # 'tclean'
        FWHM_apod=6.,
        DoApod=True,
        data_column=None,  # None: default, CORRECTED_DATA ->  DATA
        Grid=True,
        Grid_Bset=True,
        uvrange=[-1, -1],  # -1: no filtering, None: recommended uvrange
        DefaultUvrange=False,
        DoMinos=False,
        kernel_w_Bset=5,
        kernel_w_Aset=5,
        FilterWeights=True,
        wprof_factor=10.,
        min_wA=100.,  # (mJy**-2)
        min_wB=100.,
        Reset=True,
        outputdir='output_xcorr/'):
    nx = imsize
    ny = imsize

    if DefaultUvrange:
        uvrange = None

    if Reset:
        os.system('rm -rf ' + outputdir)
        os.system('mkdir ' + outputdir)

    file_gridded_vis_Aset = outputdir + 'Aset_aligned_gridded_visibilities_nat.npy'
    file_gridded_weights_Aset = outputdir + 'Aset_aligned_gridded_weights_nat.npy'
    file_gridded_vis_Bset = outputdir + 'Bset_gridded_visibilities_nat.npy'
    file_gridded_weights_Bset = outputdir + 'Bset_gridded_weights_nat.npy'

    if Grid:
        if (GridScheme == 'Pyra'):
            file_dirty = outputdir + 'dirty_' + os.path.basename(
                file_visAset) + '.fits'
            from VisAlign import Pyra_grid
            #import Pyra_grid
            from pyralysis.units import lambdas_equivalencies
            dx, Aset_gridded_visibilities_nat, Aset_gridded_weights_nat = Pyra_grid.gridvis(
                file_visAset,
                imsize=imsize,
                wantdirtymap=file_dirty,
                dx=dx,
                data_column=data_column)
            du = (1 / (imsize * dx)).to(u.lambdas,
                                        equivalencies=lambdas_equivalencies())
            duvalue = du.value
        elif (GridScheme == 'tclean'):
            from VisAlign import Tclean_grid
            #import Tclean_grid
            file_dirty = outputdir + 'dirty_' + os.path.basename(file_visAset)
            dx, Aset_gridded_visibilities_nat, Aset_gridded_weights_nat = Tclean_grid.gridvis(
                file_visAset,
                imsize=imsize,
                tcleanimagename=file_dirty,
                dx=dx,
                FWHM_apod=FWHM_apod,
                DoApod=DoApod,
                outputdir=outputdir)
        else:
            print("set GridScheme  to either tclean or Pyra")
            raise ValueError("set GridScheme  to either tclean or Pyra")

        np.save(file_gridded_vis_Aset, Aset_gridded_visibilities_nat)
        np.save(file_gridded_weights_Aset, Aset_gridded_weights_nat)
        print("sky image pixels: ", dx.to(u.arcsec))

    if Grid_Bset:
        if (GridScheme == 'Pyra'):
            file_dirty = outputdir + 'dirty_' + os.path.basename(
                file_visBset) + '.fits'
            from VisAlign import Pyra_grid
            #import Pyra_grid
            from pyralysis.units import lambdas_equivalencies
            dx, Bset_gridded_visibilities_nat, Bset_gridded_weights_nat = Pyra_grid.gridvis(
                file_visBset,
                imsize=imsize,
                wantdirtymap=file_dirty,
                dx=dx,
                data_column=data_column)
            du = (1 / (imsize * dx)).to(u.lambdas,
                                        equivalencies=lambdas_equivalencies())
            duvalue = du.value
        elif (GridScheme == 'tclean'):
            from VisAlign import Tclean_grid
            #import Tclean_grid
            file_dirty = outputdir + 'dirty_' + os.path.basename(file_visAset)
            dx, Bset_gridded_visibilities_nat, Bset_gridded_weights_nat = Tclean_grid.gridvis(
                file_visBset,
                imsize=imsize,
                tcleanimagename=file_dirty,
                dx=dx,
                outputdir=outputdir)

        np.save(file_gridded_vis_Bset, Bset_gridded_visibilities_nat)
        np.save(file_gridded_weights_Bset, Bset_gridded_weights_nat)

    du = (1 / (imsize * dx.to(u.rad).value))
    duvalue = du
    print(">>>>> du ", du, "klambda")

    Aset_gridded_visibilities_nat = np.load(file_gridded_vis_Aset)
    Aset_gridded_weights_nat = np.load(file_gridded_weights_Aset)
    Bset_gridded_visibilities_nat = np.load(file_gridded_vis_Bset)
    Bset_gridded_weights_nat = np.load(file_gridded_weights_Bset)

    #print("Aset_gridded_visibilities_nat.shape",
    #      Aset_gridded_visibilities_nat.shape)
    #print(Aset_gridded_visibilities_nat.dtype)
    #print(Aset_gridded_weights_nat.shape)

    #if GridScheme == 'Pyra':
    #    V_Aset = Aset_gridded_visibilities_nat[ :, :]
    #    V_AsetR = Aset_gridded_visibilities_nat[ :, :].real
    #    V_AsetI = Aset_gridded_visibilities_nat[ :, :].imag
    #    V_Bset = Bset_gridded_visibilities_nat[ :, :]
    #    V_BsetR = Bset_gridded_visibilities_nat[ :, :].real
    #    V_BsetI = Bset_gridded_visibilities_nat[ :, :].imag
    #    w_Aset = Aset_gridded_weights_nat[ :, :]
    #    w_Bset = Bset_gridded_weights_nat[ :, :]
    #else:
    V_Aset = Aset_gridded_visibilities_nat
    V_AsetR = Aset_gridded_visibilities_nat.real
    V_AsetI = Aset_gridded_visibilities_nat.imag
    V_Bset = Bset_gridded_visibilities_nat
    V_BsetR = Bset_gridded_visibilities_nat.real
    V_BsetI = Bset_gridded_visibilities_nat.imag
    w_Aset = Aset_gridded_weights_nat
    w_Bset = Bset_gridded_weights_nat


    Vamp_Aset = np.sqrt(V_AsetI**2 + V_AsetR**2)
    punch_vis(V_AsetR, du, outputdir + 'V_AsetR.fits')
    punch_vis(V_AsetI, du, outputdir + 'V_AsetI.fits')
    punch_vis(Vamp_Aset, du, outputdir + 'Vamp_Aset.fits')
    punch_vis(w_Aset, du, outputdir + 'w_Aset.fits')

    Vamp_Bset = np.sqrt(V_BsetI**2 + V_BsetR**2)
    punch_vis(V_BsetR, du, outputdir + 'V_BsetR.fits')
    punch_vis(V_BsetI, du, outputdir + 'V_BsetI.fits')
    punch_vis(Vamp_Bset, du, outputdir + 'Vamp_Bset.fits')
    punch_vis(w_Bset, du, outputdir + 'w_Bset.fits')

    #import PyVtools.Vtools as Vtools
    #Vtools.View(w_Bset)

    from scipy.signal import medfilt2d

    if FilterWeights:
        print('filtering V_Bset')
        wmedian = medfilt2d(w_Bset, kernel_size=kernel_w_Bset)
        #Vtools.View(wmedian)
        mask = ((w_Bset < wmedian / 2.) | (w_Bset < min_wB))
        V_Bset[mask] = 0
        V_BsetR[mask] = 0.
        V_BsetI[mask] = 0.
        w_Bset[mask] = 0.

        print('filtering V_Aset')
        wmedian = medfilt2d(w_Aset, kernel_size=kernel_w_Aset)
        mask = ((w_Aset < wmedian / 2.) | (w_Aset < min_wA))
        V_Aset[mask] = 0
        V_AsetR[mask] = 0.
        V_AsetI[mask] = 0.
        w_Aset[mask] = 0.

    #wcommon = w_Bset * w_Aset / (w_Bset + w_Aset)
    wcommon=np.zeros_like(w_Bset)
    maskfinite=((w_Bset + w_Aset)  > 0.)
    np.divide(w_Bset * w_Aset, (w_Bset + w_Aset), out=wcommon, where=maskfinite)

    
    wcommon[(w_Bset < min_wB) | (w_Aset < min_wA)] = 0.
    
    mask = (wcommon == 0.)
    wcommonA = w_Aset.copy()
    wcommonA[mask] = 0.
    maskfinite = (wcommon > 0.)
    varA = np.zeros_like(wcommonA)
    np.divide(1, wcommonA, out=varA, where=maskfinite)
    varA[mask] = np.inf

    wcommonB = w_Bset.copy()
    maskfinite = (wcommon > 0.)
    wcommonB[mask] = 0.
    varB = np.zeros_like(wcommonB)
    np.divide(1, wcommonB, out=varB, where=maskfinite)
    varB[mask] = np.inf

    wcommon = np.nan_to_num(wcommon, nan=0.)
    V_Aset = np.nan_to_num(V_Aset)
    V_Bset = np.nan_to_num(V_Bset)
    dofs = np.sum((wcommon > 0.))
    print("dofs = ", dofs)


    print("uv cell size", duvalue)
    us = -1 * (np.arange(0, nx) - (nx - 1.) / 2.) * duvalue
    vs = (np.arange(0, ny) - (ny - 1.) / 2.) * duvalue
    uus, vvs = np.meshgrid(us, vs)
    uvradss = np.sqrt(uus**2 + vvs**2)
    print("max uvrange: ", np.max(uvradss))

    import matplotlib
    import matplotlib.pyplot as plt

    #Vtools.View(w_Bset)

    print("polarexpand w_Bset")
    w_Bset_polar = polarexpand(w_Bset)
    #w_Bset_prof = np.average(w_Bset_polar, axis=1)
    w_Bset_prof = np.median(w_Bset_polar, axis=1)
    nphis, nrs = w_Bset_polar.shape
    uvrads = (np.arange(nrs)) * duvalue
    print("plotting w_Bset")
    plt.plot(uvrads, w_Bset_prof, label='w_Bset', color='C1')
    maskprof = ((w_Bset_prof > np.max(w_Bset_prof) / wprof_factor))

    #iw1=np.argmin(uvrads[maskprof])
    #iw2=np.argmax(uvrads[maskprof])
    uvmin_Bset = np.min(uvrads[maskprof])
    uvmax_Bset = np.max(uvrads[maskprof])

    print("polarexpand w_Aset")
    w_Aset_polar = polarexpand(w_Aset)
    #w_Aset_prof = np.average(w_Aset_polar, axis=1)
    w_Aset_prof = np.median(w_Aset_polar, axis=1)
    print("plotting w_Aset")
    plt.plot(uvrads, w_Aset_prof, label='w_Aset', color='C0')
    maskprof = ((w_Aset_prof > np.max(w_Aset_prof) / wprof_factor))
    #iw1=np.argmin(uvrads[maskprof])
    #iw2=np.argmax(uvrads[maskprof])
    uvmin_Aset = np.min(uvrads[maskprof])
    uvmax_Aset = np.max(uvrads[maskprof])

    uvminreco = max((uvmin_Aset, uvmin_Bset))
    uvmaxreco = min((uvmax_Aset, uvmax_Bset))
    print("recommended uvrange: ", uvminreco, uvmaxreco)
    plt.legend()
    print("plotting w profiles to: wprofs.pdf")
    plt.savefig(outputdir + 'wprofs_full.pdf', bbox_inches='tight')
    plt.xlim(uvminreco, uvmaxreco)
    plt.legend()
    print("plotting w profiles to: wprofs.pdf")
    plt.savefig(outputdir + 'wprofs.pdf', bbox_inches='tight')

    if uvrange is not None:
        uvmin = uvrange[0]
        uvmax = uvrange[1]
    else:
        uvmin = uvminreco
        uvmax = uvmaxreco

    Nw_nonnill = np.sum((wcommon > 0.))
    if uvmin > 0:
        #print("uvradss.shape", uvradss.shape)
        print("uvmin", uvmin)
        print("wcommon.shape", wcommon.shape)
        wcommon[(uvradss < uvmin)] = 0.
        print("chosen uvrange clips out uvrads < ", uvmin)
    if uvmax > 0:
        wcommon[(uvradss > uvmax)] = 0.
        print("chosen uvrange clips out uvrads > ", uvmax)
    Nw_nonnill = np.sum((wcommon > 0.))
    #print("w_nonnill ", Nw_nonnill)
    print("number of common uv cells ", Nw_nonnill)

    print(np.max(wcommon))

    wmask = (wcommon <= min_wA)
    wcommon[wmask] = 0.


    Vamp_Aset_wfilt = Vamp_Aset.copy()
    Vamp_Aset_wfilt[wmask] = 0.
    Vamp_Bset_wfilt = Vamp_Bset.copy()
    Vamp_Bset_wfilt[wmask] = 0.

    V_Aset_wfilt = V_Aset.copy()
    V_Aset_wfilt[wmask] = 0.
    V_AsetR_wfilt = V_AsetR.copy()
    V_AsetR_wfilt[wmask] = 0.
    V_AsetI_wfilt = V_AsetI.copy()
    V_AsetI_wfilt[wmask] = 0.
    w_Aset_wfilt = w_Aset.copy()
    w_Aset_wfilt[wmask] = 0.

    V_Bset_wfilt = V_Bset.copy()
    V_Bset_wfilt[wmask] = 0.
    V_BsetR_wfilt = V_BsetR.copy()
    V_BsetR_wfilt[wmask] = 0.
    V_BsetI_wfilt = V_BsetI.copy()
    V_BsetI_wfilt[wmask] = 0.
    w_Bset_wfilt = w_Bset.copy()
    w_Bset_wfilt[wmask] = 0.

    alpha_R = np.sum(wcommon *
                     (V_AsetR * V_BsetR + V_AsetI * V_BsetI)) / np.sum(
                         wcommon * (V_AsetR**2 + V_AsetI**2))

    alpha_I = np.sum(wcommon *
                     (V_BsetR * V_AsetI - V_AsetR * V_BsetI)) / np.sum(
                         wcommon * (V_AsetR**2 + V_AsetI**2))

    print("alpha_R", alpha_R, " amalytical approx")
    print("alpha_I", alpha_I)
    alpha_mod = np.sqrt(alpha_R**2 + alpha_I**2)
    alpha_phase = (180. / np.pi) * np.arctan2(alpha_I, alpha_R)
    print("alpha_mod ", alpha_mod)
    print("alpha_phase ", alpha_phase)

    print("setting up Minuit")
    Fix_alpha_R = False
    #f = lambda alpha_R, delta_x, delta_y: chi2(
    #    V_Aset_wfilt, V_Bset_wfilt, wcommon, uus, vvs, alpha_R, delta_x,
    #    delta_y)
    f = lambda alpha_R, delta_x, delta_y: chi2DERRS(
        V_Aset_wfilt, V_Bset_wfilt, varA, varB, uus, vvs, alpha_R, delta_x,
        delta_y)
    #m = Minuit(f, alpha_R=alpha_R, delta_x=0., delta_y=0.)
    m = Minuit(f, alpha_R=1., delta_x=0., delta_y=0.)
    m.errordef = Minuit.LEAST_SQUARES
    m.print_level = 1
    m.tol = 1e-4

    m.errors['alpha_R'] = 1E-3
    m.errors['delta_x'] = 1E-4
    m.errors['delta_y'] = 1E-4

    m.limits['delta_x'] = (-0.5, 0.5)
    m.limits['delta_y'] = (-0.5, 0.5)
    if Fix_alpha_R:
        m.fixed['alpha_R'] = True
    else:
        m.limits['alpha_R'] = (0., 10.)

    m.errordef = Minuit.LEAST_SQUARES

    print("start Minuit.migrad")
    m.migrad()
    m.hesse()

    #print("m.params", m.params)
    #print("m.errors", m.errors)

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

    pars = [m.values['alpha_R'], m.values['delta_x'],
            m.values['delta_y']]  # pars for best fit
    err_pars = [m.errors['alpha_R'], m.errors['delta_x'],
                m.errors['delta_y']]  #error in pars

    print("best fit %.2e  %.2e  %.2e " % tuple(pars))
    #print("errors  ", err_pars)

    # bestchi2 = chi2(V_Aset, V_Bset, wcommon, uus, vvs, m.values['alpha_R'],
    # m.values['delta_x'], m.values['delta_y'])

    bestchi2 = chi2DERRS(V_Aset, V_Bset, varA, varB, uus, vvs,
                         m.values['alpha_R'], m.values['delta_x'],
                         m.values['delta_y'])

    #print("bestchi2 ", bestchi2)
    #print("red bestchi2 ", bestchi2 / dofs)
    print("Hessian errors scaled by ", np.sqrt(bestchi2 / dofs),
          "for red chi2 = 1")
    scld_errs = np.array(err_pars) * np.sqrt(bestchi2 / dofs)
    print("errors %.2e  %.2e  %.2e  " % tuple(scld_errs.tolist()))

    file_bestfitparams = outputdir + 'bestfit_xcorr_wshift.npy'
    np.save(file_bestfitparams, pars)

    V_Bset_m = shiftvis(V_Aset, uus, vvs, m.values['alpha_R'],
                        m.values['delta_x'], m.values['delta_y'])

    V_Bset_m_wfilt = V_Bset_m.copy()
    V_Bset_m_wfilt[wmask] = 0.

    punch_vis(V_Bset_m.real, du, outputdir + 'V_BsetmR.fits')
    punch_vis(V_Bset_m.imag, du, outputdir + 'V_BsetmI.fits')
    punch_vis(w_Aset, du, outputdir + 'w_Bsetm.fits')

    punch_vis(V_Bset_m_wfilt.real, du, outputdir + 'V_BsetmR_wfilt.fits')
    punch_vis(V_Bset_m_wfilt.imag, du, outputdir + 'V_BsetmI_wfilt.fits')
    punch_vis(wcommon, du, outputdir + 'w_Bsetm_wfilt.fits')

    punch_vis(wcommon, du, outputdir + 'w.fits')

    punch_vis(V_AsetR_wfilt, du, outputdir + 'V_AsetR_wfilt.fits')
    punch_vis(V_AsetI_wfilt, du, outputdir + 'V_AsetI_wfilt.fits')
    punch_vis(Vamp_Aset_wfilt, du, outputdir + 'Vamp_Aset_wfilt.fits')
    punch_vis(wcommon, du, outputdir + 'w_Aset_wfilt.fits')

    punch_vis(V_BsetR_wfilt, du, outputdir + 'V_BsetR_wfilt.fits')
    punch_vis(V_BsetI_wfilt, du, outputdir + 'V_BsetI_wfilt.fits')
    punch_vis(Vamp_Bset_wfilt, du, outputdir + 'Vamp_Bset_wfilt.fits')
    punch_vis(wcommon, du, outputdir + 'w_Bset_wfilt.fits')


#file_visAset = 'PDS70_AsetB16_cont_chi2_casarestore.ms.selfcal.statwt'
#file_visBset = 'PDS70_cont_copy_verylowS_casarestore.ms.selfcal.statwt'
#dx = 0.004 * u.arcsec  #Bset
#imsize = 2048
#
#xcorr(file_visAset ,
#      file_visBset,
#      dx,
#      imsize,
#      Grid=True,
#      Grid_Bset=True,
#      outputdir='output_xcorr/')
#
#
#
