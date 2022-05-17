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

HOME = os.environ.get('HOME')
include_path = HOME + '/gitcommon/VisAlign/'
sys.path.append(include_path)


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


def shiftvis(V_S, uus, vvs, alpha_R, delta_x, delta_y):
    argphase = 2. * np.pi * (uus * (delta_x * np.pi / (180. * 3600.)) + vvs *
                             (delta_y * np.pi / (180. * 3600.)))
    # eulerphase = np.cos(argphase)+1j*np.sin(argphase)
    eulerphase = np.exp(1j * argphase)

    V_L_m = alpha_R * V_S * eulerphase
    return V_L_m


def chi2(V_S, V_L, w, uus, vvs, alpha_R, delta_x, delta_y):
    V_L_m = shiftvis(V_S, uus, vvs, alpha_R, delta_x, delta_y)
    diff = V_L - V_L_m
    squarediff = (diff.real**2) + (diff.imag**2)
    retval = np.sum(w * squarediff)
    if np.isnan(retval):
        print("chi2 is NaN")
        retval = np.inf
    return retval


def xcorr(
        file_visSBs,
        file_visLBs,
        dx,
        imsize,
        GridScheme='Pyra',  # 'tclean'
        Grid=True,
        Grid_LBs=True,
        uvrange=False,
        DefaultUvrange=False,
        DoMinos=False,
        kernel_w_L=5,
        kernel_w_S=5,
        wprof_factor=10.,
        min_wS=100.,
        min_wL=100.,
        outputdir='output_xcorr/'):
    nx = imsize
    ny = imsize

    os.system('mkdir ' + outputdir)

    file_gridded_vis_SBs = outputdir + 'SBs_aligned_gridded_visibilities_nat.npy'
    file_gridded_weights_SBs = outputdir + 'SBs_aligned_gridded_weights_nat.npy'
    file_gridded_vis_LBs = outputdir + 'LBs_gridded_visibilities_nat.npy'
    file_gridded_weights_LBs = outputdir + 'LBs_gridded_weights_nat.npy'

    if Grid:
        if (GridScheme == 'Pyra'):
            file_dirty = 'dirty_' + os.path.basename(file_visSBs) + '.fits'
            import Pyra_grid
            from pyralysis.units import lambdas_equivalencies
            dx, SBs_gridded_visibilities_nat, SBs_gridded_weights_nat = Pyra_grid.gridvis(
                file_visSBs, imsize=imsize, wantdirtymap=file_dirty, dx=dx)
            du = (1 / (imsize * dx)).to(u.lambdas,
                                        equivalencies=lambdas_equivalencies())
            duvalue = du.value
        elif (GridScheme == 'tclean'):
            import Tclean_grid
            file_dirty = 'dirty_' + os.path.basename(file_visSBs)
            dx, SBs_gridded_visibilities_nat, SBs_gridded_weights_nat = Tclean_grid.gridvis(
                file_visSBs,
                imsize=imsize,
                tcleanimagename=file_dirty,
                dx=dx,
                outputdir=outputdir)

        np.save(file_gridded_vis_SBs, SBs_gridded_visibilities_nat)
        np.save(file_gridded_weights_SBs, SBs_gridded_weights_nat)
        print("sky image pixels: ", dx.to(u.arcsec))

    if Grid_LBs:
        if (GridScheme == 'Pyra'):
            file_dirty = 'dirty_' + os.path.basename(file_visLBs) + '.fits'
            import Pyra_grid
            from pyralysis.units import lambdas_equivalencies
            dx, LBs_gridded_visibilities_nat, LBs_gridded_weights_nat = Pyra_grid.gridvis(
                file_visLBs, imsize=imsize, wantdirtymap=file_dirty, dx=dx)
            du = (1 / (imsize * dx)).to(u.lambdas,
                                        equivalencies=lambdas_equivalencies())
            duvalue = du.value
        elif (GridScheme == 'tclean'):
            import Tclean_grid
            file_dirty = 'dirty_' + os.path.basename(file_visSBs)
            dx, LBs_gridded_visibilities_nat, LBs_gridded_weights_nat = Tclean_grid.gridvis(
                file_visLBs,
                imsize=imsize,
                tcleanimagename=file_dirty,
                dx=dx,
                outputdir=outputdir)

        np.save(file_gridded_vis_LBs, LBs_gridded_visibilities_nat)
        np.save(file_gridded_weights_LBs, LBs_gridded_weights_nat)

    du = (1 / (imsize * dx.to(u.rad).value))
    duvalue = du
    
    SBs_gridded_visibilities_nat = np.load(file_gridded_vis_SBs)
    SBs_gridded_weights_nat = np.load(file_gridded_weights_SBs)
    LBs_gridded_visibilities_nat = np.load(file_gridded_vis_LBs)
    LBs_gridded_weights_nat = np.load(file_gridded_weights_LBs)

    print("SBs_gridded_visibilities_nat.shape",
          SBs_gridded_visibilities_nat.shape)
    print(SBs_gridded_visibilities_nat.dtype)
    print(SBs_gridded_weights_nat.shape)

    if GridScheme == 'Pyra':
        V_S = SBs_gridded_visibilities_nat[0, :, :]
        V_SR = SBs_gridded_visibilities_nat[0, :, :].real
        V_SI = SBs_gridded_visibilities_nat[0, :, :].imag
        V_L = LBs_gridded_visibilities_nat[0, :, :]
        V_LR = LBs_gridded_visibilities_nat[0, :, :].real
        V_LI = LBs_gridded_visibilities_nat[0, :, :].imag
        w_S = SBs_gridded_weights_nat[0, :, :]
        w_L = LBs_gridded_weights_nat[0, :, :]
    else:
        V_S = SBs_gridded_visibilities_nat
        V_SR = SBs_gridded_visibilities_nat.real
        V_SI = SBs_gridded_visibilities_nat.imag
        V_L = LBs_gridded_visibilities_nat
        V_LR = LBs_gridded_visibilities_nat.real
        V_LI = LBs_gridded_visibilities_nat.imag
        w_S = SBs_gridded_weights_nat
        w_L = LBs_gridded_weights_nat
        
    from scipy.signal import medfilt2d

    print('filtering V_L')
    # wmedian = np.median(w_L[(w_L > 0.)])
    wmedian = medfilt2d(w_L, kernel_size=kernel_w_L)
    #print("wmedian:", wmedian)
    mask = ((w_L < wmedian / 2.) | (w_L < min_wL))
    V_L[mask] = 0
    V_LR[mask] = 0.
    V_LI[mask] = 0.
    w_L[mask] = 0.

    print('filtering V_S')
    #wmedian = np.median(w_S[(w_S > 0.)])
    wmedian = medfilt2d(w_S, kernel_size=kernel_w_S)
    #print("wmedian:", wmedian)
    mask = ((w_S < wmedian / 2.) | (w_S < min_wS))
    V_S[mask] = 0
    V_SR[mask] = 0.
    V_SI[mask] = 0.
    w_S[mask] = 0.

    w = w_L * w_S / (w_L + w_S)
    w[(w_L < min_wL) | (w_S < min_wS)] = 0.

    w = np.nan_to_num(w)
    V_S = np.nan_to_num(V_S)
    V_L = np.nan_to_num(V_L)
    dofs = np.sum((w > 0.))
    print("dofs = ", dofs)

    Vamp_S = np.sqrt(V_SI**2 + V_SR**2)
    Vamp_L = np.sqrt(V_LI**2 + V_LR**2)

    print("uv cell size", duvalue)
    us = -1 * (np.arange(0, nx) - (nx - 1.) / 2.) * duvalue
    vs = (np.arange(0, ny) - (ny - 1.) / 2.) * duvalue
    uus, vvs = np.meshgrid(us, vs)
    uvradss = np.sqrt(uus**2 + vvs**2)
    print("max uvrange: ", np.max(uvradss))

    import matplotlib
    import matplotlib.pyplot as plt
    w_L_polar = polarexpand(w_L)
    w_L_prof = np.median(w_L_polar, axis=1)
    nphis, nrs = w_L_polar.shape
    uvrads = (np.arange(nrs)) * duvalue
    plt.plot(uvrads, w_L_prof, label='w_L', color='C1')
    maskprof = ((w_L_prof > np.max(w_L_prof) / wprof_factor))
    #iw1=np.argmin(uvrads[maskprof])
    #iw2=np.argmax(uvrads[maskprof])
    uvmin_L = np.min(uvrads[maskprof])
    uvmax_L = np.max(uvrads[maskprof])

    w_S_polar = polarexpand(w_S)
    w_S_prof = np.median(w_S_polar, axis=1)
    plt.plot(uvrads, w_S_prof, label='w_S', color='C0')
    maskprof = ((w_S_prof > np.max(w_S_prof) / wprof_factor))
    #iw1=np.argmin(uvrads[maskprof])
    #iw2=np.argmax(uvrads[maskprof])
    uvmin_S = np.min(uvrads[maskprof])
    uvmax_S = np.max(uvrads[maskprof])

    uvminreco = max((uvmin_S, uvmin_L))
    uvmaxreco = min((uvmax_S, uvmax_L))
    print("recommended uvrange: ", uvminreco, uvmaxreco)
    plt.legend()
    print("plotting w profiles to: wprofs.pdf")
    plt.savefig(outputdir + 'wprofs_full.pdf', bbox_inches='tight')
    plt.xlim(uvminreco, uvmaxreco)
    plt.legend()
    print("plotting w profiles to: wprofs.pdf")
    plt.savefig(outputdir + 'wprofs.pdf', bbox_inches='tight')

    if uvrange:
        uvmin = uvrange[0]
        uvmax = uvrange[1]
    elif DefaultUvrange:
        uvmin = uvminreco
        uvmax = uvmaxreco

    w_nonnill = np.sum((w > 0.))
    print("w_nonnill ", w_nonnill)
    if uvmin > 0:
        print("uvradss.shape", uvradss.shape)
        print("uvmin", uvmin)
        print("w.shape", w.shape)
        w[(uvradss < uvmin)] = 0.
        print("chosen uvrange clips out uvrads < ", uvmin)
    if uvmax > 0:
        w[(uvradss > uvmax)] = 0.
        print("chosen uvrange clips out uvrads > ", uvmax)
    w_nonnill = np.sum((w > 0.))
    print("w_nonnill ", w_nonnill)

    wmask = (w <= min_wS)
    w[wmask] = 0.

    Vamp_S_wfilt = Vamp_S.copy()
    Vamp_S_wfilt[wmask] = 0.
    Vamp_L_wfilt = Vamp_L.copy()
    Vamp_L_wfilt[wmask] = 0.

    V_S_wfilt = V_S.copy()
    V_S_wfilt[wmask] = 0.
    V_SR_wfilt = V_SR.copy()
    V_SR_wfilt[wmask] = 0.
    V_SI_wfilt = V_SI.copy()
    V_SI_wfilt[wmask] = 0.
    w_S_wfilt = w_S.copy()
    w_S_wfilt[wmask] = 0.

    V_L_wfilt = V_L.copy()
    V_L_wfilt[wmask] = 0.
    V_LR_wfilt = V_LR.copy()
    V_LR_wfilt[wmask] = 0.
    V_LI_wfilt = V_LI.copy()
    V_LI_wfilt[wmask] = 0.
    w_L_wfilt = w_L.copy()
    w_L_wfilt[wmask] = 0.

    alpha_R = np.sum(w *
                     (V_SR * V_LR + V_SI * V_LI)) / np.sum(w *
                                                           (V_SR**2 + V_SI**2))

    alpha_I = np.sum(w *
                     (V_LR * V_SI - V_SR * V_LI)) / np.sum(w *
                                                           (V_SR**2 + V_SI**2))

    print("alpha_R", alpha_R, "use this to scale flux calibrations")
    print("alpha_I", alpha_I)
    alpha_mod = np.sqrt(alpha_R**2 + alpha_I**2)
    alpha_phase = (180. / np.pi) * np.arctan2(alpha_I, alpha_R)
    print("alpha_mod ", alpha_mod)
    print("alpha_phase ", alpha_phase)

    print("setting up Minuit")
    Fix_alpha_R = False
    f = lambda alpha_R, delta_x, delta_y: chi2(V_S_wfilt, V_L_wfilt, w, uus,
                                               vvs, alpha_R, delta_x, delta_y)
    m = Minuit(f, alpha_R=alpha_R, delta_x=0., delta_y=0.)
    # m = Minuit(f, alpha_R=1., delta_x=0., delta_y=0.)

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

    print("m.params", m.params)
    print("m.errors", m.errors)

    if DoMinos:
        print("start Minuit.minos")
        m.minos()

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

    print("best fit ", pars)
    print("errors  ", err_pars)

    bestchi2 = chi2(V_S, V_L, w, uus, vvs, m.values['alpha_R'],
                    m.values['delta_x'], m.values['delta_y'])
    print("bestchi2 ", bestchi2)
    print("red bestchi2 ", bestchi2 / dofs)
    print("Hessian errors scaled for red chi2 = 1")
    print("errors  ", np.array(err_pars) * np.sqrt(bestchi2 / dofs))

    file_bestfitparams = outputdir + 'bestfit_xcorr_wshift.npy'
    np.save(file_bestfitparams, pars)

    V_L_m = shiftvis(V_S, uus, vvs, m.values['alpha_R'], m.values['delta_x'],
                     m.values['delta_y'])

    V_L_m_wfilt = V_L_m.copy()
    V_L_m_wfilt[wmask] = 0.

    punch_vis(V_L_m.real, du, outputdir + 'V_LmR.fits')
    punch_vis(V_L_m.imag, du, outputdir + 'V_LmI.fits')
    punch_vis(w_S, du, outputdir + 'w_Lm.fits')

    punch_vis(V_L_m_wfilt.real, du, outputdir + 'V_LmR_wfilt.fits')
    punch_vis(V_L_m_wfilt.imag, du, outputdir + 'V_LmI_wfilt.fits')
    punch_vis(w, du, outputdir + 'w_Lm_wfilt.fits')

    punch_vis(w, du, outputdir + 'w.fits')

    punch_vis(V_SR, du, outputdir + 'V_SR.fits')
    punch_vis(V_SI, du, outputdir + 'V_SI.fits')
    punch_vis(Vamp_S, du, outputdir + 'Vamp_S.fits')
    punch_vis(w_S, du, outputdir + 'w_S.fits')

    punch_vis(V_SR_wfilt, du, outputdir + 'V_SR_wfilt.fits')
    punch_vis(V_SI_wfilt, du, outputdir + 'V_SI_wfilt.fits')
    punch_vis(Vamp_S_wfilt, du, outputdir + 'Vamp_S_wfilt.fits')
    punch_vis(w, du, outputdir + 'w_S_wfilt.fits')

    punch_vis(V_LR, du, outputdir + 'V_LR.fits')
    punch_vis(V_LI, du, outputdir + 'V_LI.fits')
    punch_vis(Vamp_L, du, outputdir + 'Vamp_L.fits')
    punch_vis(w_L, du, outputdir + 'w_L.fits')

    punch_vis(V_LR_wfilt, du, outputdir + 'V_LR_wfilt.fits')
    punch_vis(V_LI_wfilt, du, outputdir + 'V_LI_wfilt.fits')
    punch_vis(Vamp_L_wfilt, du, outputdir + 'Vamp_L_wfilt.fits')
    punch_vis(w, du, outputdir + 'w_L_wfilt.fits')


#file_visSBs = 'PDS70_SB16_cont_chi2_casarestore.ms.selfcal.statwt'
#file_visLBs = 'PDS70_cont_copy_verylowS_casarestore.ms.selfcal.statwt'
#dx = 0.004 * u.arcsec  #LBs
#imsize = 2048
#
#xcorr(file_visSBs,
#      file_visLBs,
#      dx,
#      imsize,
#      Grid=True,
#      Grid_LBs=True,
#      outputdir='output_xcorr/')
#
#
#
