import os
import re
from astropy.io import fits
import scipy
import scipy.signal

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import medfilt2d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

include_path = '/home/simon/common/python/include/'
sys.path.append(include_path)
import ImUtils.Resamp as Resamp
import ImUtils.Cube2Im as Cube2Im
import PyVtools.Vtools as Vtools


def colorbar(Mappable, Orientation='horizontal', cbfmt="%.1e"):
    Ax = Mappable.axes
    fig = Ax.figure
    divider = make_axes_locatable(Ax)
    Cax = divider.append_axes("top", size="5%", pad=0.55)
    #    Cax = divider.append_axes("top", size="7%", pad=0.55)
    return fig.colorbar(mappable=Mappable,
                        cax=Cax,
                        use_gridspec=True,
                        orientation=Orientation,
                        format=cbfmt)


def addimage(iplotpos,
             label,
             atitle,
             filename_grey,
             filename_contours,
             filename_weights=False,
             VisibleXaxis=False,
             VisibleYaxis=True,
             DoGreyCont=False,
             vsyst=0.,
             nplotsx=2,
             nplotsy=2,
             SymmetricRange=False,
             MedianvalRange=False,
             DoCB=True,
             DoMask=False,
             cmap='RdBu_r',
             MedRms=True,
             Zoom=False,
             side=2.5,
             scaleunits=1E3,
             DoInterestingRegion=True,
             cbunits='mJy',
             cbfmt='%.2f',
             uvplane=False,
             DoInset=False,
             Clip=False):

    print("nplotsx ", nplotsx, iplotpos)
    ax = plt.subplot(nplotsy, nplotsx, iplotpos)
    # ax=axes[iplotpos]

    plt.setp(ax.get_xticklabels(), visible=VisibleXaxis)
    plt.setp(ax.get_yticklabels(), visible=VisibleYaxis)

    #ax.tick_params(axis='both',length = 5, width=1.3, color = 'grey',direction='in',left=True, right=True,bottom=True, top=True)
    ax.tick_params(axis='both',
                   length=5,
                   width=1.5,
                   color='black',
                   direction='in',
                   left=True,
                   right=True,
                   bottom=True,
                   top=True)

    #major_ticks = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    ## major_ticks = (-2.0, -1.0, 0.0, 1.0, 2.0)
    #ax.set_yticks(major_ticks)
    #ax.set_xticks(major_ticks)

    #ax.spines['right'].set_color('grey')
    #ax.spines['left'].set_color('grey')
    #ax.spines['top'].set_color('grey')
    #ax.spines['bottom'].set_color('grey')

    if uvplane:
        if ((iplotpos % nplotsx) == 1):
            ax.set_ylabel(r'$v$   / M$\lambda$')
        if (iplotpos > (nplotsx * (nplotsy - 1))):
            ax.set_xlabel(r'$u$   / M$\lambda$')
    else:
        if ((iplotpos % nplotsx) == 1):
            ax.set_ylabel(r'$\Delta \alpha$   / arcsec')
        if (iplotpos > (nplotsx * (nplotsy - 1))):
            ax.set_xlabel(r'$\Delta \delta$   / arcsec')

    print("loading filename_grey", filename_grey)

    #hdu_grey = fits.open(filename_grey)

    hdu_grey = Cube2Im.slice0(filename_grey)

    im_grey = hdu_grey.data * scaleunits
    hdr_grey = hdu_grey.header
    if uvplane:
        cdelt = hdr_grey['CDELT2'] * 1E-6
    else:
        cdelt = hdr_grey['CDELT2'] * 3600.

    side0 = hdr_grey['NAXIS2'] * cdelt

    #side0 = hdr_grey['NAXIS2'] * cdelt

    if filename_weights:
        print("loading filename_weights", filename_weights)
        hdu_w = Cube2Im.slice0(filename_weights)
        im_w = hdu_w.data * scaleunits
        DoMask = True
    else:
        im_w = np.ones(im_grey.shape)
    if Zoom:
        if (side > side0):
            sys.exit("side too large")

        nx = np.rint(side / cdelt)
        ny = np.rint(side / cdelt)

        print("nx ", nx, "ny", ny)

        i_star = np.rint((hdr_grey['CRPIX1'] - 1.))
        j_star = np.rint((hdr_grey['CRPIX2'] - 1.))

        j0 = int(j_star - (ny - 1.) / 2. + 1)
        j1 = int(j_star + (ny - 1.) / 2. + 1)
        i0 = int(i_star - (nx - 1.) / 2. + 1)
        i1 = int(i_star + (nx - 1.) / 2. + 1)
        print("j0:j1, i0:i1", j0, j1, i0, i1)
        subim_grey = im_grey[j0:j1, i0:i1]
        subim_w = im_w[j0:j1, i0:i1]
    else:
        side = side0
        i0 = 0
        i1 = hdr_grey['NAXIS1'] - 1
        j0 = 0
        j1 = hdr_grey['NAXIS2'] - 1

        #subim_grey=im_grey.copy()
        subim_grey = im_grey[:, :]
        subim_w = im_w[j0:j1, i0:i1]

    if DoMask:
        mask = (subim_w < 1000. * scaleunits)
        subim_grey[mask] = 0.

    a0 = side / 2.
    a1 = -side / 2.
    d0 = -side / 2.
    d1 = side / 2.

    # if 'v' in filename_grey:
    #	subim_grey = subim_grey - vsyst

    if filename_weights:
        mask = np.where(subim_w > 0.)
    else:
        mask = np.ones(subim_grey.shape)
        mask = np.where(mask > 0)

    if MedianvalRange:
        typicalvalue = np.median(subim_grey[mask])
        rms = np.std(subim_grey[mask])
        medrms = np.sqrt(np.median((subim_grey[mask] - typicalvalue)**2))

        print("typical value ", typicalvalue, " rms ", rms, "medrms", medrms)
        range1 = np.min(subim_grey[mask])
        if MedRms:
            imagerms = medrms
        else:
            imagerms = rms
        range2 = typicalvalue + 3. * imagerms
        clevs = [range1, range2]
        clabels = ['%.1f' % (clevs[0]), '%.1f' % (clevs[1])]
    else:
        range2 = np.max(subim_grey[mask])
        range1 = np.min(subim_grey[mask])
        clevs = [range1, range2]
        clabels = ['%.1f' % (clevs[0]), '%.1f' % (clevs[1])]

    if ('sigma' in filename_grey):
        cmap = 'magma_r'

    print("max:", np.max(subim_grey))
    print("min:", np.min(subim_grey))
    print("range1", range1, "range2", range2)
    if (np.isnan(subim_grey).any()):
        print("NaNs in subim_grey")
    subim_grey = np.nan_to_num(subim_grey)

    if Clip:
        subim_grey[(subim_grey < Clip)] = np.nan

    theimage = ax.imshow(
        subim_grey,
        origin='lower',
        cmap=cmap,  #norm=norm,
        extent=[a0, a1, d0, d1],
        vmin=range1,
        vmax=range2,
        interpolation='nearest')  #'nearest'  'bicubic'

    #plt.plot(0.,0.,marker='*',color='yellow',markersize=0.2,markeredgecolor='black')

    #ax.text(a0*0.9,d1*0.9,atitle,fontsize=12,ha='left',bbox=dict(facecolor='white', alpha=0.8))
    ax.text(a0 * 0.85,
            d1 * 0.8,
            atitle,
            fontsize=12,
            ha='left',
            bbox=dict(facecolor='white', alpha=0.8))

    #ax.text(a0*0.9,d0*0.9,label,weight='bold',fontsize=12,bbox=dict(facecolor='white', alpha=0.8))
    ax.text(a0 * 0.85,
            d0 * 0.8,
            label,
            weight='bold',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))

    axcb = plt.gca()

    if (DoCB):
        cb = colorbar(theimage, cbfmt=cbfmt)
        cb.ax.tick_params(labelsize='small')
        #cb.ax.tick_params(length = 5, direction='out', width=1., labelsize='small')
        print("CB label", cbunits)
        cb.set_label(cbunits)

    return clevs, clabels


def exec_summary(
        fileout,
        outputdir='./gridding_trials_megacanvas/',
        msname='PDS70_IB17_cont_copy_chi2_casarestore_2inf.ms.selfcal.statwt.4xcorr',
        Zoom=False,
        side=1.9):

    # global nplotsx
    # global nplotsy
    # global basename_log

    #matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='sans-serif')
    #matplotlib.rcParams.update({'font.size': 16})
    font = {'family': 'Arial', 'weight': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    size_marker = 10

    # cmaps = ['magma', 'inferno', 'plasma', 'viridis', 'bone', 'afmhot', 'gist_heat', 'CMRmap', 'gnuplot', 'Blues_r', 'Purples_r', 'ocean', 'hot', 'seismic_r']
    gamma = 1.0

    nplotsx = 3
    nplotsy = 4
    subsize = 3.
    figsize = (nplotsx * subsize, nplotsy * subsize * 1.2)
    #figsize = (nplotsx*4.5, 5.)

    plt.figure(figsize=figsize)

    iplotpos = 0

    cmap = 'ocean_r'
    #cmap = 'Blues'
    #cmap='inferno'
    atitle = r'Pyra $|V|$'
    label = 'a'
    filename_contours = False
    filename_grey = outputdir + 'Vamp_pyragridder.fits'
    filename_weights = outputdir + 'w_pyragridder.fits'
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                filename_weights=filename_weights,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                uvplane=True,
                                cbunits='mJy',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    #cmap='inferno'
    atitle = r'FFTclean $|V|$'
    label = 'b'
    filename_contours = False
    filename_grey = outputdir + 'Vamp_tcleangridder.fits'
    filename_weights = outputdir + 'w_tcleangridder.fits'
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_weights=filename_weights,
                                filename_contours=filename_contours,
                                VisibleXaxis=True,
                                VisibleYaxis=False,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                uvplane=True,
                                cbunits='mJy',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    #cmap='inferno'
    atitle = r'FFTPyra $|V|$'
    label = 'c'
    filename_contours = False
    filename_grey = outputdir + 'Vamp_cross.fits'
    filename_weights = outputdir + 'w_cross.fits'
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_weights=filename_weights,
                                filename_contours=filename_contours,
                                VisibleXaxis=True,
                                VisibleYaxis=False,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                uvplane=True,
                                cbunits='mJy',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    #cmap = 'Blues'
    #cmap='inferno'
    atitle = r'Pyra $w$'
    label = 'd'
    filename_contours = False
    filename_grey = outputdir + 'Vamp_pyragridder.fits'
    filename_weights = outputdir + 'w_pyragridder.fits'
    filename_grey = filename_weights
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                filename_weights=filename_weights,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                uvplane=True,
                                scaleunits=1E-6,
                                cbunits=r'$10^{6}$Jy$^{-2}$',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    #cmap='inferno'
    atitle = r'FFTclean $w$'
    label = 'e'
    filename_contours = False
    filename_grey = outputdir + 'Vamp_tcleangridder.fits'
    filename_weights = outputdir + 'w_tcleangridder.fits'
    filename_grey = filename_weights
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_weights=filename_weights,
                                filename_contours=filename_contours,
                                VisibleXaxis=True,
                                VisibleYaxis=False,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                uvplane=True,
                                scaleunits=1E-6,
                                cbunits=r'$10^{6}$Jy$^{-2}$',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    #cmap='inferno'
    atitle = r'FFTPyra $w$'
    label = 'f'
    filename_contours = False
    filename_grey = outputdir + 'Vamp_cross.fits'
    filename_weights = outputdir + 'w_cross.fits'
    filename_grey = filename_weights
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_weights=filename_weights,
                                filename_contours=filename_contours,
                                VisibleXaxis=True,
                                VisibleYaxis=False,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=Zoom,
                                side=side,
                                uvplane=True,
                                scaleunits=1E-6,
                                cbunits=r'$10^{6}$Jy$^{-2}$',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    cmap = 'RdBu_r'
    #cmap = 'Blues'
    #cmap='inferno'
    atitle = r'Pyra $I_D$'  # rms 0.032'
    label = 'g'
    filename_contours = False
    filename_grey = outputdir + 'dirty_' + msname + '_Pyra.fits'
    filename_weights = False
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                filename_weights=filename_weights,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=False,
                                side=side,
                                uvplane=False,
                                scaleunits=1E3,
                                cbunits=r'mJy beam$^{-1}$',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    cmap = 'RdBu_r'
    #cmap = 'Blues'
    #cmap='inferno'
    atitle = r'tclean $I_D$'  # rms 0.027'
    label = 'h'
    filename_contours = False
    filename_grey = outputdir + 'dirty_tclean_' + msname + '.image.fits'
    #Rms: 2.716e-05
    filename_weights = False
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                filename_weights=filename_weights,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=False,
                                side=side,
                                uvplane=False,
                                scaleunits=1E3,
                                cbunits=r'mJy beam$^{-1}$',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    #cmap = 'Blues'
    #cmap='inferno'
    atitle = r'tclean apod $I_D$'
    label = 'i'
    filename_contours = False
    filename_grey = outputdir + 'dirty_tclean_' + msname + '.image.apod.fits'
    filename_weights = False
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                filename_weights=filename_weights,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=False,
                                side=side,
                                uvplane=False,
                                scaleunits=1E3,
                                cbunits=r'mJy beam$^{-1}$',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    #cmap = 'Blues'
    #cmap='inferno'
    atitle = r'Pyra $B_D$'
    label = 'j'
    filename_contours = False
    filename_grey = outputdir + 'psf_' + msname + '_Pyra.fits'
    filename_weights = False
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                filename_weights=filename_weights,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=False,
                                side=side,
                                uvplane=False,
                                scaleunits=1E3,
                                cbunits=r'mJy beam$^{-1}$',
                                cbfmt='%.1f')

    cmap = 'ocean_r'
    #cmap = 'Blues'
    #cmap='inferno'
    atitle = r'tclean $B_D$'
    label = 'k'
    filename_contours = False
    filename_grey = outputdir + 'dirty_tclean_' + msname + '.psf.fits'
    filename_weights = False
    iplotpos += 1
    (clevs, clabels) = addimage(iplotpos,
                                label,
                                atitle,
                                filename_grey,
                                filename_contours=filename_contours,
                                filename_weights=filename_weights,
                                VisibleXaxis=True,
                                VisibleYaxis=True,
                                DoGreyCont=False,
                                nplotsx=nplotsx,
                                nplotsy=nplotsy,
                                SymmetricRange=False,
                                DoCB=True,
                                cmap=cmap,
                                Zoom=False,
                                side=side,
                                uvplane=False,
                                scaleunits=1E3,
                                cbunits=r'mJy beam$^{-1}$',
                                cbfmt='%.1f')

    #plt.subplots_adjust(hspace=0.)
    #plt.subplots_adjust(wspace=0.)

    print(fileout)
    #plt.tight_layout()

    plt.savefig(fileout, bbox_inches='tight', dpi=500)

    #plt.savefig(fileout)

    return


msname = 'PDS70_IB17_cont_copy_chi2_casarestore_2inf.ms.selfcal.statwt'
outputdir = './gridding_trials_hermit_5/'

#outputdir = './gridding_trials_residuals_nohermit/'
#outputdir = './gridding_trials_residuals/'
#msname = '_ph0_residuals.data.ms'


fileout = outputdir + 'fig_powerimages_trials.pdf'
exec_summary(fileout,
             Zoom=True,
             outputdir=outputdir,
             side=1.6,
             msname=msname)
