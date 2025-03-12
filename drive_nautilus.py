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
from copy import copy
#from multiprocessing import Pool

from multiprocessing import get_context
import time

import corner
from nautilus import Sampler

from pprint import pprint

from Likelihood import shiftvis, chi2, chi2DERRS

ctx = get_context('spawn')


def lnlike_naut(parnames, domain_dic, V_Aset_wfilt, V_Bset_wfilt, varA, varB,
                uus, vvs, x_free):
    return lnlike(x_free, parnames, domain_dic, V_Aset_wfilt, V_Bset_wfilt,
                  varA, varB, uus, vvs)


def lnlike(x_free, parnames, domain_dic, V_Aset_wfilt, V_Bset_wfilt, varA,
           varB, uus, vvs):

    domain_local = copy(domain_dic)
    for ipar, apar in enumerate(parnames):
        domain_local[apar]['default'] = x_free[ipar]

    alpha_R = domain_local['alpha_R']['default']
    delta_x = domain_local['delta_x']['default']
    delta_y = domain_local['delta_y']['default']

    achi2 = chi2DERRS(V_Aset_wfilt, V_Bset_wfilt, varA, varB, uus, vvs,
                      alpha_R, delta_x, delta_y)

    return -0.5 * achi2


def prior_transform(domain, unif_free):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[-10., 10.)`."""
    x_free = np.zeros_like(unif_free)
    bnds = list(map((lambda x: x[2]), domain))
    for iparam in list(range(len(unif_free))):
        x_free[iparam] = bnds[iparam][0] + (
            bnds[iparam][1] - bnds[iparam][0]) * unif_free[iparam]

    return x_free


def exec_naut(V_Aset_wfilt,
              V_Bset_wfilt,
              varA,
              varB,
              uus,
              vvs,
              Fix_delta_x=False,
              Fix_delta_y=False,
              Fix_alpha_R=False,
              Report=True,
              nthreads=60,
              domain=[],
              outputdir='./',
              file_bestfit="bestfitparams.txt"):

    domain_work = []
    for adomain in domain:
        if (adomain[0] == 'delta_x') and Fix_delta_x:
            continue
        if (adomain[0] == 'delta_y') and Fix_delta_y:
            continue
        if (adomain[0] == 'alpha_R') and Fix_alpha_R:
            continue
        domain_work.append(adomain)

    parnames = list(map((lambda x: x[0]), domain_work))
    sample_params = list(map((lambda x: x[1]), domain_work))
    bnds = list(map((lambda x: x[2]), domain_work))
    print("bnds", bnds)
    nvar = len(list(parnames))
    x_free = np.array(sample_params)
    x_free_init = x_free.copy()

    start_time = time.time()
    t_i = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print("check initial chi2 value")

    domain_dic = {}
    for adomain in domain:
        domain_dic[adomain[0]] = {'default': adomain[1], 'limits': adomain[2]}

    init_lnlike = lnlike(x_free_init, parnames, domain_dic, V_Aset_wfilt,
                         V_Bset_wfilt, varA, varB, uus, vvs)
    print("chi2 = %e " % (-2 * init_lnlike))
    t_f = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    Delta_t = (time.time() - start_time)
    print("Execution done in (elapsed time):" + str(Delta_t) + " s")

    nlive = 10 * nvar
    ndim = nvar

    RunSampler = True
    if RunSampler:
        lnlikeargs = [
            parnames, domain_dic, V_Aset_wfilt, V_Bset_wfilt, varA, varB, uus,
            vvs
        ]
        lnlikekwargs = {}
        ptform_args = [domain_work]

        if Report:
            print("Setting up Nautilus  sampler with n_live = ", nlive,
                  " and nthreads", nthreads, " <<")

        if nthreads > 1:
            ##ctx = get_context('spawn')
            #with ctx.Pool(processes=nthreads) as pool:
            ##with Pool(nthreads) as pool:
            #    dsampler = Sampler(prior_transform,
            #                       lnlike_naut,
            #                       pass_dict=False,
            #                       n_dim=ndim,
            #                       n_live=nlive,
            #                       prior_args=ptform_args,
            #                       n_networks=16,
            #                       likelihood_args=lnlikeargs,
            #                       pool=pool)
            print("nthreads",nthreads)
            dsampler = Sampler(prior_transform,
                               lnlike_naut,
                               pass_dict=False,
                               n_dim=ndim,
                               n_live=nlive,
                               prior_args=ptform_args,
                               n_networks=16,
                               likelihood_args=lnlikeargs,
                               pool=nthreads)

        else:
            dsampler = Sampler(
                prior_transform,
                lnlike_naut,
                pass_dict=False,
                n_dim=ndim,
                n_live=nlive,
                prior_args=ptform_args,
                n_networks=12,
                likelihood_args=lnlikeargs,
            )

        dsampler.run(verbose=True)

        points, log_w, log_l = dsampler.posterior()
        np.save(outputdir + 'dresult_points', points)
        np.save(outputdir + 'dresult_log_w', log_w)
        np.save(outputdir + 'dresult_log_l', log_l)
        points_equalw, log_w_equalw, log_l_equalw = dsampler.posterior(
            equal_weight=True)
        np.save(outputdir + 'dresult_points_equalw', points_equalw)
        np.save(outputdir + 'dresult_log_w_equalw', log_w_equalw)
        np.save(outputdir + 'dresult_log_l_equalw', log_l_equalw)

        t_f = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        Delta_t = (time.time() - start_time) / 3600.
        print("Nautilus execution done in (elapsed time):" + str(Delta_t) +
              " h")
        print("t_i = " + str(t_i))
        print("t_f = " + str(t_f))
    else:
        points = np.load(outputdir + 'dresult_points.npy', allow_pickle=True)
        log_w = np.load(outputdir + 'dresult_log_w.npy', allow_pickle=True)
        log_l = np.load(outputdir + 'dresult_log_l.npy', allow_pickle=True)
        points_equalw = np.load(outputdir + 'dresult_points_equalw.npy',
                                allow_pickle=True)
        log_w_equalw = np.load(outputdir + 'dresult_log_w_equalw.npy',
                               allow_pickle=True)
        log_l_equalw = np.load(outputdir + 'dresult_log_l_equalw.npy',
                               allow_pickle=True)

    fig = corner.corner(points,
                        weights=np.exp(log_w),
                        bins=20,
                        labels=parnames,
                        color='purple',
                        plot_datapoints=False,
                        range=np.repeat(0.999, len(parnames)))
    fig.savefig(outputdir + "triangle.png")

    ibestparams = np.argmax(log_l_equalw)
    bestparams = points_equalw[ibestparams, :]

    sampler_results = list(
        map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
            zip(*np.percentile(points_equalw, [16, 50, 84], axis=0))))

    #weights = np.exp(log_w - log_z[-1])
    posterior_samples = points_equalw

    if Report:
        print("sampler results", sampler_results)

    sampler_results_0 = np.zeros(nvar)
    sampler_results_uperrs = np.zeros(nvar)
    sampler_results_doerrs = np.zeros(nvar)
    for iparam in list(range(nvar)):
        sampler_results_0[iparam] = sampler_results[iparam][0]
        sampler_results_uperrs[iparam] = sampler_results[iparam][1]
        sampler_results_doerrs[iparam] = sampler_results[iparam][2]
    if Report:
        with open(outputdir + file_bestfit, "w") as f:
            print("%20s %12s %12s %12s %12s\n" %
                  ('#name', 'median', 'uperror', 'loerror', 'maxlnlike'))
            f.write("%20s %12s %12s %12s %12s\n" %
                    ('#name', 'median', 'uperror', 'loerror', 'maxlnlike'))
            for iparam in list(range(nvar)):
                print(parnames[iparam], sampler_results[iparam],
                      bestparams[iparam])
                print("%20s %12.4e %12.4e %12.4e %12.4e" %
                      (parnames[iparam], sampler_results[iparam][0],
                       sampler_results[iparam][1], sampler_results[iparam][2],
                       bestparams[iparam]),
                      file=f)

                #cornerquantiles=corner.quantile(points[:,iparam], [0.25, 0.50, 0.975], weights=np.exp(log_w))
                #uperr = cornerquantiles[2]-cornerquantiles[1]
                #loerr = cornerquantiles[1]-cornerquantiles[0]
                #print("corner ",cornerquantiles[0],uperr, loerr)
                #print("equalw ",sampler_results_equalw[iparam])
        f.close()

    if Report:
        print("sampler_results_uperrs", sampler_results_uperrs)

    sampler_bestparams = bestparams

    if Report:
        print("cross check initial chi2 value")
        init_lnlike = lnlike(x_free_init, parnames, domain_dic, V_Aset_wfilt,
                             V_Bset_wfilt, varA, varB, uus, vvs)
        print("chi2 = %e " % (-2 * init_lnlike))

        print("running final lnlike on median values")
        final_lnlike = lnlike(sampler_results_0, parnames, domain_dic,
                              V_Aset_wfilt, V_Bset_wfilt, varA, varB, uus, vvs)
        print("chi2 = %e " % (-2 * final_lnlike))

        print("running final lnlike on best L  values")
        final_lnlike = lnlike(bestparams, parnames, domain_dic, V_Aset_wfilt,
                              V_Bset_wfilt, varA, varB, uus, vvs)

    domain_local = copy(domain_dic)
    for ipar, apar in enumerate(parnames):
        domain_local[apar]['default'] = bestparams[ipar]

    alpha_R = domain_local['alpha_R']['default']
    delta_x = domain_local['delta_x']['default']
    delta_y = domain_local['delta_y']['default']

    bestchi2 = chi2DERRS(V_Aset_wfilt, V_Bset_wfilt, varA, varB, uus, vvs,
                         alpha_R, delta_x, delta_y)

    print("bestchi2 ", bestchi2)
    save_bestfirparams = np.array([alpha_R, delta_x, delta_y])

    file_bestfitparams = outputdir + 'bestfit_xcorr_wshift.npy'
    print("saving best fit in ", file_bestfitparams)
    np.save(file_bestfitparams, save_bestfirparams)

    return alpha_R, delta_x, delta_y
