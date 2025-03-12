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

from Likelihood import shiftvis, chi2, chi2DERRS


def exec_minuit(V_Aset_wfilt,
                V_Bset_wfilt,
                varA,
                varB,
                uus,
                vvs,
                dofs,
                domain=[],
                outputdir='./',
                DoMinos=False,
                FixKeys=[],
                Fix_delta_x=False,
                Fix_delta_y=False,
                Fix_alpha_R=False,
                file_bestfit="bestfitparams.txt"):
    """
    maxoffset=10 in arcsec, maximum value for delta_x and delta_y
    """

    print("setting up Minuit")

    domain_dic = {}
    for adomain in domain:
        domain_dic[adomain[0]] = {'default': adomain[1], 'limits': adomain[2]}

    f = lambda alpha_R, delta_x, delta_y: chi2DERRS(
        V_Aset_wfilt, V_Bset_wfilt, varA, varB, uus, vvs, alpha_R, delta_x,
        delta_y)
    #m = Minuit(f, alpha_R=alpha_R, delta_x=0., delta_y=0.)
    m = Minuit(f,
               alpha_R=domain_dic['alpha_R']['default'],
               delta_x=domain_dic['delta_x']['default'],
               delta_y=domain_dic['delta_y']['default'])
    m.errordef = Minuit.LEAST_SQUARES
    m.print_level = 1
    m.tol = 1e-4

    m.errors['alpha_R'] = 1E-2
    m.errors['delta_x'] = 1E-2
    m.errors['delta_y'] = 1E-2

    m.limits['alpha_R'] = domain_dic['alpha_R']['limits']
    m.limits['delta_x'] = domain_dic['delta_x']['limits']
    m.limits['delta_y'] = domain_dic['delta_y']['limits']

    if Fix_delta_x:
        FixKeys.append('delta_x')
    if Fix_delta_y:
        FixKeys.append('delta_y')
    if Fix_alpha_R:
        FixKeys.append('alpha_R')

    for akey in FixKeys:
        m.fixed[akey] = True

    m.errordef = Minuit.LEAST_SQUARES

    print("start Minuit.migrad")
    m.migrad()
    m.hesse()

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

    bestchi2 = chi2DERRS(V_Aset_wfilt, V_Bset_wfilt, varA, varB, uus, vvs,
                         m.values['alpha_R'], m.values['delta_x'],
                         m.values['delta_y'])

    #print("bestchi2 ", bestchi2)
    #print("red bestchi2 ", bestchi2 / dofs)
    print("Hessian errors scaled by ", np.sqrt(bestchi2 / dofs),
          "for reduced chi2 = 1")
    scld_errs = np.array(err_pars) * np.sqrt(bestchi2 / dofs)
    print("errors %.2e  %.2e  %.2e  " % tuple(scld_errs.tolist()))
    
    file_bestfitparams = outputdir + 'bestfit_xcorr_wshift.npy'
    print("saving best fit in ",file_bestfitparams)
    np.save(file_bestfitparams, pars)
    
    with open(outputdir + file_bestfit, "w") as f:
        print("%20s %12s %12s \n" % ('#name', 'best', 'error'))
        f.write("%20s %12s %12s \n" % ('#name', 'best', 'error'))
        #nvar=len(params)
        for iparam, aparam in enumerate(params):
            aparam = params[iparam]
            print("%20s %.20e %.20e" %
                  (aparam.name, aparam.value,
                   aparam.error * np.sqrt(bestchi2 / dofs)))
            print("%20s %.20e %.20e " %
                  (aparam.name, aparam.value,
                   aparam.error * np.sqrt(bestchi2 / dofs)),
                  file=f)


    return m.values['alpha_R'], m.values['delta_x'], m.values['delta_y']
