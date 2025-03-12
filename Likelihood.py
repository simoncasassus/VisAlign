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
