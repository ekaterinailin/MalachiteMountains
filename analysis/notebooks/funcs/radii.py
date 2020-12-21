"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This module contains functions used to 
calculte absolute K magnitude and errors
from Gaia distances, and transform them
to stellar radii using Mann et al. (2016)
relations.
"""


import numpy as np

def calculate_distmod(d, derr):
    distmod = 5. * np.log10(d) - 5
    distmoderr = 5. / d / np.log(10) * derr
    return distmod, distmoderr

def calculate_abs_Ks(dmod, dmoderr, Kmag, Kmagerr):
    Ks = Kmag - dmod
    Kserr = np.sqrt(Kmagerr**2 + dmoderr**2)
    return Ks, Kserr

def mann_radius_from_abs_Ks(K, Kerr):
    R = 1.9515 - 0.3520 * K + 0.01680 * (K**2)
    Rerr = (- 0.3520 + 0.01680 * 2. * K) * Kerr 
    Rerr_scatter = 0.0289 * R

    return R, np.sqrt(Rerr**2 + Rerr_scatter**2)