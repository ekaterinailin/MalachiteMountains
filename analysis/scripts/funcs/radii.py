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
    """Get distance modulus and error
    from Bailer-Jones distance in pc and
    error.
    
    Parameters:
    -----------
    d, derr : float, float
        distance to star in parsec and error
        
    Return:
    -------
    (float, float) - distance modulus in mag and error
    
    """
    distmod = 5. * np.log10(d) - 5
    distmoderr = 5. / d / np.log(10) * derr
    return distmod, distmoderr


def mann_radius_from_abs_Ks(K, Kerr):
    """Mann et al. (2016) relation for Ks to radius.
    
    Parameters:
    -----------
    K, Kerr : float, float
        2MASS Ks magnitude and error
        
    Return:
    -------
    (float, float) - radius in solar radii and error
    """
    R = 1.9515 - 0.3520 * K + 0.01680 * (K**2)
    Rerr = (- 0.3520 + 0.01680 * 2. * K) * Kerr 
    Rerr_scatter = 0.0289 * R

    return R, np.sqrt(Rerr**2 + Rerr_scatter**2)



def calculate_abs_Ks(dmod, dmoderr, Kmag, Kmagerr):
    """Apply distance modulus to measured 2MASS
    Ks magnitude and propagate errors.
    
    
    Parameters:
    -----------
    dmod, dmoderr - float, float
        distance modulus and error
    Kmag, Kmagerr - float, float
        Ks magnitude and error
        
    Return:
    --------
    (float, float) - absolute K magnitude and error
    """
    Ks = Kmag - dmod
    Kserr = np.sqrt(Kmagerr**2 + dmoderr**2)
    return Ks, Kserr

