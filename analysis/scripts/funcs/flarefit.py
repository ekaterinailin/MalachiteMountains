# Everything I need to fit the flare light curves to the model,
# sampling from the posterior distribution using MCMC.

import numpy as np

import copy

# I do not test the prior, log likelihood, or log probability functions.
# I do test the underlying functions like gaussian_prior etc.

def logit(function):
    '''Make a probability distribution (i.e., function)
    a log probability distribution.
    
    '''
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        np.seterr(divide='ignore') # ignore division by zero because you want to have the -np.inf results
        result = np.log(result)
        return result
    return wrapper


@logit
def uninformative_prior(rate, minrate, maxrate):
    '''Uninformative prior for the rates.
    Uniform within [minrate, maxrate].

    Parameters:
    -------------
    rate : float

    minrate, maxrate : float
        interval in which rate is constrained

    Return:
        Prior probability
    '''
    condition = ~(np.isfinite(maxrate) & np.isfinite(minrate))
    if ((maxrate < minrate) | condition):
        raise ValueError("maxrate must be > minrate, and a finite value")
    if ((rate >= minrate) & (rate <= maxrate)):
        return 1. / (maxrate - minrate)
    else:
        return 0



def calculate_posterior_value_that_can_be_passed_to_mcmc(lp):
    '''Do some checks to make sure MCMC will work.'''
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        return -np.inf
    else:
        return lp

# --------------- USING EMPIRICAL PRIOR FOR INCLINATION ------------------------

@logit
def empirical_prior(x, g):
    '''Evaluate an empirical prior

    Parameters:
    ------------
    x : N-array
        latitude between -pi/2 and pi/2
    g : astropy compound model 
        inclination prior
    '''
    if ((x > np.pi/2) | (x < 0)):
        return 0
    else:
        return  g(x)


# ---------------- POST-ANALYSIS -----------------------------------------------



def convert_posterior_units(res, prot, phi, time):
    """Convert radians to degrees, phases to times
    or longitudes.
    
    Required column names: ['latitude_rad', 'phase_0', 
                            'i_rad', 'phase_peak']
    Optional column names: anything with 'fwhm'
    
    Parameters:
    -----------
    res : pandas.DataFrame
        results from MCMC sampling 
    prot : float
        rotation period in days
    phi : 

    """
    r = copy.deepcopy(res)
    
    # map phi0 to phi_peak longitude still call it phi0
    r.phase_0 = (r.phase_peak - r.phase_0 + 2*np.pi) % (2. * np.pi) / np.pi * 180. # 0 would be facing the observer

    #map phi_a_distr to t0_d:
    r.phase_peak = np.interp(r.phase_peak, phi, time)

    # convert theta_f to degrees
    r.latitude_rad = r.latitude_rad / np.pi * 180.

    # convert FWHMs to days
    for fwhmcol  in [i for i in r.columns if "fwhm" in i]:
        r[fwhmcol] = r[fwhmcol] / 2 / np.pi * prot

    # convert i to degrees
    r.i_rad = r.i_rad / np.pi * 180.

    columns = ["latitude_deg", "phase_deg","i_deg","a","t0_d","fwhmi","fwhmg"]

    resultframe = r.rename(index=str, columns=dict(zip(r.columns.values,columns)))

    return resultframe




