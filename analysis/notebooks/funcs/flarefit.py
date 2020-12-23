# Everything I need to fit the flare ligh curves to the model,
# sampling from the posterior distribution using MCMC.

from scipy import optimize
import numpy as np

# from .model import full_model, full_model_2flares

# I do not test the prior, log likelihood, or log probability functions.
# I do test the underlying functions like gaussian_prior etc.

def logit(function):
    '''Make a probability distribution
    a log probability distribution.'''
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

# --------------- USING EMPIRICAL PRIOR FOR INCLINATION ----------------------------

@logit
def empirical_prior(x, g):
    '''Evaluate an empirical prior

    Parameters:
    ------------
    x : N-array
        latitude between -pi/2 and pi/2
    g : astropy compound model
    '''
    if ((x > np.pi/2) | (x < 0)):
        return 0
    else:
        return  g(x)







