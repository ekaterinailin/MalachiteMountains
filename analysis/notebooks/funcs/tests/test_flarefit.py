import pytest

import numpy as np

from ..flarefit import (uninformative_prior,
                        gaussian_prior,
                        calculate_posterior_value_that_can_be_passed_to_mcmc)



# ------------------------ TESTING calculate_posterior_value_that_can_be_passed_to_mcmc(lp) ------------------

def test_calculate_posterior_value_that_can_be_passed_to_mcmc():
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(np.inf) == -np.inf
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(np.nan) == -np.inf
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(3) == 3

# -------------------------------- TESTING gaussian_prior(rate, minrate, maxrate) ----------------------------

def test_gaussian_prior():
    # prior is 0 numerically
    assert np.isfinite(gaussian_prior(1.,.2,.01)) == False
    # invalid x
    assert np.isfinite(gaussian_prior(np.nan,.2,.01)) == False
    # x is out of range
    assert np.isfinite(gaussian_prior(4,.2,.01)) == False
    # x is out of range
    assert np.isfinite(gaussian_prior(-.2,.2,.01)) == False
    # prior is >0 numerically
    assert np.isfinite(gaussian_prior(1.,.2,.05)) 

# -------------------------------- TESTING uninformative_prior(rate, minrate, maxrate) ----------------------------

def test_uninformative_prior():
    # working example
    assert uninformative_prior(3, .4, 40) == np.log(1/39.6)

    # error in the inputs
    for i in [np.nan, np.inf]:
        assert np.isfinite(uninformative_prior(i, .4, 50)) == False
        
    # If someone just confuses minrate with maxrate, 
    # or passes invalid value
    # help them out by throwing an error:
    for i in [np.nan, np.inf, 1]:
        with pytest.raises(ValueError):
            uninformative_prior(3,5,i)