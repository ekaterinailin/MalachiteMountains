import pytest


import numpy as np
import pandas as pd

from ..flarefit import (uninformative_prior,
                        calculate_posterior_value_that_can_be_passed_to_mcmc,
		            	convert_posterior_units,
			            empirical_prior)


# Define emprical prior as superposition of two Gaussian distributions
from astropy.modeling.models import Gaussian1D


# ------------------- TESTING empirical_prior(x, g) ----------------------------

def test_empirical_prior():
    # set up some emprical prior
    x = np.linspace(0,40,100) / 180 * np.pi
    g1 = Gaussian1D(amplitude=.2, mean=21 / 180 * np.pi, stddev=0.05)
    g1.bounding_box.amplitude = (0, 1.) 
    g1.bounding_box.mean = (0, np.pi/2)
    g2 = Gaussian1D(amplitude=0.05, mean=23 / 180 * np.pi, stddev=.05)
    g2.bounding_box.amplitude = (.0, 1.)
    g2.bounding_box.mean = (0, np.pi / 2)

    # g is astropy CompoundModel
    g = g1 + g2

    # integration works
    assert g(21 / 180 * np.pi) == pytest.approx(0.239186373422815)
    assert empirical_prior(21 / 180 * np.pi, g) == np.log(0.239186373422815)

    # 0. is included
    assert g(0.) < 1e-12
    assert g(0.) > 0.
    assert empirical_prior(0., g)  == np.log(g(0.))

    # bounding box works
    assert g(50. / 180. * np.pi) < 1e-12
    assert g(50. / 180. * np.pi) > 0.
    assert empirical_prior(50. / 180. * np.pi, g) == np.log(g(50. / 180. * np.pi))

    # outside of allowed inclination of pi/2
    assert g(np.pi * .6) == pytest.approx(0.)
    assert ~ np.isfinite(empirical_prior(1.1 * np.pi, g))

    # nan returns nan
    assert np.isnan(g(np.nan))
    assert np.isnan(empirical_prior(np.nan,g))

    # cannot pass nan or float as g
    with pytest.raises(TypeError) as e:
        empirical_prior(0.,np.nan)

    with pytest.raises(TypeError) as e:
        empirical_prior(0.,4.)

    # g will accept a function like sqrt though
    assert empirical_prior(1.2,np.sqrt) == pytest.approx(0.09116077839697724)


# ---------- TESTING convert_posterior_units(res, prot, phi, time) -------------

def test_convert_posterior_units():

    # setup posterior distribution with time series and rotation period
    prot = 2.
    phi = np.linspace(0,10*np.pi,200)
    time = phi/2./np.pi * prot + 10.

    df = pd.DataFrame(data=np.array([[0.,0.,0.],
                            [1.,1.,1.],
                            [.25,.25,.25],
                            [.1,.1,.1],
                            [8.,8.,8.,],
                            [1,1,1],
                            [3,3,3,]]).T * np.pi,
                      columns=['latitude_rad', 'phase_0', 'i_rad', 
                       'a', 'phase_peak', 'fwhmi', 'fwhmg'])

    # test correct answers
    assert (convert_posterior_units(df, prot, phi, time).to_numpy() ==
            pytest.approx(np.array([[ 0., 180., 45., 0.31415927, 18., 1., 3.] for i in range(3)]), rel=1e-5))

    # check if function works with different fwhm settings
    # check nr.1
    del df["fwhmi"]
    assert convert_posterior_units(df, prot, phi, time).shape[1] == 6

    # check nr.2
    df = df.rename(index=str, columns={"fwhmg":"fwhm"})
    assert convert_posterior_units(df, prot, phi, time).shape[1] == 6

    # check nr.3
    del df["fwhm"]
    assert convert_posterior_units(df, prot, phi, time).shape[1] == 5
    
    
    # check that function fails when required columns are missing
    for col in ['latitude_rad', 'phase_0', 'i_rad', 'phase_peak']:
        with pytest.raises(AttributeError) as e:
            del df[col]
            convert_posterior_units(df, prot, phi, time)


# ---- TESTING calculate_posterior_value_that_can_be_passed_to_mcmc(lp) --------

def test_calculate_posterior_value_that_can_be_passed_to_mcmc():
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(np.inf) == -np.inf
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(np.nan) == -np.inf
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(3) == 3


# ------------ TESTING uninformative_prior(rate, minrate, maxrate) -------------

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
