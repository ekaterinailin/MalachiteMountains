import pytest
import numpy as np
import pandas as pd

import copy

import astropy.units as u
from astropy.constants import  R_sun, b_wien

# Define emprical prior as superposition of two Gaussian distributions
from astropy.modeling.models import Gaussian1D

from ..model import (aflare,
                     aflare_decoupled,
                     daylength,
                     on_off,
                     lambert,
                     black_body_spectrum,
                     great_circle_distance,
                     dot_ensemble_spherical,
                     dot_ensemble_circular,
                     calculate_specific_flare_flux,
                     calculate_angular_radius,
                     lightcurve_model,
                     calculate_ED,
                     FlareModulator)

# ------------ TESTING FlareModulator class and its methods --------------------

def test_FlareModulator():
    # TEST ALL FOUR CURRENTLY IMPLEMENTED FLARE VERSIONS 
    # (1- and 2- flare, coupled- and decoupled)

    # ------------------------------------------------------------
    # FIRST, set parameters that stay the same

    # set stellar parameters
    prot = .1
    qlum = 5e29 * u.erg / u.s
    R = 0.2 * R_sun

    # set baseline time series
    phi = np.linspace(12,30,500)
    time = phi/2./np.pi * prot
    median = 100

    # set flare parameters
    a =.4
    phia = 13.
    fwhm = .3

    # define flux
    flux = aflare(phi, phia, a, fwhm) * median + median
    flux_err = 0.01 * flux

    # define an empirical prior on inclination
    mids = np.linspace(30,80,130)
    x = mids /180 *np.pi
    g1 = Gaussian1D(amplitude=.2, mean=55/180*np.pi, stddev=0.05)
    g1.bounding_box.amplitude = (0, 1.) 
    g1.bounding_box.mean = (0,np.pi/2)
    g2 = Gaussian1D(amplitude=0.05, mean=56/180*np.pi, stddev=.05)
    g2.bounding_box.amplitude = (.0, 1.)
    g2.bounding_box.mean = (0,np.pi/2)

    gincl = g1 + g2

    # ------------------------------------------------------------
    # COUPLED 1-flare solution

    # set model specifications
    nflares = 1
    iscoupled = True

    # construct model
    FM = FlareModulator(phi, flux, flux_err, qlum, 
                        R, median, nflares, iscoupled)

    # Check some params and default values

    assert (FM.phi == phi).all()
    assert (FM.flux == flux).all()
    assert (FM.flux_err == flux_err).all()
    assert FM.qlum == qlum
    assert FM.R == R
    assert FM.median == median
    assert FM.num_pts == 100
    assert FM.Fth == calculate_specific_flare_flux("TESS", flaret=1e4)

    assert FM.nflares == nflares
    assert FM.iscoupled == iscoupled

    # set some parameters for latitude, longitude and inclination
    theta, phi0, i = 60 * np.pi / 180, 5.5, 55 * np.pi / 180
    params = [theta, phi0, i, a, phia, fwhm]

    # calculate underlying flare
    underlying_flare = FM.flare_template(params[3:])

    # do some checks
    assert np.max(underlying_flare) < .4
    assert np.max(underlying_flare) > .38
    assert underlying_flare[0] == 0.
    assert underlying_flare[-1] < 1e-3

    #calculate modulated flux
    modelflux = FM.modulated_flux(*params[:3], [params[3:]])

    # do some sanity checks
    assert (modelflux <= underlying_flare * 100 + 100.).all()
    assert (modelflux >= 100.).all()
    assert modelflux[0] == 100.
    assert modelflux[-1] < 100 + 1e-3

    # calculate log likelihood for initial parameters 
    # should be closer than underlying flare
    FM.flux = modelflux
    assert FM.log_likelihood(params) == pytest.approx(-5.7546, rel=1e-3)
    FM.flux = underlying_flare
    assert FM.log_likelihood(params) == pytest.approx(-2469724.297, rel=1e-2)

    # prior value should be the same regardless of observations
    prior = FM.log_prior(params, g=gincl)
    FM.flux = modelflux
    assert FM.log_prior(params, g=gincl) == prior

    # giving no empirical prior on inclination results in an error
    with pytest.raises(TypeError) as e:
        FM.log_prior(params, g=None)

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # DECOUPLED 1-flare solution - mostly the same results

    # set model specifications
    fwhmdec = .2
    iscoupled = False

    # keep the rest
    FM = FlareModulator(phi, flux, flux_err, qlum, R, median, nflares, iscoupled)

    # Check that the new parameter is set
    assert FM.iscoupled == iscoupled

    # redefine parameters in correct order
    params = [theta, phi0, i, a, phia, fwhm, fwhmdec]

    # calculate underlying flare
    underlying_flare = FM.flare_template(params[3:])

    # do some checks
    assert np.max(underlying_flare) < .4
    assert np.max(underlying_flare) > .38
    assert underlying_flare[0] == 0.
    assert underlying_flare[-1] < 1e-3


    #calculate modulated flux
    modelflux = FM.modulated_flux(*params[:3], [params[3:]])

    # do some sanity checks
    assert (modelflux <= underlying_flare * 100 + 100.).all()
    assert (modelflux >= 100.).all()
    assert modelflux[0] == 100.
    assert modelflux[-1] < 100 + 1e-3

    # calculate log likelihood for initial parameters 
    # should be closer than underlying flare
    FM.flux = modelflux
    assert FM.log_likelihood(params) == pytest.approx(-5.7546, rel=1e-3)
    FM.flux = underlying_flare
    assert FM.log_likelihood(params) == pytest.approx(-2469724.297, rel=1e-2)

    # prior value should be the same regardless of observations
    prior = FM.log_prior(params, g=gincl)
    FM.flux = modelflux
    assert FM.log_prior(params, g=gincl) == prior

    # giving no empirical prior on inclination results in an error
    with pytest.raises(TypeError) as e:
        FM.log_prior(params, g=None)

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # COUPLED 2-flare solution 

    a1, a2 =.4, .3
    phia1, phia2 = 13, 16.
    fwhm1, fwhm2 = .4, .5

    # define flux
    flux = (aflare(phi,phia1, a1, fwhm1) + aflare(phi,phia2, a2, fwhm2) + 1.) * median
    flux_err = 0.01 * flux

    # set model specifications
    nflares = 2
    iscoupled = True

    # keep the rest
    FM = FlareModulator(phi, flux, flux_err, qlum, R, median, nflares, iscoupled)

    # Check that the new parameter is set
    assert FM.iscoupled == iscoupled
    assert FM.nflares == nflares

    # redefine parameters in correct order
    params = [theta, phi0, i, a1, phia1, fwhm1, a2, phia2, fwhm2]

    # calculate underlying flare
    underlying_flare = FM.flare_template(params[3:]) + FM.flare_template(params[6:])

    # do some checks
    assert np.max(underlying_flare) <= .4
    assert np.max(underlying_flare) > .38
    assert underlying_flare[0] == 0.
    assert underlying_flare[-1] < 1e-3

    #calculate modulated flux
    flareparams = np.array(params[3:]).reshape(FM.nflares, len(params[3:])//FM.nflares)
    modelflux = FM.modulated_flux(*params[:3], flareparams)

    # do some sanity checks
    assert (modelflux <= underlying_flare * 100 + 100.).all()
    assert (modelflux >= 100.).all()
    assert modelflux[0] == 100.

    assert modelflux[-1] < 100.1

    # calculate log likelihood for initial parameters 
    # should be closer than underlying flare

    FM.flux = modelflux
    assert FM.log_likelihood(params) == pytest.approx(-14.4552, rel=1e-3)

    FM.flux = underlying_flare
    assert FM.log_likelihood(params) == pytest.approx(-2433738.246, rel=1e-2)

    # prior value should be the same regardless of observations
    prior = FM.log_prior(params, g=gincl)
    FM.flux = modelflux
    assert FM.log_prior(params, g=gincl) == prior

    # giving no empirical prior on inclination results in an error
    with pytest.raises(TypeError) as e:
        FM.log_prior(params, g=None)

    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # DECOUPLED 2-flare solution 

    # set model specifications
    fwhm1dec, fwhm2dec = .5, .5
    iscoupled = False

    # keep the rest
    FM = FlareModulator(phi, flux, flux_err, qlum, R, median, nflares, iscoupled)

    # Check that the new parameter is set
    assert FM.iscoupled == iscoupled

    # redefine parameters in correct order
    params = [theta, phi0, i, a1, phia1, fwhm1, fwhm1dec,  a2, phia2, fwhm2, fwhm2dec]

    # calculate underlying flare
    underlying_flare = FM.flare_template(params[3:]) + FM.flare_template(params[7:])

    # do some checks
    assert np.max(underlying_flare) <= .4
    assert np.max(underlying_flare) > .38
    assert underlying_flare[0] == 0.
    assert underlying_flare[-1] < 1e-3

    #calculate modulated flux
    flareparams = np.array(params[3:]).reshape(FM.nflares, len(params[3:])//FM.nflares)
    modelflux = FM.modulated_flux(*params[:3], flareparams)

    # do some sanity checks
    assert (modelflux <= underlying_flare * 100 + 100.).all()
    assert (modelflux >= 100.).all()
    assert modelflux[0] == 100.
    assert modelflux[-1] < 100.1

    # calculate log likelihood for initial parameters 
    # should be closer than underlying flare
    FM.flux = modelflux
    assert FM.log_likelihood(params) == pytest.approx(-14.4552, rel=1e-3)

    FM.flux = underlying_flare
    assert FM.log_likelihood(params) == pytest.approx(-2433738.246, rel=1e-2)

    # prior value should be the same regardless of observations
    prior = FM.log_prior(params, g=gincl)
    FM.flux = modelflux
    assert FM.log_prior(params, g=gincl) == prior

    # giving no empirical prior on inclination results in an error
    with pytest.raises(TypeError) as e:
        FM.log_prior(params, g=None)
    # ------------------------------------------------------------


# ---------------------- TESTING aflare(t, tpeak, dur, ampl, upsample=False, uptime=10) ---------------
# The same tests as in AltaiPony

def test_aflare_and_equivalent_duration():

    n = 1000
    time = np.arange(0, n/48, 1./48.)
    x = time * 60.0 * 60.0 * 24.0

    # Test a large flare without upsampling
    fl_flux = aflare(time, 11.400134, 1.415039, 110.981950)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(2.24e7,rel=1e-2)
    
    # Test a flare with 0 amplitude
    fl_flux = aflare(time, 11.400134, 1.415039, 0)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == 0.

    # test a large flare with upsampling
    fl_flux = aflare(time, 11.400134, 1.415039, 110.981950, upsample=True)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(2.24e7,rel=1e-2)
    
    
    # Test a smaller undersampled flare
    fl_flux = aflare(time, 11.400134, 1/48., 1.0)
    x = time * 60.0 * 60.0 * 24.0
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(3333.5296,rel=1e-2)
    
    # Test the amplitude
    fl_flux = aflare(time, 1.734, 15, 1.0)
    assert np.max(fl_flux) == pytest.approx(1,rel=1e-2)

# --------------- TESTING black_body_spectrum(wav, T) -------------------------- 

def test_black_body_spectrum():
    
    # Construct a wavelength array:
    wav = np.linspace(300,900,3000)*u.nm

    # T=0K should give 0 flux

    t = 0
    bbs =  black_body_spectrum(wav, t)
    assert bbs.unit == u.erg / u.s / u.cm**3
    assert (bbs.value == np.full(3000,0)).all()

    # Test a Sun-like star

    t = 5900
    lmaxf = wav[np.argmax(black_body_spectrum(wav, t))]
    lmax = b_wien / (t * u.K) 
    assert lmax.to("nm").value == pytest.approx(lmaxf.value, rel=1e-3)

# ----------- TESTING calculate_angular_radius(Fth, a, qlum, R, lat, lon, i, phi=0) ------- Fth, a, qlum, R

def test_calculate_angular_radius():
    # Test unphysical results: flare area > stellar surface
    with pytest.raises(ValueError) as e:
        calculate_angular_radius(3e6*u.J/u.m**2/u.s, 2, 1e30*u.erg/u.s, 
                                 .1*R_sun)
    # Test zero as input
    assert calculate_angular_radius(3e6*u.J/u.m**2/u.s, 2, 0*u.erg/u.s, 
                                    .1*R_sun,) == 0.
    # Test unit error
    with pytest.raises(u.UnitConversionError) as e:
        assert calculate_angular_radius(3e6*u.J/u.m**2/u.s, 2, 0*u.erg, 
                                        .1*R_sun,)
    # Test case where flare must cover a hemisphere
    assert (calculate_angular_radius(1*u.erg/u.cm**2/u.s, 0.5, 2*np.pi*u.erg/u.s, 
                                        1*u.cm, ) == 
            pytest.approx(90))

    
## ------------- TESTING calculate_specific_flare_flux(mission, flaret=1e4) ------------

def test_calculate_scpecific_flare_flux():
    # Are the units correct?
    assert (calculate_specific_flare_flux("TESS", flaret=1e4).unit 
            == u.erg / u.s / u.cm**2)
    
    # Doe I get zero if flare T = 0K?
    assert calculate_specific_flare_flux("TESS", flaret=0).value == 0.
    
    # Is Kepler flux higher for T>10kK?
    for t in [1e4,5e4,1e5]:
        assert (calculate_specific_flare_flux("Kepler", flaret=t) > 
                calculate_specific_flare_flux("TESS", flaret=t))
        
    # Is TESS flux higher for T=1000K?
    assert (calculate_specific_flare_flux("Kepler", flaret=1e3) <
            calculate_specific_flare_flux("TESS", flaret=1e3))
    
    # Wrong Key throws error:
    with pytest.raises(KeyError) as e:
        calculate_specific_flare_flux("Gaia")
    with pytest.raises(KeyError) as e:
        calculate_specific_flare_flux(np.nan)
        
    # NaN input throws error:
    with pytest.raises(ValueError) as e:
        calculate_specific_flare_flux("Kepler",flaret=np.nan)


## ------------- TESTING daylength(l, i) --------------------------

# Test different values on daylength(l, i) to make it fail
cases = [(np.nan,0), (np.nan,np.nan), (0,np.nan),
         (2,np.nan), (2,np.pi), (2,-np.pi),
         (1.5*np.pi, np.pi/3)]

@pytest.mark.parametrize("l,i", cases)
def test_daylength_fail(l,i):
    with pytest.raises(ValueError) as e:
        daylength(l, i)

# Test different values on daylength(l, i) to make it succeed
cases = [(0,np.pi/2,0.5),(0,0,1),(0.,70 * np.pi / 180, 0.5),
         (np.pi/2,np.pi/4,1.),(np.pi/4,np.pi/4,1.),
         (np.pi/4+.05,np.pi/4,1.),(0,np.pi/4,.5),
         (- np.pi/4,np.pi/4,0.),(-np.pi/2,np.pi/4,0.)]

@pytest.mark.parametrize("l,i,expected", cases)
def test_daylength_succeed(l,i,expected):
    assert daylength(l, i) == expected
    
## ------------- TESTING on_off(l, i) --------------------------    
    
cases = [(np.nan,0,0),(0,np.nan,0),(0,np.nan,np.nan),
           (0,0,np.nan),(np.nan,0,0),(np.inf,0,0),
           (np.inf,np.inf,0),(0,np.inf,0),(0,0,np.inf)]    
    
@pytest.mark.parametrize("p,d,p0", cases)
def test_on_off_fail(p,d,p0):
    with pytest.raises(ValueError) as e:
        on_off(p, d, phi0=p0)

# Test different values on on_off(p,d,p0) to make it succeed
cases = [(4,1,0, True),
         (0,0,0,False),
         (np.pi,0.5,0,False),
         (np.pi,0.5,np.pi,True),
         (np.pi,0.5,1.2*np.pi,True),
         (np.pi,0.5,2*np.pi,False),
         (np.linspace(3,4,20),1,0,np.full(20,1))]

@pytest.mark.parametrize("p,d,p0,expected", cases)
def test_on_off_succeed(p,d,p0,expected):
    assert on_off(p,d,phi0=p0) == pytest.approx(expected)
    
## ------------- TESTING lambert(phi, i, l)--------------------------

cases = [(0,0,np.pi/2,1),(0,0,0,0),(0,0,np.pi/6,0.5)]

@pytest.mark.parametrize("phi,i,l,expected", cases)
def test_lambert(phi,i,l,expected):
    assert lambert(phi,i,l) == pytest.approx(expected)
    
## ------------- TESTING great_circle_distance(a, la, b, lb)--------------------------    
    
cases = [(0, 0, np.pi, 0, np.pi),
         (np.pi/2, 0, np.pi/2, 0, 0),
         (0, np.pi/2, 0, 0, np.pi/2)]

@pytest.mark.parametrize("a,la,b,lb,expected", cases)
def test_great_circle_distance(a, la, b, lb, expected):
    assert great_circle_distance(a, la, b, lb) == expected
    
    
## ------------- TESTING dot_ensemble_circular(lat, lon, radius, num_pts=200)) -----------  

cases = [(0,0,6), (90,0,3), (0,90,3), (-60,-20,4)]

@pytest.mark.parametrize("lat,lon,r", cases)
def test_dot_ensemble_circular(lat, lon, r):
    
    # Call function on the case inputs
    lats, lons, (x,y,z) = dot_ensemble_circular(lat, lon, r, num_pts=100)
    
    # Make sure all arrays have correct length
    for i in (lats, lons, x, y, z):
        assert len(i) == 100
    # Make sure latitudes and longites are within allowed range
    assert (np.abs(lats) < np.pi/2).all()
    assert (np.abs(lons) < 2 * np.pi).all()

    # Make sure positions are not outside the sphere
    for i in (x, y, z):
        assert (np.abs(i) < 1).all()

    # Actually make sure positions are all ON the sphere
    assert (x**2 + y**2 + z**2) ==  pytest.approx(1.)

    # Check if error is raise if input is not finite:
    for case in [(np.nan, lon, r,),
                 (lat, np.nan, r,),
                 (lat, lon, np.inf,)]:
        with pytest.raises(ValueError) as e:
            dot_ensemble_circular(*case, num_pts=100)
    
    
## ------------- TESTING dot_ensemble_spherical(lat, lon, radius)-------------       
    
def test_dot_ensemble_spherical():
    
    # Let the ensemble cover half a sphere
    p, t = dot_ensemble_spherical(0, 0, 90)
    assert len(p) == 4999
    assert len(p) == len(t)
    assert (p < np.pi / 2).all()
    assert (p > -np.pi / 2).all()
    assert (t < 2 * np.pi).all()
    assert (t > 0 ).all()
    
    # Look at tiny radii producing only one ...
    p,t = dot_ensemble_spherical(0, 0, 1)
    assert len(p) == 1
    
    # or no dots at all.
    p,t = dot_ensemble_spherical(0, 0, .01)
    assert len(p) == 0
    
    # Test one failing case to make sure no_nan_inf is called:
    with pytest.raises(ValueError) as e:
        dot_ensemble_spherical(0, np.nan, .1)
        
## ------------- TESTING  lightcurve_model(phi, latitudes, longitudes, flare, inclination, phi=0)  -----------   
        
def test_lightcurve_model():
    
    # Set up a mock data set:
    phi = np.linspace(0,np.pi*2, 10)
    flare = np.full(10,3)*np.arange(10)
    latitudes = np.array([np.pi/2,np.pi/4,0])
    longitudes = np.array([0,0,0])
    inclination = np.pi/2
    
    # Calculate the model
    lamb, onoff, m = lightcurve_model(phi, latitudes, 
                                      longitudes,
                                      flare, inclination,)
    
    # Do some checks
    assert m[0] == 0
    assert onoff.shape == lamb.shape
    assert onoff.shape == (3,10)
    assert (lamb[0]==pytest.approx(0.)) # dot on top
    assert (lamb[1] * onoff[1] >= 0.).all()
    assert np.max(m)==m[-1]

# ------------- TESTING  calculate_ED(t, t0, dur, ampl )  -------------------------------      
    
def test_calculate_ED():
    # Define time array
    t = np.linspace(1,11,200)
    
    # unit testing
    #--------------

    # COUPLED flare model
    
    # Throw error with bad values
    with pytest.raises(ValueError):
        calculate_ED(t, np.nan, 1, 0, decoupled=False)
    with pytest.raises(ValueError):
        calculate_ED(t, 2, np.nan, 0, decoupled=False)
    with pytest.raises(ValueError):
        calculate_ED(t, 2, 5, np.nan, decoupled=False)
    
    # Return NaN if any time values are not properly defined
    t2 = copy.copy(t)
    t2[4:12]= np.nan
    assert np.isnan(calculate_ED(t2, 2, 5, 1, decoupled=False))
    
    # integration testing
    #---------------------
    
    # 0 duration
    assert calculate_ED(t, 2, 0, 1, decoupled=False) == 0.
    
    # 0 amplitude
    assert calculate_ED(t, 2, 1, 0, decoupled=False) == 0.
    
    #triangle approximation
    assert calculate_ED(t, 1, 1, .1, decoupled=False) < .2 * 60.0 * 60.0 * 24.0 

    # DECOUPLED flare model
    
    # Throw error with bad values
    with pytest.raises(ValueError):
        calculate_ED(t, np.nan, (1.,1.), 0,)
    with pytest.raises(ValueError):
        calculate_ED(t, 2, (np.nan, np.nan), 0,)
    with pytest.raises(ValueError):
        calculate_ED(t, 2, (5.,5.), np.nan,)
    
    # Return NaN if any time values are not properly defined
    t2 = copy.copy(t)
    t2[4:12]= np.nan
    assert np.isnan(calculate_ED(t2, 2, [5.,5.], 1))
    
    # integration testing
    #---------------------
    
    # 0 duration
    assert calculate_ED(t, 2, (0.,0.), 1,) == 0.
    
    # 0 amplitude
    assert calculate_ED(t, 2, (1.,1.), 0,) == 0.
    
    #triangle approximation
    assert calculate_ED(t, 1, (1.,1.), .1) < .2 * 60.0 * 60.0 * 24.0 


def test_aflare_decoupled():
    # repeat the tests for aflare

    n = 1000
    time = np.arange(0, n/48, 1./48.)
    x = time * 60.0 * 60.0 * 24.0

    # Test a large flare without upsampling
    fl_flux = aflare_decoupled(time, 11.400134, [1.415039]*2, 110.981950)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(2.24e7,rel=1e-2)

    # Test a flare with 0 amplitude
    fl_flux = aflare_decoupled(time, 11.400134, [1.415039]*2, 0)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == 0.

    # test a large flare with upsampling
    fl_flux = aflare_decoupled(time, 11.400134, [1.415039]*2, 110.981950, upsample=True)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(2.24e7,rel=1e-2)


    # Test a smaller undersampled flare
    fl_flux = aflare_decoupled(time, 11.400134, [1/48.]*2, 1.0)
    x = time * 60.0 * 60.0 * 24.0
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(3333.5296,rel=1e-2)

    # Test the amplitude
    fl_flux = aflare_decoupled(time, 1.734, [15]*2, 1.0)
    assert np.max(fl_flux) == pytest.approx(1,rel=1e-2)

    # repeat the tests for aflare but now with different gradual fwhm

    n = 1000
    time = np.arange(0, n/48, 1./48.)
    x = time * 60.0 * 60.0 * 24.0

    # Test a large flare without upsampling
    fl_flux = aflare_decoupled(time, 11.400134, [1.415039, 3.4], 110.981950)
    integral = np.sum(np.diff(x) * fl_flux[:-1])

    assert integral == pytest.approx(2.91e7,rel=1e-2)

    # Test a flare with 0 amplitude
    fl_flux = aflare_decoupled(time, 11.400134, [1.415039, 30.], 0)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == 0.

    # test a large flare with upsampling
    fl_flux = aflare_decoupled(time, 11.400134, [1.415039, 3.4], 110.981950, upsample=True)
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(2.91e7,rel=1e-2)

    # Test a smaller undersampled flare
    fl_flux = aflare_decoupled(time, 11.400134, [1/48.,1./20.], 1.0)
    x = time * 60.0 * 60.0 * 24.0
    integral = np.sum(np.diff(x) * fl_flux[:-1])
    assert integral == pytest.approx(6076.2728,rel=1e-2)

    # Test the amplitude
    fl_flux = aflare_decoupled(time, 1.734, [15,71], 1.0)
    assert np.max(fl_flux) == pytest.approx(1,rel=1e-2)
