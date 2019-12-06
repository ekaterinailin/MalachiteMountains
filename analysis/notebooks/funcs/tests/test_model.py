import pytest
import numpy as np

from ..model import (daylength,
                     on_off,
                     lambert,
                     great_circle_distance,
                     dot_ensemble_spherical,
                     dot_ensemble_circular,
                     model)



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
    
    
## ------------- TESTING dot_ensemble_spherical(lat, lon, radius, num_pts=1e4)-------------       
    
def test_dot_ensemble_spherical():
    
    # Let the ensemble cover half a sphere
    p, t = dot_ensemble_spherical(0, 0, 90, num_pts=1e4)
    assert len(p) == 4999
    assert len(p) == len(t)
    assert (p < np.pi / 2).all()
    assert (p > -np.pi / 2).all()
    assert (t < 2 * np.pi).all()
    assert (t > 0 ).all()
    
    # Look at tiny radii producing only one ...
    p,t = dot_ensemble_spherical(0, 0, 1, num_pts=1e4)
    assert len(p) == 1
    
    # or no dots at all.
    p,t = dot_ensemble_spherical(0, 0, .1, num_pts=1e4)
    assert len(p) == 0
    
    # Test one failing case to make sure no_nan_inf is called:
    with pytest.raises(ValueError) as e:
        dot_ensemble_spherical(0, np.nan, .1, num_pts=1e4)
        
## ------------- TESTING  model(phi, latitudes, longitudes, flare, inclination, phi=0)  -----------   
        
def test_model():
    
    # Set up a mock data set:
    phi = np.linspace(0,np.pi*2, 10)
    flare = np.full(10,3)*np.arange(10)
    latitudes = np.array([np.pi/2,np.pi/4,0])
    longitudes = np.array([0,0,0])
    inclination = np.pi/2
    
    # Calculate the model
    lamb, onoff, m = model(phi, latitudes, longitudes, flare, inclination,)
    
    # Do some checks
    assert m[0] == 0
    assert onoff.shape == lamb.shape
    assert onoff.shape == (3,10)
    assert (lamb[0]==pytest.approx(0.)) # dot on top
    assert (lamb[1] * onoff[1] >= 0.).all()
    assert np.max(m)==m[-1]