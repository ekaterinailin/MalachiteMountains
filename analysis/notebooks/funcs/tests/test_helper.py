import numpy as np
import pandas as pd

import pytest

from ..helper import (no_nan_inf,
                      read_custom_aperture_lc,
                      create_spherical_grid,
                      fix_mask,
                      calculate_inclination)

from astropy.io import fits

from altaipony.flarelc import FlareLightCurve

import os

CWD = "/".join(os.getcwd().split("/")[:-2])

# We don't test fetch_lightcurve, because it's just a wrapper for read_custom_aperture

# ----------------------------- TESTING calculate_inclination(s) --------------------

def test_calculate_inclination():
    
    # Create some data to test:
    df = pd.DataFrame({"rad":[.1, np.nan, .1, .1],
                   "rad_err":[.005, .005, 0.005, 0.],
                   "Prot_d":[.2, .2, .2, .2],
                   "vsini":[15, 15, 45, 15],
                   "e_vsini":[2, 2, 2, 0],
                    "i":[36.3680921216061, np.nan,np.nan,36.3680921216061],
                   "ei":[6.008265950596777,np.nan,np.nan,0.]})
    
    # run on all rows
    for i, s in df.iterrows():
        i, ei = calculate_inclination(s)
        if np.isnan(i):
            assert np.isnan(s.i)
            assert np.isnan(s.ei)
        else:
            assert i.value == s.i
            assert ei.value == s.ei

# -------------------------------- TESTING fix_mask(flc) ----------------------------

def test_fix_mask():
    # Select cadenceno range
    start, stop = int(1e5),int(3e5)

    # Define light curve
    c = np.arange(start, stop)
    t = np.linspace(1000,1030,stop-start)
    f = np.random.rand(stop-start)
    flc = FlareLightCurve(time=t, flux=f, cadenceno=c, campaign=10)

    # Call function
    flcc = fix_mask(flc)

    # Do some checks
    res = flcc.cadenceno[np.isnan(flcc.flux)] 
    assert (((res >= 246227) & (res <= 247440)) | ((res >= 255110) & (res <= 257370))).all()


    # A different case where the campaign has no custom mask
    c = np.arange(start, stop)
    t = np.linspace(1000,1030,stop-start)
    f = np.random.rand(stop-start)
    flc2 = FlareLightCurve(time=t, flux=f, cadenceno=c, campaign=18)
    flcc2 = fix_mask(flc2)
    assert (np.isnan(flcc2.flux) == False).all()

# -------------------------------- TESTING create_spherical_grid(num_pts) ----------------------------

def test_create_spherical_grid():
    
    # Create a small grid
    phi, theta = create_spherical_grid(10)
    
    # Do explicit check of results
    assert phi == pytest.approx(np.array([ 1.11976951,  0.7753975 ,  0.52359878,  0.30469265,  0.10016742,
       -0.10016742, -0.30469265, -0.52359878, -0.7753975 , -1.11976951]))
    assert theta == pytest.approx(np.array([5.08320369, 2.68324046, 0.28327723, 4.16649931, 1.76653608,
       5.64975816, 3.24979493, 0.8498317 , 4.73305378, 2.33309055]))
    
    # Create a somewhat larger grid
    phi, theta = create_spherical_grid(10000)
    
    # Do a somewhat more informative check on whether 
    # the algorithm still does the same thing as before
    x,y,z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    dist = []
    for i in range(9999): # get folde distances between subsequent data points
        dist.append(np.sqrt((x[i]-x[i+1])**2 + (y[i]-y[i+1])**2 + (z[i]-z[i+1])**2))
   
    assert np.max(dist) == pytest.approx(1.8637, rel=1e-4) # that's because the values are going in spirals
    del dist
    del phi, theta

def test_read_custom_aperture_lc():
    
    # Create a light curve with a sinusoidal modulation
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Mike'
    hdr['COMMENT'] = "Here's some commentary about this FITS file."
    start, stop, N = 1000, 1020, 1000
    c1 = fits.Column(name='TIME', array=np.linspace(start, stop, N), format='F10.4')
    c2 = fits.Column(name='FLUX', array=400 + 50*np.sin(np.linspace(start, stop, N)*2), format='F10.4')
    c3 = fits.Column(name='FLUX_ERR', array= 20*np.random.rand(N), format='F10.4')
    c4 = fits.Column(name='QUALITY', array= np.full(N,1), format='K')
    c5 = fits.Column(name='CADENCENO', array= np.arange(100,N+100), format='K')
    hdu = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5], header=hdr)
    PATH = f"{CWD}/analysis/notebooks/funcs/tests/test.fits"
    hdu.writeto(PATH,overwrite=True)
    
    # Call the function
    flc = read_custom_aperture_lc(PATH, typ="custom", mission="TESS", mode="LC",
                                sector=10, TIC=1000)
    # Do some checks
    assert (flc.flux == 400 + 50*np.sin(np.linspace(start, stop, N)*2)).all()
    assert (flc.time == np.linspace(start, stop, N)).all()
    assert flc.campaign == 10
    assert flc.targetid == 1000

cases = [((0, 0, np.nan), False),
         ((0, 0, 0), True), 
         ((0, 3, 0, np.linspace(0, 1, 10 )), True),
         ((0, 3, 0, np.full(10,np.inf)), False),
         ((np.inf, 0, 0, 0), False),
         (np.array([9, 1, np.nan]), False),
         (np.array([9, 1, np.inf]), False),]

@pytest.mark.parametrize("l,expected", cases)
def test_no_nan_inf_succeed(l,expected):
    assert no_nan_inf(l) == expected
