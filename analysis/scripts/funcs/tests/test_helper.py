import numpy as np
import pandas as pd

import pytest

from ..helper import (no_nan_inf,
                      create_spherical_grid,)

from astropy.io import fits

from altaipony.flarelc import FlareLightCurve

import os

CWD = "/".join(os.getcwd().split("/")[:-2])

# We don't test fetch_lightcurve, because it's just a wrapper for read_custom_aperture



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


# -------------------------------- TESTING no_nan_inf() ----------------------------
    
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
