""""""

import numpy as np
from altaipony.lcio import from_mast
import os

CWD = "/".join(os.getcwd().split("/")[:-3]) + "/data/lcs"


# We do not test fetch_lightcurve because it's just a wrapper for read_custom_aperture_lc

def no_nan_inf(l):
    """Check arguments in list for Inf and NaN.
    Return True if all values are finite, and not NaN.

    Parameters:
    -----------
    l : list
        contains floats, ints, and arrays of these values

    Return:
    -------
    bool
    """
    for elem in l:
        if isinstance(elem,np.ndarray):
            if (np.isnan(elem).any() |  (not np.isfinite(elem).any())):
                return False
        if (isinstance(elem, float) | isinstance(elem, int)):
            if (np.isnan(elem) |  (not np.isfinite(elem))):
                return False
    return True


def fetch_lightcurve(target, flux_type="PDCSAP_FLUX"):
    """Wrap `from_mast` to fetch a light curve from mast or
    cached one locally.

    Parameters:
    -----------
    target: pandas Series
        Description of the target.
    flux_type : str
        "PDCSAP_FLUX", "SAP_FLUX"
    """

    return from_mast(mission=target.h_mission, mode="LC",
                    targetid=target.ID, c=target.QCS,
                    flux_type=flux_type)


def create_spherical_grid(num_pts):
    """Method see:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    answered by CR Drost

    Coversion to cartesian coordinates:
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi);

    Parameters:
    -----------
    num_pts : int
        number of grid points on the full sphere

    Return:
    --------
    phi, theta - numpy arrays of latitude, longitude
    """

    # This is CR Drost's solution to the sunflower spiral:
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices/num_pts) #latitude
    theta = np.pi * (1 + 5**0.5) * indices #longitude

    # Fold onto on sphere
    phi = (np.pi/2 - phi) % (2 * np.pi) # 0th  stars at the equator
    # 2nd quadrant
    q = np.where((np.pi/2 < phi) & (phi < np.pi))
    phi[q] = np.pi-phi[q]
    # 3rd quadrant
    q = np.where((np.pi < phi) & (1.5*np.pi > phi))
    phi[q] = -(phi[q] - np.pi)
    # 4th quadrant
    q = np.where((1.5*np.pi < phi) & (2*np.pi > phi))
    phi[q] = phi[q] - np.pi*2
    theta = theta % (np.pi * 2)

    return phi, theta




