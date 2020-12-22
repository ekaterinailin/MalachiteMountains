import warnings

import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.constants import R_sun
import astropy.units as u

from altaipony.flarelc import FlareLightCurve
from altaipony.lcio import from_path

import sys, os

CWD = "/work1/eilin/MultiperiodFlares/MalachiteMountains/data/lcs"#os.getcwd()
CWD = "/home/ekaterina/Documents/001_science/MalachiteMountains/data/lcs"

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


def fetch_lightcurve(target, flux_type="FLUX", path=CWD):
    """Read in light curve from file.

    Parameters:
    -----------
    target: Series
        Description of the target.
    flux_type : str
        "PDCSAP_FLUX", "SAP_FLUX", "FLUX" or other
    """
    path = (f"{path}/{target.ID}_{target.QCS:02d}_" \
            f"{target.mission}_{target.typ}_{target.origin}.fits")

    flc = read_custom_aperture_lc(path, mission=target.h_mission,
                                  mode="LC", typ=target.origin,
                                  TIC=target.ID, sector=target.QCS,
                                  flux_type=flux_type)
    return flc


def read_custom_aperture_lc(path, typ="custom", mission="TESS", mode="LC",
                            sector=None, TIC=None, flux_type="PDCSAP_FLUX"):
    '''Read in custom aperture light curve
    from TESS or uses AltaiPony's from path for standard
    light curves. Needs specific naming convention.
    Applies pre-defined quality masks.

    Parameters:
    -----------
    path : str
        path to file
    flux_type : str
        "PDCSAP_FLUX", "SAP_FLUX", "FLUX" or other

    Returns:
    --------
    FlareLightCurve
    '''
    if typ=="custom":
        hdu = fits.open(path)
        data = hdu[1].data
        if sector==None:
            sector = int(path.split("-")[1][-2:])
        if TIC==None:
            TIC = int(path.split("-")[2])

        flc = FlareLightCurve(time=data["TIME"],
                            flux=data["FLUX"],
                            flux_err=data["FLUX_ERR"],
                            quality=data["QUALITY"],
                            cadenceno=data["CADENCENO"],
                            targetid=TIC,
                            campaign=sector)
        flc = fix_mask(flc)
    else:
        flc = from_path(path, mission=mission,
                        mode=mode, )#flux_type=flux_type)

    return flc


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




