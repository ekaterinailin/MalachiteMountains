import warnings

import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.constants import R_sun
import astropy.units as u

from altaipony.flarelc import FlareLightCurve
from altaipony.lcio import from_path

import sys, os

CWD = "/".join(os.getcwd().split("/")[:-2])


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


def fetch_lightcurve(target):
    """Read in light curve from file.

    Parameters:
    -----------
    target: Series
        Description of the target.
    """
    path = (f"{CWD}/data/lcs/{target.ID}_{target.QCS:02d}_" \
            f"{target.mission}_{target.typ}_{target.origin}.fits")

    flc = read_custom_aperture_lc(path, mission=target.h_mission,
                                  mode="LC", typ=target.origin,
                                  TIC=target.ID, sector=target.QCS)
    return flc


def read_custom_aperture_lc(path, typ="custom", mission="TESS", mode="LC",
                            sector=None, TIC=None):
    '''Read in custom aperture light curve
    from TESS or uses AltaiPony's from path for standard 
    light curves. Needs specific naming convention.
    Applies pre-defined quality masks.

    Parameters:
    -----------
    path : str
        path to file

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
        flc = from_path(path, mission=mission, mode=mode)
        
    return flc



def fix_mask(flc):
    '''Here the masks for different TESS 
    sectors are defined and applied to 
    light curve fluxes.
    
    Parameters:
    ------------
    flc : FlareLightCurve
    
    Returns:
    ----------
    FlareLightCurve
    '''
    masks = {9: [(227352, 228550), (236220, 238250)],
             10: [(246227,247440),(255110,257370)],
             11:[(265912,268250),(275210,278500)],
             8: [(208722,209250),(221400,221700)],
             6: [(179661,180680)],
             5: [(151586,151900),(160254,161353),(170000,170519)],
             4: [(139000,139800),(140700,141161),(150652,150764)],
             3: [(120940,121820)],
             12: [(286200,286300)]}

    if flc.campaign in masks.keys():
        for sta, fin in masks[flc.campaign]:
            flc.flux[np.where((flc.cadenceno >= sta) & (flc.cadenceno <= fin))] = np.nan
    else:
        warnings.warn(f"Campaign {flc.campaign} has no defined custom masks.")
    flc.quality[:] = np.isnan(flc.flux)
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


def calculate_inclination(s):
    """Determine the inclination
    vsini, stellar radius, and rotation
    period.
    
    Parameters:
    -----------
    s : pandas Series
        contains "rad", "rad_err", "Prot_d",
        "vsini_kms", and "e_vsini_kms". No uncertainties
        on "P".
    
    Return:
    -------
    inclination, uncertainty on inclination - 
        astropy Quantities
    """
    R, P, vsini = s.rad * R_sun, s.Prot_d * u.d, s.vsini_kms * u.km / u.s
    eR, eP, evsini = s.rad_err * R_sun, 0 * u.d, s.e_vsini_kms * u.km / u.s

    sini = vsini * P / 2 / np.pi / R
    print(f"sin(i)={sini.decompose()}")
    incl = np.arcsin(sini)

    _a = vsini / 2 / np.pi / R
    sigP_squared = (_a / np.sqrt(1 - (_a * P)**2))**2 * eP**2

    _b = P / 2 / np.pi / R
    sigvsini_squared = (_b / np.sqrt(1 - (_b * vsini)**2))**2 * evsini**2

    _c = vsini * P / 2 / np.pi
    sigR_squared = ( - _c / R**2 / np.sqrt(1 - (_c / R)**2))**2 * eR**2 

    eincl = np.sqrt(sigP_squared + sigR_squared + sigvsini_squared).decompose()*u.rad

    return incl.to("deg"), eincl.to("deg")
