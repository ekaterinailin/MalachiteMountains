import warnings

import pandas as pd
import numpy as np

from astropy.io import fits
from astropy.constants import R_sun
import astropy.units as u

# from altaipony.flarelc import FlareLightCurve
# from altaipony.lcio import from_path

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



def calculate_inclination(s, eP=1./24/30):
    """Determine the inclination
    vsini, stellar radius, and rotation
    period.

    Parameters:
    -----------
    s : pandas Series
        contains "rad", "rad_err", "Prot_d",
        "vsini_kms", and "e_vsini_kms". No uncertainties
        on "P". Instead, time resolution of light curve
        is used as uncertainty.
    eP : float
        Set uncertainty to 2min for TESS light curves
        automatically. Values is measured in days.
    Return:
    -------
    inclination, uncertainty on inclination -
        astropy Quantities
    """
    # Get radius und and period, plus their uncerainties
    R, P = s.rad * R_sun, s.Prot_d * u.d
    eR, eP = s.rad_err * R_sun, eP * u.d
    print(s.ID, s.e_vsini_kms)
    # Get vsini and its uncertainty
    vsini = s.vsini_kms * u.km / u.s
    evsini = s.e_vsini_kms * u.km / u.s

    # Caclulate sini
    sini = vsini * P / 2. / np.pi / R

    # Calculate rotation velocity
    v = vsini / sini

    # Calculate inclination
    incl = np.arcsin(sini)

    # Calculate uncertainty on sini
    # Propagate uncertainties on R, vsini, and Prot
    t1 = vsini * P / (2. * np.pi * R**2) * eR
    t2 = P / (2. * np.pi * R) * evsini
    t3 = vsini / (2. * np.pi * R) * eP
    esini = np.sqrt(t1**2 + t2**2 + t3**2) *u.rad

    # If inclincation is close to 90 degree
    # Use taylor expansion of d/dx(arcsin(x))
    # around x=1
    # to get uncertainty on inclination from sini
    if incl - np.pi / 2 * u.rad < 1e-3 * u.rad:
        eincl = 1 / np.sqrt(2) / np.sqrt(1 + sini) * esini
    # At inclinations lower than that calculate the
    # derivative directly
    else:
        eincl = 1 / np.sqrt(1- sini**2) * esini

    return (incl.to("deg"), eincl.to("deg"),
            sini.decompose().value, esini.decompose().value)
