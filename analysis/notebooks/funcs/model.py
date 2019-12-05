import numpy as np
import pandas as pd

from .funcs import no_nan_inf

import astropy.units as u
from astropy.constants import c, h, k_B, R_sun, L_sun

def model(phi, latitudes, longitudes, flare, inclination, phi0=0.):
    """Take a flare light curve and a rotating ensemble of latitudes
    and longitudes, and let it rotate. 
    
    Parameters:
    -----------
    phi :  N-array
        longitudes to evaluate the model at in rad
    latitudes : M-array
        latitudes of the spot grid points in rad
    longitudes : M-array
        longitudes of the spot grid points in rad
    flare : N-array
        flare shape
    inclination : float
        inclination in rad
    phi0 : float
        longitude facing the observer
        
    Return:
    --------
    lambert modifier for flux - N-array
    onoff day and night step function - N-array
    model light curve - N-array
    """
    
    if no_nan_inf([phi, latitudes, longitudes, flare, inclination, phi0]) == False:
        raise ValueError("One of your inputs in model() is or contains NaN or Inf.")
    
    # Calculate the offset:
    phi_ = phi-phi0
    
    # Check if the dimensions of the inputs are right
    l = len(latitudes)
    assert l == len(longitudes)
    assert len(phi_) == len(flare)
    # -----------------------------------------------
    
    # Get daylengths for all grid points 
    # and calculate day/night switches:
    Ds = daylength(latitudes, inclination)
    onoff = np.full((l,phi_.shape[0]),0)
    
    for i,d in enumerate(Ds):# How can I avoid this loop?
        onoff[i,:] = on_off(phi_-longitudes[1], d)
    #------------------------------------------------
        
    # Why isn't it possible to just fill an array here?
    # Refactor later ...
    # Anyways: Calculate the lambert modifier:
    latlon = np.concatenate((latitudes.reshape(l,1),
                             longitudes.reshape(l,1)),
                            axis=1)
    A = []
    for i, ll in enumerate(latlon):
        a = lambert(phi_-ll[1], inclination, ll[0])
        A.append(a)
    lamb = np.array(A)
    
    #--------------------------------------------------
    
    # Give intermediate results: lamb, onoff 
    # Also co-add all grid points and average them 
    # after folding with flare:
    return lamb, onoff, np.sum(lamb * onoff, axis=0) * flare / l

def dot_ensemble(lat, lon, radius, num_pts=1e6, s=30):
    """Create an ensemble of dots on a sphere.
    
    Method see: 
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    answered by CR Drost
    
    Parameters:
    -----------
    lat : float
        latitude of center of ensemble in rad
    lon : float
        longitude of center of ensemble in rad
    radius : float
        angular radius of the ensemble in deg
    num_pts : int 
        number of points used to generate the
        full sphere evenly covered with dots
        in a sunflower shape
    s : int
        s^2 points are generated if the patch is small
        enough to count as planar
    
    Return:
    -------
    latitudes, longitudes  -  np.arrays of dots
    that go into the ensemble.
    """
    if no_nan_inf([lat, lon, radius, num_pts]) == False:
        raise ValueError("One of your inputs in dot_ensemble is or contains NaN or Inf.")
        
    if radius > 1.5:
        
        # This is CR Drost's solution to the sunflower spiral:
        indices = np.arange(0, num_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices/num_pts) #latitude
        theta = np.pi * (1 + 5**0.5) * indices #longitude
        
        # Fold onto on sphere
        phi = np.pi / 2 - phi % (2 * np.pi)
        theta = theta % (np.pi * 2)
        
    elif radius <= 1.5:
        
        r = (radius * 3) / 180 * np.pi 
        phi = (np.full(s,1).reshape(s,1) * np.linspace(lat-r, lat+r, s).reshape(1,s)).reshape(1, s**2)[0,:]
        theta = ((np.full(s,1).reshape(1,s) * np.linspace(lon-r, lon+r, s).reshape(s,1))).reshape(1, s**2)[0,:]
        # Fold onto on sphere
        phi = phi % (2 * np.pi)
        theta = theta % (np.pi * 2)
        
#     import matplotlib.pyplot as plt
#     plt.scatter(phi,theta)
#     plt.scatter([lat],[lon])
    print(phi.shape)
    # Calculate the distance of the dots to the center of the ensemble
    gcs = great_circle_distance(lat, lon, phi, theta)
    
    # If distance is small enough, include in ensemble
    a = np.where(gcs < (radius * np.pi / 180))[0]
    print(len(a))
    return phi[a], theta[a]

def great_circle_distance(a, la, b, lb):
    """Calcultate the angular distance on
    a great circle than runs through two points
    on a globe at 
    (latitude, longitude) 
    (a,la) and 
    (b, lb).
    
    See also: https://en.wikipedia.org/wiki/Great-circle_distance
    
    Parameters:
    ------------
    a, la, b, lb : float
     latitudes and longitudes in rad
    
    Return:
    -------
    float - angular distance on a globe in rad
    """
    if no_nan_inf([a, la, b, lb]) == False:
        raise ValueError("One of your inputs is or contains NaN or Inf.")
        
    return np.arccos(np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(la-lb))

def lambert(phi, i, l, phi0=0.):
    """Calculate Lambert's law of geometric
    brightness modulation (prop. cos(incident angle))
    from known stellar inclination, and the 
    spots latitude and longitudes.
    
    Parameters:
    -----------
    phi : array or float
        longitudes
    i : float
        inlcination in rad
    l : float
        latitude in rad
        
    Return:
    -------
    Array of values between 0 and 1 that define the 
    fraction of the flux that we would receive from 
    a point at the center of the stellar disk.
    
    Wikipedia is great:
    https://en.wikipedia.org/wiki/Great-circle_distance
    https://en.wikipedia.org/wiki/Lambert%27s_cosine_law
    """
    if no_nan_inf([l,i,phi,phi0]) == False:
        raise ValueError("One of your inputs is or contains NaN or Inf.")
    return np.sin(l) * np.cos(i) + np.cos(l) * np.sin(i) * np.cos(phi-phi0)

def on_off(phi, daylength, phi0=0.):
    """Calculate the visibility step function
    of a point on a rotating sphere as a function of
    longitude phi.
    
    phi0 is facing the observer.
    
    Parameters:
    ------------
    phi : array
        longitudes
    daylength : float
        fraction of rotation period (0,1)
    phi0 : float 
        longitude facing the observer
        default 0, range [0,2pi]
    
    Return:
    -------
    array of 1 and 0 if phi is an array, 1=visible, 0=hidden
    else bool, True=visible, False=hidden
    
    """
    def condition(phi,phi0,daylength):
        
        # condition for being hidden on the back of the star
        if daylength==0.:
            return True
        else:
            return (((phi-phi0)%(2*np.pi) > daylength*np.pi) &
                ((phi-phi0)%(2*np.pi) < (2-daylength)*np.pi))
    
    if (np.isnan(daylength) | np.isnan(phi0) | (not np.isfinite(phi0)) | (not np.isfinite(daylength))):
        raise ValueError("Daylength or phi0 is NaN or not finite.")
    
    if isinstance(phi, np.ndarray):
        
        if ((np.isnan(phi).any()) | (not np.isfinite(phi).any())):
            raise ValueError("One phi value is NaN or not finite")
        
        # everything is visible by default
        res = np.full(phi.shape[0],1) 

        # if longitude is on the night side, set visibility to 0:
        res[condition(phi,phi0,daylength)] = 0
        return res
        
    elif ((isinstance(phi, float)) | (isinstance(phi, int))):
        
        if ((np.isnan(phi)) | (not np.isfinite(phi))):
            raise ValueError("Your phi value is NaN")
        # True if visible
        return not condition(phi,phi0,daylength)
    
    
    else:
        raise ValueError("Phi must be float, int, or an array of either.")

def daylength(l, i, P=1.):
    """Determine the day length, as in here: 
    http://www.math.uni-kiel.de/geometrie/klein/mpss13/do2706.pdf
    
    If P is not specified, the daylength is measured in
    rotation periods.
    
    Parameters:
    ------------
    l : array
        longitude in rad
    i : float
        inclination in rad
    P : float
        rotation period, default is 1.
        
    Return:
    -------
    float daylength in the same units as the rotation period
    """
    
    def formula(l,i):
        return np.arccos(-np.tan(l) * np.tan(np.pi/2-i)) / np.pi
    
    if ((i > np.pi/2) | (i < 0)):
        raise ValueError("Inclination must be in [0,pi/2]")
    
    if isinstance(l,np.ndarray):
        
        if np.isnan(i) | np.isnan(l).any():
            raise ValueError("Inclination or some latitude is NaN.")
            
        elif ((l > np.pi/2).any() | (l < -np.pi/2).any()):
            raise ValueError("Latitude must be in [-pi/2,pi/2]")
        
        res = formula(l,i)
        res[l>=i] = P
        res[np.abs(l) >=i] = 0
        

    
        return res * P
    
    elif ((isinstance(l, float)) | (isinstance(l, int))):
        
        if np.isnan(i) | np.isnan(l):
            raise ValueError("Inclination or latitude is NaN.")
            
        elif (l > np.pi/2) | (l < -np.pi/2):
            raise ValueError("Latitude must be in [-pi/2,pi/2]")
            
        if l >= i:
            return P
        
        elif ((l<0) & (np.abs(l) >=i)):
            return 0
        
        else:
            return formula(l,i) * P



def black_body_spectrum(wav,t):
    """Takes an array of wavelengths and
    a temperature and produces an array
    of fluxes from a thermal spectrum
    
    Parameters:
    -----------
    wav : Astropy array
        wavenlength array
    t : float
        effective temperature in Kelvin
    """
    t = t * u.K # set unit to Kelvin
    
    return (( (2 * np.pi * h * c**2) / (wav**5) / (np.exp( (h * c) / (wav * k_B * t) ) - 1)) 
            .to("erg/s/cm**3")) #simplify the units


def calculate_relative_flare_area(dist, rad, qflux, amp, mission, flaret=1e4):
    """Get the flare area in rel. unit
    
    Parameters:
    -----------
    dist : float
        distance to target in parsecs
    rad : float
        radius in solar radii
    qflux : float
        quiescent flux in erg/s/cm^2
    amp : float
        (0,inf) flare amplitude in rel. flux
    mission : string
        TESS or Kepler
    flaret : float
        flare black body temperature, default 10kK
    """
    # Give units to everything:
    dcm = dist * u.pc
    rcm = rad * R_sun
    qflux = qflux * u.erg/u.s/u.cm**2
    
    # Read in response curve:
    response_curve_path = {"TESS":"TESS.txt",
                           "Kepler":"kepler_lowres.txt"}
    df = pd.read_csv(f"static/{response_curve_path[mission]}",
                     delimiter="\s+", skiprows=8)
    df = df.sort_values(by="nm", ascending=True)
    rwav = (df.nm * 10).values * u.angstrom #convert to angstroms
    rres = (df.response).values
    
    # create an array to upsample the filter curve to
    w = np.arange(3000,13001) * u.angstrom

    # interpolate thermal spectrum onto response 
    # curve wavelength array, then sum up
    # flux times response curve:

    # Generate a thermal spectrum at the flare 
    # temperature over an array of wavelength w:
    thermf = black_body_spectrum(w, flaret) 
    
    # Interpolate response from rwav to w:
    rres = np.interp(w,rwav,rres)
    
    # Integrating the flux of the thermal 
    # spectrum times the response curve over wavelength:
    calflareflux = np.trapz(thermf * rres, x=w)
    
    return (((amp * qflux) / (calflareflux)) * (dcm / rcm)**2.).decompose()