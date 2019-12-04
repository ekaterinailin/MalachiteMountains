import numpy as np

from .funcs import no_nan_inf

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

def dot_ensemble(lat, lon, radius, num_pts=1e5):
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
    
    Return:
    -------
    latitudes, longitudes  -  np.arrays of dots
    that go into the ensemble.
    """
    if no_nan_inf([lat, lon, radius, num_pts]) == False:
        raise ValueError("One of your inputs in dot_ensemble is or contains NaN or Inf.")
        
    # This is CR Drost's solution to the sunflower spiral:
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices/num_pts) #latitude
    theta = np.pi * (1 + 5**0.5) * indices #longitue
    
    # Fold onto on sphere
    phi = np.pi / 2 - phi % (2 * np.pi)
    theta = theta % (np.pi * 2)
    
    # Calculate the distance of the dots to the center of the ensemble
    gcs = great_circle_distance(lat, lon, phi, theta)
    
    # If distance is small enough, include in ensemble
    a = np.where(gcs < (radius * np.pi / 180))[0]
    
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