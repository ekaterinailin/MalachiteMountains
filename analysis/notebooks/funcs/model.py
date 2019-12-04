import numpy as np

from .funcs import no_nan_inf

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