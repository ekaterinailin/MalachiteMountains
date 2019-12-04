import numpy as np

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