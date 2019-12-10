import numpy as np

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