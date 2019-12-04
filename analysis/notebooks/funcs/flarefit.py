from altaipony.fakeflares import aflare

from scipy import optimize


def aflare2(time, t0, fwhm, ampl, median, upsample=True, uptime=10):
    """Modified `aflare` classic flare shape.
    Originally from Davenport et al. 2014, but
    adding the median flux value to return the 
    full flux. 
    
    See `aflare` in: 
    https://github.com/ekaterinailin/AltaiPony/blob/master/altaipony/fakeflares.py
    
    Upsamples the flare model by a factor of 10 
    per default and samples back down again
    to improve the energy estimate.
    
    
    """
    return (aflare(time, t0, fwhm*2, ampl, 
                   upsample=upsample, uptime=uptime) + median)