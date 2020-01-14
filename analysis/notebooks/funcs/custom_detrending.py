import numpy as np
import pandas as pd

import copy

from collections import defaultdict

import astropy.units as u

from scipy.interpolate import UnivariateSpline
from scipy import optimize
from scipy.fftpack import fft

from altaipony.altai import find_iterative_median
from altaipony.utils import sigma_clip

import matplotlib.pyplot as plt

def custom_detrending(flc):
    """Wrapper"""
    f = flc.flux[np.isfinite(flc.flux)]
   
    if np.abs(f[0]-f[-1])/np.median(f) > .2:
        print("Do a coarse spline interpolation to remove trends.")
        flc = fit_spline(flc, spline_coarseness=12)
        flc.flux[:] = flc.detrended_flux[:]
    
    # Iteratively remove fast sines with Periods of 0.1 to 2 day periods (the very fast rotators)
    #print(1, flc.flux.shape)
    flc = iteratively_remove_sines(flc)
    flc.flux[:] = flc.detrended_flux[:]
    #print(2, flc.flux.shape)
    # remove some rolling medians on timescales of 3,33 to 10 hours time scales
    flc.flux[:] = flc.flux - pd.Series(flc.flux).rolling(300, center=True).median() + np.nanmedian(flc.flux)#15h
    #flc.flux[:] = flc.flux - pd.Series(flc.flux).rolling(200, min_periods=1).median() + np.nanmedian(flc.flux)
    #flc.flux[:] = flc.flux - pd.Series(flc.flux).rolling(100, min_periods=1).median() + np.nanmedian(flc.flux)
    #print(3, flc.flux.shape)
    # Determine the window length for the SavGol filter for each continuous observation gap
    w = search_gaps_for_window_length(flc)
    flc = flc[np.isfinite(flc.flux)]
    #print(4, flc.flux.shape)
    print("Window lengths: ", w)
    # Use lightkurve's SavGol filter while padding outliers with 25 data points around the outliers/flare candidates
    
    flc = flc.detrend("savgol", window_length=w, pad=25)#previously 25
    flc.flux[:] = flc.detrended_flux[:]
    #print(5, flc.flux.shape)
    print("Do last SavGol round.")
    
    # After filtering, always use a 2.5 hour window to remove the remaining 
    
    flcd = flc.detrend("savgol", window_length=75, pad=25)#previously 25
    #print(6,flcd.flux.shape)
    # Determine the noise properties with a rolling std, padding masked outliers/candidates
    flcd = refine_detrended_flux_err(flcd, mask_pos_outliers_sigma=1.5, 
                                     std_rolling_window_length=15, pad=25)
    #print(7, flcd.flux.shape)
    return flcd


def refine_detrended_flux_err(flcd, mask_pos_outliers_sigma=2.5, 
                              std_rolling_window_length=15, pad=25):#previously 3 >> 25
    """Attempt to recover a good estimate of the ligh curve noise.
    Start out from a simple standard deviation of the flux.
    Then filter out outliers above `mask_pos_outliers_sigma`.
    Apply rolling window standard deviation on the filtered array.
    Calculate a mean standard deviation from the result.
    Fill in this mean into the masked values.
    
    Parameters:
    -----------
    flcd : de-trended FlareLightCurve
    
    mask_pos_outliers_sigma : float
        sigma value above which to mask positive outliers
    std_rolling_window_length : int
        rolling window length for standard deviation calculation
    pad : int
        How many values to pad-mask around positive outliers.
    
    Return:
    --------
    FlareLightCurve with refined `detrended_flux_err` attribute.
    
    """
    
    # start with a first approximation to std
    flcd.detrended_flux_err[:] =  np.nanstd(flcd.detrended_flux)
    
    # and refine it:
    flcd = find_iterative_median(flcd)
    
    filtered = copy.deepcopy(flcd.detrended_flux)
    
    # mask strong positive outliers so that they don't add to std
    filtered[flcd.detrended_flux - flcd.it_med > mask_pos_outliers_sigma * flcd.detrended_flux_err] = np.nan
    
    # apply rolling window std
    flcd.detrended_flux_err[:] = pd.Series(filtered).rolling(std_rolling_window_length, min_periods=1).std()
  
    # set std to mean value if calculation fails to inf
    meanstd = np.nanmean(flcd.detrended_flux_err)
    
    # pad the excluded values not to create spikes of high error around flares
    isin = np.invert(np.isfinite(flcd.detrended_flux_err))
    x = np.where(isin)[0]
    for i in range(-pad,pad+1):
        y = x+i
        y[np.where(y>len(isin)-1)] = len(isin)-1
        isin[y] = True
            
    x = np.where(isin)[0]
    flcd.detrended_flux_err[x] = meanstd
 
    return flcd


def search_gaps_for_window_length(flc):
    """Search continuous light curve chunks for
    appropriate window_length to apply to 
    SavGol filter.
    
    Parameters:
    ------------
    flc : FlareLightCurve
    
    Return:
    -------
    list of odd ints
    """
    flc = flc[np.where(np.isfinite(flc.flux))]
    flc = flc.find_gaps()
    wls = []
    for le,ri in flc.gaps:
        wls.append(select_window_length(flc.flux[le:ri]))
    
    return wls


def select_window_length(flux):
    """Performs an FFT and defines a window
    length that is smaller than the most prominent
    frequency.
    
    Parameters:
    -----------
    flux : array
        
    Return:
    --------
    odd int
    """
    #normalize flux and FFT it:
    yf = fft(flux/np.nanmean(flux)-1.)
    
    maxfreq = len(yf)//5
    minfreq = 1

    # choose window length
    w = np.rint(len(yf)/(minfreq+np.argmax(yf[minfreq:maxfreq]))/3)

    # w must be odd
    if w%2==0:
        w += 1
    # if w is too large don't do it at all
    if w > len(yf)//2:
        return None
    else:
        return int(max(w,75))





