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
    flc = iteratively_remove_sines(flc)
    flc.flux[:] = flc.detrended_flux[:]
    
    # remove some rolling medians on a 10 hours time scale
    flc.flux[:] = flc.flux - pd.Series(flc.flux).rolling(300, center=True).median() + np.nanmedian(flc.flux)#15h
    
    # Determine the window length for the SavGol filter for each continuous observation gap
    flc = find_iterative_median(flc)
    w = search_gaps_for_window_length(flc)
    flc = flc[np.isfinite(flc.flux)]

    # Use lightkurve's SavGol filter while padding outliers with 25 data points around the outliers/flare candidates
    flc = flc.detrend("savgol", window_length=w, pad=25)
    flc.flux[:] = flc.detrended_flux[:]
    
    # After filtering, always use a 2.5 hour window to remove the remaining 
    flcd = flc.detrend("savgol", window_length=75, pad=25)
    
    # Determine the noise properties with a rolling std, padding masked outliers/candidates
    flcd = refine_detrended_flux_err(flcd, mask_pos_outliers_sigma=1.5, 
                                     std_rolling_window_length=15, pad=25)
    return flcd


def refine_detrended_flux_err(flcd, mask_pos_outliers_sigma=2.5, 
                              std_rolling_window_length=15, pad=25):
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
    for i in range(-pad, pad+1):
        y = x + i
        y[np.where(y > len(isin) - 1)] = len(isin) - 1
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
    
    maxfreq = len(yf) // 5
    minfreq = 1

    # choose window length
    w = np.rint(len(yf) / (minfreq + np.argmax(yf[minfreq:maxfreq])) / 3)

    # w must be odd
    if w%2==0:
        w += 1
        
    # if w is too large don't do it at all
    if w > len(yf) // 2:
        return None
    else:
        return int(max(w, 75))


def fit_spline(flc, spline_coarseness=12, spline_order=3):
    """Do a spline fit on a coarse sampling of data points.
    
    Parameters:
    ------------
    flc : FlareLightCurve
    
    spline_coarseness : int
        Do not spline fit every data point but use the 
        average value of spline_coarseness hours 
    spline_order : int
        order of spline fit
        
    Return:
    --------
    FlareLightCurve with new flux attribute
    """
    flc = flc[np.where(np.isfinite(flc.flux))]
    flcp = copy.deepcopy(flc)

    flcp = flcp.find_gaps()
    flux_med = np.nanmedian(flcp.flux)
    n = int(np.rint(spline_coarseness/ 24 / (flcp.time[1] - flcp.time[0])))
    k = spline_order
    #do a first round
    for le, ri in flcp.gaps:

        rip = flcp.flux[le:ri].shape[0] + le
        t, f = np.zeros((rip - le)//n+2), np.zeros((rip - le)//n+2)
        t[1:-1] = np.mean(flcp.time[le:rip - (rip - le)%n].reshape((rip - le)//n, n), axis=1)
        f[1:-1] =  np.median(flcp.flux[le:rip - (rip - le)%n].reshape((rip - le)//n, n), axis=1)
        t[0], t[-1] = flcp.time[le], flcp.time[rip-1]
        f[0], f[-1] = flcp.flux[le], flcp.flux[rip-1]
        p3 = UnivariateSpline(t, f, k=k)
        flcp.detrended_flux[le:ri] = flcp.flux[le:ri] - p3(flcp.time[le:ri]) + flux_med
        

    
    return flcp


def iteratively_remove_sines(flcd, freq_unit=1/u.day, 
                             maximum_frequency=10, 
                             minimum_frequency=0.5):
    def cosine(x, a, b, c, d):
        return a * np.cos(b * x + c) + d

    snr = 3
    flct = copy.deepcopy(flcd)
    for le, ri in flct.find_gaps().gaps:
        flc = copy.deepcopy(flct[le:ri])
        flc = find_iterative_median(flc)
        pg = flc.remove_nans().to_periodogram(freq_unit=freq_unit,
                                          maximum_frequency=maximum_frequency,
                                          minimum_frequency=minimum_frequency)
        snr = pg.flatten().max_power
    #    print("Found peak in periodogram at ", pg.frequency_at_max_power)
    #    print("SNR at ", snr)
        while snr > 2.5:
            pg = flc.remove_nans().to_periodogram(freq_unit=freq_unit,
                                                  maximum_frequency=maximum_frequency,
                                                  minimum_frequency=minimum_frequency)
            
            cond = np.invert(np.isnan(flc.time)) & np.invert(np.isnan(flc.flux)) 
            p, p_cov = optimize.curve_fit(cosine, flc.time[cond], flc.flux[cond],
                                          p0=[np.nanstd(flc.flux),
                                          2*np.pi*pg.frequency_at_max_power.value,
                                          0, np.nanmean(flc.flux)])
            flc.flux = np.nanmean(flc.flux) + flc.flux-cosine(flc.time, p[0], p[1], p[2], p[3])
           # print(snr)
            snr = pg.flatten().max_power
          #  print(snr)

        flcd.detrended_flux[le:ri] = flc.flux
    return flcd
