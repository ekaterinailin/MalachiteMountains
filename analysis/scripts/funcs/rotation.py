"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This module contains functions used to 
calculate and fit rotation periods to lightcurves.

We are only testing find_period, the rest are
scripts we call in the notebooks.
"""


# basics
import pandas as pd
import numpy as np
from scipy import optimize
import astropy.units as u

# matplotlib
import matplotlib.pyplot as plt

# funcs
from .helper import fetch_lightcurve

# mcmc
import emcee
import time as Timestamp

# data management    
import os
import copy

CWD = os.getcwd()



def find_period(target, minfreq=1, maxfreq=15, plot=True, 
                save=True, path=CWD,
                custom=True, flc=None):
    """Find dominant periodic modulation using
    Lomb-Scargle periodogram.

    Parameters:
    -----------
    target : Series
        Description of the target.
    minfreq : float
        minimum modulation period in 1/day
    maxfreq : float
        maximum modulation period in 1/day
    plot : bool
        If True, will plot the periodogram
    save : bool
        If True, will save periodogram plot to file.
    path : str
        Path to file
    custom : True
        use custom extracted LCs
    flc : FlareLightCurve
        if custom is False, use this LC
        
    Return:
    -------
    period, frequency : astropy Quantities
        dominant modulation period in hours,
        and frequency in 1/day, respectively.
    """
        # Fetch light curve
    if custom==True:
        flck = fetch_lightcurve(target)
    else: 
        flck = flc
   
    # Use Lomb-Scargle periodogram implemented in lightkurve
    pg = flck.remove_nans().to_periodogram(freq_unit=1/u.d,
                                              maximum_frequency=maxfreq,
                                              minimum_frequency=minfreq)

    # Convert frequency to period
    period = (1 / pg.frequency_at_max_power).to("h")

    if plot==True:

        # Plot the periodogram
        pg.plot(label=f"{target.prefix} {target.ID}, S{target.QCS}, {target.SpT}V \n {period:.2f}")
        plt.xlim(minfreq, maxfreq)

        # Optionally save to file
        if save==True:
            plt.savefig(f"{path}/{target.ID}_{target.QCS}_periodogram.png",dpi=300)

    # Stdout
    print(f"{target.prefix} {target.ID} modulation period: ", period)

    return period, pg.frequency_at_max_power



def rotation_period_uncertainties_mcmc(ID, sector, target, res, step=50, 
                                       maxiterations=20, CWD="", tstamp=""):
    """
    MCMC fit wrapper for rotation period uncertainty estimates.
    
    Parameters:
    -----------
    ID : int
        TESS target ID
    sector: int
        TESS Sector
    target: pandas Series
        TESS target info
    res: 
    
    
    Return:
    -------
    MCMC sampler (emcee)
    """
    # Pick sector
    target.QCS = int(sector)
    print(target.QCS)
    print(f"{target.QCS:02d}")

    # Get light curve
    flcd = fetch_lightcurve(target, flux_type="PDCSAP_FLUX")

    
    # mask the flare
    flcd.flux[((flcd.time > target.view_start) & (flcd.time < target.view_stop))] = np.nan

    # only valid values allowed
    flcd = flcd[np.where(np.isfinite(flcd.flux))]
    
    # remove long-term trends
    flcd = flcd.flatten(window_length=1001)
    
    # define flux, time and error arrays
    time = flcd.time
    flux = flcd.flux
    flux_err = flcd.flux_err

    # pick the row for MCMC inits
    pick = (res.ID == ID) & (res.sector==sector)
    row = res.loc[pick, :].iloc[0]

    # calucate the median for MCMC inits
    meanflux = np.nanmedian(flux)

    # define inits
    inits = [row.period_h/24., row.rel_amplitude * meanflux, row.phase_offset,  row.lin_trend, row.offset_y]

    # wiggle inits
    pos = inits * (1. + 1e-4 * np.random.randn(32, len(inits)))

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = f"{CWD}/analysis/results/mcmc/rotation/{tstamp}_{target.ID}_S{sector:.0f}_MCMC.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(32, len(inits))
    args = (time, flux, flux_err)

    sampler = emcee.EnsembleSampler(32, len(inits), log_probability,
                                    args=args,backend=backend)

    # run chain and track time
    start = Timestamp.time()
    sampler.run_mcmc(pos, step, progress=True, store=True)
    end = Timestamp.time()
    multi_data_time = end - start
    print("MCMC took {0:.1f} seconds".format(multi_data_time))

    # get the chain
    rounde = 1
    while rounde < maxiterations:
        try:
            tau = sampler.get_autocorr_time()
            burnin = int(2 * np.max(tau))
            thin = int(0.5 * np.min(tau))

            samples = sampler.get_chain(burnin=burnin, thin=thin,flat=True)
            
            break
        except:
            sampler.run_mcmc(None, step, progress=True, store=True)
            rounde +=1
            
        samples = sampler.get_chain(flat=True)
        

    # set up results table
    columns = ["Prot_d", "rel_amplitude","phase_offset","lin_trend","offset_y"]
    resultframe = pd.DataFrame(data=samples,
                           columns=columns)

    # calculate upper and lower percentiles
    af = resultframe.quantile([.16,0.5,.84], axis=0)
    
    # each MCMC fit is a row in the final table
    series = (pd.Series(af.loc[.16,]).
     rename(lambda x: x+"_16").
     append(pd.Series(af.loc[.5,]).
            rename(lambda x: x+"_50")).
     append(pd.Series(af.loc[.84,]).
            rename(lambda x: x+"_84")).
     append(pd.Series({"tstamp":tstamp,
                       "ID":target.ID,
                       "QCS":sector,
                       "prefix":target.prefix,
                       "Prot_d_LS":row.period_h/24.,
                       "rel_amplitude_init":row.rel_amplitude, 
                       "phase_offset_init":row.phase_offset,
                       "lin_trend_init":row.lin_trend, 
                       "offset_y":row.offset_y,
                       "steps":resultframe.shape[0]//32
                       })).
     sort_index(ascending=True))
    
    # save to file
    with open(f"{CWD}/analysis/results/mcmc/rotation/mcmc_rotation_output.csv","a") as f:
        #Add more lines here
        pd.DataFrame(series).T.to_csv(f, index=False, header=False)
        
    return sampler

def get_period_get_amplitude(target, plot=False, save=False, plotmini=False):
    """Fit a sinusoidal modulation and
    subtract it from the flux.

    Parameters:
    -----------
    target : Series
        Description of the target.
    plot : bool
        If True, will plot the periodogram
    save : bool
        If True, will save periodogram plot to file.

    Return:
    -------
    time, subtracted flux, model, period:
    array, array, array, astropy.Quantity
    """
    def cosine(x, a, b, c, d, e):
        """cosine with a linear trend"""
        return a * np.cos(b * x + c) + d * x + e

    # Fetch light curve
    flck = fetch_lightcurve(target, flux_type="PDCSAP_FLUX")

    # Get the dominant modulation period
    period, mfp = find_period(target, save=False, plot=False, 
                              custom=target.origin, flc=flck)

    # pick only valid data points
    cond = np.invert(np.isnan(flck.time)) & np.invert(np.isnan(flck.flux))
    
    # non-linear least-squares (Levenberg-Marquardt):
    p, p_cov = optimize.curve_fit(cosine, flck.time[cond],
                                  flck.flux[cond],
                                  p0=[np.nanstd(flck.flux), 
                                      2 * np.pi * mfp.value,
                                      0, 0, np.nanmean(flck.flux)],
                                  method="lm")
    
    # simple cosine model 
    model = cosine(flck.time, p[0], p[1], p[2], p[3], p[4])
    subtracted_flux = np.nanmedian(flck.flux) + flck.flux - model

    # Calculate the relative amplitude of the oscillation
    rel_amplitude = p[0] / np.nanmedian(flck.flux)
    
    # return estimate on rotational variability
    return period, rel_amplitude, p



def cosine(x, a, b, c, d, e):
    """cosine with a linear trend"""
    return a * np.cos(b * x + c) + d * x + e


def logit(function):
    '''Make a probability distribution
    a log probability distribution.'''
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        np.seterr(divide='ignore') # ignore division by zero because you want to have the -np.inf results
        result = np.log(result)
        return result
    return wrapper


@logit
def uninformative_prior(rate, minrate, maxrate):
    '''Uninformative prior for the rates.
    Uniform within [minrate, maxrate].

    Parameters:
    -------------
    rate : float

    minrate, maxrate : float
        interval in which rate is constrained

    Return:
        Prior probability
    '''
    condition = ~(np.isfinite(maxrate) & np.isfinite(minrate))
    if ((maxrate < minrate) | condition):
        raise ValueError("maxrate must be >= minrate, <= maxrate and a finite value")
    if ((rate >= minrate) & (rate <= maxrate)):
        return 1. / (maxrate - minrate)
    else:
        return 0


def calculate_posterior_value_that_can_be_passed_to_mcmc(lp):
    '''Do some checks to make sure MCMC will work.'''
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        return -np.inf
    else:
        return lp


def log_prior(p,):
    """Uniform prior
    
    Paramters:
    ----------
    p - list with 5 elements
        cosine with linear trend parameters,
        see cosine()
    
    Return:
    --------
    log prior 
    """
    
   
    prior = (uninformative_prior(p[0], 0.05, 2.) +
             uninformative_prior(p[1], 0, 1e6) +
             uninformative_prior(p[2],  -1e6, 1e6) +
             uninformative_prior(p[3], -1e6, 1e6)  +
             uninformative_prior(p[4], 0, 1e8) )

    return calculate_posterior_value_that_can_be_passed_to_mcmc(prior)




def log_likelihood(p, time, flux, flux_err):
    """Log likelihood function assuming
    Gaussian uncertainties in the data points.
    and above several hundred counts
    
        
    Paramters:
    ----------
    p - list with 5 elements
        cosine with linear trend parameters,
        see cosine()
    time: numpy array
        time series
    flux: numpy array
        flux measurements
    flux_err: numpy array
        uncertainty in flux measurements
    """

    model = cosine(time, p[1], 2*np.pi/p[0], p[2],p[3],p[4])

    fr2 = flux_err**2
    val = -0.5 * np.sum((flux - model) ** 2 / fr2 + np.log(fr2))

        
    return val


def log_probability(p, time, flux, flux_err):
    """Posterior probability to pass to MCMC sampler.
            
    Paramters:
    ----------
    p - list with 5 elements
        cosine with linear trend parameters,
        see cosine()
    time: numpy array
        time series
    flux: numpy array
        flux measurements
    flux_err: numpy array
        uncertainty in flux measurements
    """
    lp = log_prior(p)

    if not np.isfinite(lp):
        return -np.inf
    
    try:
        ll = log_likelihood(p, time, flux, flux_err)
        
    except:
        return -np.inf
    
    if np.isnan(ll):
        return -np.inf
    
    return lp + ll
