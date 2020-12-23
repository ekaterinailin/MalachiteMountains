# Python3.6
# Functions used mostly in NB4

import os
import copy

import numpy as np
import pandas as pd

from .helper import fetch_lightcurve

import astropy.units as u

from scipy import optimize

CWD = os.getcwd()

import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt



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


