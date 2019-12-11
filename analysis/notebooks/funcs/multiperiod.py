# Python3.6
# Functions used mostly in NB4

import os
import copy

import numpy as np
import pandas as pd

from .custom_detrending import custom_detrending, iteratively_remove_sines
from .funcs import read_custom_aperture_lc

import astropy.units as u

from scipy import optimize

CWD = "/".join(os.getcwd().split("/")[:-2])

import warnings
warnings.simplefilter("ignore")

import matplotlib.pyplot as plt

def fetch_lightcurve(target):
    """Read in light curve from file.

    Parameters:
    -----------
    target: Series
        Description of the target.
    """
    path = (f"{CWD}/data/lcs/{target.ID}_{target.QCS:02d}_" \
            f"{target.mission}_{target.typ}_{target.origin}.fits")

    flc = read_custom_aperture_lc(path, mission=target.h_mission,
                                  mode="LC", typ=target.origin,
                                  TIC=target.ID, sector=target.QCS)
    return flc

def show_flare(target, save=False):
    """Get light curve and plot it."""

    # Fetch light curve
    flc = fetch_lightcurve(target)

    plt.figure(figsize=(15,6))

    # Plot the light curve
    plt.plot(flc.time, flc.flux, c="grey",label=f"{target.prefix} {target.ID}, S{target.QCS}, {target.SpT}V")

    # Fix plot limits
    plt.xlim(target.view_start,target.view_stop)
    plt.ylim(target.view_min,target.view_max)

    # Layout
    plt.xlabel(f"time [BJD-{int(target.BJDoff)}]",fontsize=14)
    plt.ylabel("flux [e$^{-}]$",fontsize=14)
    plt.legend(fontsize=14)

    # Saving optional
    if save==True:
        plt.savefig(f"{CWD}/analysis/plots/{target.ID}_{target.QCS:02d}_lightcurve.png",dpi=300)


def find_period(target, minfreq=2, maxfreq=10, plot=True, save=True):
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

    Return:
    -------
    period, frequency : astropy Quantities
        dominant modulation period in hours,
        and frequency in 1/day, respectively.
    """
    # Fetch light curve
    flc = fetch_lightcurve(target)

    # Use Lomb-Scargle periodogram implemented in lightkurve
    pg = flc.remove_nans().to_periodogram(freq_unit=1/u.d,
                                              maximum_frequency=maxfreq,
                                              minimum_frequency=minfreq)

    # Convert frequency to period
    period = (1 / pg.frequency_at_max_power).to("h")

    if plot==True:

        # Plot the periodogram
        pg.plot(label=f"{target.prefix} {target.ID}, S{target.QCS}, {target.SpT}V")
        plt.xlim(minfreq, maxfreq)

        # Optionally save to file
        if save==True:
            plt.savefig(f"{CWD}/analysis/plots/{target.ID}_{target.QCS}_periodogram.png",dpi=300)

    # Stdout
    print(f"{target.prefix} {target.ID} modulation period: ", period)

    return period, pg.frequency_at_max_power

def remove_sinusoidal(target, plot=True, save=False):
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
    flck = fetch_lightcurve(target)

    # Get the dominant modulation period
    period, mfp = find_period(target, save=False, plot=False)

    # Optimize for the model parameters using
    # non-linear least-squares (Levenberg-Marquardt):
    cond = np.invert(np.isnan(flck.time)) & np.invert(np.isnan(flck.flux))
    p, p_cov = optimize.curve_fit(cosine, flck.time[cond],
                                  flck.flux[cond],
                                  p0=[np.nanstd(flck.flux), 2 * np.pi * mfp.value,
                                      0, 0, np.nanmean(flck.flux)],
                                  method="lm")
    model = cosine(flck.time, p[0], p[1], p[2], p[3], p[4])

    # Calculate the relative amplitude of the oscillation
    rel_amplitude = p[0] / np.nanmedian(flck.flux)
    print(f"Relative amplitude of modulation: {rel_amplitude:.1e}")

    # Plot the subtracted light curve
    if plot==True:

        plt.figure(figsize=(15,6))

        # Plot the original LC
        plt.plot(flck.time, flck.flux,c="grey",
                 label=(f"{target.prefix} {target.ID},"\
                        f" S{target.QCS}, {target.SpT}V"))

        # Overplot the fitted function
        plt.plot(flck.time, model,
                 c="navy",label=period)

        # Plot the flux with the model subtracted
        offset = (target.view_max - target.view_min) * .3
        subtracted_flux = np.nanmedian(flck.flux) + flck.flux - model
        plt.plot(flck.time, offset + subtracted_flux,
                 c="r",label="periodic signal subtracted")

        # Overplot the median flux value on the subtracted LC
        plt.plot(flck.time, offset + np.full(len(flck.time),np.nanmedian(flck.flux)), c="k")

        #Layout
        plt.xlim(target.view_start,target.view_stop)
        plt.ylim(target.view_min,target.view_max+offset)
        plt.xlabel(f"time [BJD-{int(target.BJDoff)}]",fontsize=14)
        plt.ylabel("flux [e$^{-}]$",fontsize=14)
        plt.legend()

        # Save optionally
        if save==True:
            plt.savefig(f"{CWD}/analysis/plots/{target.ID}_s{target.QCS}_sinusoidal.png",dpi=300)

    return flck.time, subtracted_flux, model, period