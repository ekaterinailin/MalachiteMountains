"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This script fetched LC and stellar properties, 
calculates inclination using Prot and vsini, 
and saves a plot.

The main science function here is 
`helper.calculate_inclination`
"""

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import matplotlib 
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)

from funcs.helper import calculate_inclination

# Create a time stamp for this run
import time
tstamp = time.strftime("%d_%m_%Y", time.localtime())

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


if __name__ == "__main__":

    
    # define paths    
    CWD = "/".join(os.getcwd().split("/")[:-2])
    paperpath = "/home/ekaterina/Documents/002_writing/multiperiod-flares-draft"

    # get data on LCs and stars
    lcs = pd.read_csv(f"{CWD}/data/summary/lcsi.csv")#vsin
    props = pd.read_csv(f"{CWD}/data/summary/properties.csv")

    # table engineering
    lcs = lcs.merge(props, left_on="ID", right_on="id")
    lcs = lcs.set_index("ID")
    lcs = lcs.dropna(subset=["vsini_kms"])
    lcs = lcs[lcs.index!=230120143] # Remove leftover row
    


    # Plot priors on inclinations
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
    for l, target in lcs.iterrows():
       # print(target)
        # Get inclinations
        i_mu, i_sigma, sini, esini = calculate_inclination(target)

        # Plot result as sini and i
        x = np.linspace(0, 1, 200)
        axs[1].plot(x, gaussian(x, sini, esini))

        x = np.linspace(0,np.pi/2,200)
        axs[0].plot(x / np.pi * 180.,
                    gaussian(x, i_mu.to("rad").value, i_sigma.to("rad").value),
                    label=f"{target.prefix} {target.ID}")

    # Layout figure
    axs[0].set_xlabel(r"$i$ [deg]", fontsize=14)
    axs[1].set_xlabel(r"$\sin i$", fontsize=14)
    axs[0].legend(loc=2, frameon=False)

    plt.savefig(f"{CWD}/data/summary/{tstamp}_inclination.png",dpi=300)

    # Add rows as input for MCMC later
    lcs["i_mu"] = lcs.apply(lambda x: calculate_inclination(x)[0].to("rad").value, axis=1)
    lcs["i_sigma"] = lcs.apply(lambda x: calculate_inclination(x)[1].to("rad").value, axis=1)

    print("\n Output table: \n ---------------- \n")
    print(lcs.T)
    # Output to file
    lcs.to_csv(f"{CWD}/data/summary/{tstamp}_lcsi.csv")
