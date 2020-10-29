"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This script takes the converted_mcmc_sample.csv
files from the MCMC fitting, add radius of AR 
(in degrees and as fraction of stellar surface),
flare ED, flare energy, and write out a new table
with percentiles for each posterior distribution.

If AR, flare ED, and flare energy were not in the
read file, add them and save the result to save 
computation time.
"""

import numpy as np
import pandas as pd

import os

import astropy.units as u
from astropy.constants import R_sun

from funcs.model import calculate_angular_radius, calculate_ED

import matplotlib 
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14) 

font = {'family' : 'courier',
        'weight' : 'normal',
        'size'   : 14}

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

import time



style = {'277539431' : ["b","solid"],
         '100004076' : ["orange","solid"],
         '44984200' : ["r","solid"],
         '44984200a' : ["r","dashed"],
         '44984200b' : ["r","dashed"],
         '237880881a' : ["grey","dotted"],
         '237880881b' : ["grey","dashdot"],
         '452922110':["orange","solid"]}

def write_mcmc_output(resultframe, tstamp, ID, suffix, CWD, metatstamp):
    af = resultframe.quantile([.16,0.5,.84], axis=0)

    series = (pd.Series(af.loc[.16,]).
     rename(lambda x: x+"_16").
     append(pd.Series(af.loc[.5,]).
            rename(lambda x: x+"_50")).
     append(pd.Series(af.loc[.84,]).
            rename(lambda x: x+"_84")).
     append(pd.Series({"tstamp":tstamp,
                       "ID":ID,
                       "suffix":suffix,
                       "color":style[str(ID)+suffix][0],
                       "linestyle":style[str(ID)+suffix][1]})).
     sort_index(ascending=True))

    with open(f"{CWD}/analysis/results/mcmc/mcmcoutput.csv","a") as f:
        #Add more lines here
        pd.DataFrame(series).T.to_csv(f, index=False)
        
    with open(f"{CWD}/analysis/results/mcmc/{metatstamp}_mcmcoutput.csv","a") as f:
        #Add more lines here
        pd.DataFrame(series).T.to_csv(f, index=False)

def write_meta_mcmc(CWD, tstamp, ID, burnin, steps, walkers, ndim):
    with open(f"{CWD}/analysis/results/mcmc/mcmc_meta.csv","a") as f:
        out = f"{tstamp},{ID},{burnin},{steps},{walkers},{ndim}"
        firstout = f"date,ID,burnin,steps,walkers,ndim"
        out += "\n"
        firstout += "\n"
        f.write(firstout)
        f.write(out)
        
        
def get_rad_r_and_plot(ID, resultframe, Fth, qlum, R, CWD, tstamp):
    if "rad_rsun" not in resultframe.columns:
        g = lambda x: calculate_angular_radius(Fth, x.a, qlum, R)
        resultframe["rad_rsun"] = resultframe.apply(g, axis=1)

    plt.hist(resultframe.rad_rsun.values, bins=100, 
             histtype="step", color="k", linewidth=2);
    plt.xlabel("angular radius [deg]")
    plt.savefig(f"{CWD}/analysis/plots/mcmc/"
                f"{tstamp}_{ID}_active_region_radii.png",
                dpi=300)
    return resultframe

def get_frac_area(ID, resultframe, Fth, qlum, R, CWD, tstamp):
    if "rad_rsun" not in resultframe.columns:
        g = lambda x: calculate_angular_radius(Fth, x.a, qlum, R)
        resultframe["rad_rsun"] = resultframe.apply(g, axis=1)

    resultframe["frac_area"] = np.sin(resultframe.rad_rsun / 180 * np.pi / 2.)**2
    return resultframe


def get_ED_and_plot(ID, resultframe, lc, CWD, tstamp, suffix):
    
    if "ED_s" not in resultframe.columns:
        g = lambda x: calculate_ED(lc.t.values, x.t0_d, x.fwhm1_d,x.fwhm2_d, x.a)
        resultframe["ED_s"] = resultframe.apply(g, axis=1)

    plt.hist(resultframe.ED_s.values, bins=200, 
         histtype="step", color="k", linewidth=2);
    plt.xlabel("ED [s]")
    plt.savefig(f"{CWD}/analysis/plots/mcmc/"
                f"{tstamp}_{ID}{suffix}_flare_ED.png",
                dpi=300)
    return resultframe
    

def get_E_and_plot(ID, resultframe, lc, qlum, CWD, tstamp, suffix):
    
    if "ED_s" not in resultframe.columns:
        get_ED_and_plot(resultframe, lc, CWD)
        
    if "Eflare_erg" not in resultframe.columns:
        g = lambda x: x.ED_s * qlum.value
        resultframe["Eflare_erg"] = resultframe.apply(g, axis=1)
   
    hist, bins, p = plt.hist(resultframe.Eflare_erg.values, bins=200, 
         histtype="step", color="k", linewidth=2);

    plt.xlabel(r"$E_f$ [erg]")
    plt.savefig(f"{CWD}/analysis/plots/mcmc/"
                f"{tstamp}_{ID}{suffix}_flare_E.png",
                dpi=300)
    return resultframe



if __name__ == "__main__":
    
    # time stamp for backup
    metatstamp = time.strftime("%d_%m_%Y", time.localtime())

    # Datasets we analysed 
    datasets = [#(237880881, "a", "03_10_2020_11_29"),
                #(237880881, "b", "03_10_2020_11_29"),
                #(100004076, "", "13_10_2020_10_43"),
                # (452922110, "", "24_10_2020_18_50"),
                    (452922110, "", "29_10_2020_10_58"),
              #  (277539431, "", "03_10_2020_11_30"),
#                 (277539431, "", "08_07_2020_11_48"),
#                 (237880881, "a", "11_02_2020_10_07"),
#                 (237880881, "b", "11_02_2020_10_07"),
                #(44984200, "", "02_08_2020_17_44"),
                #(44984200, "a", "04_08_2020_13_40"),
                #(44984200, "b", "04_08_2020_13_40"),
#                 (100004076, "", "18_08_2020_15_38"),
                ]

    for dataset in range(1):
        # What dataset do you want to analyse?
        ID, suffix, tstamp = datasets[dataset]

        # Pick up the input parameters
        CWD = "/".join(os.getcwd().split("/")[:-2])
        inits = pd.read_csv(f"{CWD}/data/summary/inits_decoupled.csv")
        inits = inits[(inits.ID.astype(str) == str(ID)+suffix) &
                      (inits.tstamp==tstamp)].iloc[0]

        qlum = inits.qlum_erg_s * u.erg / u.s
        Fth = inits.Fth * u.erg / u.cm**2 / u.s
        median = inits['median']
        R = inits.R_Rsun * R_sun

        # Pick up the LC
        lc = pd.read_csv(f"{CWD}/data/lcs/{tstamp}_{ID}.csv")

        # Load MCMC chain
        resultframe = pd.read_csv(f"{CWD}/analysis/results/mcmc/"
                                  f"{tstamp}_{ID}{suffix}_converted_mcmc_sample.csv")
        old = resultframe.shape[1]

        if 'Unnamed: 0' in resultframe.columns:
            resultframe = resultframe.drop('Unnamed: 0', axis=1)
        print(resultframe.columns)

        # CALCULATE RADIUS OF AR, FLARE ED, and FLARE E
        # ----------------------------------------------

        # Get radius of AR distribution and plot a histogram
        resultframe = get_rad_r_and_plot(ID, resultframe, Fth, qlum, R, CWD, tstamp)
        
        # Convert radius to fraction of stellar surface
        get_frac_area(ID, resultframe, Fth, qlum, R, CWD, tstamp)

        # Get ED distribution and plot a histogram
        resultframe = get_ED_and_plot(ID, resultframe, lc, CWD, tstamp, suffix)

        # Get E distribution and plot a histogram
        resultframe = get_E_and_plot(ID, resultframe, lc, qlum, CWD, tstamp, suffix)

        # CALCULATE PERCENTILES FOR ALL MCMC OUTPUTS AND THE NEW ONES
        # ----------------------------------------------
        print(resultframe.columns)

        # Write percentiles to table
        write_mcmc_output(resultframe, tstamp, ID, suffix, CWD, metatstamp)

        # ADD NEW COLUMNS TO MCMC SAMPLE FILE
        # -----------------------------------------------

        # But only if you added new columns
        if resultframe.shape[1] > old:
            path = (f"{CWD}/analysis/results/mcmc/"
                    f"{tstamp}_{ID}{suffix}_converted_mcmc_sample.csv")
            resultframe.to_csv(path, index=False)
