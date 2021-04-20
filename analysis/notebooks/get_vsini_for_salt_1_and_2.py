"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This script contains three procedures concerned
with the determination of vsini from SALT spectra
using CARMENES spectra of late M dwarfs as non-rotating
references.

SCRIPT 1: Just shows the spectra in the region around the Rb line.

SCRIPT 2: Calculates CCF calibrations and illustrates them for each
          model spectrum.
          
SCRIPT 3: Calibrates CCFs to FHWM, fits vsini, and saves the mean result.
          Also output a figure that goes into the paper.

"""

from scipy import interpolate

import numpy as np
import pandas as pd

import matplotlib 
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)

import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.constants import c as lightspeed

from eniric import broaden
from funcs.xcorr import correlate_maxlag, get_lags

import time
import os



def script_1_plot_spectra(carms, salt, wavmin, wavmax, tstamp):
    # Plot the model spectra together with the SALT spectra

    plt.figure(figsize=(10,7))

    off = 0

    for key, l in carms.items():
        sp = l[4]
        wav = sp.loc[(sp.lambd>wavmin) & (sp.lambd<wavmax),"lambd"].values
        mflux = sp.loc[(sp.lambd>wavmin) & (sp.lambd<wavmax),"flux"].values
        plt.plot(wav,mflux/np.median(mflux)+off , linewidth=1, label=f"{key} ({l[0]})")
        off +=2

    for l,spc in salt.items():

        sp, c = spc

        wav = sp.loc[(sp.lambd>wavmin) & (sp.lambd<wavmax),"lambd"].values
        mflux = sp.loc[(sp.lambd>wavmin) & (sp.lambd<wavmax),"flux"].values
        plt.plot(wav,mflux/np.median(mflux) + off , linewidth=1, label=f"{l}", c=c)
        off +=2

    plt.axvline(line_center, c="k")

    plt.xlim(wavmin, wavmax)
    #plt.ylim(-100,1e5)
    #plt.ylim(0,4.6)
    plt.legend(fontsize=13)
    plt.xlabel(r"$\lambda\;[\AA]$", fontsize=13)
    plt.ylabel(r"norm. flux", fontsize=13)
    plt.savefig(f"../plots/vsini/{tstamp}_salt12_{wavmin:.0f}_{wavmax:.0f}.png", dpi=300);
    
    

def script_2_rotational_broadening_and_CCF_plot(carms, salt, key, wavmin, wavmax):
    # ROTATIONAL BROADENING
    # -----------------------------------------------------------

    # Pick up the model spectrum array
    model = carms[key][4]
    wav = model.loc[(model.lambd>wavmin) & (model.lambd<wavmax),"lambd"].values 
    mflux = model.loc[(model.lambd>wavmin) & (model.lambd<wavmax),"flux"].values

    # Do rotational broadening here
    # see ENIRIC docs: https://eniric.readthedocs.io/en/latest/broadening.html#rotational-broadening

    # # Pick vsini grid
    vsinis = np.arange(33,42,.25)


    # # Set up the grid
    specs = pd.DataFrame({"wav":wav})


    # # Run the broadening with ENIRIC
    for vsini in vsinis:
        specs[vsini] = broaden.rotational_convolution(wav, model.lambd, model.flux, vsini, epsilon=0.6)


    # Path to save the grid
    grid = f"{wavmin}_{wavmax}_{carms[key][3]}.csv"

    # # If you run the broadening, save results here
    specs.to_csv(f"../../data/eniric/{grid}", index=False)


    # Read in the grid of broadened spectra
    specs = pd.read_csv(f"../../data/eniric/{grid}")

    # Set wav as index
    specs = specs.set_index("wav")



    # CROSS-CORRELATION
    # -----------------------------------------------------------

    # Convert spectral resolution to resolution in velocity space
    # Use the line center as a reference
    lagunit = (np.mean(np.diff(wav)) / line_center * lightspeed).to("km/s")
    print(f"Resolution in velocity space: {lagunit:.2f}")

    # Set up cross-correlation function (CCF) frame
    corrf = pd.DataFrame()

    # Set up CCF figure
    plt.figure(figsize=(8,7))


    # Define CC parameters
    start = len(mflux) // 2 # start at the line center
    maxlag = 200 # vsini should certainty not be more than 240 km/s...


    # Run the CC model spectrum with broadened model spectrum 
    # -----------------------------------------------------------

    # for each vsini value in the grid:
    for vsini in specs.columns.values:

        # Pick flux
        a = specs[vsini]

        # Cross-correlate
        cc1 = correlate_maxlag(a, mflux, maxlag)

        # take the lags at each step and convert to velocity
        v = get_lags(cc1)*lagunit

        # plot the CCF
        plt.plot(v, cc1 / np.max(cc1), c="grey", 
                 alpha = float(vsini) / 60, linewidth=1,
                 linestyle="dashed")

        # save to the CCF frame
        corrf[vsini] = cc1

    # ---------------------------------------------------
    # This is just a little hack to properly show the legend
    plt.plot(v, cc1/np.max(cc1),c="grey", 
                 alpha = 1., linewidth=0.5,
                 linestyle="dashed",label=f"{key}")    
    # ---------------------------------------------------


    # Run SALT spectra with non-rotating model spectrum
    # -----------------------------------------------------------    

    corrsp = {}    

    for lab, spc in salt.items():    
        # Pick up spectrum and color for plotting
        sp, c = spc

        # Pick the spectral region for which CCF was computed
        sflux = sp.loc[(sp.lambd>wavmin) & (sp.lambd<wavmax),"flux"].values
        swav = sp.loc[(sp.lambd>wavmin) & (sp.lambd<wavmax),"lambd"].values

        # Interpolate values to be the same as the model spectra
        # Note: 
        # CARMENES AND SALT HAVE SIMILAR RESOLUTION, 
        # SO INTERPOLATION IS FINE.
        f = interpolate.interp1d(swav, sflux, fill_value="extrapolate")
        sfluxinterp = f(wav)
        sfluxinterp = sfluxinterp / np.median(sfluxinterp)

        # Correlate non-rotating model with SALT spectrum
        cc1 = correlate_maxlag(sfluxinterp, mflux, maxlag)

        # Convert CCF lag to velocity
        v = get_lags(cc1)*lagunit

        # Center on 0 km/s and plot the CCF
        ccf_ = cc1/np.max(cc1)
        maxarg = np.argmax(ccf_)
        v_ = v - v[maxarg]
        plt.plot(v_, ccf_, label=f"{lab}", c=c)

        # Save to the CCF frame
        corrsp[lab] = cc1


    # Layout figure    
    plt.xlim((-maxlag*lagunit).value, (maxlag*lagunit).value)
    plt.legend(fontsize=13)
    plt.xlabel(r"$v$ [km/s]", fontsize=13)
    plt.ylabel(r"cross-correlation function", fontsize=13)
    plt.savefig(f"../plots/vsini/{tstamp}_{key}_salt12_crosscorr.png", dpi=300)
    

def script_3_ccf_and_vsini_fit(carms, salt, wavmin, wavmax):

    # set up results dict
    res = {}

    # set up figure for paper
    plt.figure(figsize=(7,6))

    # iterate through all model spectra
    for key, col, ls in [('CN Leo',"silver", "solid"), 
                         ('LP 731-058',"grey","dashed"), 
                         ("Teegarden's Star", "dimgrey", "dotted")]:

        # init sub-dict for model spectrum
        res[key] = {}

        # Pick up model spectrum
        model = carms[key][4]

        # Constrain the spectral region of interest
        wav = model.loc[(model.lambd>wavmin) & (model.lambd<wavmax), "lambd"].values 
        mflux = model.loc[(model.lambd>wavmin) & (model.lambd<wavmax), "flux"].values

        # Convert spectral resolution to resolution in velocity space
        # Use the line center as a reference
        lagunit = (np.mean(np.diff(wav)) / line_center * lightspeed).to("km/s")
        print(f"\nResolution in velocity space: {lagunit:.2f}")

        # Load grid of broadened spectra
        grid = f"{wavmin}_{wavmax}_{carms[key][3]}.csv"
        specs = pd.read_csv(f"../../data/eniric/{grid}")
        specs = specs.set_index("wav")

        # Set up cross-correlation function (CCF) frame
        corrf = pd.DataFrame()

        # Define CC parameters
        start = len(mflux) // 2 # start at the line center
        maxlag = 200 # vsini should certainty not be more than 240 km/s...

        # Run CC on all model spectra with the non-broadened version
        for vsini in specs.columns.values:

            # pick broadened spectrum
            a = specs[vsini]

            # run CC
            cc1 = correlate_maxlag(a, mflux, maxlag)

            # save CCF to frame
            corrf[vsini] = cc1


        # Run SALT spectra with non-rotating model spectrum
        # -----------------------------------------------------------    

        corrsp = {}    

        for lab, spc in salt.items():    
            # Pick up spectrum and color for plotting
            sp, c = spc

            # Pick the spectral region for which CCF was computed
            sflux = sp.loc[(sp.lambd>wavmin) & (sp.lambd<wavmax),"flux"].values
            swav = sp.loc[(sp.lambd>wavmin) & (sp.lambd<wavmax),"lambd"].values

            # Interpolate values to be the same as the model spectra
            # Note: 
            # CARMENES AND SALT HAVE SIMILAR RESOLUTION, 
            # SO INTERPOLATION IS FINE.
            f = interpolate.interp1d(swav, sflux, fill_value="extrapolate")
            sfluxinterp = f(wav)
            sfluxinterp = sfluxinterp / np.median(sfluxinterp)

            # Correlate non-rotating model with SALT spectrum
            cc1 = correlate_maxlag(sfluxinterp, mflux, maxlag)

            # Save to the CCF frame
            corrsp[lab] = cc1


        # FIT SALT CCF TO MODEL AND FIND VSINI
        # -----------------------------------------------------------------

        # Calculate the FWHM calibration for the model spectrum
        FWHM = corrf.apply(lambda x:np.where(x>np.max(x)/2.)[0][-1]-np.where(x>np.max(x)/2.)[0][0], axis=0)
        FWHM = FWHM.sort_values()

        # Pick a range where you want to fit 3rd order polynomial 
        wh = np.where(FWHM.values>39)[0]
        x, y = FWHM.index.astype(float).values[wh], FWHM.values[wh]

        # Fit a 3rd order polynomial
        z = np.polyfit(y,x,3, cov=False)
        p = np.poly1d(z)

        # Get the derivate of polynomial fit to propagate
        # uncertainty on FWHM to vsini uncertainty
        pdiv = np.polyder(p)

        # Plot the polynomial fit
        plt.plot(y, p(y), label=key, c=col, linestyle=ls)

        # 
        for lab in salt:

            # Get FWHM from SALT vs. model CCFs:
            cc = corrsp[lab]
            overhalf = np.where(cc>np.max(cc)/2.)[0]
            fwhm = overhalf[-1] - overhalf[0]

            # Define FWHM and uncertainty on FWHM
            # SPECTRAL RESOLUTION DEFINES THE FWHM UNCERTAINTY
            y0, ey0 = fwhm, lagunit.value

            # UNCERTAINTY ON VSINI IS PROPAGATED FROM
            # UNCERTAINTY ON FWHM
            dfwhm_dvsini = pdiv(y0)
            yerr = dfwhm_dvsini * ey0

            # ----------------------------------------------
            # This is a hack to make the legend look right
            if key == 'CN Leo':
                label = lab
            else:
                label = None
            # ----------------------------------------------

            # Plot the vsini fits:
            plt.errorbar([y0], [p(y0)], xerr=ey0, yerr=yerr,
                         c=salt[lab][1], linewidth=2,
                         label=label)
            print(f"{key} vs. {lab}: vsini = {p(y0):.2f} +/- {yerr:.2f} km/s")

            # Write out results
            res[key][lab] = [p(y0), yerr]


    # Layout figure for paper
    plt.ylim(20,60)
    plt.xlim(40,65)
    plt.xlabel("FWHM [km/s]", fontsize=14)
    plt.ylabel(r"$v\sin i$ [km/s]", fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.tight_layout()
    plt.savefig(f"../plots/vsini/{tstamp}_vsini.png", dpi=300)

    # UPDATE THE PAPER FIGURE HERE:
    #plt.savefig(f"{figdir}vsini.png", dpi=300)



    # Convert res dict to pandas table
    reform = {(innerKey, outerKey): values for 
              outerKey, innerDict in 
              res.items() for 
              innerKey, values in 
              innerDict.items()}

    result = pd.DataFrame(reform).T.sort_index()
    result = result.reset_index()
    result = result.rename(index=str, columns={0:"vsini",
                                               1:"err", 
                                               "level_0":"TIC", 
                                               "level_1":"template"})

    tdr = f"{CWD}/data/summary/{tstamp}_vsinis.csv"
    result.to_csv(tdr, index=False)
    # Merge results with the parameters table that 
    # is used elsewhere in the project
    # CALCULATE FINAL RESULT AND SAVE IT
    # ------------------------------------------------------------

    # Pick up parameters table
    tabdir = f"{CWD}/data/summary/{tstamp}_lcs.csv"
    lcs = pd.read_csv(tabdir)

    print("\nMean vsini results:\n")

    for l, g in result.groupby("TIC"):  

        # Use MEAN VALUE OF THREE FITS
        # PROPAGATE THE ERROR ON EACH
        err = np.sqrt((g.err**2).sum())/3
        val = g.vsini.mean()
        print(f"{l}: vsini = {val} +/- {err} km/s")

        # Write out values to quote in paper
        with open(f"{valdir}{l[4:7]}_vsini.txt", "w") as f:
            s = f"${val:.1f}\pm{err:.1f}$ km/s"
            f.write(s)

        # Write result to parameters table
        lcs.loc[lcs.ID == int(l[4:]),"vsini_kms"] = val
        lcs.loc[lcs.ID == int(l[4:]),"e_vsini_kms"] = max(3., err)

    # Define a different output path to avoid overwriting with wrong results
    tabdir2 = f"{CWD}/data/summary/{tstamp}_lcsvsini.csv"
    lcs.to_csv(tabdir2, index=False)
    
    

if __name__ == "__main__":
    
    # Time stamp
    tstamp = time.strftime("%d_%m_%Y", time.localtime())

    # Current working directory
    CWD = "/".join(os.getcwd().split("/")[:-2])

    # Define paper directory:
    CPD = "/home/ekaterina/Documents/002_writing/multiperiod-flares-draft"

    # Define in- and output paths
    valdir = f'{CPD}/values/'
    figdir = f'{CPD}/figures/'
    dirr = f'{CWD}/data'

    # GET ALL THE SPECTRA:

    # SALT1 and SALT2
    # ------------------------------------------------------------------------

    sp1 = pd.read_csv(f"{dirr}/salt/R202002090014.1.ffc.hwl.txt", 
                      delimiter="\s+", names=["lambd", "flux", "flux_err"])
    # mask an emission line that is only seen in SALT but not in CARMENES
    sp1.loc[((sp1["lambd"] > 7949.2) & (sp1["lambd"] < 7949.8)), "flux"] = np.nan
    sp1 = sp1.dropna(how="any")

    sp2 = pd.read_csv(f"{dirr}/salt/R202002080017.1.ffc.hwl.txt", 
                      delimiter="\s+", names=["lambd", "flux", "flux_err"])
    # mask an emission line that is only seen in SALT but not in CARMENES
    sp2.loc[((sp2["lambd" ]> 7949.2) & (sp2["lambd"] < 7949.8)), "flux"] = np.nan
    sp2 = sp2.dropna(how="any")


    salt = {"TIC 44984200": [sp1, "r"],
            "TIC 277539431": [sp2, "b"]}    

    # Get non-rotating spectra from CARMENES
    # ------------------------------------------------------------------------

    # Spectral types come from the CARMENES data base on Vizier
    carms = {"CN Leo": 
             ["M6", "carmenes_cnleo_vis.fits", 3.05, "cnleo"],
             "LP 731-058": 
             ["M6.5", "carmenes_lp731_m65_vis.fits", 1.05, "lp731"],
             "Teegarden's Star": 
             ["M7", "carmenes_teegarden_vis.fits", 3.45, "teega"],}

    for s, l in carms.items():
        hdu = fits.open(f"{dirr}/carmenes/{l[1]}")
        lambd = hdu[4].data.flatten().byteswap().newbyteorder() - l[2]
        flux = hdu[1].data.flatten().byteswap().newbyteorder()
        sp4 = pd.DataFrame({"lambd":lambd, "flux":flux}).sort_values(by="lambd", ascending=True)
        sp4 = sp4.dropna(how="any")
        l.append(sp4)


    # --------------------------------------
    # Pick spectral region


    # Pick the Rubidium line region
    # (left edge, right edge, center of line)
    wavmin, wavmax, line_center = (7938,7955,7948) 
    # --------------------------------------

    # --------------------------------------
    # SCRIPT 1: JUST PLOT SALT and MODEL SPECTRA

    script_1_plot_spectra(carms, salt, wavmin, wavmax, tstamp)
    # --------------------------------------

    # --------------------------------------
    # SCRIPT 2: GET CCFs and PLOT THEM FOR EVERY MODEL SPECTRUM
    # TOGETHER WITH SALT CCFs

    for key in ['CN Leo', 'LP 731-058', "Teegarden's Star"]:
        script_2_rotational_broadening_and_CCF_plot(carms, salt, key, wavmin, wavmax)
    # --------------------------------------

    # --------------------------------------
    # SCRIPT 3: RUN THE FULL FWHM CALIBRATION, FIT VSINIs,
    # and WRITE OUT RESULTS

    script_3_ccf_and_vsini_fit(carms, salt, wavmin, wavmax)
    # --------------------------------------
