"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This script converts the mcmcoutput.csv table
to the latex table that appears in the paper.
"""

import pandas as pd
import numpy as np
import os

def add_val_with_percentiles(df, val, out, suff = ["_16","_50","_84"]):
    """Convert table with percentiles 
    to latex compatible table.
    
    Parameters:
    ------------
    df : pandas table
        table with three columns per value
    val : str
        column name without suffixes
    out : str
        formatted column name for latex output
    suff : list of three strings
        suffixes to val
    """
    
    for s in suff:
        df[val+s] = df[val+s].apply(lambda x: float(x))
    
    df["vplus"] = df.apply(lambda x: x[val+suff[2]]-x[val+suff[1]], axis=1)
    df["vminus"] = df.apply(lambda x: x[val+suff[1]]-x[val+suff[0]], axis=1)
    power = int(np.round(np.log10(np.abs(float(np.min(df.vplus))))))
    print(power, "POWER")
    powerabs = int(np.round(np.log10(np.abs(float(np.min( df[val+suff[1]]))))))
    print(powerabs, "POWERABS")
    
    if abs(powerabs) > 4:
        
        out = out + f"$\cdot 10^{powerabs}"
        out = out.replace("^","^{")
        out = out + "}$"
        df.vminus = df.vminus/(10**(powerabs))
        df.vplus = df.vplus/(10**(powerabs))
        df[val+suff[1]] = df[val+suff[1]]/(10**powerabs)
        print(powerabs, power, df.vplus, df.vminus, df[val+suff[1]])
        df[out] = df.apply(lambda x: f"${x[val+suff[1]]:.{powerabs-power}f}\left(^{x.vplus:.{powerabs-power}f}_{x.vminus:.{powerabs-power}f}\right)$", axis=1)
        df[out] = df[out].apply(lambda x: x.replace("^","^{+").replace("_","}_{-").replace("\right)","}\right)"))
    else:
        r = powerabs-power
        df[out] = df.apply(lambda x: f"${x[val+suff[1]]:.{r}f}\left(^{x.vplus:.{r}f}_{x.vminus:.{r}f}\right)$", axis=1)
        df[out] = df[out].apply(lambda x: x.replace("^","^{+").replace("_","}_{-").replace("\right)","}\right)"))
                                        
    
    del df['vminus'], df['vplus']
    for s in suff:
        del df[val+s]
    return df



if __name__ == "__main__":
    # Get results table that you want to convert:
    CWD = "/".join(os.getcwd().split("/")[:-2])
    df = pd.read_csv(f"{CWD}/analysis/results/mcmc/mcmcoutput.csv").iloc[:7]
    df = df.drop_duplicates(keep=False).fillna("")
    
    # Get properties, and add them to the table
    prop = pd.read_csv(f"{CWD}/data/summary/inclination_input.csv")
    prop['id'] = prop['id'].astype(str) 
    
    # get inclinations frm Elisabeth
    incl = pd.read_csv(f"{CWD}/data/inclinations/inclination_output.dat", delimiter=r"\s+")
    incl['id'] = incl['id'].astype(str)
    
    df = df.merge(prop, left_on="ID", right_on="id", how="left")
    df = df.merge(incl, left_on="ID", right_on="id", how="left")
    
    print(df.columns, df.shape, df.T)

    # Make sure that FWHM is FWHM as normal people understand it and not 2*FWHM!
    for col in ['fwhm_d_16', 'fwhm_d_50', 'fwhm_d_84']:
        df[col] = df[col].astype(float) / 2.

    # Drop columns
    for i in ["_16","_50","_84"]:
        df = df.drop(f"rad_rsun{i}", axis=1)
        df = df.drop(f"i_deg{i}", axis=1)
    
    # What values do you want to convert and how to call them                                
    valout = [("t0_d", "$t_0$ (BJD)"),
              ("Eflare_erg","$E_{f}$ (erg)"),
              ("ED_s", "$ED$ (s)"),
              ("frac_area","$A/A_*$"),
             # ("rad_rsun", "$\omega/2$ (deg)"),
              ("phase_deg","$\phi_0$ (deg)"),
              ("a","$a$"),
             # ("i_deg","$i$ (deg)"),
              ("fwhm_d", "FWHM (d)"),
              ("latitude_deg", "$\theta_f$ (deg)")]

    # Do the conversion
    cp = df.copy(deep=True)
    for val, out in valout:
        print("Convert to LaTex: ", val)
        cp = add_val_with_percentiles(df, val, out)

    # Merge ID and suffix
    # Suffix should not resemble exoplanets
    mapsuffix = {"a":" (I)", "b": " (II)", "":"", np.nan:""}
    cp["ID"] = cp.prefix_y + " " + cp.ID.astype(str).str[:3] + cp.suffix.map(mapsuffix)                               


    # convert Prot, vsini, Rstar, and later i
   # df[r"$i$ (deg)"] = df.apply(f"${x.i_deg:.1f}({g(x.i_sigma):.1f})$",
    #                              axis=1)

    # write vsini with uncertainties to str
    cp[r"$v \sin i$ (km/s)"] = cp.apply(lambda x: 
                                          fr"${x.vsini_kms:.1f}$\pm${x.e_vsini_kms :.1f}$",
                                          axis=1)
    
    # write Rstart with uncertainties to str
    cp[r"$R_*/R_\odot$"] = cp.apply(lambda x: 
                                          fr"${x.rad_rsun:.3f}$\pm${x.e_rad_rsun:.3f}$",
                                          axis=1)

    # write rotation period to string
    cp[r"$P$ (min)"] = cp.apply(lambda x:
                                       fr"{x.prot_d * 24. *60.:.3f}$\pm${x.e_prot_d * 24. *60.:.3f}", axis=1)
    
    #write inclination
    cp[r"$i$ (deg)"] = cp.apply(lambda x: f"${x.inclination:.1f}\left(^{x.inclination_uperr:.1f}_{abs(x.inclination_lowerr):.1f}\right)$", axis=1)
    cp[r"$i$ (deg)"]  = cp[r"$i$ (deg)"].apply(lambda x: x.replace("^","^{+").replace("_","}_{-").replace("\right)","}\right)"))
    del cp['inclination']
    del cp['inclination_uperr']
    del cp['inclination_lowerr']
    
    # rename spt to SpT
    cp = cp.rename(index=str, columns={"spt":"SpT"})
    
    
    
    # Remove helper columns
    del cp['tstamp']
    del cp['color']
    del cp['linestyle']
    del cp['suffix']
    for _ in ['x', 'y']:
        del cp[f'id_{_}']
        del cp[f'prefix_{_}']

    
    # Drop columns
    for i in ["","e_"]:
        for j in ["prot_d", "rad_rsun", "vsini_kms"]:
            cp = cp.drop(f"{i}{j}", axis=1)

    # Convert pandas to latex string
    df3 = cp.copy()
    
    # select columns
    df3 = df3[["ID","SpT", r"$P$ (min)", r"$v \sin i$ (km/s)", 
               "$R_*/R_\odot$", r"$i$ (deg)", r"$E_{f}$ (erg)$\cdot 10^{33}$",
               r"$a$", "$\theta_f$ (deg)", ]]# r"$A/A_*$",

   # nc = 'c' * (df3.shape[1]-2) #number of middle columns

    footnote_pref= "\multicolumn{" + str(df3.shape[1]-2) + "}{l}" 
    footnote_suf = "\n"

    stri = df3.to_latex(index=False,escape=False, column_format=f"lc|cccc|cccr")
    stri = stri.replace("\\toprule","\hline\hline")
    stri = stri.replace("\\midrule","\hline")
    stri = stri.replace("\\bottomrule","\hline\n" )


    # Path to paper table directory
    PWD = "/home/ekaterina/Documents/002_writing/multiperiod-flares-draft"

    # Write out latext string
    with open(f"{PWD}/nature/nature_results.tex", "w") as f:
        f.write(stri)
