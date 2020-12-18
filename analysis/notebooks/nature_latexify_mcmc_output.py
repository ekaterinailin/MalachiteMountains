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

def get_multicolumntab(df, ID, columns, both, both_to, same, same_to, fill=" "*50):
    dfn = {}
    
    for label, g in df.groupby(ID):
        res =  np.array([fill] * len(columns))
        if len(g)>1:
            vals1 = g.iloc[0].values
            vals2 = g.iloc[1].values
            
            res[both_to] = [item for sublist in list(zip(vals1[both], vals2[both])) for item in sublist]
            res[same_to] = vals1[same]
            dfn[label] = res
        else:
            vals1 = g.iloc[0].values

            res[both_to] = [item for sublist in list(zip(vals1[both], [fill]*len(vals1))) for item in sublist]
            res[same_to] = vals1[same]
            dfn[label] = res

    ddf = pd.DataFrame(dfn)
    ddf.index = columns

    return ddf



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
    power = int(np.floor(np.log10(np.abs(float(np.min(df.vplus))))))
    print(val, power, "POWER")
    powerabs = int(np.round(np.log10(np.abs(float(np.min( df[val+suff[1]]))))))
    print(val, powerabs, "POWERABS")
    
    if abs(powerabs) > 4:
        
        out = out + f"$\cdot 10^{powerabs}"
        out = out.replace("^","^{")
        out = out + "}$"
        df.vminus = df.vminus/(10**(powerabs))
        df.vplus = df.vplus/(10**(powerabs))
        df[val+suff[1]] = df[val+suff[1]]/(10**powerabs)
      #  print(powerabs, power, df.vplus, df.vminus, df[val+suff[1]])
        df[out] = df.apply(lambda x: f"${x[val+suff[1]]:.{powerabs-power}f}\left(^{x.vplus:.{powerabs-power}f}_{x.vminus:.{powerabs-power}f}\right)$", axis=1)
        df[out] = df[out].apply(lambda x: x.replace("^","^{+").replace("_","}_{-").replace("\right)","}\right)"))
    else:
        r = np.abs(power)
        rabs = powerabs
        print(val, "r", r)
        df[out] = df.apply(lambda x: f"${x[val+suff[1]]:{rabs}.{r}f}\left(^{x.vplus:{rabs}.{r}f}_{x.vminus:{rabs}.{r}f}\right)$", axis=1)
        df[out] = df[out].apply(lambda x: x.replace("^","^{+").replace("_","}_{-").replace("\right)","}\right)"))
                                        
    
    del df['vminus'], df['vplus']
    for s in suff:
        del df[val+s]
    return df



if __name__ == "__main__":
    # Get results table that you want to convert:
    CWD = "/".join(os.getcwd().split("/")[:-2])
    df = pd.read_csv(f"{CWD}/analysis/results/mcmc/15_12_2020_GP_deprecated_mcmcoutput.csv")
    df = df.drop_duplicates(keep=False).fillna("")
    
    # Get properties, and add them to the table
    prop = pd.read_csv(f"{CWD}/data/summary/inclination_input.csv")
    prop['id'] = prop['id']#.astype(str) 
    
    # get inclinations frm Elisabeth
    incl = pd.read_csv(f"{CWD}/data/inclinations/inclination_output.dat", delimiter=r"\s+")
    incl['id'] = incl['id']#.astype(str)
    
    df = df.merge(prop, left_on="ID", right_on="id", how="left")
    df = df.merge(incl, left_on="ID", right_on="id", how="left")
    
    print(df.columns, df.shape, df.i_deg_16)

    # convert energy to log10 value, hours to minutes
    for i in ["_16","_50","_84"]:

#        df = df.drop(f"rad_rsun{i}", axis=1)
        df[f"Eflare_erg{i}"] = np.log10(df[f"Eflare_erg{i}"])
        df[f"fwhm1_d{i}"] = df[f"fwhm1_d{i}"] * 60.
        df[f"fwhm2_d{i}"] = df[f"fwhm2_d{i}"] * 60.
        #df = df.drop(f"i_deg{i}", axis=1)
    
    # What values do you want to convert and how to call them                                
    valout = [("t0_d", "$t_0$ (BJD)"),
              ("Eflare_erg","$\log_{10} E_{f}$ (erg)"),
              ("ED_s", "$ED$ (s)"),
              ("frac_area","$A/A_*$"),
             # ("rad_rsun", "$\omega/2$ (deg)"),
              ("phase_deg","$\phi_0$ (deg)"),
              ("a","$A$"),
              ("i_deg","$i$ (deg)"),
              ("fwhm1_d", "FWHM$_i$ (min)"),
	       ("fwhm2_d", "FWHM$_g$ (min)"),
              ("latitude_deg", "$\theta_f$ (deg)")]

    # Do the conversion
    cp = df.copy(deep=True)

    print(cp.columns)
    for val, out in valout:
        print("Convert to LaTex: ", val)
        cp = add_val_with_percentiles(df, val, out)
   
    # Merge ID and suffix
    # Suffix should not resemble exoplanets
    mapsuffix = {"a":" (2-flare)", "b": " (2-flare)", "c": " (0h. u.)","":"", np.nan:""}
    print(cp["ID"], cp.suffix)
    cp["TIC"] = "TIC " + cp.ID.astype(str).str[:3] + cp.suffix.map(mapsuffix)                               


    # convert Prot, vsini, Rstar, and later i
   # df[r"$i$ (deg)"] = df.apply(lambdaf"${x.i_deg:.1f}({g(x.i_sigma):.1f})$",
   #                               axis=1)

    # write vsini with uncertainties to str
    cp[r"$v \sin i$ (km s$^{-1}$)"] = cp.apply(lambda x: 
                                          fr"${x.vsini_kms:.1f} \pm {x.e_vsini_kms :.1f}$",
                                          axis=1)
    
    # write Rstart with uncertainties to str
    cp[r"$R_*/R_\odot$"] = cp.apply(lambda x: 
                                          fr"${x.rad_rsun:.3f} \pm {x.e_rad_rsun:.3f}$",
                                          axis=1)

    # write rotation period to string
    cp[r"$P$ (min)"] = cp.apply(lambda x:
                                       fr"${x.prot_d * 24. *60.:.3f} \pm {x.e_prot_d * 24. *60.:.3f}$", axis=1)
    
    #write inclination
  #  cp[r"$i$ (deg)"] = cp.apply(lambda x: f"${x.inclination:.1f}\left(^{x.inclination_uperr:.1f}_{abs(x.inclination_lowerr):.1f}\right)$", axis=1)
   # cp[r"$i$ (deg)"]  = cp[r"$i$ (deg)"].apply(lambda x: x.replace("^","^{+").replace("_","}_{-").replace("\right)","}\right)"))
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
    del cp["ID"]
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
    df3 = df3[["TIC","SpT", r"$P$ (min)", r"$v \sin i$ (km s$^{-1}$)", 
               "$R_*/R_\odot$", r"$i$ (deg)","$\log_{10} E_{f}$ (erg)",
               r"$A$","FWHM$_i$ (min)","FWHM$_g$ (min)" ,"$\theta_f$ (deg)", ]]

    columns = ["SpT", r"$P$ (min)", r"$v \sin i$ (km s$^{-1}$)", 
               "$R_*/R_\odot$", r"$i$ (deg)",
               "$\log_{10} E_{f,1}$ (erg)",r"$A_1$", "FWHM$_{i,1}$ (min)","FWHM$_{g,1}$ (min)" ,
               "$\log_{10} E_{f,2}$ (erg)",r"$A_2$", "FWHM$_{i,2}$ (min)","FWHM$_{g,2}$ (min)" ,
                "$\theta_f$ (deg)", ]
    #df3 = df3[columns]
    same = [1,2,3,4,5,10]
    same_to = [0,1,2,3,4,13]
    both = [6,7,8,9]
    both_to = [5,9,6,10,7,11,8,12]
    
    df3 = get_multicolumntab(df3, "TIC", columns, both, both_to,same, same_to)
    df3 = df3.sort_values(by=r"$i$ (deg)", ascending=False, axis=1)
   # nc = 'c' * (df3.shape[1]-2) #number of middle columns
   # df3 = df3.set_index("TIC")
   # df3 = df3.T
    footnote_pref= "\multicolumn{" + str(df3.shape[1]-2) + "}{l}" 
    footnote_suf = "\n"

#    stri = df3.to_latex(index=False,escape=False, column_format=f"lc|cccc|ccccr")
    stri = df3.to_latex(index=True,escape=False, column_format=f"l|cccccc")
    stri = stri.replace("\\toprule","\hline\hline")
    stri = stri.replace("\\midrule","\hline")
    stri = stri.replace("\\bottomrule","\hline\n" )


    # Path to paper table directory
    PWD = "/home/ekaterina/Documents/002_writing/multiperiod-flares-draft/nature"
    PWD = "/home/ekaterina/Documents/002_writing/multiperiod-flares-draft/nature/multiperiodflaresnature/tables"

    # Write out latex string
    with open(f"{PWD}/nature_results_decoupled_GP_deprecated.tex", "w") as f:
        f.write(stri)
