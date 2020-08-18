"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This script converts the lcsi.csv table
to the latex table geometry.tex that appears
in the paper.
"""

import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    # Get results table that you want to convert:
    CWD = "/".join(os.getcwd().split("/")[:-2])

    df = pd.read_csv(f"{CWD}/data/summary/lcsi.csv")

    dff = df[["prefix","ID", "Prot_d","vsini_kms",
              "e_vsini_kms","i_mu","i_sigma","SpT"]]

    # allow to sort by SpT
    dff["SpT"] = dff.SpT.apply(lambda x: x[-1])
    dff.loc[np.where(dff.SpT=="1")[0],"SpT"] = "10"
    dff.SpT = dff.SpT.astype(int)
    dff.sort_values(by="SpT", ascending=True)


    # throw away 230
    dff = dff[dff.ID != 230120143]

    # add references
    sources = {100004076 : r"\citet{gizis2013}", 
               237880881 : r"\citet{kraus2014}",  
               44984200 : "this work",
               277539431: "this work",}
    dff[r"$v\sin i$ ref."] = dff.ID.map(sources)


    # generate ID for printin
    dff["ID"] = dff.apply(lambda x: 
                          x.prefix + " " + f"{str(x.ID)[:3]}", 
                          axis=1)

    # convert radian to degrees in inclination and write a string
    g = lambda x: x/np.pi*180.
    dff[r"$i$ (deg)"] = dff.apply(lambda x: 
                                  fr"${g(x.i_mu):.1f}({g(x.i_sigma):.1f})$",
                                  axis=1)

    # write vsini with uncertainties to str
    dff[r"$v \sin i$ (km/s)"] = dff.apply(lambda x: 
                                          fr"${x.vsini_kms:.1f}({x.e_vsini_kms:.1f})$",
                                          axis=1)

    # write rotation period to string
    dff[r"$P$ (h)"] = dff.Prot_d.apply(lambda x:
                                       f"{x * 24.:.1f}(0.03)")
    # Kepler
    dff.loc[dff.ID == "KIC 100", r"$P$ (h)"] = dff.loc[dff.ID == "KIC 100", r"$P$ (h)"].str.replace("0.03","0.00") 

    # Sort by SpT and pick the relevant columns
    dfff = dff.sort_values(by="SpT", ascending=True)
    dfff = dfff[["ID",r"$P$ (h)",r"$v \sin i$ (km/s)",
                 r"$i$ (deg)", r"$v\sin i$ ref."]]

    # Convert pandas to latex string
    nc = 'c' * (dfff.shape[1]-2) #number of middle columns

    footnote_pref= "\multicolumn{" + str(dfff.shape[1]-2) + "}{l}" 
    footnote_suf = "\n"

    stri = dfff.to_latex(index=False,escape=False, column_format=f"l{nc}r")
    stri = stri.replace("\\toprule","\hline\hline")
    stri = stri.replace("\\midrule","\hline")
    stri = stri.replace("\\bottomrule","\hline\n" )


    # Path to paper table directory
    PWD = "/home/ekaterina/Documents/002_writing/multiperiod-flares-draft"

    # Write out latext string
    with open(f"{PWD}/tables/geometry.tex", "w") as f:
        f.write(stri)
    
    print("\n----------------\nSuccessfully latexified lcsi.csv to geometry.tex")

