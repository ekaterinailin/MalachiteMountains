"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This script fetches all light curves of
the targets that showed multiperiod flares,
detrends them, and performs the injection and
recovery of synthetic flares.

The resulting table of injected (and recovered)
flares is saved to file for further analysis, and
flare characterization in particular.

"""
import os
import copy

import numpy as np
import pandas as pd

from funcs.multiperiod import remove_sinusoidal
from altaipony.lcio import from_mast
from altaipony.flarelc import FlareLightCurve

CWD = "/".join(os.getcwd().split("/")[:-2])

import warnings
warnings.simplefilter("ignore")

def custom_detrend(flc, target=None):
    t, sflux, model, period = remove_sinusoidal(target, save=False, flc=flc, plot=False, custom=False)
    f = FlareLightCurve(time=t.byteswap().newbyteorder(), 
                        flux = flc.flux,
                        detrended_flux=sflux.byteswap().newbyteorder(), 
                        detrended_flux_err=flc.flux_err.byteswap().newbyteorder())
    f = f[np.isfinite(f.detrended_flux)]
    return f

def sample_flare_recovery(target, sec, mission, iterations=10):
    flcs = from_mast(f"{target.prefix} {target.ID}", cadence="short", c=sec, mission=mission)
    if mission != "Kepler":
        flcs = [flcs]
 
    for i, flc in enumerate(flcs):

        f = custom_detrend(flc, target=target)

        flares = f.find_flares().flares
        flares["ID"] = target.ID
        flares["sector"] = sec
        df = flares

        # Inj/Rec
        df["dur"] = df.tstop - df.tstart
        flc, fake_flc = flc.sample_flare_recovery(inject_before_detrending=True, mode="custom",
                                                  iterations=iterations, fakefreq=1e-3, 
                                                  ampl=[df.ampl_rec.min()/2, df.ampl_rec.max()*1.5],
                                                  dur=[df.dur.min()/6., df.dur.max()/4.], 
                                                  func=custom_detrend)
        
        outpath = f"{CWD}/analysis/results/flarefind/{target.ID}_s{sec}_{i}_fake_flares.csv"
        
        if os.path.exists(outpath):
            header=False
        else:
            header=True
            
        with open(outpath, "a") as F:
            flc.fake_flares.to_csv(F, index=False, header=header)

        print("Before loading extra events: New events: ", flc.fake_flares.shape[0])

        flc.load_injrec_data(outpath)
        flc.fake_flares = flc.fake_flares.drop_duplicates(keep=False).astype(float)

        print("After loading extra events: Total events: ", flc.fake_flares.shape[0])


if __name__ == "__main__":
    
    sectors = {44984200: [[8, 9, 10], "TESS"],
               277539431: [[12],"TESS"],
               237880881: [[1,2],"TESS"],
               100004076:[[14],"Kepler"],}
    lcs = pd.read_csv(f"{CWD}/data/summary/lcsi.csv")
    

    for l, target in lcs.iloc[0:-1].iterrows():
        print(f"Running {target.ID}")
        secs, mission = sectors[target.ID]
        for sec in secs:
            sample_flare_recovery(target, sec, mission, iterations=2000)
            
