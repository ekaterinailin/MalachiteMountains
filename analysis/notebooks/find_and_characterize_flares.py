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

def custom_detrend(flc):
    t, sflux, model, period = remove_sinusoidal(target, save=False, flc=flc, plot=False, custom=False)
    f = FlareLightCurve(time=t.byteswap().newbyteorder(), 
                        flux = flc.flux,
                        detrended_flux=sflux.byteswap().newbyteorder(), 
                        detrended_flux_err=flc.flux_err.byteswap().newbyteorder())
    f = f[np.isfinite(f.detrended_flux)]
    return f

def sample_flare_recovery(target, sec, mission, iterations=10):
    flc = from_mast(f"{target.prefix} {target.ID}", cadence="short", c=sec, mission=mission)
    
    f = custom_detrend(flc)

    flares = f.find_flares().flares
    flares["ID"] = target.ID
    flares["sector"] = sec
    df = df.append(flares)
    
    # Inj/Rec
    df["dur"] = df.tstop - df.tstart
    flc, fake_flc = flc.sample_flare_recovery(inject_before_detrending=True, mode="custom",
                                              iterations=iterations, fakefreq=1e-3, 
                                              ampl=[df.ampl_rec.min()/2, df.ampl_rec.max()*1.5],
                                              dur=[df.dur.min()/6., df.dur.max()/4.], 
                                              func=custom_detrend)
    with open(f"{CWD}/analysis/results/flarefind/{target.ID}_s{sec}_fake_flares.csv", "a") as F:
        flc.fake_flares.to_csv(F, index=False)
        
    print("Before loading extra events: New events: ", flc.fake_flares.shape[0])
    
    flc.load_injrec_data(f"{CWD}/analysis/results/flarefind/{target.ID}_s{sec}_fake_flares.csv")
    flc.fake_flares = flc.fake_flares.drop_duplicates(keep=False).astype(float)
    
    print("After loading extra events: Total events: ", flc.fake_flares.shape[0])


if __name__ == "__main__":
    
    sectors = {44984200: [[8, 9, 10], "TESS"],
               277539431: [[12],"TESS"],
               237880881: [[1,2],"TESS"],
               100004076:[[14],"Kepler"],}
    lcs = pd.read_csv(f"{CWD}/data/summary/lcsi.csv")
    
    for l, target in lcs.iterrows():
        secs, mission = sector[target.ID]
        for sec in secs:
            sample_flare_recovery(target, sec, mission, iterations=1000)
            
