import os
import pickle

import numpy as np
import pandas as pd

import emcee

from funcs.flarefit import (log_probability,
                            log_probability_2flares)
from get_mcmc_outputs import write_meta_mcmc

from astropy.constants import R_sun
import astropy.units as u

from multiprocessing import Pool
import time

log_probs = {"log_probability":[6, log_probability],
             "log_probability_2flares":[9, log_probability_2flares]}

CWD = "/".join(os.getcwd().split("/")[:-2])
nwalkers = 32

def get_inits_one(ID, tstamp):
    
    inits = pd.read_csv(f"{CWD}/data/summary/inits.csv")
    target = inits.loc[(inits.ID == ID) & (inits.tstamp==tstamp),:].iloc[0]
    ndim = int(target.nparam)
    
    assert ndim == log_probs[target.log_prob][0]
    
    try:
        lc = pd.read_csv(f"{CWD}/data/lcs/{tstamp}_{ID}.csv")
    except:
        lc = pd.read_csv(f"{CWD}/data/lcs/{tstamp}_{ID[:-1]}.csv")
        target.ID = ID[:-1]
        
    phi = lc.phi.values
    flux = lc.flux.values
    flux_err = lc.flux_err.values
    
    qlum = target.qlum_erg_s * u.erg / u.s
    Fth = target.Fth * u.erg / u.s / (u.cm)**2
    
    inits = [target.phi_a, target.theta_a, 
                        target.a, target.fwhm, 
                        target.i_mu, target.phi0]
    
    return ndim, inits, phi, flux, flux_err, qlum, Fth, target

def get_inits_multi(ID, tstamp, nars=1):
    
    ndim, inits_a, phi, flux, flux_err, qlum, Fth, target = get_inits_one(ID+"a", tstamp)
    ndim, inits_b, phi, flux, flux_err, qlum, Fth, target = get_inits_one(ID+"b", tstamp)
    
    inits_a.insert(1, inits_b[0])
    inits_a.insert(4, inits_b[2])
    inits_a.insert(6, inits_b[3])
    
    
    assert inits_b[1]==inits_a[2]
    assert inits_b[3]==inits_a[6]
    assert inits_b[5]==inits_a[8]
    assert inits_b[0]==inits_a[1]
    assert inits_b[4]==inits_a[7]
    
    if nars == 2:
        inits_a.insert(3, inits_b[1])
        assert inits_a[-1]==inits_b[-1]
        assert inits_a[7]==inits_b[3]
    print(target.ID, "ID")
    
    return ndim, inits_a, phi, flux, flux_err, qlum, Fth, target


def get_inits(ID, tstamp, nflares, nars):
    
    if nflares == 1:
        return get_inits_one(ID, tstamp), ''
        
    elif nflares ==2 :
        return get_inits_multi(ID, tstamp, nars=nars), 'b'
        
    
def run_mcmc(ID, tstamp, nflares, nars, Nsteps=50000, wiggle=1e-3):

    parameters, suffix = get_inits(ID, tstamp, nflares, nars)
    ndim, inits, phi, flux, flux_err, qlum, Fth, target = parameters
    
    inits = [a for a in inits]
    pos = inits * (1. + wiggle * np.random.randn(nwalkers, ndim))

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = f"{CWD}/analysis/results/mcmc/{target.tstamp}_{target.ID}{suffix}_MCMC.txt"
    
    # Get prior distribution for inclination 
    inclination_path =  f"{CWD}/data/inclinations/{target.ID}_post_compound.p"
    gload = pickle.load( open( inclination_path, "rb" ) )
    
    backend = emcee.backends.Backend()
    backend.reset(nwalkers, ndim)
    args = (phi, flux, flux_err, qlum,
                Fth, (target.R_Rsun*R_sun).to("cm"), 
                target['median'], {})
    kwargs = {"g":gload}

    with Pool(7) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probs[target.log_prob][1],
                                        args=args,kwargs=kwargs, backend=backend,pool=pool)#
        start = time.time()
        sampler.run_mcmc(pos, Nsteps, progress=True, store=True)
        end = time.time()
        multi_data_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_data_time))
        
    tau = 2#sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))    
    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    
    columns = ["phase_peak","latitude_rad","a","fwhm_periods","i_rad","phase_0"]
    rawsamples = pd.DataFrame(data=samples, columns=columns)
    rawsamples.to_csv(f"{CWD}/analysis/results/mcmc/{tstamp}_{target.ID}_raw_mcmc_sample.csv",index=False)

    
    write_meta_mcmc(CWD, tstamp, target.ID, burnin, samples.shape[0], samples.shape[1], ndim)
    


def continue_mcmc(ID, tstamp, nflares, nars, Nsteps=50000):
    
    parameters, suffix = get_inits(ID, tstamp, nflares, nars)
    ndim, inits, phi, flux, flux_err, qlum, Fth, target = parameters
    
    filename = f"{CWD}/analysis/results/mcmc/{tstamp}_{ID}{suffix}_MCMC.h5"
    new_backend = emcee.backends.HDFBackend(filename)
    
    # Get prior distribution for inclination 
    inclination_path =  f"{CWD}/data/inclinations/{target.ID}_post_compound.p"
    gload = pickle.load( open( inclination_path, "rb" ) )
    
    print(f"Initial size: {filename} {new_backend.iteration}")
    print(f"Dimensions: {ndim}")
          
    YN = input("Continue? (1/0)")
    if YN != "1":
        print("Do not continue.")
        return
    elif YN == "1":
        with Pool(7) as pool:
            new_sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probs[target.log_prob][1],
                                        args=(phi, flux, flux_err, qlum,
                                              Fth, (target.R_Rsun*R_sun).to("cm"), 
                                              target['median'],
                                              {"g":gload}),
                                        backend=new_backend,pool=pool)
            start = time.time()
            new_sampler.run_mcmc(None, Nsteps, progress=True, store=True)
            end = time.time()
            multi_data_time = end - start
            print("Final size: {0}".format(new_backend.iteration))
            print("Multiprocessing took {0:.1f} seconds".format(multi_data_time))
        

if __name__ == "__main__":
# Read ID from keyboard here
    
    ID = '44984200'#input("ID? ")
    tstamp = '22_09_2020_09_37'#input("tstamp? ")
    Nsteps = 50#input("Number of steps? ")
    nflares = 1
    nars = 1
    filename = f"{CWD}/analysis/results/mcmc/{tstamp}_{ID}_MCMC.h5"

    if os.path.isfile(filename):
        print ("File exist")
        continue_mcmc(ID, tstamp, nflares, nars, Nsteps=int(Nsteps))
    else:
        print ("File does not exist")
        run_mcmc(ID, tstamp, nflares, nars, Nsteps=int(Nsteps))


    
    
