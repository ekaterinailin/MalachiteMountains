import os

import numpy as np
import pandas as pd

import emcee

from funcs.flarefit import (log_probability,
                            log_probability_2flares,
                            log_probability_2flares2ars)

from astropy.constants import R_sun
import astropy.units as u

from multiprocessing import Pool
import time

log_probs = {"log_probability":[6, log_probability],
             "log_probability_2flares":[9, log_probability_2flares],
             "log_probability_2flares2ars":[10, log_probability_2flares2ars],}

CWD = "/".join(os.getcwd().split("/")[:-2])
nwalkers = 32

def get_inits(ID, tstamp):
    
    inits = pd.read_csv(f"{CWD}/data/summary/inits.csv")
    target = inits.loc[(inits.ID == ID) & (inits.tstamp==tstamp),:].iloc[0]
    ndim = int(target.nparam)
    
    assert ndim == log_probs[target.log_prob][0]
    
    lc = pd.read_csv(f"{CWD}/data/lcs/{tstamp}_{ID}.csv")
    phi = lc.phi.values
    flux = lc.flux.values
    flux_err = lc.flux_err.values
    
    qlum = target.qlum_erg_s * u.erg / u.s
    Fth = target.Fth * u.erg / u.s / (u.cm)**2
    
    return ndim, target, phi, flux, flux_err, qlum, Fth
    
def run_mcmc(ID, tstamp, Nsteps=50000, wiggle=1e-3):

    ndim, target, phi, flux, flux_err, qlum, Fth = get_inits(ID, tstamp)

    inits = np.array([target.phi_a, target.theta_a, 
                      target.a, target.fwhm, 
                      target.i_mu, target.phi0]) 
    pos = inits * (1. + wiggle * np.random.randn(nwalkers, ndim))

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = f"{CWD}/analysis/results/mcmc/{target.tstamp}_{target.ID}_MCMC.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    

    with Pool(5) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probs[target.log_prob][1],
                                    args=(phi, flux, flux_err, qlum,
                                          Fth, (target.R_Rsun*R_sun).to("cm"), 
                                          target['median'],
                                          {"i_mu":target.i_mu,
                                          "i_sigma":target.i_sigma}),
                                    backend=backend,pool=pool)
        start = time.time()
        sampler.run_mcmc(pos, Nsteps, progress=True, store=True)
        end = time.time()
        multi_data_time = end - start
        print("Multiprocessing took {0:.1f} seconds".format(multi_data_time))


def continue_mcmc(ID, tstamp, Nsteps=50000):
    
    ndim, target, phi, flux, flux_err, qlum, Fth = get_inits(ID, tstamp)
    
    filename = f"{CWD}/analysis/results/mcmc/{tstamp}_{ID}_MCMC.h5"
    new_backend = emcee.backends.HDFBackend(filename)
    print("Initial size: {0}".format(new_backend.iteration))
    YN = input("Continue? (1/0)")
    if YN != "1":
        print("Do not continue.")
        return
    elif YN == "1":
        with Pool(5) as pool:
            new_sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probs[target.log_prob][1],
                                        args=(phi, flux, flux_err, qlum,
                                              Fth, (target.R_Rsun*R_sun).to("cm"), 
                                              target['median'],
                                              {"i_mu":target.i_mu,
                                              "i_sigma":target.i_sigma}),
                                        backend=new_backend,pool=pool)
            start = time.time()
            new_sampler.run_mcmc(None, Nsteps, progress=True, store=True)
            end = time.time()
            multi_data_time = end - start
            print("Final size: {0}".format(new_backend.iteration))
            print("Multiprocessing took {0:.1f} seconds".format(multi_data_time))
        

if __name__ == "__main__":
# Read ID from keyboard here
    ID = '100004076'#input("ID? ")
    tstamp = '04_02_2020_12_01'#input("tstamp? ")
    Nsteps = 10#input("Number of steps? ")
    filename = f"{CWD}/analysis/results/mcmc/{tstamp}_{ID}_MCMC.h5"

    if os.path.isfile(filename):
        print ("File exist")
        continue_mcmc(ID, tstamp, Nsteps=int(Nsteps))
    else:
        print ("File does not exist")
        run_mcmc(ID, tstamp, Nsteps=int(Nsteps))


    
    
