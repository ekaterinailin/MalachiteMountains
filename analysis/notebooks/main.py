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

log_probs = {"log_probability":[7, log_probability],
             "log_probability_2flares":[11, log_probability_2flares]}

CWD = "/".join(os.getcwd().split("/")[:-2])
nwalkers = 32

def get_inits_one(ID, tstamp):

    inits = pd.read_csv(f"{CWD}/data/summary/inits_decoupled.csv")
    print(inits, inits.ID, ID, inits.tstamp, tstamp)
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
                        target.a, target.fwhm1,target.fwhm2,
                        target.i_mu, target.phi0]

    return ndim, inits, phi, flux, flux_err, qlum, Fth, target, lc

def get_inits_multi(ID, tstamp, nars=1):

    ndim, inits_a, phi, flux, flux_err, qlum, Fth, target, lc = get_inits_one(ID+"a", tstamp)
    ndim, inits_b, phi, flux, flux_err, qlum, Fth, target, lc = get_inits_one(ID+"b", tstamp)
    print("INITS ", inits_a, inits_b)

    inits = [inits_a[0], inits_b[0], inits_a[1], inits_a[2], inits_b[2],
             inits_a[3], inits_b[3], inits_a[4], inits_b[4], inits_a[5],
             inits_a[6]]

    print(target.ID, "ID")

    return ndim, inits, phi, flux, flux_err, qlum, Fth, target, lc


def get_inits(ID, tstamp, nflares, nars):

    if nflares == 1:
        return get_inits_one(ID, tstamp), ''

    elif nflares ==2 :
        return get_inits_multi(ID, tstamp, nars=nars), 'b'


def run_mcmc(ID, tstamp, nflares, nars, Nsteps=50000, wiggle=1e-3):

    parameters, suffix = get_inits(ID, tstamp, nflares, nars)
    ndim, inits, phi, flux, flux_err, qlum, Fth, target, lc = parameters

    inits = [a for a in inits]
    pos = inits * (1. + wiggle * np.random.randn(nwalkers, ndim))

    # Get Prot_d
    props = pd.read_csv(f"{CWD}/data/summary/inclination_input.csv")

    Prot_d = props[props["id"] == int(ID)].iloc[0].prot_d
    print(Prot_d)

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

    samples = sampler.get_chain(discard=Nsteps//10, flat=True, thin=50)

    if ndim==7:

        columns = ["phase_peak","latitude_rad","a","fwhm1_periods","fwhm2_periods","i_rad","phase_0"]
        rawsamples = pd.DataFrame(data=samples, columns=columns)
        rawsamples.to_csv(f"{CWD}/analysis/results/mcmc/{tstamp}_{target.ID}_raw_mcmc_sample.csv",index=False)

        # map phi0 to phi_peak longitude, still call it phi0
        samples[:, -1] = (samples[:, 0]%(2.*np.pi) - samples[:, -1]) / np.pi * 180. # 0 would be facing the observer

        #map phi_a_distr to t0_distr:
        samples[:, 0] = np.interp(samples[:,0],lc.phi,lc.t)

        # convert theta_f to degrees
        samples[:, 1] = samples[:, 1] / np.pi * 180.

        # convert FWHM1 to days
        samples[:, 3] = samples[:, 3]/2/np.pi * Prot_d

        # convert FWHM2 to days
        samples[:, 4] = samples[:, 4]/2/np.pi * Prot_d

        # convert i to degrees
        samples[:, -2] = samples[:, -2] / np.pi * 180.

        columns = ["t0_d","latitude_deg","a",
                  "fwhm1_d","fwhm2_d","i_deg","phase_deg"]

        resultframe = pd.DataFrame(data=samples,
                                columns=columns)

        resultframe.to_csv(f"{CWD}/analysis/results/mcmc/"
                        f"{tstamp}_{target.ID}_converted_mcmc_sample.csv",
                        index=False)

    elif ndim==11:

        columns = ["phase_peak_a", "phase_peak_b", "latitude_rad",
                   "a_a", "a_b", "fwhm_periods_a1","fwhm_periods_a2",
                   "fwhm_periods_b1","fwhm_periods_b2",
                   "i_rad","phase_0"]

        rawsamples = pd.DataFrame(data=samples, columns=columns)

        for suffix in ["a","b"]:
            rawsamples1 = rawsamples[[f"phase_peak_{suffix}",
                                    "latitude_rad",
                                    f"a_{suffix}",
                                    f"fwhm_periods_{suffix}1",
                                      f"fwhm_periods_{suffix}2",
                                    "i_rad",
                                    "phase_0"]]
            rawsamples2 = rawsamples1.rename(index=str,
                                            columns=dict(zip([f"phase_peak_{suffix}",
                                                            f"a_{suffix}",
                                                            f"fwhm_periods_{suffix}1",
                                                             f"fwhm_periods_{suffix}1"],
                                                            ["phase_peak",
                                                            "a",
                                                            "fwhm1_periods",
                                                            "fwhm2_periods",])))
            rawsamples2.to_csv(f"{CWD}/analysis/results/mcmc/"
                            f"{tstamp}_{target.ID}{suffix}"
                            f"_raw_mcmc_sample.csv",
                            index=False)

        # map phi0 to phi_peak longitude, still call it phi0
        samples[:, -1] = (samples[:, 0]%(2.*np.pi) - samples[:, -1]) / np.pi * 180. # 0 would be facing the observer

        #map phi_a_distr to t0_distr:
        for i in [0,1]:
            samples[:, i] = np.interp(samples[:,i],lc.phi,lc.t)

        # convert theta_f to degrees
        samples[:, 2] = samples[:, 2] / np.pi * 180.

        # convert FWHM to days
        for i in [5,6,7,8]:
            samples[:, i] = samples[:, i]/2/np.pi * Prot_d

        # convert i to degrees
        samples[:, -2] = samples[:, -2] / np.pi * 180.



        columns = ["t0_d_a","t0_d_b","latitude_deg",
                   "a_a","a_b","fwhm_d_a1","fwhm_d_a2",
                   "fwhm_d_b1","fwhm_d_b2","i_deg","phase_deg"]

        rawsamples = pd.DataFrame(data=samples, columns=columns)


        for suffix in ["a","b"]:
            rawsamples1 = rawsamples[[f"t0_d_{suffix}",
                                    "latitude_deg",
                                    f"a_{suffix}",
                                    f"fwhm_d_{suffix}1",
                                      f"fwhm_d_{suffix}2",
                                    "i_deg",
                                    "phase_deg"]]
            rawsamples2 = rawsamples1.rename(index=str,
                                            columns=dict(zip([f"t0_d_{suffix}",
                                                            f"a_{suffix}",
                                                            f"fwhm_d_{suffix}1",
                                                             f"fwhm_d_{suffix}2"],
                                                            ["t0_d",
                                                            "a",
                                                            "fwhm1_d",
                                                            "fwhm2_d"])))
            rawsamples2.to_csv(f"{CWD}/analysis/results/mcmc/"
                            f"{tstamp}_{target.ID}{suffix}"
                            f"_converted_mcmc_sample.csv",
                            index=False)





    write_meta_mcmc(CWD, tstamp, target.ID, 1000, samples.shape[0], samples.shape[1], ndim)



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
    tstamp = '06_11_2020_15_47'#input("tstamp? ")30_10_2020_11_30
    Nsteps =10000#input("Number of steps? ")
    nflares = 1
    nars = 1
    filename = f"{CWD}/analysis/results/mcmc/{tstamp}_{ID}_MCMC.h5"

    if os.path.isfile(filename):
        print ("File exist")
        continue_mcmc(ID, tstamp, nflares, nars, Nsteps=int(Nsteps))
    else:
        print ("File does not exist")
        run_mcmc(ID, tstamp, nflares, nars, Nsteps=int(Nsteps))




