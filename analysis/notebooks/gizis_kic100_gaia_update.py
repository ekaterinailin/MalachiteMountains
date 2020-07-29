"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This script checks if Gaia distances gave any
update on Gizis+2013 estimate on the inclination
of KIC 100004076.

We do this by reproducing their model, and fitting the
same model but with Gaia distances.

posterior: L, R, sini
observables: L, Teff, vsini

The answer is: i value is improved 
The Gaia distance is consistent with Gizis' distance within 1.5 sigma,
but has much small uncertainties, giving better constraint on i.


"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from astropy.constants import R_sun
import astropy.units as u

from funcs.flarefit import uninformative_prior, calculate_posterior_value_that_can_be_passed_to_mcmc

import os

CWD = "/".join(os.getcwd().split("/")[:-2]) 

import corner
import emcee

import time
# Create a time stamp for this run
tstamp = time.strftime("%d_%m_%Y", time.localtime())


def model(d, R, i, P):
    """Get L, Teff, and vsini observables
    from generative model of KIC 1000.
    """
    # Calculate magnitude and DM from absolute Ks magnitude
    Ks = 11.77 # Gizis+2013 Table 1
    ms = Ks - 5 * np.log10(d) + 5 +3.22
    dm = 4.83 - ms
    
    # Get luminosity from DM
    L_o_L_sun = np.power(2.512, dm)
    
    # Get Teff from Stefan-Boltzmann
    Teff = 5778 * np.sqrt(1/R) * np.sqrt(np.sqrt(L_o_L_sun))
    
    # Get vsini from inclination, period, and radius
    vsini = R*R_sun * np.sin(i) * np.pi * 2. / (P*u.d)
    
    return L_o_L_sun, Teff, vsini.to("km/s").value

def log_prior(theta, R_min=0.0,
              R_max=1, d_min=0,
              d_max=200):
    """
    Uninformative priors on P, distance, and inclination.
    
    Parameters:
    ------------
    theta : tuple
        d, R, i
    """
    d, R, i =  theta

    prior = (uninformative_prior(R, R_min, R_max) +
             uninformative_prior(d, d_min, d_max) +
             uninformative_prior(i, 0, np.pi/2))

    return calculate_posterior_value_that_can_be_passed_to_mcmc(prior)


def log_likelihood(theta, P, logLobs, logLerr):
    """Log likelihood function assuming
    Gaussian uncertainties in the data points.
    
    Parameters:
    -----------
    theta : 
    P : 
    """

    d, R, i = theta
 
    L_o_L_sun, Teff, vsini = model(d, R, i, P)
 
    logL = np.log10(L_o_L_sun)

    # hard coded value from Gizis+2013 Table 1
    Teffobs = 2300 
    Tefferr = 75 
    
    # hard coded value from Gizis+2013 Table 1
    vsiniobs = 11.2
    vsinierr = .2 

    val = -0.5 * ((logL - logLobs) ** 2 / logLerr**2 + np.log(logLerr**2) +
                  (vsini - vsiniobs) ** 2 / vsinierr**2 + np.log(vsinierr**2) +
                  (Teff - Teffobs) ** 2 / Tefferr**2 + np.log(Tefferr**2) )
    return val


def log_probability(theta, P, logLobs, logLerr):
    """Posterior probability to pass to MCMC sampler.
    """
    lp = log_prior(theta)
    
    # Check if prior is okay
    if not np.isfinite(lp):
        return -np.inf
    
    # Calculate log likelihood
    try:
        ll = log_likelihood(theta, P, logLobs, logLerr)
    
    # If something goes wrong throw value away
    except:
        return -np.inf
    
    # If NaN is returned throw value away
    if np.isnan(ll):
        return -np.inf
    
    # Return posterior
    return lp + ll



# Note

#     # Data from Gizis+2013 Table 1
#     # Use to reproduce their results:  
#     logLobs = -3.67
#     logLerr = .03

# Calculate luminosity from Gaia distance


if __name__ == "__main__":

    p = 59.671 # Gaia parallax
    ep = 0.1363 # Gaia parallax error
    
    # number of steps in MCMC
    N = 100

    gaia_d = []
    gaia_logL = []

    for x in [p, p+ep, p-ep]:
        d = 1 / ((x) * 1e-3) 
        Ks = 11.77 # Gizis+2013 Table 1
        ms = Ks - 5 * np.log10(d) + 5 +3.22
        dm = 4.83 - ms

        # Get luminosity from DM
        L_o_L_sun_gaia = np.log10(np.power(2.512, dm))

        gaia_d.append(d)
        gaia_logL.append(L_o_L_sun_gaia)


    # Create a table with Gaia derived distance and bolometric luminosity
    data = np.array([[gaia_d[0], gaia_d[0]-gaia_d[1], gaia_d[2]- gaia_d[0]],
              [gaia_logL[0], gaia_logL[2]- gaia_logL[0], gaia_logL[0]-gaia_logL[1]]]).T

    df = pd.DataFrame(data=data, columns = ["gaia_d","gaia_L"], index=["val","uperr","loerr"])


    # guess starting point 
    inits = np.array([16.7867, .088, 1.05])
    pos = inits * (1. + 1e-3 * np.random.randn(32, 3))
    nwalkers, ndim = pos.shape

    # Set up the backend
    # Don't forget to clear it in case the file already exists
    filename = (f"{CWD}/analysis/results/mcmc/Gizis_KIC100_Gaia_Update/"
                f"{tstamp}_Gizis_KIC100_Gaia_Update_uninformative_5500_steps.h5")

    backend = emcee.backends.HDFBackend(filename)
    #backend.reset(nwalkers, ndim)

    # Set up MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    # Prot, logL, logLerr:
                                    args=(0.37017, df.loc["val","gaia_L"], df.loc["uperr","gaia_L"]),
                                    backend=backend)

    # Run MCMC
    #sampler.run_mcmc(pos, N, progress=True, store=True);
    sampler.run_mcmc(None, N, progress=True, store=True);
    
    print("Final size: {0}".format(backend.iteration))

    # Plot chain
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels=[r"$d$ [pc]", r"$R$ [$R_\odot$]", r"$i$ [rad]"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    chainpath = (f"{CWD}/analysis/plots/Gizis_KIC100_Gaia_Update/"
                 f"{tstamp}_KIC1000_MCMC_uninformative_gaia_{backend.iteration}_steps_chain.png")
    plt.savefig(chainpath, dpi=300)

    # Plot corner

    tau = sampler.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    thin = int(0.5 * np.min(tau))
    print(f"tau = {tau}, burnin={burnin}, thin={thin}")

    samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)

    # convert i from rad to deg
    samples[:,2] = samples[:,2] / np.pi * 180. 

    fig = corner.corner(samples)

    cornerpath = (f"{CWD}/analysis/plots/Gizis_KIC100_Gaia_Update/"
                  f"{tstamp}_KIC1000_MCMC_uninformative_gaia_{backend.iteration}_steps.png")
    plt.savefig(cornerpath, dpi=300)


    # Get values for d, R and i from posterior    
    for num, val in [(0,"d"), (1,"R"),(2,"i")]:
        vals = np.percentile(samples[:, num], [50, 84, 16])
        df[f"{val}_posterior"] = [vals[0], vals[1] - vals[0], vals[0] - vals[2]]


    # For completeness: add Gizis+2013 Table 1 values for Teff and vsini
    df["Teff_Gizis"] = [2300, 75, 75]
    df["vsini"] = [11.2, .2, .2]

    # Write out full table

    tablepath = (f"{CWD}/analysis/results/mcmc/Gizis_KIC100_Gaia_Update/"
                 f"{tstamp}_KIC1000_MCMC_uninformative_gaia_{backend.iteration}_steps.csv")
    with open(tablepath, "w") as f:
        f.write("# Update on Gizis+2013 estimate of L1 "
                "dwarf inclination with Gaia parallax\n")
        f.write("# In the posteriors uperr and loerr designate \n" 
                "# the difference between 84th and 50th;"
                " and 16th and 50th percentiles.\n")
        df.to_csv(f)