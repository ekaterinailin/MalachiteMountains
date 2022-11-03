"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This simple script creates a model light curve of a
flare that is geometrically modulated by stellar rotation, 
and shows a plot.

The flare shape is empirical from Davenport et al. (2014).
"""
import numpy as np

import matplotlib.pyplot as plt

from astropy.constants import R_sun
import astropy.units as u

from funcs.model import FlareModulator, calculate_specific_flare_flux, aflare


if __name__ == "__main__":

    # Set the model parameters
    #------------------------

    # LIGHT CURVE PARAMETERS
    #------------------------

    # Number of data points
    N = 200

    # Time array
    phi = np.linspace(0, 30*np.pi, N)

    # Time when flare peaks
    phi_a = 6.1 * np.pi

    # Rotation phase at the beginning of observations
    phi0= 7 * np.pi / 180

    # Median quiescent flux
    median = 500.

    # FLARE PARAMETERS
    #------------------------

    # what wavelength band are we using? TESS or Kepler
    mission = "TESS"

    # what temperature does the flare have in K
    flaret = 1e4

    # Flare latitude in rad
    theta_a = 35*np.pi/180

    # Flare amplitude
    a = 2.

    # FWHM of flare
    fwhm = 1. * np.pi

    # STELLAR PARAMETERS
    #------------------------

    # Stellar inclination
    i = 45 * np.pi / 180

    # Stellar luminosity
    qlum = 1e28 * u.erg / u.s

    # Stellar radius
    R = .15 * R_sun


    # Create the model flux
    #------------------------

    # modulated flux
    modulator = FlareModulator(phi, qlum, R, median=median,
                               mission=mission, flaret=flaret)

    flareparams = [(a, phi_a, fwhm)]
    modulated_flare = modulator.modulated_flux(theta_a, phi0, i, flareparams)

    # underlying flare
    flare = aflare(phi, phi_a, fwhm, a*median)

    # Create a figure that shows a typical LC
    flux = modulated_flare + np.random.rand(N) * 10.
    flux_err = np.full(N, 5.)

    plt.figure(figsize=(7,5))
    plt.plot(phi, flux)
    plt.plot(phi, flare+median)
    plt.xlabel("time")
    plt.ylabel("flux")
    plt.show()
    plt.savefig("model_lc.png")