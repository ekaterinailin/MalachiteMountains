"""
UTF-8, Python 3

------------------
MalachiteMountains
------------------

Ekaterina Ilin, 2020, MIT License


This simple script creates a model light curve of a
flare that is geometrically modulated by stellar rotation,
 and shows a plot.

The flare shape is empirical from Davenport et al. (2014)
"""
import matplotlib.pyplot as plt

from astropy.constants import R_sun
import astropy.units as u

import numpy as np

from funcs.model import full_model, calculate_specific_flare_flux, aflare


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

    # Specific flare flux 
    Fth = calculate_specific_flare_flux("TESS", flaret=1e4)

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
    qlum = 1e28 * u.erg/u.s

    # Stellar radius
    R = .15 * R_sun


    # Create the model flux
    #------------------------

    # modulated flux
    m = full_model(phi_a, theta_a, a, fwhm, i, phi0=phi0,
                  phi=phi, num_pts=50, qlum=qlum,
                  Fth=Fth, R=R, median=median)

    # underlying flare
    flare = aflare(phi, phi_a, fwhm, a*median,)

    # Create a figure that shows a typical LC
    flux = m + np.random.rand(N) * 10.
    flux_err = np.full(N, 5.)

    plt.figure(figsize=(7,5))
    plt.plot(phi, flux)
    plt.plot(phi, flare+median)
    plt.xlabel("time")
    plt.ylabel("flux")
    plt.show()
    #plt.savefig("model_lc.png")