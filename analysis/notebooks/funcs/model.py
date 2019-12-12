import numpy as np
import pandas as pd

from .helper import no_nan_inf

import astropy.units as u

from astropy.constants import c, h, k_B, R_sun, L_sun

import matplotlib.pyplot as plt

# Read in response curve ---------------------------------------------
response_curve = {"TESS":"TESS.txt",
                  "Kepler":"kepler_lowres.txt"}
for key, val in response_curve.items():
    df = pd.read_csv(f"static/{val}",
                     delimiter="\s+", skiprows=8)
    df = df.sort_values(by="nm", ascending=True)
    rwav = (df.nm * 10).values * u.angstrom #convert to angstroms
    rres = (df.response).values
    response_curve[key] = (rwav,rres)
#----------------------------------------------------------------------

def big_model(phi_a, theta_a, a, fwhm, i, phi0=0,
              phi=None, num_pts=100, qlum=None,
              Fth=None, R=None, median=0):
    """Full model.

    Parameters:
    ------------
    phi_a : float (0,2pi)
        longitude of the flare peak in rad
    theta_a : float (0, pi/2)
        latitude of the flaring region in rad
    a : float >0
        relative amplitude of the flare
    fwhm : float >0
        FWHM of the flare in fractions of 2pi
    i : float
        inclination in rad
    phi0 : float (0,2pi)
        longitude that is facing the observer a t=0
    phi : array of floats >0
        longitudes
    num_pts : int
        number of grid points
    qlum : astropy Quantity
        quiescent luminosity in defined band in erg/s
    Fth : astropy Quantity
        specific flux of the flare at a given temperature
        and in a defined band in erg/s/cm^2
    R : astropy Quantity
        stellar radius
    median : float
        quiescent flux of the light curve

    Return:
    -------
    array of floats -  model light curve
    """
    #Fth, a, qlum, R, lon, lat, i, phi0=0 __  phi_a, theta_a, i, phi0=phi0
    radius = calculate_angular_radius(Fth, a, qlum, R, 0, 0, np.pi/2, phi0=0) # the amplitude is the real one observed from the front
   # print(radius, "Radius")
    flare = aflare(phi, phi_a, fwhm, a*median,)
  #  plt.plot(phi, flare)
    latitudes, longitudes, pos = dot_ensemble_circular(theta_a, 0, radius, num_pts=num_pts)

    lamb, onoff, m = model(phi, latitudes, longitudes, flare, i, phi0=phi0)
    #plt.plot(phi, m)
   # print(lamb, np.max(lamb), np.min(lamb))
    #print(lamb, onoff)
    return m + median

def model(phi, latitudes, longitudes, flare, inclination, phi0=0.):
    """Take a flare light curve and a rotating ensemble of latitudes
    and longitudes, and let it rotate.

    Parameters:
    -----------
    phi :  N-array
        longitudes to evaluate the model at in rad
    latitudes : M-array
        latitudes of the spot grid points in rad
    longitudes : M-array
        longitudes of the spot grid points in rad
    flare : N-array
        flare shape
    inclination : float
        inclination in rad
    phi0 : float
        longitude facing the observer

    Return:
    --------
    lambert modifier for flux - N-array
    onoff day and night step function - N-array
    model light curve - N-array
    """

    if no_nan_inf([phi, latitudes, longitudes, flare, inclination, phi0]) == False:
        raise ValueError("One of your inputs in model() is or contains NaN or Inf.")



    # Check if the dimensions of the inputs are right
    l = len(latitudes)
    assert l == len(longitudes)
    assert len(phi) == len(flare)
    # -----------------------------------------------

    # Get daylengths for all grid points
    # and calculate day/night switches:
    Ds = daylength(latitudes, inclination)
    onoff = np.full((l,phi.shape[0]),0)

    for i,(d,lon) in enumerate(zip(Ds,longitudes)):# How can I avoid this loop?
        onoff[i,:] = on_off(phi-lon, d, phi0=phi0)
    #------------------------------------------------

    # Why isn't it possible to just fill an array here?
    # Refactor later ...
    # Anyways: Calculate the lambert modifier:
    latlon = np.concatenate((latitudes.reshape(l,1),
                             longitudes.reshape(l,1)),
                            axis=1)
    A = []
    for i, ll in enumerate(latlon):
        a = lambert(phi-ll[1], inclination, ll[0], phi0=phi0)
        A.append(a)
    lamb = np.array(A)

    #--------------------------------------------------

    # Give intermediate results: lamb, onoff
    # Also co-add all grid points and average them
    # after folding with flare:
    return lamb, onoff, np.sum(lamb * onoff, axis=0) * flare / l



def dot_ensemble_circular(lat, lon, radius, num_pts=200):
    """For radii smaller than 5 deg, you can approximate
    a uniform grid on a sphere by projecting a uniform
    grid on a circular surface (sunflower shape).

    Here we create a grid on a pole and then rotate it
    around the y- and then around the z- axis to center
    it on the latitute and longitude we need.


    Parameters:
    ------------
    lat : float
        latitude in rad
    lon : float
        longutude in rad
    radius : float
        radius of the spot in deg
    num_pts: int
        number of grid points

    Return:
    -------
    latitudes, longitudes: arrays of floats
        positions of grid points in rad
    pos : tuple of arrays of float
        cartesian coordinates of grid points

    """
    if no_nan_inf([lat, lon, radius]) == False:
        raise ValueError("One of your inputs in "\
                         "dot_ensemble_circular is"\
                         " or contains NaN or Inf.")

    # Create grid as a circle in z=1 plane centered on the z-axis
    # The grid sits on the pole, that is.
    lon = -lon
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    radius = radius / 180 * np.pi #radius in rad
    rad = np.tan(radius) # project radius to the unit sphere
    r = np.sqrt(indices / num_pts) * rad #apply projected radius to grid
    theta = np.pi * (1 + 5**0.5) * indices #use formula

    # Now rotate the grid down to the position of the spot

    # Rotation matrix [R_z(lon) x R_y(pi/2-lat)]
    cl, sl, ca, sa = np.cos(lon), np.sin(lon), np.cos(lat), np.sin(lat)
    Mrot = np.array([[cl * sa,   -sl, cl * ca],
                     [sl * sa,    cl, sl * ca],
                     [    -ca,    0,      sa]])

    # Vectors of projected grid dots on the pole (x,y,z)
    sina, cosa,  = r / np.sqrt(r**2 + 1), np.cos(np.arctan(r))
    st, ct = np.sin(theta), np.cos(theta)
    vec = np.array([sina * ct, sina * st, cosa])

    # Matrix multiplication
    x,y,z = (Mrot@vec)

    # Convert cartesian positions back to latitudes and longitudes:
    lats = np.arcsin(z)
    lons = np.arctan2(y, x)

    return lats, lons, (x,y,z)


def dot_ensemble_spherical(lat, lon, radius, num_pts=5e4):
    """Create an ensemble of dots on a sphere.

    Method see:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    answered by CR Drost

    Parameters:
    -----------
    lat : float
        latitude of center of ensemble in rad
    lon : float
        longitude of center of ensemble in rad
    radius : float
        angular radius of the ensemble in deg
    num_pts : int
        number of points used to generate the
        full sphere evenly covered with dots
        in a sunflower shape

    Return:
    -------
    latitudes, longitudes  -  np.arrays of dots
    that go into the ensemble.
    """
    if no_nan_inf([lat, lon, radius, num_pts]) == False:
        raise ValueError("One of your inputs in "\
                         "dot_ensemble_spherical is"\
                         " or contains NaN or Inf.")

    # This is CR Drost's solution to the sunflower spiral:
    indices = np.arange(0, num_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices/num_pts) #latitude
    theta = np.pi * (1 + 5**0.5) * indices #longitude

    # Fold onto on sphere
    phi = np.pi / 2 - phi % (2 * np.pi)
    theta = theta % (np.pi * 2)


    # Calculate the distance of the dots to the center of the ensemble
    gcs = great_circle_distance(lat, lon, phi, theta)

    # If distance is small enough, include in ensemble
    a = np.where(gcs < (radius * np.pi / 180))[0]
    return phi[a], theta[a]

def great_circle_distance(a, la, b, lb):
    """Calcultate the angular distance on
    a great circle than runs through two points
    on a globe at
    (latitude, longitude)
    (a,la) and
    (b, lb).

    See also: https://en.wikipedia.org/wiki/Great-circle_distance

    Parameters:
    ------------
    a, la, b, lb : float
     latitudes and longitudes in rad

    Return:
    -------
    float - angular distance on a globe in rad
    """
    if no_nan_inf([a, la, b, lb]) == False:
        raise ValueError("One of your inputs is or contains NaN or Inf.")

    return np.arccos(np.sin(a) * np.sin(b) + np.cos(a) * np.cos(b) * np.cos(la-lb))

def lambert(phi, i, l, phi0=0.):
    """Calculate Lambert's law of geometric
    brightness modulation (prop. cos(incident angle))
    from known stellar inclination, and the
    spots latitude and longitudes.

    Parameters:
    -----------
    phi : array or float
        longitudes
    i : float
        inlcination in rad
    l : float
        latitude in rad

    Return:
    -------
    Array of values between 0 and 1 that define the
    fraction of the flux that we would receive from
    a point at the center of the stellar disk.

    Wikipedia is great:
    https://en.wikipedia.org/wiki/Great-circle_distance
    https://en.wikipedia.org/wiki/Lambert%27s_cosine_law
    """
    if no_nan_inf([l,i,phi,phi0]) == False:
        raise ValueError("One of your inputs is or contains NaN or Inf.")
    return np.sin(l) * np.cos(i) + np.cos(l) * np.sin(i) * np.cos(phi-phi0)

def on_off(phi, daylength, phi0=0.):
    """Calculate the visibility step function
    of a point on a rotating sphere as a function of
    longitude phi.

    phi0 is facing the observer.

    Parameters:
    ------------
    phi : array
        longitudes
    daylength : float
        fraction of rotation period (0,1)
    phi0 : float
        longitude facing the observer
        default 0, range [0,2pi]

    Return:
    -------
    array of 1 and 0 if phi is an array, 1=visible, 0=hidden
    else bool, True=visible, False=hidden

    """
    def condition(phi,phi0,daylength):

        # condition for being hidden on the back of the star
        if daylength==0.:
            return True
        else:
            return (((phi-phi0)%(2*np.pi) > daylength*np.pi) &
                ((phi-phi0)%(2*np.pi) < (2-daylength)*np.pi))

    if (np.isnan(daylength) | np.isnan(phi0) | (not np.isfinite(phi0)) | (not np.isfinite(daylength))):
        raise ValueError("Daylength or phi0 is NaN or not finite.")

    if isinstance(phi, np.ndarray):

        if ((np.isnan(phi).any()) | (not np.isfinite(phi).any())):
            raise ValueError("One phi value is NaN or not finite")

        # everything is visible by default
        res = np.full(phi.shape[0],1)

        # if longitude is on the night side, set visibility to 0:
        res[condition(phi,phi0,daylength)] = 0
        return res

    elif ((isinstance(phi, float)) | (isinstance(phi, int))):

        if ((np.isnan(phi)) | (not np.isfinite(phi))):
            raise ValueError("Your phi value is NaN")
        # True if visible
        return not condition(phi,phi0,daylength)


    else:
        raise ValueError("Phi must be float, int, or an array of either.")

def daylength(l, i, P=1.):
    """Determine the day length, as in here:
    http://www.math.uni-kiel.de/geometrie/klein/mpss13/do2706.pdf

    If P is not specified, the daylength is measured in
    rotation periods.

    Parameters:
    ------------
    l : array
        latitude in rad
    i : float
        inclination in rad
    P : float
        rotation period, default is 1.

    Return:
    -------
    float daylength in the same units as the rotation period
    """

    def formula(l,i):
        return np.arccos(-np.tan(l) * np.tan(np.pi/2-i)) / np.pi

    if ((i > np.pi/2) | (i < 0)):
        raise ValueError("Inclination must be in [0,pi/2]")

    if isinstance(l,np.ndarray):

        if np.isnan(i) | np.isnan(l).any():
            raise ValueError("Inclination or some latitude is NaN.")

        elif ((l > np.pi/2).any() | (l < -np.pi/2).any()):
            raise ValueError("Latitude must be in [-pi/2,pi/2]")

        res = formula(l,i)
      #  res[l>=i] = P
        res[np.abs(l) >=i] = 0
        res[l>=i] = P

     #   print(res*P)
        return res * P

    elif ((isinstance(l, float)) | (isinstance(l, int))):

        if np.isnan(i) | np.isnan(l):
            raise ValueError("Inclination or latitude is NaN.")

        elif (l > np.pi/2) | (l < -np.pi/2):
            raise ValueError("Latitude must be in [-pi/2,pi/2]")

        if l >= i:
            return P

        elif ((l<0) & (np.abs(l) >=i)):
            return 0

        else:
            return formula(l,i) * P



def black_body_spectrum(wav,t):
    """Takes an array of wavelengths and
    a temperature and produces an array
    of fluxes from a thermal spectrum

    Parameters:
    -----------
    wav : Astropy array
        wavenlength array
    t : float
        effective temperature in Kelvin
    """
    t = t * u.K # set unit to Kelvin

    return (( (2 * np.pi * h * c**2) / (wav**5) / (np.exp( (h * c) / (wav * k_B * t) ) - 1))
            .to("erg/s/cm**3")) #simplify the units


# def calculate_relative_flare_area(dist, rad, qflux, amp, mission, flaret=1e4):
#     """Get the flare area in rel. unit

#     Parameters:
#     -----------
#     dist : float
#         distance to target in parsecs
#     rad : float
#         radius in solar radii
#     qflux : float
#         quiescent flux in erg/s/cm^2
#     amp : float
#         (0,inf) flare amplitude in rel. flux
#     mission : string
#         TESS or Kepler
#     flaret : float
#         flare black body temperature, default 10kK
#     """
#     # Give units to everything:
#     dcm = dist * u.pc
#     rcm = rad * R_sun
#     qflux = qflux * u.erg/u.s/u.cm**2

#     # Read in response curve:
#     response_curve_path = {"TESS":"TESS.txt",
#                            "Kepler":"kepler_lowres.txt"}
#     df = pd.read_csv(f"static/{response_curve_path[mission]}",
#                      delimiter="\s+", skiprows=8)
#     df = df.sort_values(by="nm", ascending=True)
#     rwav = (df.nm * 10).values * u.angstrom #convert to angstroms
#     rres = (df.response).values

#     # create an array to upsample the filter curve to
#     w = np.arange(3000,13001) * u.angstrom

#     # interpolate thermal spectrum onto response
#     # curve wavelength array, then sum up
#     # flux times response curve:

#     # Generate a thermal spectrum at the flare
#     # temperature over an array of wavelength w:
#     thermf = black_body_spectrum(w, flaret)

#     # Interpolate response from rwav to w:
#     rres = np.interp(w,rwav,rres)

#     # Integrating the flux of the thermal
#     # spectrum times the response curve over wavelength:
#     calflareflux = np.trapz(thermf * rres, x=w)

#     return (((amp * qflux) / (calflareflux)) * (dcm / rcm)**2.).decompose()

def calculate_specific_flare_flux(mission, flaret=1e4):
    """Get the flare area in rel. unit

    Parameters:
    -----------
    mission : string
        TESS or Kepler
    flaret : float
        flare black body temperature, default 10kK

    Return:
    -------
    specific flare flux in units erg/
    """
    if no_nan_inf([flaret]) == False:
        raise ValueError("flaret is NaN or Inf.")

    try:
        # Read in response curve:
        rwav, rres = response_curve[mission]
    except KeyError:
        raise KeyError("Mission can be either Kepler or TESS.")

    # create an array to upsample the filter curve to
    w = np.arange(3000,13001) * u.angstrom

    # interpolate thermal spectrum onto response
    # curve wavelength array, then sum up
    # flux times response curve:

    # Generate a thermal spectrum at the flare
    # temperature over an array of wavelength w:
    thermf = black_body_spectrum(w, flaret)

    # Interpolate response from rwav to w:
    rres = np.interp(w,rwav,rres, left=0, right=0)

    # Integrating the flux of the thermal
    # spectrum times the response curve over wavelength:
    return np.trapz(thermf * rres, x=w).to("erg/cm**2/s")


def calculate_angular_radius(Fth, a, qlum, R, lon, lat, i, phi0=0):
    """Calculate angular radius in degrees from.

    Parameters:
    ------------
    Fth : astropy value > 0
        specific flare flux in erg/cm^2/s
    a : float > 0
        relative flare amplitude
    qlum : astropy value > 0
        quiescent luminosity in erg/s
    R : float > 0
        stellar radius in solar radii
    lon : float or array (0,2pi)
        longitude in rad
    lat : float or array(-pi/2, pi/2)
        latitude in rad
    i : float (0, pi/2)
        inclination

    Return:
    -------
    float - radius of flaring area in deg
    """
    #phi, i, l, phi0=0.
    A = (a * qlum) / (Fth * lambert(lon%(2*np.pi), i, lat, phi0=phi0))
   # print("AREA", A, 4 * np.pi * R**2)
    if np.sqrt( A / (4 * np.pi * R**2)) > 1:
        raise ValueError("Flare area seems larger than stellar surface.")

    return (np.arcsin(np.sqrt( A / (4 * np.pi * R**2) )) * 2.).to("deg").value

def aflare2(time, t0, fwhm, ampl, median, upsample=True, uptime=10):
    """Modified `aflare` classic flare shape.
    Originally from Davenport et al. 2014, but
    adding the median flux value to return the
    full flux.

    See `aflare` in:
    https://github.com/ekaterinailin/AltaiPony/blob/master/altaipony/fakeflares.py

    Upsamples the flare model by a factor of 10
    per default and samples back down again
    to improve the energy estimate.


    """
    return (aflare(time, t0, fwhm*2, ampl,
                   upsample=upsample, uptime=uptime) + median)

def aflare(t, tpeak, dur, ampl, upsample=False, uptime=10):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Use this function for fitting classical flares with most curve_fit
    tools.
    Note: this model assumes the flux before the flare is zero centered
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    dur : float
        The duration of the flare
    ampl : float
        The amplitude of the flare
    upsample : bool
        If True up-sample the model flare to ensure more precise energies.
    uptime : float
        How many times to up-sample the data (Default is 10)
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    fwhm = dur/2. # crude approximation for a triangle shape would be dur/2.

    if upsample:
        dt = np.nanmedian(np.diff(t))
        timeup = np.linspace(min(t)-dt, max(t)+dt, t.size * uptime)

        flareup = np.piecewise(timeup, [(timeup<= tpeak) * (timeup-tpeak)/fwhm > -1.,
                                        (timeup > tpeak)],
                                    [lambda x: (_fr[0]+                       # 0th order
                                                _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                                _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                                _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                                _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                     lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                                _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                    ) * np.abs(ampl) # amplitude

        # and now downsample back to the original time...
        ## this way might be better, but makes assumption of uniform time bins
        # flare = np.nanmean(flareup.reshape(-1, uptime), axis=1)

        ## This way does linear interp. back to any input time grid
        # flare = np.interp(t, timeup, flareup)

        ## this was uses "binned statistic"
        downbins = np.concatenate((t-dt/2.,[max(t)+dt/2.]))
        flare,_,_ = binned_statistic(timeup, flareup, statistic='mean',
                                     bins=downbins)

    else:
        flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm > -1.,
                                 (t > tpeak)],
                                [lambda x: (_fr[0]+                       # 0th order
                                            _fr[1]*((x-tpeak)/fwhm)+      # 1st order
                                            _fr[2]*((x-tpeak)/fwhm)**2.+  # 2nd order
                                            _fr[3]*((x-tpeak)/fwhm)**3.+  # 3rd order
                                            _fr[4]*((x-tpeak)/fwhm)**4. ),# 4th order
                                 lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm)*_fd[1] ) +
                                            _fd[2]*np.exp( ((x-tpeak)/fwhm)*_fd[3] ))]
                                ) * np.abs(ampl) # amplitude

    return flare
