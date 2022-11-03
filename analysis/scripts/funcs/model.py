import numpy as np
import pandas as pd

from .helper import no_nan_inf, create_spherical_grid
from .flarefit import (uninformative_prior, 
                       empirical_prior, 
                       calculate_posterior_value_that_can_be_passed_to_mcmc)

import astropy.units as u
from astropy.constants import c, h, k_B, R_sun, L_sun

from scipy.stats import binned_statistic


# Read in response curve ---------------------------------------------

response_curve = {"TESS" : "TESS.txt",
                  "Kepler" : "kepler_lowres.txt"}

for key, val in response_curve.items():
    df = pd.read_csv(f"static/{val}",
                     delimiter="\s+", skiprows=8)
    df = df.sort_values(by="nm", ascending=True)
    rwav = (df.nm * 10).values * u.angstrom #convert to angstroms
    rres = (df.response).values
    response_curve[key] = (rwav,rres)

#----------------------------------------------------------------------

# Create a spherical grid only once for the entire analysis ------------

PHI, THETA = create_spherical_grid(int(1e4)) #lat, lon


class FlareModulator:
    """Class to calculate the flare light curve as it is modulated by stellar rotation.

    Attributes
    ----------
    phi : array-like
        The time array of the light curve in longitudinal angles
    flux : array-like
        The flux array of the light curve.
    flux_err : array-like
        The flux error array of the light curve.
    qlum : float
        The bolometric luminosity of the star.
    R : float   
        The radius of the star.
    median : float
        The median flux of the light curve.
    num_pts : int
        The number of points to use for the grid on the stellar surface.
    Fth : float
        Specific flare flux, depending on mission.
    nflares : int
        Number of flares in the light curve.
    iscoupled : bool
        Whether the flares have coupled rise and decay phases or not or not.
    mission : str
        The mission the light curve is from. Kepler or TESS.
    flaret : float
        The flare temperature in K.
    """
    def __init__(self, phi, flux, flux_err, qlum, R,
                 median, nflares, iscoupled, num_pts=100,
                 mission="TESS", flaret=1e4):
        
        self.phi = phi
        self.flux = flux
        self.flux_err = flux_err
        self.qlum = qlum
        self.R = R
        self.median = median
        self.num_pts = num_pts

        self.Fth = calculate_specific_flare_flux(mission, flaret=flaret)

        self.nflares = nflares
        self.iscoupled = iscoupled

    def flare_template(self, params):
        """
        Picks a flare tmeplate based whether the flares' rise and 
        decay phase are coupled or not. If it is coupled, only
        one FWHM is used. If it is not coupled, two FWHM are used, 
        one for the rise and one for the decay phase.
        
        Parameters
        ----------
        params : array-like
            The parameters of the flare.

        """
    
        if self.iscoupled == True:
            # params = [ampl, tpeak, fwhm, (fwhm2)]
            return aflare(self.phi, params[1], params[2], params[0])
        elif self.iscoupled == False:
            return aflare_decoupled(self.phi, params[1], params[2:4], params[0])

    def modulated_flux(self, theta, phi0, i, flareparams):
        

        ms = [] # list of flares

        # model each flare
        for params in flareparams:
            # calculate the angular radius of the flare based on its amplitude, 
            # specific flare flux, stellar luminosity and radius
            # (the amplitude is the real one observed from the front)
            radius = calculate_angular_radius(self.Fth, params[0], self.qlum, self.R) 

            # get a flare template, either with coupled or decoupled FWHM
            flare = self.flare_template(params)
      
            # calculate the latitudes and longitudes of a flaring spot grid
            # if the radius is relatively small, use a circular approximation
            if radius < 10: #deg
                latitudes, longitudes, pos = dot_ensemble_circular(theta, 0, radius, num_pts=self.num_pts)

            # for a big spot, use a cap on the sphere approximation
            else:
                latitudes, longitudes = dot_ensemble_spherical(theta, 0, radius)

            # apply the flare model model to each grid point at each time of the light curve 
            lamb, onoff, m = lightcurve_model(self.phi, latitudes, longitudes, flare, i, phi0=phi0)
            ms.append(m * self.median)
        
        # sum contributions from each flare and add the median flux
        return sum(ms, self.median)

    def log_likelihood(self, params, sta=0, sto=None):
        """
        Calculate the log of the likelihood function of the model with
        given parameters.


        Parameters

        Structure of params:

        index | value
        -----------------------------------------------------------------------
        0     | latitude
        1     | longitude at t0
        2     | inclination
        3+    | (number of flares f1, f2, ...) x (number of parameters a,b,c...) 
                as in [f1a, f1b, f1c, f2a, f2b, f2c]
 
        """
        theta, phi0, i =  params[:3]
        flareparams = np.array(params[3:]).reshape(self.nflares, len(params[3:])//self.nflares)
        model = self.modulated_flux(theta, phi0, i, flareparams)

        fr2 = self.flux_err[sta:sto]**2
 
        val = -0.5 * np.sum((self.flux[sta:sto] - model[sta:sto]) ** 2 / fr2 + np.log(fr2))
        return val
    
    
    def chi_square(self, params, sta=0, sto=None):
        """structure of params:

        index | value
        -----------------------------------------------------------------------
        0     | latitude
        1     | longitude at t0
        2     | inclination
        3+    | (number of flares f1, f2, ...) x (number of parameters a,b,c...) 
                as in [f1a, f1b, f1c, f2a, f2b, f2c]
 
        """
        theta, phi0, i =  params[:3]
        flareparams = np.array(params[3:]).reshape(self.nflares, len(params[3:])//self.nflares)
        model = self.modulated_flux(theta, phi0, i, flareparams)
        
        return np.sum((self.flux[sta:sto] - model[sta:sto]) ** 2 / model[sta:sto])


    def log_prior(self, params, phi_a_min=(0,0.),
                  phi_a_max=(1e9,1e9), theta_a_min=-np.pi/2.,
                  theta_a_max=np.pi/2., a_min=(0,0), a_max=(1e9,1e9),
                  fwhm_min=(0,0), fwhm_max=(1e9,1e9), phi0_min=-2*np.pi,
                  phi0_max=2*np.pi, g=None):

        """

        Parameters:


        """
        theta, phi0, i =  params[:3]
        flareparams = np.array(params[3:]).reshape(self.nflares, len(params[3:])//self.nflares)

        prior = (uninformative_prior(theta, theta_a_min, theta_a_max) + 
                 uninformative_prior(phi0, phi0_min, phi0_max) +  
                 empirical_prior(i, g)) 

        for i, flare in enumerate(flareparams): 
            prior += (uninformative_prior(flare[0], a_min[i], a_max[i]) +
                      uninformative_prior(flare[1], phi_a_min[i], phi_a_max[i]))
        
            for fwhm in flare[2:]:
                prior += uninformative_prior(fwhm, fwhm_min[1], fwhm_max[1])

        return calculate_posterior_value_that_can_be_passed_to_mcmc(prior)


def lightcurve_model(phi, latitudes, longitudes, flare, inclination, phi0=0.):
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

    # Mould phi0 into -pi:pi range

    phi0 = phi0 % (2*np.pi)


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


def dot_ensemble_circular(lat, lon, radius, num_pts=100):
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


def dot_ensemble_spherical(lat, lon, radius):
    """Create an ensemble of dots on a sphere. Use the pre-defined
    grid in phi and theta.

    Parameters:
    -----------
    lat : float
        latitude of center of ensemble in rad
    lon : float
        longitude of center of ensemble in rad
    radius : float
        angular radius of the ensemble in deg


    Return:
    -------
    latitudes, longitudes  -  np.arrays of dots
    that go into the ensemble.
    """
    if no_nan_inf([lat, lon, radius]) == False:
        raise ValueError("One of your inputs in "\
                         "dot_ensemble_spherical is"\
                         " or contains NaN or Inf.")

    # Calculate the distance of the dots to the center of the ensemble
    gcs = great_circle_distance(lat, lon, PHI, THETA)

    # If distance is small enough, include in ensemble
    a = np.where(gcs < (radius * np.pi / 180))[0]
    return PHI[a], THETA[a]


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
    def condition(phi, phi0, daylength):

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

        res = np.full_like(l, np.nan)
        # polar night
        res[np.abs(l) >=i] = 0
        # polar day
        res[l>=i] = P
        # rest
        res[np.isnan(res)] = formula(l[np.isnan(res)], i)

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


def black_body_spectrum(wav, t):
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
            .to("erg*s**(-1)*cm**(-3)")) #simplify the units

# ---------------------------------------------------------------------------------

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
    return np.trapz(thermf * rres, x=w).to("erg*cm**(-2)*s**(-1)")

# ---------------------------------------------------------------------------------

def calculate_angular_radius(Fth, a, qlum, R):
    """Calculate angular radius in degrees from. Do the integration
        over the polar cap.

    Parameters:
    ------------
    Fth : astropy value > 0
        specific flare flux in erg/cm^2/s
    a : float > 0
        relative flare amplitude
    qlum : astropy value > 0
        projected quiescent luminosity in erg/s
    R : float > 0
        stellar radius in solar radii

    Return:
    -------
    float - radius of flaring area in deg
    """
    sin = np.sqrt((a * qlum) / (np.pi * R**2 * Fth))

    if sin > 1:
        raise ValueError("Flare area seems larger than stellar hemisphere.")

    return np.arcsin(sin).to("deg").value



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

    fwhm = dur # crude approximation for a triangle shape would be dur/2.

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

def aflare_decoupled(t, tpeak, dur, ampl, upsample=False, uptime=10):
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
    dur : float, float
        Rise and decay fwhm of the flare
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

    fwhm1, fwhm2 = dur # crude approximation for a triangle shape would be dur/2.

    if upsample:
        dt = np.nanmedian(np.diff(t))
        timeup = np.linspace(min(t)-dt, max(t)+dt, t.size * uptime)

        flareup = np.piecewise(timeup, [(timeup<= tpeak) * (timeup-tpeak)/fwhm1 > -1.,
                                        (timeup > tpeak)],
                                    [lambda x: (_fr[0]+                       # 0th order
                                                _fr[1]*((x-tpeak)/fwhm1)+      # 1st order
                                                _fr[2]*((x-tpeak)/fwhm1)**2.+  # 2nd order
                                                _fr[3]*((x-tpeak)/fwhm1)**3.+  # 3rd order
                                                _fr[4]*((x-tpeak)/fwhm1)**4. ),# 4th order
                                     lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm1)*_fd[1] ) +
                                                _fd[2]*np.exp( ((x-tpeak)/fwhm2)*_fd[3] ))]
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
        flare = np.piecewise(t, [(t<= tpeak) * (t-tpeak)/fwhm1 > -1.,
                                 (t > tpeak)],
                                [lambda x: (_fr[0]+                       # 0th order
                                            _fr[1]*((x-tpeak)/fwhm1)+      # 1st order
                                            _fr[2]*((x-tpeak)/fwhm1)**2.+  # 2nd order
                                            _fr[3]*((x-tpeak)/fwhm1)**3.+  # 3rd order
                                            _fr[4]*((x-tpeak)/fwhm1)**4. ),# 4th order
                                 lambda x: (_fd[0]*np.exp( ((x-tpeak)/fwhm1)*_fd[1] ) +
                                            _fd[2]*np.exp( ((x-tpeak)/fwhm2)*_fd[3] ))]
                                ) * np.abs(ampl) # amplitude

    return flare

def calculate_ED(t, t0, fwhm, ampl, decoupled=True):
    """Calculate equiavlent duration
    of model flare.

    Parameters:
    -----------
    t : numpy.array
        observation times in days
    t0 : float
        flare peak time
    dur : float
        flare FWHM
    ampl : float
        relative flare amplitude

    Return:
    --------
    ED in seconds - float
    """
    if decoupled==True:
        fwhm_ = fwhm
        model = aflare_decoupled
    else:
        fwhm_ = [fwhm]
        model = aflare

    if no_nan_inf([t0, *fwhm_, ampl]) == False:
        raise ValueError("flaret is NaN or Inf.")
    return np.sum(np.diff(t) * model(t, t0, fwhm, ampl)[:-1]) * 60.0 * 60.0 * 24.0
