import pytest

import numpy as np
import pandas as pd

import os

from altaipony.flarelc import FlareLightCurve

from ..multiperiod import find_period

CWD = os.getcwd()

def test_find_period():
    
    def create_lc(hours):
        # Create a light curve with a sinusoidal modulation
        hdr = fits.Header()
        hdr['OBSERVER'] = 'Mike'
        hdr['COMMENT'] = "Here's some commentary about this FITS file."
        start, stop, N = 1000, 1020, 10000
        c1 = fits.Column(name='TIME', array=np.linspace(start, stop, N), format='F10.4')
        c2 = fits.Column(name='FLUX', array=400 + 50*np.sin(np.linspace(start, stop, N)*np.pi*48/hours), format='F10.4')
        c3 = fits.Column(name='FLUX_ERR', array= 20*np.random.rand(N), format='F10.4')
        c4 = fits.Column(name='QUALITY', array= np.full(N,0), format='K')
        c5 = fits.Column(name='CADENCENO', array= np.arange(100,N+100), format='K')
        hdu = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5], header=hdr)
        PATH = f"{CWD}/1000_10_tess_custom_flc.fits"
        hdu.writeto(PATH,overwrite=True)

    # Create a target description
    target = pd.Series({"h_mission": "TESS",
                        "origin": "flc",
                        "ID": 1000,
                        "QCS": 10,
                        "typ": "custom",
                        "mission": "tess",
                        "SpT": "M8",
                        "prefix": "TIC"})
    
    # Test a case
    start, stop, N, hours = 1000, 1020, 10000, 6
    flc = FlareLightCurve(time=np.linspace(start, stop, N),
                          flux=400 + 50*np.sin(np.linspace(start, stop, N)*np.pi*48/hours),
                          flux_err=20*np.random.rand(N),
                          quality=np.full(N,0),
                          cadenceno= np.arange(100,N+100))
    period, mfp = find_period(target, minfreq=.1, maxfreq=40, plot=False, save=False, flc=flc, custom=False)

    # Do some checks
    assert period.value == pytest.approx(hours)
    assert mfp.value == pytest.approx(24/hours)
    

