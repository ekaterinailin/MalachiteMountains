import numpy as np

import pytest

from ..helper import no_nan_inf, read_custom_aperture_lc

from astropy.io import fits

import os

CWD = "/".join(os.getcwd().split("/")[:-2])

# We don't test fetch_lightcurve, because it's just a wrapper for read_custom_aperture

def test_read_custom_aperture_lc():
    
    # Create a light curve with a sinusoidal modulation
    hdr = fits.Header()
    hdr['OBSERVER'] = 'Mike'
    hdr['COMMENT'] = "Here's some commentary about this FITS file."
    start, stop, N = 1000, 1020, 1000
    c1 = fits.Column(name='TIME', array=np.linspace(start, stop, N), format='F10.4')
    c2 = fits.Column(name='FLUX', array=400 + 50*np.sin(np.linspace(start, stop, N)*2), format='F10.4')
    c3 = fits.Column(name='FLUX_ERR', array= 20*np.random.rand(N), format='F10.4')
    c4 = fits.Column(name='QUALITY', array= np.full(N,1), format='K')
    c5 = fits.Column(name='CADENCENO', array= np.arange(100,N+100), format='K')
    hdu = fits.BinTableHDU.from_columns([c1,c2,c3,c4,c5], header=hdr)
    PATH = f"{CWD}/analysis/notebooks/funcs/tests/test.fits"
    hdu.writeto(PATH,overwrite=True)
    
    # Call the function
    flc = read_custom_aperture_lc(PATH, typ="custom", mission="TESS", mode="LC",
                                sector=10, TIC=1000)
    # Do some checks
    assert (flc.flux == 400 + 50*np.sin(np.linspace(start, stop, N)*2)).all()
    assert (flc.time == np.linspace(start, stop, N)).all()
    assert flc.campaign == 10
    assert flc.targetid == 1000

cases = [((0, 0, np.nan), False),
         ((0, 0, 0), True), 
         ((0, 3, 0, np.linspace(0, 1, 10 )), True),
         ((0, 3, 0, np.full(10,np.inf)), False),
         ((np.inf, 0, 0, 0), False),
         (np.array([9, 1, np.nan]), False),
         (np.array([9, 1, np.inf]), False),]

@pytest.mark.parametrize("l,expected", cases)
def test_no_nan_inf_succeed(l,expected):
    assert no_nan_inf(l) == expected
