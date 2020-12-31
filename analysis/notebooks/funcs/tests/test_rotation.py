import pytest

import numpy as np
import pandas as pd

import os

from altaipony.flarelc import FlareLightCurve

from ..rotation import find_period

CWD = os.getcwd()

def test_find_period():

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
    

