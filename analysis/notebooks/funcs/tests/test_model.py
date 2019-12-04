import pytest
import numpy as np

from ..model import daylength 


# Test different values on daylength(l, i) to make it fail
cases = [(np.nan,0), (np.nan,np.nan), (0,np.nan),
         (2,np.nan), (2,np.pi), (2,-np.pi),
         (1.5*np.pi, np.pi/3)]

@pytest.mark.parametrize("l,i", cases)
def test_daylength_fail(l,i):
    with pytest.raises(ValueError) as e:
        daylength(l, i)

# Test different values on daylength(l, i) to make it succeed
cases = [(0,np.pi/2,0.5),(0,0,1),(0.,70 * np.pi / 180, 0.5),
         (np.pi/2,np.pi/4,1.),(np.pi/4,np.pi/4,1.),
         (np.pi/4+.05,np.pi/4,1.),(0,np.pi/4,.5),
         (- np.pi/4,np.pi/4,0.),(-np.pi/2,np.pi/4,0.)]

@pytest.mark.parametrize("l,i,expected", cases)
def test_daylength_succeed(l,i,expected):
    assert daylength(l, i) == expected