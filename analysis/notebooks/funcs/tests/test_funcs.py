import numpy as np
import pytest

from ..funcs import no_nan_inf

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