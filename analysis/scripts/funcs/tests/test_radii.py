import pytest

import numpy as np

from ..radii import (mann_radius_from_abs_Ks,
	                 calculate_distmod,
                     calculate_abs_Ks,)

def test_mann_radius_from_abs_Ks():
    """Just some simple unit tests and integrations."""
    assert mann_radius_from_abs_Ks(0., 0.) == pytest.approx((1.9515, 1.9515*0.0289))
    assert np.isnan(mann_radius_from_abs_Ks(np.nan, 0.)).all()
    assert mann_radius_from_abs_Ks(0., np.nan)[0] == pytest.approx(1.9515)
    assert np.isnan(mann_radius_from_abs_Ks(0., np.nan)[1])

    assert mann_radius_from_abs_Ks(1., 1.) == pytest.approx((1.9515 - 0.3520 + 0.01680, 
                                                     np.sqrt((- 0.3520 + 0.01680 * 2.)**2 +
                                                             (0.0289*(1.9515 - 0.3520 + 0.01680))**2)) )



def test_calculate_distmod():
    """Just some simple unit tests and integrations."""
    with pytest.raises(ZeroDivisionError) as e:
        calculate_distmod(0., 0.)

    assert calculate_distmod(1., 0.) == (-5.,0.)
    assert np.isnan(calculate_distmod(np.nan, 0.)).all()
    assert calculate_distmod(1., np.nan)[0] == -5.
    assert np.isnan(calculate_distmod(1., np.nan)[1])


def test_calculate_abs_Ks():
    """Just some simple unit tests and integrations."""
    assert calculate_abs_Ks(1.,0.,1.,0.) == (0.,0.)
    assert np.isnan(calculate_abs_Ks(np.nan,0.,1.,0.)[0])
    assert np.isnan(calculate_abs_Ks(1.,np.nan,1.,0.)[1]) 
