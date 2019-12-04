import pytest 
import numpy as np

from ..custom_detrending import (#custom_detrending,
                                 select_window_length,
                                 search_gaps_for_window_length,
                                 refine_detrended_flux_err,
                                 fit_spline,
                                 iteratively_remove_sines)

                                 
from altaipony.flarelc import FlareLightCurve

def test_refine_detrended_flux_err():
    # ---------------- No gaps and a flare -------------------------------------

    N = int(1e3)
    time = np.linspace(2000,2050,N)
    np.random.seed(2050)
    detrended_flux = np.sin(time / 2.) * 2. + 1e3 + np.random.rand(N) * 15. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
    detrended_flux[500:510] = detrended_flux[500:510] + np.array([1500,650,450,280,160,90,20,10,7,4])

    detrended_flux_err = np.full(N, np.nanstd(detrended_flux))
    flc = FlareLightCurve(targetid=10000009, time=time, flux=detrended_flux,
                          flux_err=detrended_flux_err, detrended_flux=detrended_flux,
                          detrended_flux_err=detrended_flux_err)

    flcd = refine_detrended_flux_err(flc, mask_pos_outliers_sigma=2,
                                     std_rolling_window_length=3,
                                     pad=3)

    # Make sure the procedure does not fail
    assert np.isfinite(flcd.detrended_flux_err).all()
    # Make sure that we recover the original noise level from the synthetic data
    assert np.std(np.random.rand(N) * 15.) == pytest.approx(np.mean(flcd.detrended_flux_err),rel=.2)

    # -----------------Three gaps and a slow sine-----------------------------------
    N = int(1e4)
    time = np.linspace(2000,2050,N)
    np.random.seed(2050)
    flux = np.sin(time / 2.) * 10. + 1e3 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
    flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
    flux[4000:4010] = flux[4000:4010] + np.array([400,250,150,80,60,30,20,10,7,4])
    flux[9000:9010] = flux[9000:9010] + np.array([100,150,50,30,20,10,5,4,3,1])
    flux[4500:4809] = np.nan
    flux[7000:8000] = np.nan
    flux_err = np.random.rand(N) * 35.
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, 
                          flux_err=flux_err, detrended_flux_err=flux_err,
                          detrended_flux=flux)

    flcd = refine_detrended_flux_err(flc, mask_pos_outliers_sigma=2,
                                     std_rolling_window_length=3,
                                     pad=3)

    # Make sure the procedure does not fail
    assert np.isfinite(flcd.detrended_flux_err).all()
    # Make sure that we recover the original noise level from the synthetic data
    assert np.std(np.random.rand(N) * 35.) == pytest.approx(np.mean(flcd.detrended_flux_err),rel=.2)

def test_search_gaps_for_window_length():
    # Set up for all:
    
    N = int(1e4)
    time = np.linspace(2000,2050,N)
    # Three gaps and a slow sine

    np.random.seed(2050)
    flux = np.sin(time / 2.) * 20. + 1e3 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
    flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
    flux[4000:4010] = flux[4000:4010] + np.array([400,250,150,80,60,30,20,10,7,4])
    flux[9000:9010] = flux[9000:9010] + np.array([100,150,50,30,20,10,5,4,3,1])
    flux[4500:4809] = np.nan
    flux[7000:8000] = np.nan
    flux_err = np.random.rand(N) * 10.
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

    res = search_gaps_for_window_length(flc)
    assert res == [751, 25, 667]


    # Three gaps and no sine
    np.random.seed(251)
    flux = 1e3 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
    flux[5000:5015] = flux[5000:5015] + np.array([2500,1400,1000,550,350,
                                                  250,150,80,60,30,20,10,7,4,3])
    flux[4000:4010] = flux[4000:4010] + np.array([400,250,150,80,60,30,20,10,7,4])
    flux[9000:9010] = flux[9000:9010] + np.array([100,150,50,30,20,10,5,4,3,1])
    flux[4500:4809] = np.nan
    flux[7000:8000] = np.nan
    flux_err = np.random.rand(N) * 10.
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

    res = search_gaps_for_window_length(flc)
    assert res == [25,25,25]

    # Only one gap
    np.random.seed(2050)
    flux = 1e3 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
    flux[5000:5015] = flux[5000:5015] + np.array([2500,1400,1000,550,350,
                                                  250,150,80,60,30,20,10,7,4,3])
    flux[4000:4010] = flux[4000:4010] + np.array([400,250,150,80,60,30,20,10,7,4])
    flux[9000:9010] = flux[9000:9010] + np.array([100,150,50,30,20,10,5,4,3,1])
    flux_err = np.random.rand(N) * 10.
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

    res = search_gaps_for_window_length(flc)
    assert res == [111]



def test_select_window_length():
    for freq, res in [(.001,25), (0.01,29), (.1,279), (.3,833), (1.,3333)]:
        flux = np.sin(np.linspace(3970,4000, 40000)/freq)*10. + 200.
        assert select_window_length(flux) == res
    

# def test_custom_detrending():

#     # Test a standard case:

#     N = int(1e4)
#     time = np.linspace(2000,2050,N)
#     flux = np.sin(time / 2.) * 30. + 5e4 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 30 * (time-2004)**2)
#     flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
#     flux_err = np.random.rand(N) * 10.
#     flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

#     flcd = custom_detrending(flc)

#     assert flcd.detrended_flux.std() < 1.5 * np.std(np.random.rand(N) * 35.)
#     assert flcd.detrended_flux[5000] == pytest.approx(50500, abs=60)
#     assert flcd.detrended_flux[5001] == pytest.approx(50250, abs=60)
#     assert flcd.detrended_flux[5002] == pytest.approx(50150, abs=60)

#     # Test another

#     N = int(1e4)
#     time = np.linspace(2000,2050,N)
#     flux = np.sin(time / 6.) * 20. + 1e3 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
#     flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
#     flux_err = np.random.rand(N) * 10.
#     flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

#     flcd = custom_detrending(flc)

#     assert flcd.detrended_flux.std() < 1.5 * np.std(np.random.rand(N) * 35.)
#     assert flcd.detrended_flux[5000] == pytest.approx(np.mean(flux) + 500, abs=60)
#     assert flcd.detrended_flux[5001] == pytest.approx(np.mean(flux) + 250, abs=60)
#     assert flcd.detrended_flux[5002] == pytest.approx(np.mean(flux) + 150, abs=60)

#     # Test one with a gap

#     N = int(1e4)
#     time = np.linspace(2000,2050,N)
#     flux = np.sin(time / 6.) * 20. + 1e3 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
#     flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
#     flux[4500:4809] = np.nan
#     flux_err = np.random.rand(N) * 10.
#     flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

#     flcd = custom_detrending(flc)
#     print(np.nanmean(flux),flcd.detrended_flux[5001-309]-250)
#     assert flcd.detrended_flux.std() < 1.5 * np.std(np.random.rand(N) * 35.)
#     assert flcd.detrended_flux[5000-309] == pytest.approx(np.nanmedian(flux) + 500, abs=60)
#     assert flcd.detrended_flux[5001-309] == pytest.approx(np.nanmedian(flux) + 250, abs=60)
#     assert flcd.detrended_flux[5002-309] == pytest.approx(np.nanmedian(flux) + 150, abs=60)


#     # Test one with a gap and a diverging part

#     N = int(1e4)
#     time = np.linspace(2000,2050,N)
#     flux = np.sin(time / 6.) * 20. + 1e3 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
#     flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
#     flux[4500:4809] = np.nan
#     flux[4000:4499] = flux[4000:4499] + 20*((time[4000:4499]-time[4000]))**2
#     flux_err = np.random.rand(N) * 10.
#     flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

#     flcd = custom_detrending(flc)
#     assert flcd.detrended_flux.std() < 1.5 * np.std(np.random.rand(N) * 35.)
#     assert flcd.detrended_flux[5000-309] == pytest.approx(np.nanmean(flux) + 500, abs=60)
#     assert flcd.detrended_flux[5001-309] == pytest.approx(np.nanmean(flux) + 250, abs=60)
#     assert flcd.detrended_flux[5002-309] == pytest.approx(np.nanmean(flux) + 150, abs=60)

#     # Test one with a gap and a diverging part
#     np.random.seed(2005)
#     N = int(1e4)
#     time = np.linspace(2000, 2050, N)
#     flux = np.sin(time / 0.05) * 60. + 1e3 + np.random.rand(N) * 35. + 5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2)
#     flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
#     flux[4500:4809] = np.nan
#     flux[4000:4500] = flux[4000:4500] + 20*((time[4000:4500]-time[4000]))**2
#     flux_err = np.random.rand(N) * 10.
#     flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, flux_err=flux_err)

#     flcd = custom_detrending(flc, d={10000009:25})
#     assert flcd.detrended_flux.std() < 1.5 * np.std(np.random.rand(N) * 35.)
#     assert flcd.detrended_flux[5000-309] == pytest.approx(np.nanmedian(flux) + 500, abs=100)
#     assert flcd.detrended_flux[5001-309] == pytest.approx(np.nanmedian(flux) + 250, abs=110)
#     assert flcd.detrended_flux[5002-309] == pytest.approx(np.nanmedian(flux) + 150, abs=110)
    
def test_fit_spline():
    N = int(1e4)
    time = np.linspace(2000,2050,N)
    np.random.seed(2050)
    flux = (np.sin(time / .08) * 40. + 
            np.sin(time / .05) * 10. + 
            1e4 + 
            np.random.rand(N) * 35. + 
            5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2))
    flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
    flux[4000:4010] = flux[4000:4010] + np.array([400,250,150,80,60,30,20,10,7,4])
    flux[9000:9010] = flux[9000:9010] + np.array([100,150,50,30,20,10,5,4,3,1])
    flux[4500:4809] = np.nan
    flux[7000:8000] = np.nan
    flux_err = np.random.rand(N) * 35.
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, 
                          flux_err=flux_err, detrended_flux_err=flux_err,
                          detrended_flux=flux)

    flcd = fit_spline(flc)

    assert flcd.flux.shape[0] == 1e4-309-1e3
    assert not np.isnan(flcd.flux).any()
    
def test_iteratively_remove_sines():
    N = int(1e4)
    time = np.linspace(2000,2050,N)
    np.random.seed(2050)
    flux = (np.sin(time / .08) * 40. + 
            np.sin(time / .05) * 10. + 
            1e4 + 
            np.random.rand(N) * 35. + 
            5e-4 * ((time-2004.)**3 - 300 * (time-2004)**2))
    flux[5000:5010] = flux[5000:5010] + np.array([500,250,150,80,60,30,20,10,7,4])
    flux[4000:4010] = flux[4000:4010] + np.array([400,250,150,80,60,30,20,10,7,4])
    flux[9000:9010] = flux[9000:9010] + np.array([100,150,50,30,20,10,5,4,3,1])
    flux[4500:4809] = np.nan
    flux[7000:8000] = np.nan
    flux_err = np.random.rand(N) * 35.
    flc = FlareLightCurve(targetid=10000009, time=time, flux=flux, 
                          flux_err=flux_err, detrended_flux_err=flux_err,
                          detrended_flux=flux)

    flcd = iteratively_remove_sines(flc)

    assert flcd.flux.shape[0] == flc.flux.shape[0]
    assert np.where(np.isnan(flcd.detrended_flux))[0].shape[0] == 1309
