# Python3
# Some functions used for flare finding, data management


import pandas as pd
import numpy as np

from astropy.io import fits
from altaipony.flarelc import FlareLightCurve
from altaipony.lcio import from_path

import sys, os

from collections import defaultdict


def get_window_length_dict():
    """Collection of window lengths
    customized to individual LCs.
    
    Returns:
    --------
    dictionary with defaul return value None
    """
    l25 = [(x, 25) for x in   [44984200, 98874143, 388903843, 332623751, 44892011, 
                               29780677, 340703996, 395130640, 441000085, 
                               53603145, 144776281,]]
    l55 = [(x, 55) for x in   [471012770, 5630425, 140478472, 142052876,
                               272349442, 277539431, 293561794, 369555560,
                               464378628]]
    l75 = [(x, 75) for x in   [29928567,298907057, 366567664, 369863567,
                               420001446]]
    l115 = [(x, 115) for x in [328254412,]]
    l555 = [(x, 555) for x in [471012740, 125835702, 30101427, 415839928, 
                               398985964, 322568489, 2470992, 1539914,
                               117733581, 73118477]]
    L = [l25, l55, l75, l115,l555]
    L = [x for a in L for x in a]
    l = dict(L)
    assert len(l) == len(L)
    ht = defaultdict(lambda:None, l)
    return ht


def write_flares_to_file(flc, cluster):
    '''
    Write the resulting flare table to file, adding
    it  to the rest.
    '''
    cols = ['TIC', 'ampl_rec', 'cstart', 'cstop', 
            'ed_rec', 'ed_rec_err', 'istart', 'istop', 
            'tstart', 'tstop', 'Campaign']
            #'saturation_f10']
    try:
        df = pd.read_csv('{0}_flares.csv'.format(cluster),
            usecols=cols)
    except:
        print('Create a file named {}_flares.csv'.format(cluster))
        df = pd.DataFrame(columns=cols)
        
    lbefore = flc.flares[~flc.flares.ed_rec.isnull()].shape[0]
    if lbefore == 0:
        emptydf = pd.DataFrame(dict(zip(cols, [flc.targetid] + [np.nan] * (len(cols) - 1))), index=[0])
        flc.flares = flc.flares.append(emptydf, ignore_index=True)
        print('Added an empty row for TIC {}'.format(flc.targetid))
    flc.flares['TIC'] = flc.targetid
    flc.flares['Campaign'] = flc.campaign
    #flc.flares = flc.get_saturation(return_level=True) maybe later
    print(flc.flares)
    df = df.append(flc.flares, ignore_index=True)
    lafter = df[~df.ed_rec.isnull()].shape[0]
   # print('Added {} flares to the {} found so far in {}.'.format(lafter-lbefore, lbefore, cluster))
    df.to_csv('{0}_flares.csv'.format(cluster), index=False)
    return
    
def read_custom_aperture_lc(path, typ="custom", mission="TESS", mode="LC",
                           sector=None,TIC=None):
    '''Read in custom aperture light curve
    from TESS. Needs specific naming convention.
    Applies pre-defined quality masks.
    
    Parameters:
    -----------
    path : str
        path to file
    
    Returns:
    --------
    FlareLightCurve
    '''
    if typ=="custom":
        hdu = fits.open(path)
        data = hdu[1].data
        if sector==None:
            sector = int(path.split("-")[1][-2:])
        if TIC==None:
            TIC = int(path.split("-")[2])

        flc = FlareLightCurve(time=data["TIME"],
                            flux=data["FLUX"],
                            flux_err=data["FLUX_ERR"],
                            quality=data["QUALITY"],
                            cadenceno=data["CADENCENO"],
                            targetid=TIC,
                            campaign=sector)
        flc = fix_mask(flc)
    else:
        flc = from_path(path, mission=mission, mode=mode)
    return flc

def fix_mask(flc):
    '''Here the masks for different TESS 
    sectors are defined and applied to 
    light curve fluxes.
    
    Parameters:
    ------------
    flc : FlareLightCurve
    
    Returns:
    ----------
    FlareLightCurve
    '''
    masks = {9: [(227352, 228550), (236220, 238250)],
             10: [(246227,247440),(255110,257370)],
             11:[(265912,268250),(275210,278500)],
             8: [(208722,209250),(221400,221700)],
             6: [(179661,180680)],
             5: [(151586,151900),(160254,161353),(170000,170519)],
             4: [(139000,139800),(140700,141161),(150652,150764)],
             3: [(120940,121820)],
             12: [(286200,286300)]}

    if flc.campaign in masks.keys():
        for sta, fin in masks[flc.campaign]:
            flc.flux[np.where((flc.cadenceno >= sta) & (flc.cadenceno <= fin))] = np.nan
            
    flc.quality[:] = np.isnan(flc.flux)
    return flc

def get_window_length_dict():
    l15 = [(x, 15) for x in   [44984200]]
    l25 = [(x, 25) for x in   [98874143, 388903843, 332623751, 44892011, 
                               29780677, 340703996, 395130640, 441000085, 
                               53603145, 144776281,]]
    l55 = [(x, 55) for x in   [471012770, 5630425, 140478472, 142052876,
                               272349442, 277539431, 293561794, 369555560,
                               464378628]]
    l75 = [(x, 75) for x in   [29928567,298907057, 366567664, 369863567,
                               420001446]]
    l115 = [(x, 115) for x in [328254412,]]
    l555 = [(x, 555) for x in [471012740, 125835702, 30101427, 415839928, 
                               398985964, 322568489, 2470992, 1539914,
                               117733581, 73118477]]
    L = [l15, l25, l55, l75, l115,l555]
    L = [x for a in L for x in a]
    l = dict(L)
    assert len(l) == len(L)
    return l
