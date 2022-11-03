"""
UTF-8, Python 3

------------------
TESS UCD Flares
------------------

Ekaterina Ilin, 2020, MIT License


This script reads in TIC and Sector,
fetches a light curve from MAST,
performs custom detrending on the light curve, 
searches for flares in the de-trended
light curve, and writes the solution to file.

"""

import numpy as np
import pandas as pd


import time
import os

from custom_detrending import custom_detrending

from altaipony.lcio import from_mast


def write_flares_to_file(flc, tstamp):
    '''
    Write the resulting flare table to file, adding
    it  to the rest.
    '''
    cols = ['ampl_rec', 'cstart', 'cstop', 
            'ed_rec', 'ed_rec_err', 'istart', 'istop', 
            'tstart', 'tstop', "total_n_valid_data_points", "dur",
           'TIC', 'Sector', ]
    
    path = f'{tstamp}_flares.csv'
    if os.path.exists(f'{tstamp}_flares.csv'):
        ignore=False
    else:
        ignore=True
        print('Create a file named {}_flares.csv'.format(tstamp))
    
    lbefore = flc.flares[~flc.flares.ed_rec.isnull()].shape[0]
    
    if lbefore == 0:
        flc.flares = pd.DataFrame(dict(zip(cols,[np.nan] * (len(cols)-3) +
                                           [flc.flux[~np.isnan(flc.flux)].shape[0]] +
                                           [flc.targetid] +
                                           [flc.sector] )), 
                                  index=[0])
        print('Added an empty row for TIC {}'.format(flc.targetid))
    else:
        # Add columns to identify flares later
        flc.flares['TIC'] = flc.targetid
        flc.flares['Sector'] = flc.sector
    
    # save to file
    with open(path, "a") as f:
        flc.flares.to_csv(f,header=ignore, index=False)
        
    return
    
    

if __name__ == "__main__":

    # time stamp for backup
    tstamp = time.strftime("%d_%m_%Y", time.localtime())
    
    # get input table
    inp = pd.read_csv("gotlc.csv")
    
    # De-trend and find all flares
    for i, target in inp.iloc[4119:].iterrows():
        
        try:
            # download light curve if available
            flc = from_mast(f"TIC {target.tic_id}", c=target.sector, mission="TESS", 
                            download_dir="/home/ekaterina/Documents/001_science/TESS_UCD_flares/lcs")
            # apply custom de-trending
            flcd = flc.detrend("custom", func=custom_detrending)

            # Find all flare candidates
            flcd = flcd.find_flares()

            # write to file
            write_flares_to_file(flcd, tstamp)
            
            # note
            print(f"\n--------------\nTIC {target.tic_id}, Sector {target.sector} finished.\n----------------\n")

            # take a break
            time.sleep(5)


        except:
            # if something goes wrong, flag the target that did not work
            with open(f"{tstamp}_fails.txt", "a") as f:
                f.write(f"{target.tic_id},{target.sector}\n")
        
        
        
