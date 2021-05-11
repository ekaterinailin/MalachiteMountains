![Malachite Mountains](mm_hl.png)

# MalachiteMountains

This repository contains the necessary data, model code and scripts to reproduce the results, figures, and tables that appear in Ilin et al. (submitted to MNRAS).

The name of this repository refers to Russian folklore, a fairy tale about the [Malachite Casket](https://en.wikipedia.org/wiki/The_Malachite_Casket_(fairy_tale)).

## Relevant contents of this repository

- data/
  - inclinations/
    - *post_compound.p
    - *post.dat
    - inclination_output.dat
  - solar/
    - sars.csv (solar superactive regions)
  - summary/
    - inclination_input.csv (rotation period, vsini, stellar radius)
    - inits\_decoupled\_GP.csv (inits for MCMC model fit)
    - lcs.csv (other stellar and light curve properties)
  - lcs/
    - *.csv (light curves including quiescent model flux array)

- analysis/
  - notebooks/
    - funcs/
        - `model.py` - FlareModulator class
    - *.ipynb
        - the majority of scripts used in this work as notebooks
    - *.py
        - some pure python scripts
    
## Plans


- Make all the notebooks work with the refactored FlareModulator class and its outputs.
- Branch out a version that contains the packaged model only.
