import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gamma, gammainc
import takahe
from tqdm import tqdm

def MadauDickinson(Z, z):
    """Computes the Madau & Dickinson SFRD at metallicity Z and redshift z.

    Implements the SFRD given by eqn(15) of [1]. Returns a value in
    M_sun / yr / Mpc^3.

    Assumes Z_sun = 0.020, and that input metallicity is NOT already
    measured relative to this.

    [1] https://www.annualreviews.org/doi/pdf/10.1146/annurev-astro-081811-125615

    Arguments:
        Z {float} -- The metallicity under consideration.
        z {float} -- The redshift under consideration.

    Returns:
        {float} -- The SFRD at metallicity Z and redshift z.
    """
    GAM = gammainc(0.84, (Z / 0.02)**2 * 10**(0.3*z))
    NUM = 0.015 * (1+z)**2.7
    DEM = (1+((1+z)/2.9)**5.6)

    SFRDi = GAM * (NUM / DEM)

    return SFRDi

def MilkyWay(Z, z):
    """Implements the SFR equation from (Wiktorowicz et. al. 2020) [1]
    for the Milky Way Galaxy.

    Piecewise function for the SFR in the Milky Way galaxy. Assumes a
    four-component formalism - consisting of a thin disk, thick disk,
    bulge, and halo. Precise values of the SFR come from
    (Olejak et. al 2019) [2].

    [1] https://arxiv.org/pdf/2006.08317.pdf
    [2] https://arxiv.org/pdf/1908.08775.pdf

    Arguments:
        Z {float} -- The metallicity under consideration.
        z {float} -- The redshift under consideration.

    Returns:
        {float} -- The SFRD at metallicity Z and redshift z.
    """
    SFR_arr = np.zeros(len(z))
    for i in range(len(z)):
        zi = z[i]
        tL = takahe.helpers.redshift_to_lookback(zi)

        Z_sun = takahe.constants.SOLAR_METALLICITY

        SFR = 0

        if Z == Z_sun:
            # Thin Disk
            if 0 <= tL <= 10:
                SFR += 4.7
            # Bulge
            if 0 <= tL <= 10:
                SFR += 0.45
            elif 10 <= tL <= 12:
                SFR += 2.3
        elif Z == Z_sun / 10:
            # Thick Disk
            if 9 <= tL <= 11:
                SFR += 2.5
        elif Z == 1e-4:
            # Halo
            if 10 <= tL <= 12:
                SFR += 0.5

        SFR_arr[i] += SFR

    return SFR_arr
