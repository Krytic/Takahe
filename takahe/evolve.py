from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import pandas as pd
import pkgutil
from scipy import stats
from scipy.optimize import minimize
import takahe
from tqdm import tqdm

def evolve_system(a0, e0, m1, m2,
                  weight=1, SFRD=None, beta=1,
                  alpha=0, only_arrays=False,
                  return_value=False):
    """Evolves a binary system until merger or the age of the Universe.

    A note about normalisation & units
    ----------------------------------

    time_in_bin is our resultant matrix for this. It has units of:
    => # * [yr] * ([M_sun] / [yr] / [Mpc^3]) / (1e6 * [M_sun]) / 1e6
    => # / [Mpc^3]

    If we use takahe.SFR.MilkyWay for the SFR, then this just becomes
    a pure # plot -- because (Wiktorowicz et. al. 2020) is a "whole
    galaxy" model and returns M_sun / yr for its SFR.

    This is a wrapper for a Julia function (see src/integrator.jl for
    details)

    Decorators:
        np.vectorize

    Arguments:
        a0 {float}     -- The initial semimajor axis in Solar Radii
        e0 {float}     -- The initial eccentricity
        m1 {float}     -- The mass of the primary in Solar Masses
        m2 {float}     -- The mass of the secondary in Solar Masses

    Keyword Arguments:
        weight {float} -- the BPASS weight of the system (# / 10^6
                          solar masses)

    Returns:
        {mixed}
    """

    if not return_value:
        global matrix_elements

    params = [m1, m2, beta, alpha]
    a, e, h = takahe.helpers.integrate(a0, e0, params)

    if only_arrays:
        return a, e, h

    af = a[-1]

    e0 = e0
    ef = e[-1]
    P = takahe.helpers.compute_period(a, m1, m2)
    P0, Pf = P[0], P[-1]

    logP = np.log10(P)

    per_bins = int(8 // takahe.constants.PE_BINS_PER_W)
    ecc_bins = int(1 // takahe.constants.PE_BINS_ECC_W)

    binx = np.linspace(-2, 6, per_bins)
    biny = np.linspace(0,  1, ecc_bins)

    h_matrix = np.zeros((per_bins, ecc_bins))

    # Convert step sizes to years
    h     /= takahe.constants.SECONDS_PER_YEAR
    h_cum  = np.cumsum(h)

    for i in range(len(a)):
        if binx[0] <= logP[i] <= binx[-1]:
            # Determine which bins the current point is
            j = np.where(logP[i] >= binx)[0][-1]
            k = np.where(e[i] >= biny)[0][-1]

            # need the SFR for here too
            time = h_cum[i]
            if SFRD is not None and SFRD.inbounds(time / 1e9):
                bin_nr = SFRD.getBin(time / 1e9)
                sfr = SFRD.getBinContent(bin_nr) * SFRD.getBinWidth(bin_nr) * 1e9
            else:
                sfr = 1

            h_matrix[j][k] += (sfr * 1e9)

    return_age_matrix = h_matrix * weight

    if return_value:
        return return_age_matrix
    else:
        if 'matrix_elements' in globals().keys():
            matrix_elements.append(return_age_matrix)

        return

def period_eccentricity(in_df, Z=1, beta=1, alpha=0):
    """Computes the period-eccentricity distribution for an ensemble.

    Evolves the ensemble from 0 to the Hubble time to see how systems
    behave.

    Arguments:
        in_df {pd.DataFrame} -- The DataFrame representing your ensemble.

    Returns:
        {np.matrix}          -- A matrix of the binned Period-Eccentricty
                                distribution.
    """

    global pbar, matrix_elements

    # Don't know why we have to do this, computation breaks otherwise.
    pd.set_option('compute.use_numexpr', False)

    # Highly eccentric orbits lead to division by zero in the integrator.
    df = in_df.drop(in_df[in_df['e0'] == 1].index, inplace=False)

    if len(df) == 0:
        per_bins = int(8 // takahe.constants.PE_BINS_PER_W)
        ecc_bins = int(1 // takahe.constants.PE_BINS_ECC_W)
        return np.zeros((per_bins, ecc_bins))

    matrix_elements = []

    Z = float(Z)

    bins = takahe.constants.LINEAR_BINS
    SFRD_data = takahe.event_rates.generate_sfrd(bins)[Z]
    SFRD = takahe.histogram.histogram(edges=takahe.constants.LINEAR_BINS)
    SFRD.fill(SFRD_data)

    evolve_system(df['a0'].values,
                  df['e0'].values,
                  df['m1'].values,
                  df['m2'].values,
                  weight=df['weight'].values,
                  SFRD=SFRD,
                  beta=beta,
                  alpha=alpha)

    C = np.sum(matrix_elements, 0)

    return C, df

def coalescence_time(star, nyadzani=False, a=None):
    """Computes the coalescence time for a BSS.

    Uses eqn(16) of [1] to compute the coalescence time to 1PN accuracy.

    [1] https://arxiv.org/pdf/1905.06086.pdf

    Arguments:
        star {pd.Series} -- The star to compute the CT for.

    Returns:
        {float} -- the coalescence time in years.
    """

    if nyadzani:
        G = takahe.constants.G
        c = takahe.constants.c
        M_sun = takahe.constants.SOLAR_MASS
        m1 = star.m1 * M_sun
        m2 = star.m2 * M_sun

        M = m1 + m2

        mu = m1 * m2 / (m1 + m2)
        nu = mu / M

        beta = (64/5) * G**3 * m1 * m2 * (m1 + m2) / (c**5)
        beta1 = G * M * (7 - nu) / (4 * c**2)
        beta2 = G * M * (13-840*nu) / (336*c**2)

        # Not from paper - derived by hand
        a1 = G*M/(1-G*M) / (4*c**2/(7-nu))

        C1 = 3*beta1 - beta2
        C2 = 5*beta1**2 - 2*beta1*beta2 + beta2**2
        C3 = 5*beta1**3 - 2*beta1**2*beta2 + beta1*beta2**2 -beta2 **3
        C4 = beta1**6*(6*beta1 - beta2)
        C5 = beta1**4*(14*beta1**2 - 4*beta1*beta2 + beta2**2)
        C6 = beta1**2*(14*beta1**3 -5*beta1**2*beta2 + 2*beta1*beta2**2 - beta2**3)

        Tc  = a1**4 / (4*beta)
        Tc += a1**3 * C1 / (3*beta)
        Tc += a1**2 * C2 / (2*beta)
        Tc += a1 * C3 / beta

        valtolog = np.abs((a1**2+a1*beta2-beta1*beta2)/(a1-beta1))

        Tc += beta2**4 / beta * np.log10(valtolog)
        Tc += beta1**8 / (4*beta*(a1-beta1)**4)
        Tc += C4 / (3*beta*(a1-beta1)**3)
        Tc += C5 / (2*beta*(a1-beta1)**2)
        Tc += C6 / (  beta*(a1-beta1)   )

        return Tc / takahe.constants.SECONDS_PER_YEAR
    else:
        # https://link.springer.com/article/10.12942/lrr-2012-8
        # eqn(1), 2.5PN, assume point masses
        if a is None:
            a = star.a0

        q = star.m2 / star.m1
        Rsun = takahe.constants.SOLAR_RADIUS
        Msun = takahe.constants.SOLAR_MASS
        T_GW = 2.2e8 / (q*(1+q)) * (a/Rsun)**4 / (star.m1/(1.4*Msun))**3

        return T_GW / 4

def find_nearest(df, column, needle):
    index = abs(df[column] - needle)
    nearest = df[index.isin([index.nsmallest(1)])]

    return nearest

def _integrate_worker(p0, e0, m1=1.4, m2=1.4, pbar=None):
    pbar.update(1)
    return takahe.integrate_timescale(m1, m2, p0, e0) / takahe.constants.SECONDS_PER_GYR

def constant_coalescence_isocontour(ct):
    """Computes the isocontour of the coalescence time in
    period-eccentricity space.

    Uses a precomputed grid to determine the isocontours representing
    a given coalescence time. Can extract isocontours for arbitrarily
    many such times.

    Arguments:
        ct {mixed} -- the isocontour(s) required. If:
                        array: all requested isocontours plotted
                        int/float: just that isocontour plotted
    """
    if isinstance(ct, float) or isinstance(ct, int):
        ct = np.array([ct])

    p = np.linspace(1e-2, 1e2, 5000) # days
    e = np.linspace(0.0 , 1.0, 5000) # no dim.

    P, E = np.meshgrid(p, e, indexing='ij')

    fobj = BytesIO(pkgutil.get_data(__name__, 'data/isocontour_data.npy'))
    Z = np.load(fobj)

    return takahe.helpers.find_contours(P, E, Z, ct)

evolve_system = np.vectorize(evolve_system, excluded=['SFRD', 'beta', 'alpha', 'only_arrays', 'return_value'])

_integrate_worker = np.vectorize(_integrate_worker, excluded=['m1', 'm2', 'pbar'])
