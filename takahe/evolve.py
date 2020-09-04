import bisect

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import pandas as pd
from scipy import stats
import takahe
from tqdm import tqdm

def find_between(a, low, high):
    i = bisect.bisect_left(a, low)
    g = bisect.bisect_right(a, high)
    if i != len(a) and g != len(a):
        return a[i:g]
    raise ValueError

@np.vectorize
def evolve_system(a0, e0, m1, m2):
    """Evolves a binary system until merger or the age of the Universe.

    # TODO: [description]

    Decorators:
        np.vectorize

    Arguments:
        a0 {float} -- The initial semimajor axis in Solar Radii
        e0 {float} -- The initial eccentricity
        m1 {float} -- The mass of the primary in Solar Masses
        m2 {float} -- The mass of the secondary in Solar Masses

    Returns:
        {tuple} -- The initial and final parameters:
                   - The initial eccentricity
                   - The initial period (days)
                   - The final eccentricity
                   - The final period (days)
                   - The number of samples
    """
    global pbar

    params = [m1, m2]

    a, e = takahe.helpers.integrate(a0, e0, params)

    af = a[-1]

    e0 = e0
    ef = e[-1]
    P = takahe.helpers.compute_period(a, m1, m2)
    P0, Pf = P[0], P[-1]

    nbin_width = 1e-2 # Todo: customisable
    mbin_width = 1e-2

    logP = np.log10(P)

    nbins = int(1 // nbin_width)
    mbins = int(8 // mbin_width)

    binx = np.linspace(-2, 6, nbins)
    biny = np.linspace(0,  1, mbins)

    ret = stats.binned_statistic_2d(logP, e, logP, 'count', bins=[binx, biny])

    bin_counts = ret.statistic

    pbar.update(1)

    return bin_counts

def period_eccentricity(in_df, transient_type="NSNS"):
    """Computes the period-eccentricity distribution for an ensemble.

    Evolves the ensemble from 0 to the Hubble time to see how systems
    behave.

    Strictly this is an SMA-eccentricity routine but the period is
    computable from the SMA using Kepler's third law.

    Adds FOUR new columns to the DataFrame:
        - a (the array of SMAs for this star)
        - e (the array of eccentricities for this star)
        - af (the final SMA for this star)
        - ef (the final eccentricity for this star)

    # Todo: Correct implementation.
    # Todo: Return a period array.

    Arguments:
        in_df {pd.DataFrame} -- The DataFrame representing your ensemble.

    Keyword Arguments:
        transient_type {str} -- The transient type (NSNS, NSBH, BHBH)
                                under consideration. (default: {"NSNS"})

    Returns:
        {pd.DataFrame} -- The DataFrame with the a and e arrays added
                          as new columns.
    """

    global pbar

    # Don't know why we have to do this, computation breaks otherwise.
    pd.set_option('compute.use_numexpr', False)

    # Now we mask out what we're not interested in.
    df = takahe.event_rates.filter_transients(in_df, transient_type)

    # Highly eccentric orbits lead to division by zero.
    df.drop(df[df['e0'] == 1].index, inplace=True)

    with tqdm(total=len(df)) as pbar:
        c = evolve_system(df['a0'].values,
                          df['e0'].values,
                          df['m1'].values,
                          df['m2'].values)

    Po = np.log10(np.array([4.072, 0.102, 0.421, 0.320, 0.323, 0.206, 0.184, 8.634, 18.779, 1.176, 45.060, 13.638, 2.616, 0.078]))
    eo = [0.113, 0.088, 0.274, 0.181, 0.617, 0.090, 0.606, 0.249,  0.828, 0.139,  0.399,  0.304, 0.169, 0.064]

    C = np.sum(c, 0)

    shp = C.shape

    x = np.linspace(-2, 6, shp[0])
    y = np.linspace(0, 1, shp[1])

    X, Y = np.meshgrid(x, y, indexing='ij')

    return X, Y, C, Po, Eo

def compute_ct(coalescence_time, star):
    # TODO: This needs work, there is a circular definition
    G = takahe.constants.G
    c = takahe.constants.c

    M = star.m1 + star.m2

    mu = star.m1 * star.m2 / (star.m1 + star.m2)
    nu = mu / M

    beta = star.beta
    beta1 = G * M * (7 - nu) / (4 * c**2)
    beta2 = G * M * (13-840*nu) / (336*c**2)

    a1 = 1 # Something????
    E1 = -G * M / (2*a1) + G * M / (8*c**2*a1**2) * (7-nu)

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
    Tc += beta2**4 / beta * np.log10((a1**2+a1*beta2-beta1*beta2)/(a1-beta1))
    Tc += beta1**8 / (4*beta*(a1-beta1)**4)
    Tc += C4 / (3*beta*(a1-beta1)**3)
    Tc += C5 / (2*beta*(a1-beta1)**2)
    Tc += C6 / (beta*(a1-beta1))

    return Tc
