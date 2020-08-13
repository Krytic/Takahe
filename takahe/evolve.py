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

def evolve_system(star, pbar, Pv, Ev):
    nbin_width = 1e-4 # Todo: customisable
    mbin_width = 1e-4

    t_eval = np.linspace(0, takahe.constants.HUBBLE_TIME, 1000)

    a, e = takahe.helpers.integrate(t_eval, star.a0, star.e0, star.beta)
    P = takahe.helpers.compute_period(a, star.m1, star.m2)

    star['af'] = a[-1]
    star['ef'] = e[-1]
    star['Pf'] = takahe.helpers.compute_period(a[-1], star.m1, star.m2)

    star['P0'] = takahe.helpers.compute_period(a[0],  star.m1, star.m2)

    star['P'] = P
    star['e'] = e

    # ecc_bin_switches = [np.argmin(np.abs(ed - Ev)) for ed in e]
    # per_bin_switches = [np.argmin(np.abs(Pd - Pv)) for Pd in P]

    # bin_switches = ecc_bin_switches or per_bin_switches

    # h = t_eval[1] - t_eval[0]

    # t = [h*bin_switches[i] - h*bin_switches[i-1] for i in range(0, len(bin_switches)-1, -1)]

    logP = np.log10(P)

    nbins = int((max(e)-min(e)) // nbin_width)
    mbins = int((max(logP)-min(logP)) // mbin_width)

    binx = np.linspace(min(e), max(e), nbins)
    biny = np.linspace(min(logP), max(logP), mbins)


    # Todo: WTF why is this "empty"???
    try:
        ret = stats.binned_statistic_2d(e, logP, e, 'count', bins=[binx, biny])
    except ValueError as ex:
        raise ValueError(str(ex) + "\n\nFailed on e:\n" + str(list(e)) + "\nlogP:\n" + str(list(logP)))

    bin_counts = ret.statistic

    lt = star.lifetime

    k_points = len(e)

    time_per_point = lt / k_points

    time_per_2d_bin = bin_counts * time_per_point

    pbar.update(1)

    return star

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
    histogram_edges = np.linspace(6.05, 11.05, 51)
    eccentricity_bins = np.linspace(0, 1, 1000)
    period_bins = np.linspace(-6, 4, 1000)

    bins = [0.0]
    bins.extend(10**histogram_edges / 1e9)

    # Now we mask out what we're not interested in.

    df = takahe.event_rates.filter_transients(in_df, transient_type)

    # This is just shorthand
    G = takahe.constants.G # m^3 / kg*s
    c = takahe.constants.c # m /s

    # Highly eccentric orbits lead to division by zero.
    df.drop(df[df['e0'] == 1].index, inplace=True)

    # Unit Conversions:
    df['a0'] *= (69550 * 1000) # Solar Radius -> Metre
    df['m1'] *= 1.989e30 # Solar Mass -> Kilogram
    df['m2'] *= 1.989e30 # Solar Mass -> Kilogram

    # Introduce some temporary terms, to make computation easier
    df['beta'] = ((64/5) * G**3 * df['m1'] * df['m2']
                         * (df['m1'] + df['m2'])
                         / (c**5))

    # m^4 / s

    df.drop(df[df['beta'] == 0].index, inplace=True)

    df['circ'] = df['a0']**4 / (4*df['beta'])

    df['divisor'] = ((1 - df['e0'] ** (7/4)) ** (1/5)
                  *  (1+121/304 * df['e0'] ** 2))

    df['coalescence_time'] = ((df['circ'] * (1-df['e0']**2)**(7/2)
                           / df['divisor'])
                           / (1e9 * 60 * 60 * 24 * 365.25))

    df['lifetime'] = (df['evolution_age'] / 1e9
                   +  df['rejuvenation_age'] / 1e9
                   +  df['coalescence_time']
                     )

    # Unit Conversions (back):
    df['a0'] /= (69550 * 1000) # Metre -> Solar Radius
    df['m1'] /= (1.989e30) # Kilogram -> Solar Mass
    df['m2'] /= (1.989e30) # As above
    df['beta'] /= ((69550 * 1000) ** 4 / (1e9 * 60 * 60 * 24 * 365.25))
    # m^4 / s -> Solar Radius^4 / Gyr

    # The minimum lifetime of a star is ~3 Myr, so introduce
    # an artificial cutoff there.
    df['lifetime'] = np.maximum(df.lifetime, 0.003)

    # I don't think it's required but just in case
    df.reset_index(drop=True, inplace=True)

    # Remove temporary columns
    df = df.drop(columns=['coalescence_time',
                          'evolution_age',
                          'rejuvenation_age',
                          'circ',
                          'divisor'
                         ],
                        inplace=False)

    # Remove systems with lifetime > Hubble time
    df.drop(df[df['lifetime'] > takahe.constants.HUBBLE_TIME].index, inplace=True)

    with tqdm(total=len(df)) as pbar:
        df, bins = df.apply(evolve_system, axis=1, args=(pbar,period_bins,eccentricity_bins))

    return df
