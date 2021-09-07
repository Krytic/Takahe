from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from os import path
import pandas as pd
import pkgutil
from scipy import stats
from scipy.optimize import minimize
import takahe
from tqdm import tqdm

def evolve_system(a0, e0, m1, m2, beta=1, alpha=0, evotime=0):
    """
    Evolves a binary system until merger or the age of the Universe.

    This is a wrapper for a Julia function (see src/integrator.jl for
    details)

    Arguments:
        a0 {float}     -- The initial semimajor axis in Solar Radii
        e0 {float}     -- The initial eccentricity
        m1 {float}     -- The mass of the primary in Solar Masses
        m2 {float}     -- The mass of the secondary in Solar Masses

    Keyword Arguments:
        weight {float} -- the BPASS weight of the system (# / 10^6
                          solar masses)

    Returns:
        {tuple}        -- A tuple containing the semimajor axes (in solar
                          radii), eccentricities, and timesteps (in
                          seconds) of the binary star as it classically
                          decays.
    """

    params = [m1, m2, beta, alpha, evotime]
    a, e, h, reason = takahe.helpers.integrate(a0, e0, params)

    return a, e, h

def period_eccentricity(in_df, Z, transient_type='NSNS', outdir=None):
    """
    Computes the period-eccentricity-time cube for an ensemble of data.

    Uses the method outlined in (Richards 2021) to compute the
    period-eccentricity distribution of a given dataset.

    Formally, this computes the period-eccentricity-time distribution -
    the period-eccentricity distribution can be resolved by compressing
    the cube along the time axis.

    Arguments:
        in_df {pd.DataFrame} -- The input dataframe.
        Z {mixed}            -- The metallicity corresponding to this
                                file. Must be in a format we can coerce
                                into a Takahe-formatted metallicity (see
                                takahe.helpers.format_metallicity() for
                                more details).

    Keyword Arguments:
        transient_type {string} -- The transient type (NSNS, BHBH, NSBH)
                                   to compute for.
        outdirectory {string}   -- a path to an output directory
                                   to save. If None, Takahe will
                                   not save its output.
                                   (Default: None)

    Returns:
        {tuple} -- Either a 2-tuple or a 1-tuple depending on if the
                   cube was saved. It always returns the cube as the
                   first element of the tuple. If outdir is set, then it
                   returns the name of the output file as the second
                   element in the tuple.

    Raises:
        AssertionError -- on malformed input.
    """

    assert isinstance(in_df, pd.DataFrame), "Expected in_df to be a DataFrame"
    assert isinstance(Z, (str, float)), "Expected Z to be a ..." # Complete
    assert transient_type in ['NSNS', 'NSBH', 'BHBH'], ("Expected"
                                                        " transient_type to be"
                                                        " one of: NSNS, NSBH,"
                                                        " or BHBH.")
    assert isinstance(outdir, str), "Expected outdir to be a string"
    assert path.exists(outdir), "Outdir must exist."

    df = takahe.event_rates.filter_transients(in_df, transient_type)

    df['age'] = df['evolution_age'] + df['rejuvenation_age']
    df['e'] = df['e0']

    hist = takahe.histogram.histogram_2d((-2, 6), (0, 1), 80, 100)

    x_ex, y_ex = hist.to_extent()

    cube = takahe.frame.FrameCollection((x_ex, y_ex), (dt, unit))

    Z_So_Far = np.zeros((80, 100))

    SFR_obj = takahe.event_rates.generate_sfrd(takahe.constants.LINEAR_BINS)[float(metallicity)]
    SFRD = takahe.histogram.histogram(edges=takahe.constants.LINEAR_BINS)
    SFRD.fill(SFR_obj)

    for t in tqdm(range(0, int(np.ceil(takahe.constants.HUBBLE_TIME * 1e9)), int(dt))):
        frame = takahe.frame.Frame(t, np.zeros((80, 100)))
        cube.insert(frame)

    i = 0
    N = int(len(df))

    takahe.debug('info', 'Beginning numerical integration')

    with tqdm(total=N) as pbar:
        for _, row in df.iterrows():
            a0, e0, m1, m2, weight = row.a0, row.e0, row.m1, row.m2, row.weight
            evotime = row.age
            a, e, h = evolve_system(a0, e0, m1, m2, evotime=row.age)

            h = np.cumsum(h)

            for j in range(len(h)):
                t   = (h[j] + evotime) / takahe.constants.SECONDS_PER_GYR
                P   = np.log10(takahe.helpers.compute_period(a[j], m1, m2))
                ecc = e[j]

                if P < -2 or P > 6:
                    continue
                if np.isnan(P) or ecc < 0 or ecc > 1:
                    takahe.debug("warning", f"Invalid P ({P}) or e ({ecc}) - skipping")
                    continue

                frame = cube.find(t)
                bins = hist.getBin(P, ecc)

                binnr = SFRD.getBin(np.min([t, takahe.constants.HUBBLE_TIME]))
                sfr = SFRD.getBinContent(binnr) * SFRD.getBinWidth(binnr) * 1e9

                frame.z[bins[0]][bins[1]] += (weight * 1)

            i += 1

            pbar.update(1)

    if outdir != None:
        takahe.debug('info', "Saving Cube...")

        fname = f"{outdir}/Period_eccentricity_cube-{kick}-{alpha}-{beta}.fr"

        cube.save(fname)

        return cube, fname

    return (cube, )

def coalescence_time(star):
    """Computes the coalescence time of a star.

    Computes the coalescence time of a binary star by explicitly evolving
    the binary in time until coalesence, or well past the age of the
    Universe. Returns min(coalescence_time, 100) Gyr.

    Arguments:
        star {pd.Series} -- The binary star system under consideration

    Returns:
        {float} -- The coalesence time in gigayears.
    """

    assert isinstance(star, pd.Series), ("Expected a Series object to be "
                                         "passed to coalescence_time.")

    evo = star.evolution_age + star.rejuvenation_age
    _, _, h, _ = evolve_system(star.a0,
                               star.e0,
                               star.m1,
                               star.m2,
                               evotime=evo)

    return np.sum(h) / takahe.constants.SECONDS_PER_GYR

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
    if isinstance(ct, [np.float, np.int]):
        ct = np.array([ct])

    p = np.linspace(1e-2, 1e2, 5000) # days
    e = np.linspace(0.0 , 1.0, 5000) # no dim.

    P, E = np.meshgrid(p, e, indexing='ij')

    fobj = BytesIO(pkgutil.get_data(__name__, 'data/isocontour_data.npy'))
    Z = np.load(fobj)

    return takahe.helpers.find_contours(P, E, Z, ct)

evolve_system = np.vectorize(evolve_system, excluded=['beta', 'alpha'])

