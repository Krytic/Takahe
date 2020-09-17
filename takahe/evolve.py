import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import pandas as pd
from scipy import stats
import takahe
from tqdm import tqdm

def evolve_system(a0, e0, m1, m2, lifetime, weight=1, SFRD=None):
    """Evolves a binary system until merger or the age of the Universe.

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
        {tuple}        -- The initial and final parameters:
                          - The initial eccentricity
                          - The initial period (days)
                          - The final eccentricity
                          - The final period (days)
                          - The number of samples
    """

    global pbar, matrix_elements

    params = [m1, m2]

    a, e, h = takahe.helpers.integrate(a0, e0, params)

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
            bin_nr = SFRD.getBin(h_cum[i-1] / 1e9)
            sfr = SFRD.getBinContent(sfr)

            h_matrix[j][k] += (time * sfr)

    time_in_bin = h_matrix * weight

    # # * yr * / M_sun * (M_sun / yr / Mpc^3)
    # = # / Mpc^3

    pbar.update(1)

    matrix_elements.append(time_in_bin)

def period_eccentricity(in_df, transient_type="NSNS", Z=1):
    """Computes the period-eccentricity distribution for an ensemble.

    Evolves the ensemble from 0 to the Hubble time to see how systems
    behave.

    Arguments:
        in_df {pd.DataFrame} -- The DataFrame representing your ensemble.

    Keyword Arguments:
        transient_type {str} -- The transient type (NSNS, NSBH, BHBH)
                                under consideration. (default: {"NSNS"})

    Returns:
        {np.matrix}          -- A matrix of the binned Period-Eccentricty
                                distribution.
    """

    global pbar, matrix_elements

    # Don't know why we have to do this, computation breaks otherwise.
    pd.set_option('compute.use_numexpr', False)

    # Now we mask out what we're not interested in.
    df = takahe.event_rates.filter_transients(in_df, transient_type)

    # Highly eccentric orbits lead to division by zero in the integrator.
    df.drop(df[df['e0'] == 1].index, inplace=True)

    if len(df) == 0:
        per_bins = int(8 // takahe.constants.PE_BINS_PER_W)
        ecc_bins = int(1 // takahe.constants.PE_BINS_ECC_W)
        return np.zeros((per_bins, ecc_bins))

    matrix_elements = []

    df['lifetime'] = df['evolution_age'] + df['rejuvenation_age']

    Z = float(Z)

    bins = takahe.constants.LINEAR_BINS
    SFRD_data = takahe.event_rates.generate_sfrd(bins)[Z]
    SFRD = takahe.histogram.histogram(edges=takahe.constants.LINEAR_BINS)
    SFRD.fill(SFRD_data)

    with tqdm(total=len(df)) as pbar:
        evolve_system(df['a0'].values,
                      df['e0'].values,
                      df['m1'].values,
                      df['m2'].values,
                      df['lifetime'].values,
                      weight=df['weight'].values,
                      SFRD=SFRD)


    C = np.sum(matrix_elements, 0)

    return C

def compute_ct(coalescence_time, star):
    # TODO: This needs work, there is a circular definition in the paper
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

evolve_system = np.vectorize(evolve_system, excluded=['SFRD'])
