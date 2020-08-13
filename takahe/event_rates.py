import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
from scipy.special import gamma, gammainc
import takahe
from tqdm import tqdm

def chirp_mass_distribution(input_dataframes,
                            transient_type='NSNS',
                            redshift=0):
    """Computes the chirp mass distribution for an ensemble.

    Computes the chirp mass distribution. Returns a histogram object which
    contains the number of events in each bin per year per cubic Gpc.
    The mass bins are selected to be from 10^0 - 10^2 in steps of 0.1 dex

    Arguments:
        input_dataframes {dict} -- A dictionary of dataframes, keyed by
                                   metallicity (relative to Z_solar).

    Keyword Arguments:
        transient_type {str} -- The type of transient we wish to consider
                                from NSNS, NSBH, BHBH. {default: NSNS}
        redshift {float} -- The redshift we want to compute at. This is
                            converted into a lookback time internally.
                            {default: 0}

    Returns:
        {takahe.histogram} -- a histogram containing the chirp mass
                              distribution.
    """

    tL = takahe.helpers.redshift_to_lookback(redshift)
    log_mass_bins = np.linspace(0, 2, 41)
    chirp_mass_bins = 10 ** log_mass_bins

    CMD = takahe.histogram.histogram(edges=chirp_mass_bins)

    edges = takahe.constants.LINEAR_BINS

    dataframes = input_dataframes.copy()

    primed_dfs = dict()

    for Z, in_df in dataframes.items():
        df = filter_transients(in_df, transient_type)
        constrain_masses(df, transient_type)
        df['M_chirp'] = (df['m1']*df['m2'])**(3/5) / (df['m1']+df['m2'])**(1/5)
        primed_dfs[Z] = df

    for i in tqdm(range(1, CMD.getNBins())):
        low = chirp_mass_bins[i-1]
        high = chirp_mass_bins[i]

        true_primed_dfs = dict()

        for Z, df in primed_dfs.items():
            df_prime = df[df['M_chirp'].between(low, high)]
            true_primed_dfs[Z] = df_prime

        events = composite_event_rates(true_primed_dfs, transient_type=None)

        CMD.fill(chirp_mass_bins[i-1], events.getBinContent(events.getBin(tL)))

    return CMD

def filter_transients(in_df, transient_type):
    """Filters a dataset by transient type.

    Filters a dataset by transient type (NSNS, NSBH, BHBH). Selects based
    on customisable values.

    Arguments:
        in_df {pd.DataFrame} -- The input dataframe.
        transient_type {str} -- The transient type (NSNS, NSBH, BHBH)

    Returns:
        {pd.DataFrame} -- The DataFrame containing the filtered data.

    Raises:
        AssertionError -- on malformed input.
    """

    assert isinstance(in_df, pd.DataFrame), ("Expected in_df to be a DataFrame"
                                             " in call to filter_transients.")
    assert transient_type in ['NSNS', 'NSBH', 'BHBH'], ("Expected"
                                                        " transient_type to"
                                                        " be NSNS, NSBH, or"
                                                        " BHBH in call to"
                                                        " filter_transients.")

    MASS_NS = takahe.constants.MASS_CUTOFF_NS
    MASS_BH = takahe.constants.MASS_CUTOFF_BH

    if transient_type == 'NSNS':
        # Both M1 and M2 are NS
        df = in_df[(in_df['m1'] < MASS_NS) & (in_df['m2'] < MASS_NS)].copy()
    elif transient_type == 'BHBH':
        # M1 and M2 are both BHs
        df = in_df[(in_df['m1'] > MASS_BH) & (in_df['m2'] > MASS_BH)].copy()
    elif transient_type == 'NSBH':
        df = in_df[
            ( # M1 is an NS, and M2 is a BH
                (in_df['m1'] < MASS_NS) & (in_df['m2'] > MASS_BH)
            )
            | # Or
            ( # M1 is a BH and M2 is an NS
                (in_df['m1'] > MASS_BH) & (in_df['m2'] < MASS_NS)
            )
        ].copy()

    return df

@njit
def _mass_worker_nsns(m, M):
    for i in range(len(m)):
        mi = -1 + np.sqrt(1+4*0.084*m[i]) / (2*0.084)
        Mi = -1 + np.sqrt(1+4*0.084*M[i]) / (2*0.084)

        mi = min(0.9*m[i], mi)
        Mi = min(0.9*M[i], Mi)

        m[i] = mi
        M[i] = Mi

    return m, M

@njit
def _mass_worker_nsbh(m, M):
    for i in range(len(m)):
        if m[i] < M[i]:
            m[i] = -1 + np.sqrt(1+4*0.084*m[i]) / (2*0.084)
            M[i] = 0.9 * M[i]
        else:
            m[i] = 0.9 * m[i]
            M[i] = -1 + np.sqrt(1+4*0.084*M[i]) / (2*0.084)

    return m, M

def constrain_masses(df, transient_type):
    if transient_type == 'NSNS':
        df['m1'], df['m2'] = _mass_worker_nsns(df['m1'].to_numpy(), df['m2'].to_numpy())
    elif transient_type == 'NSBH':
        df['m1'], df['m2'] = _mass_worker_nsbh(df['m1'].to_numpy(), df['m2'].to_numpy())
    elif transient_type == 'BHBH':
        df['m1'] = 0.9 * df['m1']
        df['m2'] = 0.9 * df['m2']

def generate_sfrd(tL_edges, func=None):
    """Generates the SFRD at every BPASS metallicity.

    Generates the SFRD for a given lookback time/s at every BPASS
    metallicity. This optionally takes a custom SFRD function, which must
    be of the form:
      >>> func(Metallicity, redshift)
    and returns the SFRD at that metallicity & redshift in units of
    M_sun / yr / Mpc^3.

    The BPASS metallicities are bin centers, not bin edges. To compute
    the SFRD at Z_i, we average Z_i and Z_{i-1}, and compute at this
    metallicity.

    Arguments:
        tL_edges {ndarray} -- An array of the edges of your histogram.

    Keyword Arguments:
        func {callable} -- The SFRD computation function to use. Takes
                           two arguments: the signature should be
                           func(Z, z). (default: {MadauDickinson})

    Returns:
        dict -- The SFRD at each BPASS metallicity. Indexed by relative
                to solar metallicity -- that is, the SFRD corresponding
                to Z = 0.040 will be indexed as:
                >>> SFRD[2.0] = np.array(...)
    """

    assert isinstance(tL_edges, np.ndarray), ("Expected tL_edges to be an"
                                              " ndarray in call to"
                                              " generate_sfrd()")

    assert callable(func) or func == None, ("Expected func to be a"
                                            " callable type in call to"
                                            " generate_sfrd()")

    if func == None:
        func = MadauDickinson

    total_SFRD = np.zeros(len(tL_edges))

    SFRD = dict()

    Z_fmts = takahe.constants.BPASS_METALLICITIES_F

    # Compute the array of means.
    # This sets means_arr[i] = np.mean(Z_fmts[i], Z_fmts[i+1])
    means_arr = [np.mean([Z_fmts[i], Z_fmts[i+1]]) for i in range(12)]

    # Prepend 0 to the means array
    means = [0.0]
    means.extend(means_arr)

    z = takahe.helpers.lookback_to_redshift(tL_edges)

    for i in range(13):
        # transform the BPASS metallicities into fractions of solar
        Z = takahe.constants.BPASS_METALLICITIES[i]
        Z = takahe.helpers.format_metallicity(Z)

        # Compute the *culmulative* metallicity up to this metallicity
        SFRD_here = func(means[i], z)

        # and remove all prior contributions
        SFRD_here = SFRD_here - total_SFRD
        total_SFRD = total_SFRD + SFRD_here

        SFRD[Z] = list(SFRD_here)

    return SFRD

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


def compute_dtd(in_df, extra_lt=None, transient_type='NSNS'):
    """Computes the DTD for a given transient type.

    Computes the Delay-Time distribution for a given transient type.
    Assumes that the maximum mass of a Neutron Star is given by
    takahe.constants.MASS_CUTOFF_NS and the minimum mass of a Black Hole
    is given by takahe.constants.MASS_CUTOFF_BH.

    Arguments:
        in_df {pd.DataFrame} -- The pandas DataFrame corresponding to the
                                input data (usually the output from
                                takahe.load.from_directory).

    Keyword Arguments:
        extra_lt {callable} -- A lambda representing the extra lifetime
                               term you wish to specify. This lambda
                               receives one argument - the dataframe
                               under consideration (so you can specify
                               an extra lifetime term that depends on
                               the stellar properties). If None, then an
                               empty lambda will be used.
                               (default: {None})

        transient_type {str} -- The transient type (NSNS, NSBH, BHBH)
                                that you wish to exclusively consider.

    Returns:
        {pd.DataFrame} -- The pandas DataFrame representing the DTD of
                          the system.
    """

    # First we set up some basic variables.

    histogram_edges = np.linspace(6.05, 11.05, 51)

    bins = [0.0]
    bins.extend(10**histogram_edges / 1e9)

    # Now we mask out what we're not interested in.

    if transient_type != None:
        df = filter_transients(in_df, transient_type)
    else:
        df = in_df.copy()

    # This is just shorthand
    G = takahe.constants.G
    c = takahe.constants.c

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

    if extra_lt != None:
        df['lifetime'] = df['lifetime'].apply(extra_lt, args=(df,))

    # Unit Conversions (back):
    df['a0'] /= (69550 * 1000) # Metre -> Solar Radius
    df['m1'] /= (1.989e30) # Kilogram -> Solar Mass
    df['m2'] /= (1.989e30) # As above

    # The minimum lifetime of a star is ~3 Myr, so introduce
    # an artificial cutoff there.
    df['lifetime'] = np.maximum(df.lifetime, 0.003)

    # I don't think it's required but just in case
    df.reset_index(drop=True, inplace=True)

    # Remove temporary columns
    population_at = df.drop(columns=['beta',
                                     'coalescence_time',
                                     'evolution_age',
                                     'rejuvenation_age',
                                     'circ',
                                     'divisor'
                                    ],
                            inplace=False)

    population_at['bins'] = pd.cut(population_at.lifetime,
                                   bins,
                                   right=False)

    out_df = population_at[["bins", "weight"]].groupby("bins").sum()

    out_df = out_df.values.ravel() / 1e6 / np.diff(bins)

    return out_df # events/Msun/Gyr

def single_event_rate(in_df,
                      Z,
                      SFRDi,
                      lin_edges,
                      extra_lt=None,
                      transient_type='NSNS',
                      as_hist=False
                     ):
    """Computes the event rate of a single metallicity.

    Computes the event rate for metallicity Z, using Langer & Norman's [1]
    FMD expression and Madau & Dickinson's [2] SFRD expression. In a future
    version this will be customisable.

    [1] https://iopscience.iop.org/article/10.1086/500363

    [2] https://www.annualreviews.org/doi/pdf/10.1146/annurev-astro-081811-125615

    Arguments:
        in_df {pandas.dataframe} -- The dataframe of stellar data
                                    (typically the output from
                                    takahe.load.from_file())

        Z {float} -- The metallicity to compute at, expressed as a
                     fraction of solar metallicity.

    Keyword Arguments:
        extra_lt {callable} -- A lambda representing the extra lifetime
                               term you wish to specify. This lambda
                               receives one argument - the dataframe
                               under consideration (so you can specify
                               an extra lifetime term that depends on
                               the stellar properties). If None, then an
                               empty lambda will be used.
                               (default: {None})

        transient_type {str} -- The transient type (NSNS, NSBH, BHBH)
                                that you wish to exclusively consider.

    Returns:
        {takahe.histogram} -- A histogram representing the total event
                              rate.
    """

    if extra_lt == None:
        extra_lt = lambda lt, df: lt

    # Numexpr causes our computations to break.
    # This is a "feature" of numexpr as far as I can tell
    pd.set_option('compute.use_numexpr', False)

    LOG_edges = [0.0]
    LOG_edges.extend(10**np.linspace(6.05, 11.05, 51)/1e9)

    # TODO: Change to integrating the EOMs

    DTD = takahe.histogram.histogram(edges=LOG_edges)
    SFRD = takahe.histogram.histogram(edges=lin_edges)
    events = takahe.histogram.histogram(edges=lin_edges)

    DTDi = compute_dtd(in_df, extra_lt, transient_type)

    DTD.fill(LOG_edges[:-1], DTDi)

    SFRD.fill(lin_edges, SFRDi)

    for i in range(1, len(lin_edges)):
        t1 = lin_edges[i-1]

        t2 = lin_edges[i]

        this_SFR = SFRD.integral(t1, t2) * 1e9

        this_SFR /= (1e-3)**3

        # Convolve the SFH with the DTD to get the event rates
        for j in range(i):
            t1_prime = t2 - lin_edges[j]
            t2_prime = t2 - lin_edges[j+1]

            events_in_bin = DTD.integral(t2_prime, t1_prime)

            events.fill(lin_edges[j], events_in_bin * this_SFR)

    # Normalise to years:
    events /= (np.diff(lin_edges) * 1e9)

    events._values = np.append(events._values, events._values[-1])

    if as_hist:
        return events
    else:
        return events._values

def composite_event_rates(dataframes, extra_lt=None, transient_type='NSNS'):
    """Computes the event rate for a variety of stars at different
    metallicities.

    Computes an ensemble event rate. Makes the same assumptions that
    single_event_rate does (because it calls it), namely:

    - Uses Langer & Norman's [1] FMD expression and Madau & Dickinson's
    [2] SFRD expression. In a future version this will be customisable.

    [1] https://iopscience.iop.org/article/10.1086/500363

    [2] https://www.annualreviews.org/doi/pdf/10.1146/annurev-astro-081811-125615

    Arguments:
        dataframes {dict} -- A dictionary of dataframes of stellar
                             properties, indexed by fractional
                             metallicity (as string).

    Keyword Arguments:
        extra_lt {callable} -- A lambda representing the extra lifetime
                               term you wish to specify. This lambda
                               receives one argument - the dataframe
                               under consideration (so you can specify
                               an extra lifetime term that depends on
                               the stellar properties). If None, then an
                               empty lambda will be used.
                               (default: {None})

    Returns:
        {takahe.histogram} -- A histogram representing the total event
                              rate for your ensemble.
    """

    assert isinstance(dataframes, dict), ("Expected dataframes to be a dict "
                                          "in call to composite_event_rates.")

    key = list(dataframes.keys())[0]

    assert isinstance(dataframes[key], pd.DataFrame), ("Expected dataframes "
                                                       "to contain dataframe "
                                                       "objects in call to "
                                                       "composite_event_rates."
                                                       )

    assert callable(extra_lt) or extra_lt is None, ("Expected extra_lt to be "
                                                    "callable or None in call "
                                                    "to composite_event_rates")

    types = ['NSNS', 'NSBH', 'BHBH', None]

    assert transient_type in types, ("Expected transient_type to be "
                                     + (" or ".join(types))
                                     + "in call to composite_event_rates")

    if extra_lt == None:
        extra_lt = lambda lt, df: lt

    # lin_edges = [0.0]
    # lin_edges.extend(10**np.linspace(6.05, 11.05, 51) / 1e9)
    edges = takahe.constants.LINEAR_BINS

    total_event_rate = np.zeros(len(edges))
    SFRD = generate_sfrd(edges)

    N_datapoints = np.zeros(len(edges)-1)

    for i in range(13):
        Z = takahe.constants.BPASS_METALLICITIES[i]
        Z = takahe.helpers.format_metallicity(Z)
        df = dataframes[str(Z)]

        SFRD_here = SFRD[Z]

        event_rate = single_event_rate(df,
                                       Z,
                                       SFRD_here,
                                       edges,
                                       extra_lt,
                                       transient_type,
                                       as_hist=True
                                      )

        total_event_rate = total_event_rate + event_rate._values

        N_datapoints = N_datapoints + event_rate._hits

    TER = takahe.histogram.histogram(edges=edges)
    TER.fill(edges, total_event_rate)
    TER.reregister_hits(N_datapoints)
    return TER

def single_event_rate_save_result(datafile, transient_type, output_dir):
    kick = 'Bray' if 'Bray' in datafile else 'Hobbs'
    edges = takahe.constants.LINEAR_BINS

    SFRD = generate_sfrd(edges)

    Z = takahe.helpers.extract_metallicity(datafile)
    df = takahe.load.from_file(datafile)

    SFRD_here = SFRD[Z]

    event_rate = single_event_rate(df,
                                   Z,
                                   SFRD_here,
                                   edges,
                                   None,
                                   transient_type,
                                   as_hist=True
                                  )

    event_rate.to_pickle(f"{output_dir}/{kick}-{transient_type}-{Z}.p")
