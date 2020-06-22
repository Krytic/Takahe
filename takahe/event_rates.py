import numpy as np
import pandas as pd
from scipy.special import gamma, gammainc
import takahe
from tqdm import tqdm

def single_event_rate(df, Z, extra_lt=None):
    """Computes the event rate of a single metallicity.

    Computes the event rate for metallicity Z, using Langer & Norman's [1]
    FMD expression and Madau & Dickinson's [2] SFRD expression. In a future
    version this will be customisable.

    [1] https://iopscience.iop.org/article/10.1086/500363

    [2] https://www.annualreviews.org/doi/pdf/10.1146/annurev-astro-081811-125615

    Arguments:
        df {pandas.dataframe} -- The dataframe of stellar data
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

    Returns:
        {takahe.histogram} -- A histogram representing the total event
                              rate.
    """

    assert isinstance(df, pd.DataFrame), ("Expected df to be a pandas "
                                          "DataFrame in call to "
                                          "single_event_rate.")

    assert isinstance(Z, float), ("Expected Z to be a float in call to "
                                  "single_event_rate.")

    assert callable(extra_lt) or extra_lt is None, ("Expected extra_lt to be "
                                                    "callable or None in call "
                                                    "to single_event_rate.")

    if extra_lt == None:
        extra_lt = lambda df: 0

    lin_edges = np.linspace(0, takahe.constants.HUBBLE_TIME, 51)

    total_event_rate = takahe.histogram.histogram(edges=lin_edges)
    total_SFRD = takahe.histogram.histogram(edges=lin_edges)
    Z_prev = 0.0

    events_histograms = dict()

    df.drop(df[df['m1'] > takahe.constants.MASS_CUTOFF].index, inplace=True)
    df.drop(df[df['m2'] > takahe.constants.MASS_CUTOFF].index, inplace=True)

    G = 6.67430e-11
    c = 299792458

    # Highly eccentric orbits lead to division by zero.
    df.drop(df[df['e0'] == 1].index, inplace=True)

    # Unit Conversions:
    df['a0'] *= (69550 * 1000) # Solar Radius -> Metre
    df['m1'] *= 1.989e30 # Solar Mass -> Kilogram
    df['m2'] *= 1.989e30 # Solar Mass -> Kilogram

    # Introduce some temporary terms, to make computation easier
    temp_series = pd.to_numeric((64/5) * (G**3*df['m1']*df['m2']*(df['m1']+df['m2'])) / (c**5))
    df['beta'] = temp_series
    df['circ'] = df['a0']**4 / (4*df['beta'])
    df['divisor'] = ((1-df['e0']**(7/4))**(1/5)*(1+121/304 * df['e0']**2))
    df['coalescence_time'] = (df['circ'] * (1-df['e0']**2)**(7/2) / df['divisor']) / (1e9 * 60 * 60 * 24 * 365.25)
    df['lifetime'] = (df['evolution_age'] + df['rejuvenation_age']) / 1e9 + df['coalescence_time'] + extra_lt(df)



    # Unit Conversions (back):
    df['a0'] /= (69550 * 1000) # Metre -> Solar Radius
    df['m1'] /= (1.989e30) # Kilogram -> Solar Mass
    df['m2'] /= (1.989e30) # As above

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

    # population_at['bins'] = pd.cut(population_at.lifetime, np.append([0.0], 10**lin_edges / 1e9), right=False)
    # That's a quick reminder to myself about BPASS binning potentially

    # Perform the binning
    # Thank you to Max Briel for the tip here
    population_at['bins'] = pd.cut(population_at.lifetime,
                                   lin_edges,
                                   right=False)

    bin_widths = lin_edges

    out_df = population_at[["bins", "weight"]].groupby("bins").sum()

    z = takahe.helpers.lookback_to_redshift(bin_widths)

    for Zi in takahe.constants.BPASS_METALLICITIES:
        Zi = takahe.helpers.format_metallicity(Zi)
        Zc = np.mean([Zi, Z_prev])

        if Zi > Z:
            break

        SFRDi = gammainc(0.84, Zc**2 * 10**(0.3*z)) \
                               * 0.015 * (1+z)**2.7 / \
                               (1+((1+z)/2.9)**5.6)

        Z_prev = Zi

        SFRD = takahe.histogram.histogram(edges=z)
        SFRD.fill(z, w=SFRDi)

        SFRD = SFRD - total_SFRD # remove prior SFRD contributions

        total_SFRD += SFRD

    DTDi = out_df.values.ravel() / 1e6 / np.diff(bin_widths)

    DTDi = np.append(DTDi, DTDi[-1]) # Because DTDi isn't long enough?

    DTD = takahe.histogram.histogram(edges=lin_edges)
    DTD.fill(lin_edges, w=DTDi)

    events = takahe.histogram.histogram(edges=lin_edges)

    for i in range(1, len(bin_widths)):
        t1 = max(bin_widths[i-1], 0.0000001)

        t2 = bin_widths[i]

        this_SFR = SFRD.integral(t1, t2) * 1e9

        this_SFR /= (1e-3)**3

        # Convolve the SFH with the DTD to get the event rates
        for j in range(i):
            t1_prime = t2 - bin_widths[j]
            t2_prime = t2 - bin_widths[j+1]

            events_in_bin = DTD.integral(t2_prime, t1_prime)

            events.fill(bin_widths[j], events_in_bin * this_SFR)

    # Normalise to years:
    events /= (np.diff(bin_widths) * 1e9)

    events_histograms[str(Z)] = events

    total_SFRD._values += (SFRD._values)
    total_event_rate._values += events._values

    return total_event_rate

def composite_event_rates(dataframes, extra_lt=None):
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
                                                       "to contain dataframes "
                                                       " incall to "
                                                       "composite_event_rates."
                                                       )

    assert callable(extra_lt) or extra_lt is None, ("Expected extra_lt to be "
                                                    "callable or None in call "
                                                    "to composite_event_rates")

    if extra_lt == None:
        extra_lt = lambda df: 0

    lin_edges = np.linspace(0, takahe.constants.HUBBLE_TIME, 51)

    total_event_rate = takahe.histogram.histogram(edges=lin_edges)
    total_SFRD = takahe.histogram.histogram(edges=lin_edges)

    for index in range(len(takahe.constants.BPASS_METALLICITIES)):
        Zi = takahe.constants.BPASS_METALLICITIES[index]
        Z = takahe.helpers.format_metallicity(Zi)

        df = dataframes[str(Z)]

        events = single_event_rate(df, Z, extra_lt)
        total_event_rate += events

    return total_event_rate
