import linecache

import numpy as np
from hoki import load
import pandas as pd
import takahe
from takahe.constants import *

def from_data(data):
    """
    Loads a binary star system from a dictionary of data.

    Arguments:
        data {dict} -- A dictionary containing the 4 elements necessary
        to solve the two body problem:
                       - M1 (mass of primary star)
                       - M2 (mass of secondary star)
                       - e0 (current eccentricity)
                       - a0/T (current semimajor axis / Period)
                    Period takes precedence over the SMA, so if one is provided, we use Kepler's law to compute the SMA.

                    Note that extra parameters can be provided in the
                    data dictionary, these are passed through to
                    BSS.create as the extra_terms array.

    Returns:
        BinaryStarSystem -- An ADT to represent the BSS.

    Raises:
        KeyError -- if data is not well-formed.
    """

    # force column names to lowercase so we can compare them
    keys = [k.lower() for k in data.keys()]

    # minimum keys required
    base_array = ['m1', 'm2', 'e0', 'a0']

    # we *need* m1 m2 and e0, we can infer a0 from kepler's 3rd law
    # if we know T
    if 'm1' in keys and 'm2' in keys and 'e0' in keys:
        if 'T' in keys:
            # Kepler's 3rd law
            first_term = (G * (data['m1'] + data['m2']))/(4*np.pi**2)
            data['a0'] = (first_term * data['t'] ** 2) ** (1/3)

        if 'a0' in keys:
            # Extra terms is a dict of data that isn't in {m1, m2, e0, a0}
            extra_terms = {k:v for k,v in data.items() if k not in base_array}
            return takahe.BSS.create(data['m1'],
                             data['m2'],
                             data['a0'],
                             data['e0'],
                             extra_terms
                            )

    # If we reach here, the data is not well-formed.
    raise KeyError("data must contain definitions for M1, M2, e0, and a0/T!")

def from_list(data_list):
    """Creates a binary star system ensemble from a list of configs

    Arguments:
        data_list {list} -- A list of config values. Each item in the
                            list must be acceptable by from_data; i.e.,
                            must contain M1, M2, e0, and a0/T.

    Returns:
        BinaryStarSystemEnsemble -- an ensemble representing the
                                    collection of Binary Star System
                                    objects.
    """
    ensemble = takahe.ensemble.create()

    for data in data_list:
        binary_star = from_data(data)
        ensemble.add(binary_star)

    return ensemble

def from_file(fname, name_hints=[], n_stars=100, mass=1e6):
    """
    Loads the first n_stars in a given file into a pandas dataframe.

    General utility loader for most cases. Can be replaced with a more
    flexible one, such as from_bpass().

    Arguments:
        fname {string} -- the path to the file we wish to open.

    Keyword Arguments:
        name_hints {list} -- A list of column names for pandas.
                             (default: {[]})
        n_stars {number} -- The number of stars (rows in file) to load
                            (default: {100})
        mass {number} -- The total mass of the ensemble. This is used to
                         populate the ensemble with weight*mass stars of
                         a given stellar configuration (default: {1e6})
    """

    # If we request all stars, set n_stars to None
    # pandas interprets this as "load entire file"
    if n_stars == 'all':
        n_stars = None

    # Read the file into a dataframe
    # Sample file format is:
    # m1   m2   a0    e0    weight   evolution_age   rejuvenation_age   coalescence_time
    # Note that the number of spaces can vary
    df = pd.read_csv(fname,
                     names=name_hints,
                     nrows=n_stars,
                     sep=r"\s+",
                     engine='python')

    # initialize an ensemble (glorified list)
    ensemble = takahe.ensemble.create()

    for row in df.iterrows():
        # weight represents the number of stars of this kind per 10^6 solar masses.
        number_of_stars_of_type = int(np.ceil(row[1]['weight'] * mass))

        # generate this many stars
        for n in range(number_of_stars_of_type):
            star = from_data(dict(row[1])) # create the BSS object
            ensemble.add(star) # add to ensemble -- O(1) operation

    return ensemble

def random_from_file(fname, draw_from, name_hints=[], n_stars=100, mass=1e6):
    """
    Loads a random sample of stars from a file.

    Arguments:
        fname {string} -- The path to the file you want to load
        draw_from {int} -- How many lines of the file to sample from. For
                           instance, random_from_file("somefile.dat",
                                                      10,
                                                      n_stars=7)
                           means "uniformly draw 7 stars from the first
                           10 lines of the file somefile.dat."

    Keword Arguments:
        name_hints {list} -- A list of hints for each column name for
                             pandas. Takahe can infer the name hints in
                             some instances however the name_hints
                             provided will take precedence if provided.
        n_stars {int} -- How many star *types* to sample from. Note that
                         the size of the ensemble will be different to
                         the size you pass in here, this is because
                         takahe accounts for the weight (number of systems
                         of this type per 10^6 solar masses).
        mass {float} -- The total mass to create.

    Returns:
        {BinaryStarSystemEnsemble} -- An ensemble object representing
                                      the ensemble of objects,
    """

    ensemble = takahe.ensemble.create()

    df = pd.read_csv(fname,
                     nrows=draw_from,
                     names=name_hints,
                     sep=r'\s+',
                     engine='python')

    sample = df.sample(n_stars)

    for star_line in sample.iterrows():
        star_line = star_line[1] # pandas returns a tuple from iterrows()
        for n in range(int(np.ceil(star_line['weight'] * mass))):
            binary_star = from_data(star_line)
            ensemble.add(binary_star)

    return ensemble
