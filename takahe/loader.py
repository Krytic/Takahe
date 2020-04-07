import numpy as np
from hoki import load
import pandas as pd
import takahe
from takahe.constants import *

def from_data(data, extra_terms=dict()):
    """
    Loads a binary star system from a dictionary of data.

    Arguments:
        data {dict} -- A dictionary containing the 4 elements necessary
        to solve the two body problem:
                       - M1 (mass of primary star)
                       - M2 (mass of secondary star)
                       - e0 (current eccentricity)
                       - a0/T (current semimajor axis / Period)
                    Period takes precedence over the SMA, so if one is provided, we use Kepler's law
                    to compute the SMA.

    Keyword Arguments:
        extra_terms {dict} -- A dictionary containing any extra terms
                              you wish to add to the BSS.

    Returns:
        BinaryStarSystem -- An ADT to represent the BSS.

    Raises:
        KeyError -- if data is not well-formed.
    """

    set_of_data_keys = set(data.keys())

    # Need to refactor. M1, M2, e0 are requried. a0 can be subbed for T
    if not set_of_data_keys.issubset({'M1', 'M2', 'e0', 'a0', 'T'}):
        raise KeyError("data must contain definitions for M1, M2, e0, \
                        and a0/T!")

    if 'T' in data.keys():
        first_term = (G * (data['M1'] + data['M2']))/(4*np.pi**2)
        data['a0'] = (first_term * data['T'] ** 2) ** (1/3)

    return takahe.BSS.create(data['M1'],
                             data['M2'],
                             data['a0'],
                             data['e0'],
                             extra_terms
                            )

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

    if n_stars == 'all':
        n_stars = None

    df = pd.read_csv(fname,
                     names=name_hints,
                     nrows=n_stars,
                     sep="   ",
                     engine='python')

    ensemble = takahe.ensemble.create()

    for row in df.iterrows():
        number_of_stars_of_type = int(np.ceil(row[1]['weight'] * mass))

        for n in range(number_of_stars_of_type):
            extra_terms = {k:v for k,v in row[1].items()
                               if k not in ['m1', 'm2', 'a0', 'e0']
                          }

            star = from_data(row[1], extra_terms)

            ensemble.add(star)

    return ensemble

def random_from_file(fname, name_hints=[], n_stars=100, mass=1e6):
    """
    Loads a random sample of stars from a file.

    This code is particularly hacky. We should find a better way to
    accomplish this.

    Arguments:
        fname {string} -- The path to the file you want to load

    Keyword Arguments:
        limit {number} -- The number of stars to load (default: {10})
        n {number} -- The number of lines in the file (default: {1000})

    Returns:
        {BinaryStarSystemEnsemble} -- An ensemble object representing
                                      the ensemble of objects,
    """
    import mmap, linecache
    ensemble = takahe.ensemble.create()

    # Determine the number of lines in the file requested.
    n_lines = 0
    f = open(fname, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    readline = buf.readline
    while readline():
        n_lines += 1
    f.close()
    lines = np.random.randint(1, n_lines, n_stars)

    for line in lines:
        l = linecache.getline(fname, line).strip()
        m = l.strip()
        n = m.split(' ')

        star = list(map(float, n))
        number_of_stars_of_type = int(np.ceil(star[4] * mass))

        for i in range(number_of_stars_of_type):
            binary_star = takahe.BSS.create(*star[0:4])
            ensemble.add(binary_star)

    return ensemble
