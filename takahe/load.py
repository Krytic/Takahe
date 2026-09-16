import gzip
import time
from os.path import isfile

import numpy as np
import pandas as pd
from tqdm import tqdm
import takahe


def from_file(filepath, name_hints=None, options=dict()):
    """Loads a single file into memory.

    Loads a single file from a directory into memory. Makes a few assumptions:
        1. That the file structure is:
           m1  m2  a0  e0  weight  evolution_age  rejuvenation_age
        2. The presence of "_ct" in the filename indicates that the
           eighth column is the coalescence time.

    Arguments:
        filepath {str} -- The location of the file to load.

    Keyword Arguments:
        name_hints {list | None} -- A list containing the names of each of
                                    the columns to load (see Notes). If None,
                                    the default spec is used. (default: None)
        options {dict} -- A dict of options. So far the only option supported
                          is key_to_load, which specifies which HDF file key
                          to use if you are loading an hdf5 file.

    Returns:
        pd.DataFrame -- A dataframe of the data.
    """
    if name_hints is None:
        name_hints = []
        name_hints.extend(['m1', 'm2', 'a0', 'e0'])
        name_hints.extend(['weight', 'evolution_age', 'rejuvenation_age'])

        if "_ct" in filepath:
            name_hints.extend(['coalescence_time'])

    # this is wrong
    if ".h5" in filepath:
        df = pd.read_hdf(filepath, options['key_to_load'])
    else:
        df = pd.read_csv(filepath,
                         names=name_hints,
                         sep=r'\s+',
                         )

    return df


def from_gzip(filepath):
    """Loads a single gzip-compressed file into memory.

    Behaves like from_file(), but decompresses the file first. Appends
    a ".gz" extension to filepath if it is not already present.

    Arguments:
        filepath {string} -- The path to the (optionally already
                             ".gz"-suffixed) file to load.

    Returns:
        {pd.DataFrame} -- The loaded data.

    Raises:
        IOError -- if the (gzip-suffixed) file cannot be found.
    """
    if ".gz" not in filepath:
        filepath = filepath + ".gz"

    if isfile(filepath):
        with open(filepath, 'rb') as f:
            df = from_file(gzip.GzipFile(fileobj=f))

        return df

    raise IOError(f"File {filepath} not found.")
