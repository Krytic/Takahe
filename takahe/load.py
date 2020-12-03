import gzip
import time
from os.path import isfile

import numpy as np
import pandas as pd
from tqdm import tqdm
import takahe

def from_file(filepath, options=dict()):
    """Loads a single file into memory.

    Loads a single file from a directory into memory. Makes a few assumptions:
        1. That the file structure is:
           m1  m2  a0  e0  weight  evolution_age  rejuvenation_age
        2. The presence of "_ct" in the filename indicates that the
           eighth column

    Arguments:
        filepath {[type]} -- [description]

    Keyword Arguments:
        options {[type]} -- [description] (default: {dict()})

    Returns:
        [type] -- [description]
    """
    name_hints = []
    name_hints.extend(['m1','m2','a0','e0'])
    name_hints.extend(['weight','evolution_age','rejuvenation_age'])

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
    if ".gz" not in filepath:
        filepath = filepath + ".gz"

    if isfile(filepath):
        with open(filepath, 'rb') as f:
            df = from_file(gzip.GzipFile(fileobj=f))

        return df

    raise IOError(f"File {filepath} not found.")


def from_directory(path):
    """Loads the contents of a directory into memory.

    [description]

    Arguments:
        path {[type]} -- [description]

    Returns:
        [type] -- [description]

    Raises:
        IOError -- [description]
    """
    dataframes = dict()

    n_files = len(takahe.constants.BPASS_METALLICITIES)

    for index in tqdm(range(n_files)):
        Zi = takahe.constants.BPASS_METALLICITIES[index]
        filepath = f"{path}/Remnant-Birth-bin-imf135_300-z{Zi}_StandardJJ.dat"

        if not isfile(filepath):
            # We may be dealing with a large file, if that is the
            # case we should check if its zipped
            if isfile(filepath + ".gz"):
                filepath = filepath + ".gz"

                with open(filepath, 'rb') as f:
                    df = from_file(gzip.GzipFile(fileobj=f))
            else:
                raise IOError((f"File {filepath} not found (tried looking "
                                "for gzip file too)."))
        else:
            df = from_file(filepath)

        Z = takahe.helpers.format_metallicity(Zi)

        dataframes[Z] = df

    return dataframes
