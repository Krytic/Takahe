import time
from os.path import isfile

import numpy as np
import pandas as pd
from tqdm import tqdm
import takahe

def from_file(filepath):
    name_hints = []
    name_hints.extend(['m1','m2','a0','e0'])
    name_hints.extend(['weight','evolution_age','rejuvenation_age'])

    df = pd.read_csv(filepath,
                     nrows=None,
                     names=name_hints,
                     sep=r'\s+',
                     engine='python',
                     dtype=np.float64,
                     skiprows=lambda i: bool(i%10))

    return df

def from_directory(path):
    dataframes = dict()

    n_files = len(takahe.constants.BPASS_METALLICITIES)

    for index in tqdm(range(n_files)):
        Zi = takahe.constants.BPASS_METALLICITIES[index]
        filepath = f"{path}/Remnant-Birth-bin-imf135_300-z{Zi}_StandardJJ.dat"

        if not isfile(filepath):
            filepath = filepath + ".gz"

        df = from_file(filepath)
        Z = takahe.helpers.format_metallicity(Zi)

        dataframes[str(Z)] = df

    return dataframes
