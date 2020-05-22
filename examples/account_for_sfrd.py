import time
import multiprocessing as mp
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import takahe

n_stars = 1000
#plt.rcParams['figure.figsize'] = (40, 40)

#plt.style.use('krytic')

data_dir = 'data/newdata'

files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

def execute(file):

    universe = takahe.universe.create('real')

    universe.populate(f"{data_dir}/{file}", n_stars=n_stars)

    z = universe.get_metallicity()

    size = universe.populace.size()

#    universe.set_nbins(51)
    universe.event_rate()

    end = time.time()
    print(f"Completed z={z} in {end-start} seconds. {n_stars} requested, {size} generated.")

    return 1

# cpus_to_use = min(mp.cpu_count(), len(files))

# print(f"Running on {cpus_to_use} CPUs")

# pool = mp.Pool(cpus_to_use)

# results = [pool.apply_async(execute, args=(file,)) for file in files]

# results = [r.get() for r in results]

# pool.close()

execute(files[0])
