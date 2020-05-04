import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import takahe

n_stars = 1000

plt.style.use('krytic')

def main(z, *args, **kwargs):
    global n_stars
    print(z)
    universe = takahe.universe.create('real')
    universe.populate(f'data/newdata/Remnant-Birth-bin-imf135_300-z{z}_StandardJJ.dat',
        n_stars=n_stars,
        name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'],
        load_type='random')

    universe.plot_event_rate()
    plt.savefig(f'output/SFRD/{z}.png')

    return 1

cpus = mp.cpu_count()
pool = mp.Pool(cpus)

z_array = ['020', '002']
results = [pool.apply_async(main, args=(z,)) for z in z_array]

results = [r.get() for r in results]
print(results)

pool.close()
