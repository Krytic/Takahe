from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import takahe

n_stars = 100

plt.style.use('krytic')

data_dir = 'data/newdata'

files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

for file in files:
    universe = takahe.universe.create('real')

    universe.populate(f"{data_dir}/{file}", n_stars=n_stars)

    z = universe.get_metallicity()

    universe.set_nbins(51)
    universe.plot_event_rate()
    plt.tight_layout()
    plt.savefig(f"output/{z}.png")

    universe.plot_event_rate_BPASS()
    plt.tight_layout()
    plt.savefig(f"output/{z}_BPASS.png")
