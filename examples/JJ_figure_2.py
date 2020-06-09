import time
from os import listdir
from os.path import isfile, join
import cycler

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import takahe
from kea.hist import histogram


n_stars = 2000
#plt.rcParams['figure.figsize'] = (40, 40)
data_dir = 'data/newdata'
files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

color = plt.cm.tab20(np.linspace(0.1,0.9,len(files)+1))

plt.style.use('krytic')
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

file_array = dict()

for i in range(len(files)):
    file = files[i]

    fname = file.split("/")[-1].split(".")[0].rsplit("_", 1)[0]
    parts = fname.split("-")
    z = None

    for part in parts:
        if part[0] == "z":
            # metallicity term in fname
            if "_" in part:
                part = part.split("_")[0]
            z = takahe.helpers.format_metallicity(part[1:])

    file_array[z] = file

universe = takahe.universe.create("LCDM")
z = np.linspace(0, universe.tH, universe.get_nbins())
event_rates = np.array([0 for _ in range(len(z))])

keys = list(file_array.keys())
keys.sort()

start = time.time()

x = np.linspace(0, universe.tH, universe.get_nbins()+1)
SFRD_so_far = histogram(edges=x)

rows_for_results = []

plt.figure()

n = len(keys)

for i in range(n):
    Z_frac = keys[i]
    Z_frac_prev = keys[i-1] if i > 0 else 0

    Z_compute = np.mean([Z_frac, Z_frac_prev])

    if i == n:
        Z_compute = 1 - np.sum(keys[:-1])

    file = file_array[Z_frac]

    universe.populate(f"{data_dir}/{file}", n_stars=n_stars, load_type='random')
    universe.set_nbins(100)
    dtd, sfrd, events = universe.event_rate(SFRD_so_far=SFRD_so_far, Z_compute=Z_compute)

    z = universe.get_metallicity()

    sfrd.plot(label=z)

    SFRD_so_far = sfrd

    print(f'completed z={z}')

SFRD_so_far.plot(label="Total")
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
plt.xlabel("Lookback time / Gyr")
plt.ylabel(r"SFRD by metallicity in $M_\odot$ / yr / Mpc$^3$")
plt.ylim(0, 0.04)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
