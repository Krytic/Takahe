from os import listdir
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import takahe

n = 1000

files = listdir('data/newdata')
files = [f'data/newdata/{file}' for file in files]

i = 0
nfiles = len(files)

# nrows = int(np.floor(np.sqrt(nfiles)))
# ncols = int(np.ceil(np.sqrt(nfiles)))

# fig, axes = plt.subplots(nrows, ncols)
# axes = axes.flatten()

# if len(axes) > nfiles:
#     fig.delaxes(axes[-1])

def format_z(z):
    if z[:2] == "em":
        exp = z[-1]
        fmt = rf"$1\times 10^{{-{exp}}}$"
    else:
        fmt = f"0.{z}"
    return fmt

for file in files:
    # file = f'data/newdata/Remnant-Birth-bin-imf135_300-z{z}_StandardJJ.dat'

    parts = file.split('/')[-1].split("-")
    z = None
    for part in parts:
        if part[0] == 'z':
            z = part.split("_")[0][1:]
    if z == None:
        print("Unable to compute z for this file, aborting!")
        print(file)
        break

    # ax = axes[i]
    # plt.sca(ax)

    universe = takahe.universe.create('eds')
    universe.populate(file, name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'], n_stars=n)

    universe.populace.compute_delay_time_distribution(label=f"z=0.{z}")

    # plt.text(0.98, 0.98, rf'z={format_z(z)}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    # plt.yscale('log')
    # plt.xlabel(" ")

    i += 1

    print(f'Completed z={z}. ({i}/{nfiles})')

plt.suptitle(f'delay time distribution for different $z$ values')
# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.yscale("log")
plt.ylabel(r"Mergers [# of events / $M_\odot$ / Gyr]")
plt.xlabel("log(age/yrs)")

plt.show()
