import cycler
import time
import multiprocessing as mp

from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import takahe
import pandas as pd


from kea.hist import histogram
from scipy.special import gammaincc

n_stars = 'all'
#plt.rcParams['figure.figsize'] = (40, 40)
data_dir = 'data/newdata'
files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

color = plt.cm.tab20(np.linspace(0.1,0.9,len(files)))

plt.style.use('krytic')
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

file_array = dict()

for i in range(len(files)):
    file = files[i]

    z = takahe.helpers.extract_metallicity(file)

    file_array[z] = file

universe = takahe.universe.create("LCDM")
z = np.linspace(0, universe.tH, universe.get_nbins())
event_rates = np.array([0 for _ in range(len(z))])

all_metallicities = list(file_array.keys())
all_metallicities.sort()

start = time.time()

x = np.linspace(0, universe.tH, universe.get_nbins()+1)
SFRD_so_far = histogram(edges=x)

rows_for_results = []

n = len(all_metallicities)

for i in range(n):
    this_start = time.time()

    Z_frac = all_metallicities[i]
    Z_frac_prev = all_metallicities[i-1] if i > 0 else 0

    avg = np.mean([Z_frac, Z_frac_prev])

    if i < n-1:
        Z_frac_next = all_metallicities[i+1]
    else:
        # take the upper bound to be as far away from Z_frac as Z_frac_prev
        # is
        Z_frac_next = all_metallicities[i] + Z_frac - avg

    Z_compute = avg#[Z_frac_prev, Z_frac_next] # if i < n else 1-np.sum(all_metallicities[:-1])

    file = file_array[Z_frac]

    universe.populate(f"{data_dir}/{file}", n_stars=n_stars, load_type='random')

    # universe.set_nbins(100)

    dtd, sfrd, events, sfh = universe.event_rate(Z_compute, SFRD_so_far)

    z = universe.get_metallicity()

    events_list = events.getValues()

    event_rates = event_rates + events_list

    this_today = np.round(np.log10(events_list[0]), 2)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0 , 0])
    ax2 = plt.subplot(gs[1 , 0])
    ax3 = plt.subplot(gs[: , 1])

    plt.sca(ax1)
    dtd.plot()
    plt.ylabel(r"DTD [$M_\odot/yr/Mpc^3$")
    plt.xlabel("Lookback Time [Gyr]")
    plt.yscale('log')
    ax1.xaxis.tick_top()

    plt.sca(ax2)
    events.plot()
    plt.ylabel(r"Event Rate [#/yr/Gpc$^3$]")
    plt.yscale('log')

    plt.sca(ax3)
    sfrd.plot()
    plt.ylabel(r"SFRD [$M_\odot/yr/Mpc^3$]")
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')
    plt.yscale('log')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(rf'$Z={takahe.helpers.format_metallicity(z)}Z_\odot\quad\log\mathcal{{R}}_0 = {this_today}$')

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid("off")
    plt.xlabel("Lookback time [Gyr]")

    plt.savefig(f'output/figures/linear_{z}.png', dpi=150, bbox_inches='tight')
    plt.close()

    SFRD_so_far._values += sfrd._values

    rows_for_results.append({'mfrac': takahe.helpers.format_metallicity(z), 'R': this_today, 'n': universe.populace.size()})

    this_end = time.time()
    print(f"Completed z={z} in {this_end-this_start} seconds")

# plt.xlabel("Lookback Time [Gyr]")
# plt.ylabel(r"Stellar Formation Rate [$M_\odot$ / yr / Mpc$^3$]")
# # plt.yscale("log")
# plt.ylim(0, 0.04)
# plt.title("Metallicity-Dependent SFRD")
# ax = plt.gca()
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

result_table = pd.DataFrame(rows_for_results, columns=rows_for_results[0].keys())

result_table.to_latex(index=False, buf="output/Merger_Result_Table_MMS.tex")

event_rate_histogram = histogram(edges=x)

event_rate_histogram.Fill(x[:-1], w=event_rates)

today = np.round(np.log10(event_rates[0]), 2)
end = time.time()
print(f"Completed in {end-start} seconds.")
plt.figure()
plt.xlabel("Lookback Time [Gyr]")
plt.ylabel(r"Event rate [events / yr / Gpc$^{-3}$]")
plt.title(rf"Mixed Metallicity Event Rates$\quad n={n_stars}\quad\log\mathcal{{R}}={today}$")
plt.yscale('log')

event_rate_histogram.plot()
plt.show()

