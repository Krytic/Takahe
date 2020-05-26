import time
import multiprocessing as mp
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import takahe

import faulthandler
faulthandler.enable()

n_stars = 1000
#plt.rcParams['figure.figsize'] = (40, 40)

plt.style.use('krytic')

def _express_z(z):
    if z[:2] == "em":
        div = 1*10**(-int(z[-1]))
    else:
        div = float("0." + z)
    res = div / 0.020

    return res

def _format_z(z):
    res = _express_z(z)
    return rf"{res}Z_\odot"

data_dir = 'data/newdata'

files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

def execute(file):
    start = time.time()

    universe = takahe.universe.create('real')

    universe.populate(f"{data_dir}/{file}", n_stars=n_stars)
    print(f"{file}: populated")

    z = universe.get_metallicity()

    size = universe.populace.size()

#    universe.set_nbins(51)
    print(f"z={z}: events")
    events = universe.event_rate()
    print(f"z={z}: events done")

    today = int(np.nan_to_num(events.getBinContent(0), posinf=0))
    print(f"z={z}: {today} today computed")

    today = np.log10(today) if today > 0 else 0
    print(f"z={z}: plotting")

    events.plot()
    plt.xlabel("Lookback Time [Gyr]")
    plt.ylabel(r"Events [# / yr / Gpc$^3$]")
    plt.title(rf"Z=${_format_z(z)}$ log(R(z=0))=${today:.2f}$")
    # plt.yscale("log")
    plt.savefig(f"output/figures/linear_{z}.png")

    end = time.time()
    print(f"Completed z={z} in {end-start} seconds. {n_stars} requested, {size} generated.")

    plt.show()

    return _express_z(z), today

execute(files[0])

# cpus_to_use = min(mp.cpu_count(), len(files))

# print(f"Running on {cpus_to_use} CPUs")

# pool = mp.Pool(cpus_to_use)

# results = [pool.apply_async(execute, args=(file,)) for file in files]

# results = [r.get() for r in results]

# pool.close()

# print("Table of results")

# results.sort(key=lambda tup: tup[0])

# output = r"""
# \begin{table}[h]
#     \centering
#     \begin{tabular}{@{}ll@{}}
#         \toprule
#         $Z$ & $R(z=0)$ \\
#         \midrule
# """

# for result in results:
#     output += fr"        ${result[0]}Z_\odot$ & {result[1]:.2f} \\"
#     output += "\n"

# output.rstrip("\n")
# output += r"""
#         \bottomrule
#     \end{tabular}
# \end{table}
# """

# print(output)
