import numpy as np
import matplotlib.pyplot as plt

from takahe import BinaryStarSystemLoader as load

import random

binning = False

limit = 1

ensemble = load.from_file('data/Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat', limit=limit)

i = 0

cts = np.empty((1, limit))[0]
wts = np.empty((1, limit))[0]

for star in ensemble:
    t, a_array, e_array = star.evolve_until_merger()

    if binning:
        a_array = np.round(10*np.log10(a_array))
        ls = None
        marker = "."

        dp_list = np.array([])

        for j in range(int(min(a_array)), int(max(a_array))+1):
            dp_list = np.append(dp_list, np.sum(a_array == j))

        time_step = t[1] - t[0]

        estimate = np.sum(time_step * np.array(dp_list))
        calc = star.coalescence_time()

        cts[i] = estimate
        wts[i] = star.weight

    print(star.coalescence_time())

    plt.plot(a_array, e_array)

    i += 1

# plt.plot(cts, wts, 'r.')

# plt.xlabel("coalescence_time [ga]")
# plt.ylabel("weight")

plt.xlabel("SMA (solar radii)")
plt.ylabel("Eccentricity")

plt.grid('on')
plt.tight_layout()
plt.show()
