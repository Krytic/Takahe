import numpy as np
import matplotlib.pyplot as plt

from takahe import BinaryStarSystemLoader as load

import random

binning = True

ensemble = load.from_file('data/Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat', limit=1)

i = 0

for star in ensemble:
    t, a_array, e_array = star.evolve_until_merger()

    col = random.randint(0, 0xFFFFFF)

    if binning:
        a_array = np.round(10*np.log10(a_array))
        ls = None
        marker = "."

        dp_list = []

        for j in range(int(min(a_array)), int(max(a_array))+1):
            dp_list.append(np.sum(a_array == j))

        time_step = t[1] - t[0]

        print("coalescence time estimate =", np.sum(time_step * np.array(dp_list)))
        print("coalescence time calculated =", star.coalescence_time())

        plt.subplot(211)
        plt.plot(dp_list, color="#%06x" % col)
        plt.xlabel("Bin")
        plt.ylabel("Number of data points in bin")

        print("Sum:", np.sum(dp_list), 'len:', len(a_array))

    if binning:
        plt.subplot(212)

    plt.scatter(a_array,
             e_array,
             color="#%06x" % col
            )

    i += 1

axlabel = r"a ($R_\odot$)" if not binning else r"log(a) (binned) ($R_\odot$)"
plt.xlabel(axlabel)
plt.ylabel("eccentricity")

plt.grid('on')
plt.tight_layout()
plt.show()
