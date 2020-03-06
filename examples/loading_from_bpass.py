import numpy as np
import matplotlib.pyplot as plt

from takahe import BinaryStarSystemLoader as load

import random

# colors = 'bgrcmyk'

ensemble = load.from_file('data/Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat', limit=17)

i = 0
for star in ensemble:
    t, a_array, e_array = star.evolve_until_merger()

    # a_array = np.round(10*np.log10(a_array))

    plt.plot(a_array, e_array, color="#%06x" % random.randint(0, 0xFFFFFF))

    i += 1

plt.xlabel(r"a ($R_\odot$)")#log(a) (binned) ($R_\odot$)")
plt.ylabel("eccentricity")

plt.grid('on')
plt.show()
