import numpy as np
import matplotlib.pyplot as plt
import takahe

universe = takahe.universe.create('eds')

file = 'data/newdata/Remnant-Birth-bin-imf135_300-z040_StandardJJ.dat'

filename = file.split("/")[-1].split('.')[0]
parameters = file.split("-")
for p in parameters:
    if p[0] == "z":
        metallicity = "0." + p.split("_")[0][1:]

universe.populate(file, name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'], n_stars=1000)

stars = universe.populace.size()

# SFRD = universe.stellar_formation_rate()
universe.populace.compute_event_rate_plot()

plt.title(rf"$z={metallicity}$, $n={stars}$")
plt.yscale('log')
# plt.xscale('log')
plt.xlabel("log(age of universe/Gyr)")
plt.ylabel(r"Mergers [# of events / $M_\odot$ / Gyr]")
plt.show()
