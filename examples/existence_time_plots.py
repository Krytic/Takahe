import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import takahe

universe = takahe.universe.create('real')

n_stars = 4000

universe.populate('data/newdata/Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat',
    n_stars=n_stars,
    name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'])

plt.subplot(212)
cts = universe.populace.get_cts()
plt.hist(cts, bins=[x for x in range(0, int(13.8e9), int(3e7))])
plt.xlabel("Coalescence Time")
plt.ylabel("Frequency")

plt.subplot(221)
plt.title("Existence time plot")
universe.populace.compute_existence_time_distribution()
plt.ylabel(r"Events [# / $M_\odot$ / GYr]")
plt.yscale('log')

plt.subplot(222)
plt.title("Delay time plot")
universe.populace.compute_delay_time_distribution()
plt.yscale("log")

plt.suptitle(rf"$Z=Z_\odot, n\approx {universe.populace.size()}$")
plt.subplots_adjust(wspace=0)

plt.show()
