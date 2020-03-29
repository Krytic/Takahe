import numpy as np
import matplotlib.pyplot as plt
import takahe

ensemble = takahe.load.from_file('data/newdata/Remnant-Birth-bin-imf135_300-z040_StandardJJ.dat', name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'], n_stars=10000)

ensemble.compute_event_rate_plot()
plt.yscale('log')
plt.xlabel("time [Gyr]")
plt.ylabel(r"Mergers [events / M$_\odot$ / GYr]")
plt.show()
