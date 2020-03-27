import numpy as np
import matplotlib.pyplot as plt
import takahe

ensemble = takahe.load.from_file('data/newdata/Remnant-Birth-bin-imf135_300-z040_StandardJJ.dat', name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'], n_stars=10000)

x, y = ensemble.compute_event_rate_plot()

plt.step(x, y)
plt.yscale('log')
plt.xlabel("log(age/yrs)")
plt.ylabel(r"Event rate / events/M$_\odot$/yr")
plt.show()
