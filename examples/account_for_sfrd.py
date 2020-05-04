import numpy as np
import matplotlib.pyplot as plt
import takahe

n_stars = 1000

plt.style.use('krytic')

universe = takahe.universe.create('real')
universe.populate(f'data/newdata/Remnant-Birth-bin-imf135_300-z014_StandardJJ.dat',
    n_stars=n_stars,
    name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'],
    load_type='random')

universe.plot_event_rate()
plt.show()
