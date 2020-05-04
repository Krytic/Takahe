import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import takahe
from kea.hist import histogram

"""
1. Begin at t=0
2. Seed universe with some initial stars.
3. For every time t=t' (0 < t' < 13.8), add suffcient stars such that
    dN/dt = SFRD * N and the weight is respected.
    3b. If dN/dt = SFRD * N, then N(t) = Ae^(SFRD*t) where A is # of
    stars at t=0.
4. Compute the current merge rate.
5. Plot, and profit?

eqn(30) in Hogg

300 events / GPc / yr

Existence time (?) distribution?
When it forms -- "how many systems have not merged yet"

Mass formed in each time bin

"""

n_stars = 1000

plt.style.use('krytic')

universe = takahe.universe.create('real')
universe.populate('data/newdata/Remnant-Birth-bin-imf135_300-z002_StandardJJ.dat',
    n_stars=n_stars,
    name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'],
    load_type='random')

universe.plot_event_rate()

plt.show()
