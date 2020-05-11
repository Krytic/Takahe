import numpy as np
import takahe
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from takahe.constants import *

n_stars = 1000

universe = takahe.universe.create('real')
universe.populate('data/newdata/Remnant-Birth-bin-imf135_300-z014_StandardJJ.dat',
    n_stars=n_stars,
    name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'])

print(universe.populace.size())

ax = universe.populace.track_through_phase_space(in_range=[5, 25])
plt.show()
