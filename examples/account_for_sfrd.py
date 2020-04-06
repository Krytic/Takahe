import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import takahe

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

n_stars = 4000

universe = takahe.universe.create('real')

def integrand(z):
    def E(z):
        return np.sqrt(universe.omega_m * (1+z)**3
                     + universe.omega_k * (1+z)**2
                     + universe.omega_lambda)
    return 1 / ((1+z) * E(z))

z = np.linspace(0, 20)

t_arr = []

for zi in z:
    result, err = quad(integrand, 0, zi)

    tL = universe.tH * result / 31557600000000000

    t_arr.append(tL)

# a star at distance z will be tL Gyr old
# ENDS SFRD CODE

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

