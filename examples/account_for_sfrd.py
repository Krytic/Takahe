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

# takahe.load.random_from_file("data/newdata/Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat", 100,
#     name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'])

n_stars = 4000

universe = takahe.universe.create('real')

def integrand(z):
    def E(z):
        return np.sqrt(universe.omega_m * (1+z)**3
                     + universe.omega_k * (1+z)**2
                     + universe.omega_lambda)
    return 1 / ((1+z) * E(z))

z = np.linspace(0, 5, n_stars)

t_arr = []
SFR_arr = []

for zi in z:
    result, err = quad(integrand, 0, zi)

    tL = universe.tH * result / 31557600000000000

    t_arr.append(tL)

    V_C = universe.comoving_volume(z=zi)

    SFRD = universe.stellar_formation_rate(z=zi) / 1e9

    SFR = V_C * SFRD

    SFR_arr.append(SFR)

# a star at distance z will be tL Gyr old

plt.plot(t_arr, SFR_arr)
plt.xlabel("Lookback time [Gyr]")
plt.ylabel(r"Stellar Formation Rate [$M_\odot$ / Gyr]")
plt.title(r"SFR against $t_L$ for $0 < z < 5$")
plt.yscale('log')
plt.show()
