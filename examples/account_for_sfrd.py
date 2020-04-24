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

n_stars = 4000

universe = takahe.universe.create('real')
universe.populate('data/newdata/Remnant-Birth-bin-imf135_300-z014_StandardJJ.dat',
    n_stars=n_stars,
    name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'])

print(universe.populace.get('mass'))

universe.plot_event_rate()

# def integrand(z):
#     def E(z):
#         return np.sqrt(universe.omega_m * (1+z)**3
#                      + universe.omega_k * (1+z)**2
#                      + universe.omega_lambda)
#     return 1 / ((1+z) * E(z))

# znum = 100
# z = np.linspace(0, 5, znum)

# t_arr = []
# SFR_arr = []
# mr_arr = []

# i = 0

# hist = histogram(xlow=0, xup=14, nr_bins=50)
# edges = hist.getBinEdges()
# old_mr = 0

# for zi in z:
#     result, err = quad(integrand, 0, zi)

#     tL = universe.tH * result / 31557600000000000

#     t_arr.append(tL)

#     V_C = universe.comoving_volume(z=zi)

#     SFRD = universe.stellar_formation_rate(z=zi) / 1e9

#     SFR = V_C * SFRD

#     SFR_arr.append(SFRD)

#     mr = universe.populace.merge_rate(tL)

#     mr_arr.append(mr)

#     # TODO: make binary search (speed)
#     this_mr = mr - old_mr
#     for j in range(len(edges)):
#         if edges[j] < tL and tL < edges[j+1]:
#             # edges[j] or edges[j+1]?
#             hist.Fill(edges[j], this_mr)

#     old_mr += this_mr
#     i += 1

#     print(f"{i/znum*100:.2f}%", end="\r")

# bin_widths = [hist.getBinWidth(i) for i in range(0,hist.getNBins())]
# hist = hist / 1e6 / bin_widths
# # hist.plot()

# # # plt.yscale('log')
# # plt.ylabel(r'Merge Rate [events / $M_\odot$ / Gyr]')
# # plt.xlabel("Lookback time")
# # plt.show()

# plt.plot(t_arr, np.log10(SFR_arr))
# plt.xlabel("Lookback time (Gyr) (Inverted axis)")
# plt.xlim(t_arr[-1], t_arr[0])
# plt.ylabel(r"$\log(\psi(z)/Gyr)$")
# plt.show()

# # a star at distance z will be tL Gyr old
