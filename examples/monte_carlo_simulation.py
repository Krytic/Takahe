import numpy as np
import linecache
import takahe
from takahe.constants import *
import matplotlib.pyplot as plt
import pickle
from os import path

k_simulations = 200
n_stars = 1000

# if path.exists('examples/computermodernstyle.mplstyle'):
#     plt.style.use('examples/computermodernstyle.mplstyle')

merge_rate_array = []
ct = []

regenerate = True

cnt = 0

if regenerate:
    for k in range(k_simulations):
        ensemble = takahe.load.random_from_file('data/Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat', n_stars=n_stars)

        merge_rate_array.append(ensemble.merge_rate(13.8))
        ct.append(ensemble.average_coalescence_time())

        cnt += 1

        print(f"Completed simulation {cnt} of {k_simulations}", end="\r")

    save_data = [merge_rate_array, ct]

    pickle.dump(save_data, open("monte_carlo.pickle", 'wb'))
else:
    saved_data = pickle.load(open('monte_carlo.pickle', 'rb'))

    merge_rate_array, ct = saved_data

mean_merge_rate = np.cumsum(merge_rate_array) / (np.array(range(k_simulations))+1)

print("mean merge rate:", mean_merge_rate[-1])

plt.suptitle(f"Monte Carlo Simulation of Binary Star Systems (Ensemble size: {n_stars}, Simulations: {k_simulations})")

plt.subplot(221)
plt.title(r"Merge rate (\%)")
plt.plot(range(k_simulations), np.array(merge_rate_array) * 100)
ax = plt.gca()
plt.axhline(mean_merge_rate[-1] * 100, linestyle='dashed', color='black')
plt.text(k_simulations / 2, mean_merge_rate[-1] * 100+0.1, f"{mean_merge_rate[-1]*100:.2f}%", horizontalalignment='center')

plt.subplot(222)
plt.plot(range(k_simulations), ct)
plt.title(r"Average $\tau_c$ [ga]")
ax = plt.gca()
ax.yaxis.tick_right()

plt.subplot(212)
plt.title(r"Cumulative Mean of Merge Rate (\%)")
plt.plot(range(k_simulations), mean_merge_rate * 100)

plt.subplots_adjust(top=0.87, wspace=0, hspace=0.4)
plt.show()
