import numpy as np
import matplotlib.pyplot as plt

import takahe
import random

merge_rate_threshold = 13.8 #gigayears
stars_in_ensemble = 1

number_of_ensembles_to_simulate = list(range(5))
merge_rates = []

for k in number_of_ensembles_to_simulate:
    ensemble = takahe.load.from_file('data/Remnant-Birth-bin-imf135 \
                               _300-z020_StandardJJ.dat',
                               limit=stars_in_ensemble,
                               n=263454)

    i = 0
    mergers_in_ensemble = 0

    for star in ensemble:
        t, a_array, e_array = star.evolve_until_merger()

        a_array = np.round(10*np.log10(a_array))

        dp_list = np.array([])

        ## Compute the number of mergers in a given bin
        for j in range(int(min(a_array)), int(max(a_array))+1):
            dp_list = np.append(dp_list, np.sum(a_array == j))

        time_step = t[1] - t[0]

        estimate = np.sum(time_step * np.array(dp_list))

        if estimate <= merge_rate_threshold:
            # Star has merged
            mergers_in_ensemble += 1

        i += 1

    merge_rates.append(mergers_in_ensemble)

plt.plot(k_arr, merge_rates, 'r.')

plt.xlabel("coalescence_time [ga]")
plt.ylabel("merge rate")

plt.grid('on')
plt.tight_layout()
plt.show()
