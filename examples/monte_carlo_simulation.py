import numpy as np
import linecache
from takahe import BinaryStarSystemEnsemble, BinaryStarSystem
import matplotlib.pyplot as plt
import pickle
import matplotlib.transforms as transforms

k_simulations = 200
n_stars = 1000

total_iterations = k_simulations * n_stars

merge_rate_array = []
ct = []

regenerate = False

cnt = 0

if regenerate:
    for k in range(k_simulations):
        ensemble = BinaryStarSystemEnsemble.BinaryStarSystemEnsemble()
        lines = np.random.randint(1, 263454, n_stars)

        for line in lines:
            star = list(map(float, linecache.getline('data/Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat', line).split()))

            binary_star = BinaryStarSystem.BinaryStarSystem(*star)

            ensemble.add(binary_star)

            cnt += 1
            if cnt % 100 == 0:
                print(f"{cnt/total_iterations*100:.2f}% complete", end="\r")

        merge_rate_array.append(ensemble.merge_rate(13.8))
        ct.append(ensemble.average_coalescence_time())

    save_data = [merge_rate_array, ct]

    pickle.dump(save_data, open("monte_carlo.p", 'wb'))
else:
    saved_data = pickle.load(open('monte_carlo.p', 'rb'))

    merge_rate_array, ct = saved_data

mean_merge_rate = np.cumsum(merge_rate_array) / (np.array(range(k_simulations))+1)

print("mean merge rate:", mean_merge_rate[-1])

plt.suptitle(f"Monte Carlo Simulation of Binary Star Systems\n Ensemble size: {n_stars}, Simulations: {k_simulations}")

plt.subplot(221)
plt.ylabel("Merge rate (%)")
plt.plot(range(k_simulations), np.array(merge_rate_array) * 100)
ax = plt.gca()
trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
plt.axhline(mean_merge_rate[-1] * 100, linestyle='dashed', color='black')
plt.text(k_simulations / 2, mean_merge_rate[-1] * 100+0.1, f"{mean_merge_rate[-1]*100:.2f}%", horizontalalignment='center')

plt.subplot(222)
plt.plot(range(k_simulations), ct)
plt.ylabel(r"Average $\tau_c$ [ga]")

plt.subplot(212)
plt.ylabel("Cumulative Mean of Merge Rate (%)")
plt.plot(range(k_simulations), mean_merge_rate * 100)
plt.show()
