import numpy as np
import linecache
from takahe import BinaryStarSystemEnsemble, BinaryStarSystem
import matplotlib.pyplot as plt
import pickle
import matplotlib.transforms as transforms
from os import path

G = 6.67e-11

k_simulations = 1
n_stars = 1

Solar_Radii = 695500

total_iterations = k_simulations * n_stars

# if path.exists('examples/computermodernstyle.mplstyle'):
#     plt.style.use('examples/computermodernstyle.mplstyle')

merge_rate_array = []
ct = []

regenerate = True
cnt = 0

if regenerate:
    for k in range(k_simulations):
        ensemble = BinaryStarSystemEnsemble.BinaryStarSystemEnsemble()
        lines = np.random.randint(1, 263454, n_stars)

        for line in lines:
            star = list(map(float, linecache.getline('data/Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat', line).split()))

            binary_star = BinaryStarSystem.BinaryStarSystem(*star)

            t, a, e = binary_star.evolve_until_merger()

            first = G * (binary_star.m1 +binary_star.m2)/(4*np.pi**2)

            T = ((a / Solar_Radii) ** 3 / first)**(1/2)

            plt.plot(e, T)
            plt.xlabel("eccentricity")
            plt.ylabel("period")
            plt.show()

            raise ValueError()

            ensemble.add(binary_star)

            cnt += 1

            if cnt % 100 == 0:
                print(f"{cnt/total_iterations*100:.2f}% complete", end="\r")

        merge_rate_array.append(ensemble.merge_rate(13.8))
        ct.append(ensemble.average_coalescence_time())

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
