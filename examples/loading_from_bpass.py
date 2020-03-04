import numpy as np
import matplotlib.pyplot as plt

from takahe import BinaryStarSystemLoader as load

mass_fraction = np.linspace(0.001, 0.5, 100)
merge_rate = np.array([])

for fraction in mass_fraction:
    ensemble = load.from_bpass('data/starmass-bin-imf_chab100.z001.dat', fraction, a0_range=(0,100))

    this_merge_rate = ensemble.merge_rate(ensemble.average_coalescence_time())

    merge_rate = np.append(merge_rate, this_merge_rate)

p = np.polyfit(mass_fraction, list(merge_rate), 1)

yhat = np.poly1d(p)

plt.plot(mass_fraction, merge_rate)
plt.plot(mass_fraction, yhat(mass_fraction), 'k--')
plt.xlabel("Mass Fraction")
plt.ylabel("Merge Rate")
plt.show()
