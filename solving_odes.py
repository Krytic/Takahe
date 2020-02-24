import numpy as np
import matplotlib.pyplot as plt

from BinaryStarSystem import BinaryStarSystemLoader

BSS = BinaryStarSystemLoader("data/starmass-bin-imf_chab100.z001.dat")

# Specify the age of the universe as an integration span
t_span = (0, 13.8e9)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

plt.figure()
t, a, e = BSS.evolve(t_eval)

# Plotting the SMA
plt.subplot(121)
plt.plot(t, a)
plt.xlabel("Time [yrs]")
plt.ylabel("Semimajor Axis [km]")

# Plotting the Eccentricity
plt.subplot(122)
plt.plot(t, e)
plt.xlabel("Time [yrs]")
plt.ylabel("Eccentricity [dimensionless]")

# Boilerplate
plt.suptitle(r"$\beta$ Centauri")
plt.show()