import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

from BinaryStarSystem import BinaryStarSystemLoader

Solar_Mass = 1.989e30

cfg = {
	'M1' : 1.33 * Solar_Mass, # Kilogram
	'M2' : 1.35 * Solar_Mass, # Kilograms
	'a0' : 3.28 * 696340, # Kilometers
	'e0' : 0.274 # 0 <= e < 1
}

BSS = BinaryStarSystemLoader(data=cfg)

# Specify the age of the universe as an integration span
t_span = (12.8e9, 13.8e9)
t_eval = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0]) / 1000)

plt.figure()
t, a, e = BSS.evolve(t_eval)

print(BSS.coalesce())

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
plt.suptitle("B1534+12")
plt.show()
