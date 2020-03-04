import numpy as np
import matplotlib.pyplot as plt

from takahe import BinaryStarSystemLoader as load

cfg = {
	'M1' : 1.33, # Solar masses
	'M2' : 1.35, # Solar masses
	'a0' : 3.28, # Solar radii
	'e0' : 0.274 # 0 <= e < 1
}

BSS = load.from_data(data=cfg)

# Specify  an integration span
t_span = (0, BSS.coalescence_time() * 1e9 * 60 * 60 * 24 * 365.25)

plt.figure()
t, a, e = BSS.evolve_until(t_span)

print(t,a,e)

print("coalescence_time: ", BSS.coalescence_time())

plt.rcParams['axes.formatter.useoffset'] = False

# Plotting the SMA
plt.subplot(121)
plt.plot(t, a)
plt.xlabel("Time [ga]")
plt.ylabel("Semimajor Axis [solar radii]")
plt.axvline(BSS.coalescence_time(), linestyle='dashed', color='black')
# plt.ylim(0, a[0])

# Plotting the Eccentricity
plt.subplot(122)
plt.plot(t, e)
plt.xlabel("Time [ga]")
plt.ylabel("Eccentricity [dimensionless]")
# plt.ylim(0, e[0])
plt.axvline(BSS.coalescence_time(), linestyle='dashed', color='black')

# Boilerplate
plt.suptitle("Star: B1534+12")
plt.show()
