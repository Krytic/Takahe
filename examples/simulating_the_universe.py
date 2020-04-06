import numpy as np
import matplotlib.pyplot as plt
import takahe

uni = takahe.universe.create('eds')

z = np.linspace(0, 10, 1000)
SFRD = []
d = []
for zi in z:
    SFRD.append(uni.stellar_formation_rate(z=zi))
    d.append(uni.compute_comoving_distance(zi))

print(rf"SFRD peaks at $\psi(z)={max(SFRD):.2f} M_\odot/yr/Mpc^3$, corresponding to z={z[np.argmax(SFRD)]:.2f}.")

plt.plot(d, SFRD)
plt.xlabel("E(z) (")
plt.ylabel(r"$\psi(z)$")
plt.show()
