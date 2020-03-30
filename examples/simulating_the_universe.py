import numpy as np
import matplotlib.pyplot as plt
import takahe

uni = takahe.universe.create('eds')

z = np.linspace(0, 10, 1000)
SFRD = []
for zi in z:
    SFRD.append(uni.stellar_formation_rate(z=zi))

plt.plot(z, SFRD)
plt.xlabel("redshift")
plt.ylabel(r"$\psi(z)$")
plt.show()
