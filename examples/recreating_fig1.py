"""

This code recreates Figure 1 of https://arxiv.org/pdf/1905.06086.pdf

This code (so far) only recreates the solid blue curve.

# todo: recreate the other curves.

"""

import numpy as np
from scipy.constants import c, G
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Sample Parameters for beta centauri
M1 = 10.7 * 2e30
M2 = 10.3 * 2e30
a0 = 454368025.65
e0 = 0.825

# cfg = {
# 	'M1' : 1.33 * Solar_Mass, # Kilograms
# 	'M2' : 1.35 * Solar_Mass, # Kilograms
# 	'a0' : 3.28 * 696340, # Kilometers
# 	'e0' : 0.274 # 0 <= e < 1
# }

# Specify the age of the universe as an integration span
t_span = (0, 13.8e9)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Constant term for use in the ODEs
beta = 64/5 * (G**3*M1*M2*(M1+M2))/(c**5)

# Recreating Fig 1 of https://arxiv.org/pdf/1905.06086.pdf
e = np.linspace(0, 0.995, 1000)
T = []

for e0 in e:
	# At every iteration of the eccentricity, compute T_e / T_c
	# T_c is the coalescence time for a circular orbit with the same SMA
	T_e = a0**4 / (4*beta) * (1-e0**2)**(7/2) / ((1-e0**(7.4))**(1/5)*(1+121/304 * e0**2))
	T_c = a0**4 / (4*beta)

	T_ratio = T_e / T_c
	T.append(T_ratio)

# Regular plotting stuff
plt.semilogy(e, T)
plt.ylim(10**-3, 10**0)
plt.xlabel(r"$e_0$")
plt.ylabel(r"$T_e(e_0, a_0)/T_c(a_0)$")
plt.show()