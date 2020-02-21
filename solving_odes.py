import numpy as np
from scipy.constants import c, G
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Sample Parameters for beta centauri
M1 = 10.7 * 2e30
M2 = 10.3 * 2e30
a0 = 454368025.65
e0 = 0.825

# Specify the age of the universe as an integration span
t_span = (0, 13.8e9)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Constant term for use in the ODEs
beta = 64/5 * (G**3*M1*M2*(M1+M2))/(c**5)

# Equation (3) from https://arxiv.org/pdf/1905.06086.pdf
def dadt(t, e, a):
	"""
	Auxiliary function to compute Equation 3.

	Params:
		t [ndarray] A vector of times.
		e [float] The current eccentricity
		a [float] The current semimajor axis

	Output:
		The quantity da/dt - how the semimajor axis is changing with time.
	"""

	first_term = -beta / (a**3 * (1-e**2)**(7/2))
	second_term = 1 + 73/24 * e**2 + 37 / 96 * e ** 4

	return first_term * second_term

# Equation (4) from ibid
def dedt(t, e, a):
	"""
	Auxiliary function to compute Equation 4.

	Params:
		t [ndarray] A vector of times.
		e [float] The current eccentricity
		a [float] The current semimajor axis

	Output:
		The quantity de/dt - how the eccentricity is changing with time.
	"""
	first_term = -19/12 * beta / (a**4*(1-e**2)**(5/2))
	second_term = (e + 121/304 * e ** 3)

	return first_term * second_term

def coupled_eqs(t, params):
	"""
	Primary workhorse function. Computes the vector [da/dt, de/dt] for use in
	solve_ivp.

	Params:
		t [ndarray] A vector of times
		params [list] A list or 2-tuple of arguments. Must take the form [a, e]

	Output:
		A list containing da/dt and de/dt
	"""
	a = params[0]
	e = params[1]
	return [dadt(t, e, a), dedt(t, e, a)]

# Numerically solve the coupled ODEs
res = solve_ivp(coupled_eqs, t_span, [a0, e0], t_eval=t_eval)

a = res.y[0]
e = res.y[1]

# Plotting the SMA
plt.subplot(121)
plt.plot(res.t, a)
plt.xlabel("Time [yrs]")
plt.ylabel("Semimajor Axis [km]")

# Plotting the Eccentricity
plt.subplot(122)
plt.plot(res.t, e)
plt.xlabel("Time [yrs]")
plt.ylabel("Eccentricity [dimensionless]")

# Boilerplate
plt.suptitle(r"$\beta$ Centauri")
plt.show()