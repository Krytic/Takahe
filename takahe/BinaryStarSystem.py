import numpy as np
from scipy.constants import c, G
from scipy.integrate import solve_ivp
from hoki import load

Solar_Mass = 1.989e30

class BinaryStarSystem:
	"""Represents a binary star system."""

	def __init__(self, primary_mass, secondary_mass, a0, e0):
		self.m1 = primary_mass * Solar_Mass # Units: kg
		self.m2 = secondary_mass * Solar_Mass # Units: kg
		self.a0 = a0 * 696340 # Units: km
		self.e0 = e0 # Units: dimensionless
		self.beta = (64/5) * ((G / (1000**3))**3*self.m1*self.m2*(self.m1+self.m2))/((c/1000)**5) # units: km^4 s^-1

	def coalescence_time(self):
		"""Computes the coalescence time for the BSS in gigayears.
		
		Uses Eqn 10 from Nyadzani and Razzaque (https://arxiv.org/pdf/1905.06086.pdf) to compute the coalescence
		time for the binary star system. Although this is strictly speaking an approximation, it converges on the
		result gained by numerically by Peters in 1964.
		
		Returns:
			float -- the coalescence time for the binary star (units: ga).
		"""

		return (self.a0**4 / (4*self.beta) * (1-self.e0**2)**(7/2) / ((1-self.e0**(7/4))**(1/5)*(1+121/304 * self.e0**2))) / (31557600000000000) # Units: ga

	def evolve(self, t_span):
		"""Evolve the binary star system in time
		
		Uses a Runge-Kutta algorithm to evolve the binary star system over
		a specified range evolve_over.
		
		Arguments:
			t_span {tuple} -- A 2-tuple that corresponds to the start and end points of integration.
		
		Returns:
			mixed -- A 3-tuple containing the resultant time array,
			and the resultant SMA and eccentricity arrays.
		"""

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
			
			da = (-self.beta / (a**3 * (1-e**2)**(7/2))) * (1 + 73/24 * e**2 + 37 / 96 * e ** 4) # Units: km/s
			
			return da

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

			de = (-19/12 * self.beta / (a**4*(1-e**2)**(5/2))) * (e + 121/304 * e ** 3) # Units: s^-1
 
			return de

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

		evolve_over = np.linspace(t_span[0], t_span[1], 10000)

		res = solve_ivp(coupled_eqs, t_span, [self.a0, self.e0], t_eval=evolve_over)

		print(res)

		return res.t / 1e9, np.round(res.y[0] / 696340, 3), np.round(res.y[1], 3)