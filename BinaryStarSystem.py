import numpy as np
from scipy.constants import c, G
from scipy.integrate import solve_ivp
from hoki import load

def BinaryStarSystemLoader(bpass_from, mass_fraction=None):
	"""Loads a binary star system from the BPASS dataset.
	
	Opens the BPASS file you wish to use, uses hoki to load it into a
	dataframe, and returns a list of BSS objects.
	
	Arguments:
		bpass_from {str} -- the filename of the BPASS file to use
	
	Keyword Arguments:
		mass_fraction {int} -- If not None, assumes that M1 = mass_fraction * M2
		and uses this to compute the masses. (default: {None})
	
	Returns:
		[mixed] -- A list of BinaryStarSystem objects (if BPASS data is used).
				   A singular BinaryStarSystem object (if BPASS data is NOT used).
	"""
	if mass_fraction is None:
		# Temporary hack, will be supplanted by something better in a future commit	
		# Also this is treating beta centauri as a binary system not a triple binary
		M1 = 10.7 * 2e30 # Kilograms
		M2 = 10.3 * 2e30 # Kilograms
		a0 = 454368025.65 # Kilometers
		e0 = 0.825 # 0 <= e < 1

		return BinaryStarSystem(M1, M2, a0, e0)

	else:
		# Todo: Following code returns a flatline on BPASS data. Why?
		data = load._stellar_masses(bpass_from)

		if mass_fraction is None:
			raise ValueError

		star_systems = []

		for mass in data['stellar_mass']:
			M2 = mass / (2 * mass_fraction)
			M1 = mass_fraction * M2

			a0 = np.random.uniform(454368024.65, 454368026.65)
			e0 = np.random.uniform(0, 1)

			BSS = BinaryStarSystem(M1, M2, a0, e0)

			star_systems.append(BSS)

		return star_systems

class BinaryStarSystem:
	def __init__(self, primary_mass, secondary_mass, a0, e0):
		self.m1 = primary_mass
		self.m2 = secondary_mass
		self.a0 = a0
		self.e0 = e0
		self.beta = 64/5 * (G**3*self.m1*self.m2*(self.m1+self.m2))/(c**5)

	def evolve(self, evolve_over):
		"""Evolve the binary star system in time
		
		Uses a Runge-Kutta algorithm to evolve the binary star system over
		a specified range evolve_over.
		
		Arguments:
			evolve_over {ndarray} -- An ndarray specifying the time that
			should be integrated over. Each data point in evolve_over
			corresponds to a datapoint in the output.
		
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

			return -self.beta / (a**3 * (1-e**2)**(7/2)) * (1 + 73/24 * e**2 + 37 / 96 * e ** 4)

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
			return -19/12 * self.beta / (a**4*(1-e**2)**(5/2)) * (e + 121/304 * e ** 3)

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

		res = solve_ivp(coupled_eqs, (evolve_over[0], evolve_over[-1]), [self.a0, self.e0], t_eval=evolve_over)

		return res.t, res.y[0], res.y[1]