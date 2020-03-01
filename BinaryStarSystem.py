import numpy as np
from scipy.constants import c, G
from scipy.integrate import solve_ivp
from hoki import load

Solar_Mass = 1.989e30

def BinaryStarSystemLoader(bpass_from="", mass_fraction=None, data=dict()):
	"""Loads a binary star system from the BPASS dataset.
	
	Opens the BPASS file you wish to use, uses hoki to load it into a
	dataframe, and returns a list of BSS objects.
	
	Arguments:
		bpass_from {str} -- the filename of the BPASS file to use
	
	Keyword Arguments:
		mass_fraction {int} -- If not None, assumes that M1 = mass_fraction * M2
		and uses this to compute the masses. (default: {None})

		data {dict} -- A dictionary of data for the BSS. Keywords used are M1 (primary mass),
		M2 (secondary mass), e0 (initial eccentricity), and a0 (initial SMA) or T (period).
		(default: empty)
	
	Returns:
		[mixed] -- A list of BinaryStarSystem objects (if BPASS data is used).
				   A singular BinaryStarSystem object (if BPASS data is NOT used).
	"""
	if mass_fraction is None:
		if 'T' in data.keys():
			data['a0'] = ((G * (data['M1'] + data['M2']))/(4*np.pi**2) * data['T'] ** 2) ** (1/3)

		return BinaryStarSystem(data['M1'], data['M2'], data['a0'], data['e0'])

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
		self.beta = (64/5) * (G**3*self.m1*self.m2*(self.m1+self.m2))/(c**5)

	def coalesce(self):
		"""Computes the coalescence time for the BSS.
		
		Uses Eqn 10 from Nyadzani and Razzaque (https://arxiv.org/pdf/1905.06086.pdf) to compute the coalescence
		time for the binary star system. Although this is strictly speaking an approximation, it converges on the
		result gained by numerically by Peters in 1964.
		
		Returns:
			float -- the coalescence time for the binary star.
		"""

		return self.a0**4 / (4*self.beta) * (1-self.e0**2)**(7/2) / ((1-self.e0**(7/4))**(1/5)*(1+121/304 * self.e0**2))

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
			
			da = (-self.beta / (a**3 * (1-e**2)**(7/2))) * (1 + 73/24 * e**2 + 37 / 96 * e ** 4)
			
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

			de = (-19/12 * self.beta / (a**4*(1-e**2)**(5/2))) * (e + 121/304 * e ** 3)
 
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

		res = solve_ivp(coupled_eqs, (evolve_over[0], evolve_over[-1]), [self.a0, self.e0], t_eval=evolve_over, atol=696.340)

		return res.t, res.y[0], res.y[1]