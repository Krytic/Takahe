import numpy as np
from scipy.constants import c, G
from scipy.integrate import solve_ivp
from hoki import load

from takahe import BinaryStarSystem

Solar_Mass = 1.989e30

def from_data(data):
	"""Loads a binary star system from a dictionary of data.
	
	Arguments:
		data {dict} -- A dictionary containing the 4 elements necessary to solve the 2body problem:
					   - M1 (mass of primary star)
					   - M2 (mass of secondary star)
					   - e0 (current eccentricity)
					   - a0/T (current semimajor axis / Period)
					Peroid takes precedence over the SMA, so if one is provided, we use Kepler's law
					to compute the SMA.

	tw	Returns:
		BinaryStarSystem -- An ADT to represent the BSS.
	"""
	if 'T' in data.keys():
		data['a0'] = ((G * (data['M1'] + data['M2']))/(4*np.pi**2) * data['T'] ** 2) ** (1/3)

	return BinaryStarSystem.BinaryStarSystem(data['M1'], data['M2'], data['a0'], data['e0'])

def from_bpass(bpass_from, mass_fraction):
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

		BSS = BinaryStarSystem.BinaryStarSystem(M1, M2, a0, e0)

		star_systems.append(BSS)

	return star_systems