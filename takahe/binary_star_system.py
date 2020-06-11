import matplotlib.pyplot as plt
import numpy as np
import takahe.helpers

from takahe.constants import *
from mpl_toolkits.mplot3d import Axes3D

def create(primary_mass, secondary_mass, a0, e0, extra_terms=dict()):
    """Creates a given Binary Star System from provided data.

        Represents a binary star system in the Universe.

        All physical parameters are expected to be in Solar units. i.e.,
        - Mass in Solar Mass
        - a0 in Solar Radii

        Arguments:
            primary_mass {float} -- The mass of the primary star
            secondary_mass {float} -- The mass of the secondary star
            a0 {float} -- The semimajor axis (SMA) of the binary
            e0 {float} -- The eccentricity of the binary system.

        Keyword Arguments:
            extra_terms {dict} -- A dictionary of extra parameters for
                                  the system. Acceptable extra parameters
                                  are: weight, evolution_age,
                                  and rejuvenation_age
                                  (default: {empty dict})

        Raises:
            ValueError -- if e0 is not in the interval [0, 1]

        Returns:
            BinaryStarSystem
        """
    return BinaryStarSystem(primary_mass,
                            secondary_mass,
                            a0,
                            e0,
                            extra_terms)

class BinaryStarSystem:
    """Represents a binary star system."""

    def __init__(self,
                 primary_mass,
                 secondary_mass,
                 a0,
                 e0,
                 extra_terms=dict()):

        if e0 > 1 or e0 < 0:
            raise ValueError("Eccentricity must be between 0 and 1.")

        # Unit conversion into quasi-SI units.
        self.m1 = primary_mass * Solar_Mass # Units: kg
        self.m2 = secondary_mass * Solar_Mass # Units: kg
        self.a0 = np.float64(a0 * Solar_Radii * 1000) # Units: km
        self.e0 = np.float64(e0) # Units: dimensionless

        extra_keys = ['weight', 'evolution_age',
                      'rejuvenation_age', 'coalescence_time']

        # only permit terms that are in the extra_keys list to appear
        # in self.extra_terms
        # @TODO: this seems arbitrary and restrictive.
        self.extra_terms = {k: v for k, v in extra_terms.items()
                                 if k in extra_keys}

        self.beta = (64/5) * (G**3*self.m1*self.m2*(self.m1+self.m2)) / (c**5)

        # If weight, evolution_age or rejuvenation)age are missing,
        # assign them default values. weight is 1 by default, others
        # are zero.
        for key in extra_keys:
            if key not in self.extra_terms.keys():
                if key == 'weight':
                    self.extra_terms[key] = 1
                elif key == "coalescence_time":
                    self.extra_terms[key] = self.coalescence_time()
                else:
                    self.extra_terms[key] = 0
        # units: km^4 s^-1

        self.__parameter_array = {
            'beta': self.beta,
            'm1': self.m1,
            'm2': self.m2,
            'a0': self.a0,
            'e0': self.e0,
            'weight': self.extra_terms['weight'],
            'evolution_age': self.extra_terms['evolution_age'],
            'rejuvenation_age': self.extra_terms['rejuvenation_age'],
            'coalescence_time': self.extra_terms['coalescence_time']
        }

    def track_evolution(self, ax=None):
        """Tracks the evolution of a BSS in phase space.

        Propagates the BSS in time, generating the SMA and eccentricity
        arrays, and then plots this in 3D space. The units are Gyr (time),
        and Yr (period).

        Keyword Arguments:
            ax {matplotlib.axes3D} -- An axis object to plot on. Set to
                                      None to allow takahe to generate
                                      its own. (default: {None})

        Returns:
            {matplotlib.axes3D} -- The axis object generated.
        """
        ct = self.get('coalescence_time')

        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        t, a, e = self.evolve_until_merger()

        m1 = self.get('m1')
        m2 = self.get('m2')
        a *= (Solar_Radii * 1000)

        k = G*(m1 + m2) / (4*np.pi**2)

        T = np.sqrt(a**3 / k)

        T /= 31557600000000000
        T *= 1e9

        ax.plot(t, T, e)
        ax.set_xlabel('age (Gyr)')
        ax.set_ylabel("period (yr)")
        ax.set_zlabel("eccentricity")

        return ax

    def get(self, parameter):
        """Retrieve a given parameter.

        Fast and clean fetching of a given parameter in the config of
        the BSS.

        Arguments:
            parameter {string} -- The parameter to fetch. Call
                                  star.get('show') to see all allowed
                                  parameters.

        Returns:
            {float} -- the given parameter.

        Raises:
            ValueError -- If the parameter requested does not exist.
        """
        if parameter == 'show':
            return self.__parameter_array.keys()
        elif parameter not in self.__parameter_array.keys():
            raise ValueError("Key does not exist!")

        return self.__parameter_array[parameter]

    def get_mass(self):
        """Returns the total mass of the system in solar masses.

        Returns:
            {float} -- the total mass of the binary system (units: M_sun)
        """
        return (self.get('m1') + self.get('m2')) / Solar_Mass

    def lifetime(self):
        """Computes the total lifetime of a binary star system.

        Calculates the total lifetime of the BSS. Assumes the total
        lifetime is the rejuvenation age + evolution_age + coalescence
        time.

        Returns:
            {float} -- The lifetime ofcoalescence_ the BSS, in gigayears.
        """
        early_lifetime = self.get('rejuvenation_age') \
                       + self.get('evolution_age')

        early_lifetime /= (1e9)
        return early_lifetime + self.get('coalescence_time')

    def coalescence_time(self):
        """Computes the coalescence time for the BSS in gigayears.

        Uses Eqn 10 from Nyadzani and Razzaque [1] to compute the
        coalescence time for the binary star system. Although this is
        strictly speaking an approximation, it converges on the
        result gained by numerically by Peters in 1964.

        [1] https://arxiv.org/pdf/1905.06086.pdf

        Returns:
            float -- the coalescence time for the binary star (units: ga).
        """

        circ = self.a0**4 / (4*self.beta)
        divisor = ((1-self.e0**(7/4))**(1/5)*(1+121/304 * self.e0**2))

        return (circ * (1-self.e0**2)**(7/2) / divisor) / 31557600000000000

    def circularises(self, thresholds=(0.0, 2*Solar_Radii)):
        """Determines if the orbit in question circularises or not.

        By default, this function assumes a BSS circularises if it's
        eccentricity becomes arbitrarily close to 0 and it's final SMA
        is more than 2 solar radii (such that it does not merge).

        Keyword Arguments:
            thresholds {tuple} -- The thresholds to determine if the
                                  system merges. The first entry in the
                                  tuple is the threshold for the
                                  eccentricity, the second is the
                                  threshold for the SMA.
                                  Default: (0, 2*Solar_Radii)

        Returns:
            bool -- True if the orbit circularises, False otherwise.
        """
        t, a, e = self.evolve_until_merger()

        if np.isclose(e[-1], thresholds[0]) and a[-1] > thresholds[1]:
            return True

        return False

    def evolve_until_merger(self):
        """Syntactic sugar for BSS.evolve_until(BSS.coalescence_time())

        Evolves a BSS in time until its coalescence time.

        Returns:
            mixed -- A 3-tuple containing the resultant time
                     array (in gigayears), and the resultant SMA (in
                     solar radii) and eccentricity arrays.
        """
        t_span = (0, self.get('coalescence_time') * 1e9 * 60 * 60 * 24 * 365.25)
        return self.evolve_until(t_span)

    def evolve_until(self, t_span):
        """Evolve the binary star system in time.

        Uses a Runge-Kutta algorithm to evolve the binary star system
        over a specified range evolve_over.

        This function adopts the Peters prescription, with customisable
        terms for da/dt and de/dt. This function is compiled using
        numba to achieve performance gains

        Arguments:
            t_span {tuple} -- A 2-tuple that corresponds to the start
                              and end points of integration.

        Returns:
            mixed -- A 3-tuple containing the resultant time
                     array (in gigayears), and the resultant SMA (in
                     solar radii) and eccentricity arrays.
        """

        t0 = np.float64(t_span[0])
        t1 = np.float64(t_span[1])

        evolve_over = np.linspace(t0, t1, 10000)

        a, e = takahe.helpers.integrate(evolve_over,
                                        self.a0,
                                        self.e0,
                                        self.beta)

        # Convert quantities back into Solar Units
        evolve_over /= 31557600000000000
        a /= (Solar_Radii * 1000)

        return evolve_over, a, e

    def __str__(self):
        return f"""
Binary Star System Parameters:
    - M1: {self.get('m1')} kg
    - M2: {self.get('m2')} kg
    - a0: {self.get('a0')} km
    - e0: {self.get('e0')}
    - Coalescence Time: {self.get('coalescence_time')} Gyr
        """
