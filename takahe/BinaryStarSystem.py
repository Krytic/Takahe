import numpy as np
from scipy.constants import c, G
from scipy.integrate import solve_ivp
from hoki import load

Solar_Mass = 1.989e30 #kg
Solar_Radii = 696340 #km

class BinaryStarSystem:
    """Represents a binary star system."""

    def __init__(self, primary_mass, secondary_mass, a0, e0):
        self.m1 = primary_mass * Solar_Mass # Units: kg
        self.m2 = secondary_mass * Solar_Mass # Units: kg
        self.a0 = np.float128(a0 * Solar_Radii * 1000) # Units: km
        self.e0 = np.float128(e0) # Units: dimensionless

        self.beta = (64/5) * (G**3*self.m1*self.m2*(self.m1+self.m2))/(c**5)
        # units: km^4 s^-1

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

    def evolve(self, t_span):
        """Evolve the binary star system in time

        Uses a Runge-Kutta algorithm to evolve the binary star system
        over a specified range evolve_over.

        Arguments:
            t_span {tuple} -- A 2-tuple that corresponds to the start
                              and end points of integration.

        Returns:
            mixed -- A 3-tuple containing the resultant time
                     array (in gigayears), and the resultant SMA (in
                     solar radii) and eccentricity arrays.
        """

        def dadt(t, a, e):
            """
            Auxiliary function to compute Equation 3.

            Params:
                t [ndarray] A vector of times.
                e [float] The current eccentricity
                a [float] The current semimajor axis

            Output:
                The quantity da/dt - how the semimajor axis is changing
                                     with time.
            """

            initial_term = (-self.beta / (a**3 * (1-e**2)**(7/2)))

            da = initial_term * (1 + 73/24 * e**2 + 37 / 96 * e ** 4)
            # Units: km/s

            return da

        # Equation (4) from ibid
        def dedt(t, a, e):
            """
            Auxiliary function to compute Equation 4.

            Params:
                t [ndarray] A vector of times.
                e [float] The current eccentricity
                a [float] The current semimajor axis

            Output:
                The quantity de/dt - how the eccentricity is changing
                                     with time.
            """

            initial_term = (-19/12 * self.beta / (a**4*(1-e**2)**(5/2)))

            de = inital_term * (e + 121/304 * e ** 3) # Units: s^-1

            return de

        def coupled_eqs(t, params):
            """
            Primary workhorse function. Computes the vector
            [da/dt, de/dt] for use in our integrator.

            Params:
                t [ndarray] A vector of times
                params [list] A list or 2-tuple of arguments. Must take
                              the form [a, e]

            Output:
                A list containing da/dt and de/dt
            """

            return [dadt(t, *params), dedt(t, *params)]

        def integrate(t_eval):
            """
            Auxilary function which uses the euler method to integrate
            the system of ODEs

            Arguments:
                t_eval {ndarray} -- An array of timesteps to compute
                                    the integrals over

            Returns:
                a_arr {ndarray} -- An array representing the SMA of the
                                   binary orbit
                e_arr {ndarray} -- An array representing the
                                   eccentricity of the binary orbit
            """
            h = t_eval[1] - t_eval[0]
            a, e = self.a0, self.e0

            a_arr = []
            e_arr = []

            for t in t_eval:
                da = dadt(t, a, e)
                de = dedt(t, a, e)
                a = a + h * da
                e = e + h * de

                a_arr.append(a)
                e_arr.append(e)

            return np.array(a_arr), np.array(e_arr)

        evolve_over = np.linspace(t_span[0], t_span[1], 10000)

        a, e = integrate(evolve_over)

        evolve_over /= 31557600000000000
        a /= (Solar_Radii * 1000)

        return evolve_over, a, e
