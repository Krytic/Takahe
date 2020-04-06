import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
import takahe
from takahe.constants import *

def create(model, hubble_parameter=70):
    """

    "if you want to make an apple pie from scratch,
    you must first invent the universe" -- Carl Sagan

    Represents the universe writ large and contains all of the
    important physical parameters.

    """
    return Universe(model, hubble_parameter)

class Universe:
    """

    "if you want to make an apple pie from scratch,
    you must first invent the universe" -- Carl Sagan

    Represents the universe writ large and contains all of the
    important physical parameters.

    """
    def __init__(self, model, hubble_parameter=70):
        """

        Creates our Universe, conforming to a given set of physical laws.
        For instance, one may create an Einstein-de Sitter universe,
        one with a large cosmological constant (high lambda), low density
        or real (in which parameters are aligned as closely to reality
        as possible)

        Arguments:
            model {str} -- the model (eds, lowdensity, highlambda, real)
                           of the universe under consideration.

        Keyword Arguments:
            hubble_parameter {float} -- the current value of H0.
                                        units are km/s/Mpc.
                                        (default: 70)

        Raises:
            ValueError -- if model is not eds / lowdensity /
                                          highlambda / real
        """

        if model.lower() == 'eds':
            Omega_M = 1
            Omega_Lambda = 0
        elif model.lower() == 'lowdensity':
            Omega_M = 0.05
            Omega_Lambda = 0
        elif model.lower() == 'highlambda':
            Omega_M = 0.2
            Omega_Lambda = 0.8
        elif model.lower() == 'real':
            Omega_M = 0.286
            Omega_Lambda = 0.714
            hubble_parameter = 69.6
        else:
            raise ValueError("Incorrect model type!")

        # Universal physical constants.
        self.omega_m = Omega_M
        self.omega_lambda = Omega_Lambda
        self.omega_k = 1 - self.omega_m - self.omega_lambda
        self.H0 = hubble_parameter

        self.DH = (c/1000) / self.H0 # Megaparsecs

        self.tH = 1 / (self.H0 / 3.086e+19) # seconds

        self.__count = 0

    def stellar_formation_rate(self, z=None, d=None):
        """Computes the SFRD for the universe at a given redshift.

        Uses eqn(15) of [1] to compute the SFRD at redshift z. You may
        specify either z (redshift) or d (comoving distance) as a keyword
        argument. If d is provided, z is computed via self.compute_redshift.

        [1] https://www.annualreviews.org?cid=75#/doi/pdf/10.1146/annurev-astro-081811-125615

        Keyword Arguments:
            z {float} -- The redshift to consider
                         (default: {None})
            d {float} -- The comoving distance to consider
                         (default: {None})

        Returns:
            float -- the SFRD for the universe at z

        Raises:
            ValueError -- If z or d are not provided.
        """
        if z == None and d == None:
            raise ValueError("Either z or d must be provided!")
        elif z == None:
            z = self.compute_redshift(d)

        SFRD = 0.015 * (1+z)**2.7 / (1+((1+z)/2.9)**5.6)

        return SFRD

    def compute_redshift(self, d):
        """Computes the redshift at a given comoving distance d.

        Uses scipy.optimize.root_scalar to find the redshift, via
        eqn(15) and eqn(14) of [1].

        Note that as this involves finding a root of a function that
        must be continually numerically evaluated, this function can be
        both unstable and computationally expensive.

        [1] https://arxiv.org/pdf/astro-ph/9905116.pdf

        Arguments:
            d {float} -- The comoving distance to compute redshift at.
                         (units: Mpc)

        Returns:
            float -- The redshift at distance d.
        """
        f = lambda x: self.compute_comoving_distance(x) - d
        res = root_scalar(f, x0=0, x1=0.001)

        return res.root

    def compute_comoving_distance(self, z):
        """Computes the comoving distance between two objects in this
        universe.

        Uses eqn(15) and eqn(14) from [1], and numerically integrates
        this using scipy.integrate.quad.

        [1] https://arxiv.org/pdf/astro-ph/9905116.pdf

        Arguments:
            z {float} -- The redshift to compute DC for.

        Returns:
            number -- The comoving radial distance (units: Mpc)
        """
        def integrand(z):
            return 1 / np.sqrt(self.omega_m * (1+z)**3
                             + self.omega_k * (1+z)**2
                             + self.omega_lambda)

        result, err = quad(integrand, 0, z)

        return self.DH * result

    def populate(self, loader, mass=1e6, name_hints=None, n_stars=1000):
        """
        Populates the Universe with stars.

        Arguments:
            loader {str} -- a path to the file to load

        Keyword Arguments:
            mass {float} -- the total mass you wish to generate. This
                            generates a total of mass*weight stars for
                            each star type in the dataset.
            name_hints {list} -- a list of column names to pass to the
                                 loader.
        """

        self.populace = takahe.load.from_file(loader,
                                              name_hints=name_hints,
                                              mass=mass,
                                              n_stars=n_stars)
