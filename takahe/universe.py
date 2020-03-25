import numpy as np
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
        For instance, one may create an Einstein-de Sitter Universe.

        Arguments:
            model {str} -- the model (eds, lowdensity, highlambda) of
                           the universe under consideration.

        Keyword Arguments:
            hubble_paramter {float} -- the current value of H0.
                                       (default: 70)

        Raises:
            ValueError -- if model is not eds / lowdensity / highlambda
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
        else:
            raise ValueError("model must be eds (Einstein-de Sitter), \
                              lowdensity (Low Density), or highlambda \
                              (High lambda).")

        self.omega_m = Omega_M
        self.omega_lambda = Omega_Lambda
        self.omega_k = 1 - self.omega_m - self.omega_lambda
        self.H0 = hubble_parameter

        self.DH = (c/1000) / self.H0 # Megaparsecs

        self.__count = 0

    def compute_comoving_distance(self, z):
        """Computes the comoving distance between two objects in this
        universe.

        Uses eqn(15) and eqn(14) from [1], and numerically integrates
        this using scipy.integrate.quad.

        [1] https://arxiv.org/pdf/astro-ph/9905116.pdf

        Arguments:
            z {float} -- The redshift to compute DC for.

        Returns:
            number -- The comoving radial distance
        """
        def integrand(z):
            return 1 / np.sqrt(self.omega_m * (1+z)**3
                             + self.omega_k * (1+z)**2
                             + self.omega_lambda)

        result, err = quad(integrand, 0, z)

        return self.DH * result

    def populate(self, mass=1e6):
        """
        Populates the Universe with stars.

        Loads from a standard file at the moment. Will be extended in
        future, to support arbitrary files.
        """

        self.populace = takahe.load.from_file('data/newdata/Remnant-Birth-bin-imf135_300-z040_StandardJJ.dat', name_hints=['m1', 'm2', 'a0', 'e0', 'weight', 'evolution_age', 'rejuvenation_age'])
