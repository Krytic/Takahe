import numpy as np
import matplotlib.pyplot as plt
import takahe
from kea.hist import histogram, BPASS_hist
from takahe.constants import *
from scipy.optimize import root_scalar, fminbound
from scipy.integrate import quad
from numba import njit

def create(model, hubble_parameter=70):
    """

    "if you want to make an apple pie from scratch,
    you must first invent the universe" -- Carl Sagan

    Represents the universe writ large and contains all of the
    important physical parameters.

    """
    return Universe(model, hubble_parameter)

"""
Internal functions.

These functions are defined here such that we can numba-fy them to make
them computationally faster.
"""

def _format_z(z):
    if z[:2] == "em":
        div = 1*10**int(-z[-1])
    else:
        div = float("0." + z)
    res = div / 0.020
    return rf"{res}Z_\odot"

@njit
def _comoving_vol(DH, omega_k, DC):
    if omega_k > 0:
        OK = np.sqrt(omega_k)
        DM = DH / OK * np.sinh(OK * DC / DH)
    elif omega_k == 0:
        DM = DC
    elif omega_k < 0:
        OK = np.sqrt(np.abs(omega_k))
        DM = DH / OK * np.sin(OK * DC / DH)

    if omega_k == 0:
        VC = 4*np.pi/3 * DM**3
    else:
        DH = DH
        OK = np.sqrt(np.abs(omega_k))

        coeff = 4*np.pi * DH**3 / (2*omega_k)
        term1 = DM / DH * np.sqrt(1+omega_k*(DM/DH))**2

        if omega_k > 0:
            term2 = 1/OK * np.arcsinh(OK * DM / DH)
        else:
            term2 = 1/OK * np.arcsin(OK * DM / DH)

        VC = coeff * (term1 - term2)

    return VC

"""
Begin definition of Universe class.
"""

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

        self.tH = 1 / (self.H0 / 3.086e+19 * 31557600000000000) # seconds

        self.__count = 0
        self.__z = None

    def comoving_volume(self, z=None, d=None):
        """Computes the comoving volume, all-sky, out to redshift z.

        Rather chonky function that uses eqn(29) of [1] to compute the
        comoving volume of a region of space out to redshift z.

        If a redshift is provided, takahe computes the comoving distance
        via eqn(15) and eqn(14) of [1]. Alternatively, it uses the
        comoving distance provided to it.

        As an auxiliary calculation it does also compute the transverse
        comoving distance D_M defined by eqn(16) of [1].

        As with all functions in this class, the value will vary
        depending on the type of universe created (eds / real / lowdens
        / highlambda).

        This function is a wrapper for an internal Numba-compiled
        function.

        [1] https://arxiv.org/pdf/astro-ph/9905116.pdf

        Keyword Arguments:
            z {float} -- The redshift to compute V_C for
                         (default: {None})
            d {float} -- The comoving distance to compute V_C for
                         (default: {None})

        Returns:
            float -- The comoving volume element in units of Mpc^3

        Raises:
            ValueError -- If one of z or d are missing
        """

        if z == None and d == None:
            raise ValueError("Either z or d must be provided!")
        elif z == None:
            DC = d
        else:
            DC = self.compute_comoving_distance(z)

        return _comoving_vol(self.DH, self.omega_k, DC)

    def plot_event_rate(self):
        """Generates and plots the event rate distribution for this universe.

        Computes the event rate distribution for this universe. Assumes
        SFRD as given by eqn(15) in Madau & Dickinson 2014 [1], with
        u = 5.6 (see self.stellar_formation_rate for details).

        Returns the given histogram for further manipulation, if required.

        [1] https://www.annualreviews.org?cid=75#/doi/pdf/10.1146/annurev-astro-081811-125615

        Returns:
            {kea.hist.histogram} -- the generated histogram.
        """
        plt.figure()

        dtd_hist = self.populace.compute_delay_time_distribution(color='blue', label=r'DTD [events / $M_\odot$ / Gyr]')

        edges = dtd_hist.getBinEdges()

        events = histogram(0, self.tH, len(edges)-1)
        ev_edges = events.getBinEdges()

        for i in range(1, events.getNBins()+1):
            t1 = ev_edges[i-1]
            t2 = ev_edges[i]

            z_low = self.__lookback_to_redshift(t1)
            z_high = self.__lookback_to_redshift(t2)

            SFRD, _ = quad(self.stellar_formation_rate, z_low, z_high)

            SFRD /= (1e-3)**3

            for j in range(i):
                t1_prime = t2 - ev_edges[j]
                t2_prime = t2 - ev_edges[j+1]
                events_in_bin = dtd_hist.integral(t2_prime, t1_prime)
                events.Fill(ev_edges[j], events_in_bin*SFRD)

        plt.clf()

        bins = np.array([events.getBinWidth(i)*1e9 for i in range(0, events.getNBins())])
        events /= bins # Normalise to years

        events.plot(color='red', label="Mergers [# of events / yr / Gpc^3]")

        plt.yscale('log')
        plt.legend()

        if self.__z != None:
            plt.title(rf"$Z={_format_z(self.__z)}, n={self.populace.size()}$")

        return events

    def __lookback_to_redshift(self, tL):
        """Internal function to convert a lookback time into a redshift.

        Used by plot_merge_rate in furtherance of computing the SFRD.

        Arguments:
            tL {float} -- A lookback time within the range (0, 14).

        Returns:
            {float} -- The redshift z, corresponding to the lookback time
                       tL
        """

        f = lambda z: np.abs(self.__redshift_to_lookback(z) - tL)

        zbest, _, _, _ = fminbound(f, 1e-8, 1000, maxfun=500, full_output=1, xtol=1e-8)

        return zbest

    def __redshift_to_lookback(self, z):
        """Internal function to convert a redshift into a lookback time.

        Used by plot_merge_rate in furtherance of computing the SFRD.

        Arguments:
            z {float} -- A redshift value in the range (0, 100).

        Returns:
            {float} -- The redshift z, corresponding to the lookback time
                       tL
        """

        def integrand(z):
            def E(z):
                return np.sqrt(self.omega_m * (1+z)**3
                             + self.omega_k * (1+z)**2
                             + self.omega_lambda)
            return 1 / ((1+z) * E(z))

        rest, err = quad(integrand, 0, z)

        return self.tH*rest

    def stellar_formation_rate(self, z=None, d=None, u=5.6):
        """Computes the SFRD for the universe at a given redshift.

        Uses eqn(15) of [1] to compute the SFRD at redshift z. You may
        specify either z (redshift) or d (comoving distance) as a keyword
        argument. If d is provided, z is computed via self.compute_redshift.

        Following the argument in [2], we use u as a parameter of the eqn
        to adjust the peak of SFRD. To reproduce [1], leave u = 5.6.

        [1] https://www.annualreviews.org?cid=75#/doi/pdf/10.1146/annurev-astro-081811-125615

        [2] Tang, N (2019). Uncertainty in the Gravitational Wave Event Rates from the History of Star Formation in the Universe (MSc Thesis, The University of Auckland, Auckland, New Zealand). Retrieved from http://hdl.handle.net/2292/47490.

        Keyword Arguments:
            z {float} -- The redshift to consider
                         (default: {None})
            d {float} -- The comoving distance to consider
                         (default: {None})
            u {float} -- A parameterisation relating to when the peak
                         SFRD occurs.
                         (default: 5.6)

        Returns:
            float -- the SFRD for the universe at z
                     (units: M_sun / yr / Mpc^3).

        Raises:
            ValueError -- If z or d are not provided.
        """
        if z == None and d == None:
            raise ValueError("Either z or d must be provided!")
        elif z == None:
            z = self.compute_redshift(d)

        SFRD = 0.015 * (1+z)**2.7 / (1+((1+z)/2.9)**u)

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

    def populate(self, loader, mass=1e6, name_hints=None, n_stars=1000, load_type='linear'):
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

        if load_type == 'linear':
            self.populace = takahe.load.from_file(loader,
                                                  name_hints=name_hints,
                                                  mass=mass,
                                                  n_stars=n_stars)
        elif load_type == 'random':
            self.populace = takahe.load.random_from_file(loader,
                                                         10 * n_stars,
                                                         name_hints=name_hints,
                                                         mass=mass,
                                                         n_stars=n_stars)

        fname = loader.split("/")[-1].split(".")[0].rsplit("_", 1)[0]
        parts = fname.split("-")
        self.__z = None

        for part in parts:
            if part[0] == "z":
                # metallicity term in fname
                self.__z = part[1:]
