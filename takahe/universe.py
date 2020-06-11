import pickle

import matplotlib.pyplot as plt
import numpy as np
import takahe

from kea.hist import histogram
from scipy.optimize import root_scalar, fminbound
from scipy.integrate import quad
from scipy.special import gammainc, gamma
from takahe.constants import *

def create(model, hubble_parameter=70):
    """

    "if you want to make an apple pie from scratch,
    you must first invent the universe" -- Carl Sagan

    Represents the universe writ large and contains all of the
    important physical parameters.

    """
    return Universe(model, hubble_parameter)


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
                                        units are km/s/Mpc. You must
                                        specify this if you are not using
                                        the lcdm model.
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
        elif model.lower() == "lcdm":
            Omega_M = 0.3
            Omega_Lambda = 0.7
            hubble_parameter = 70
        else:
            raise ValueError("Incorrect model type!")

        # Universal physical constants.
        self.omega_m = Omega_M
        self.omega_lambda = Omega_Lambda
        self.omega_k = 1 - self.omega_m - self.omega_lambda
        self.H0 = hubble_parameter

        self.DH = (c/1000) / self.H0 # Megaparsecs

        self.tH = 1 / (self.H0 / 3.086e+19 * 31557600000000000) # Gyr

        self.__resolution = 51

        self.__count = 0
        self.__z = None

        self.SFR = self.stellar_formation_rate

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

        return takahe.helpers.comoving_vol(self.DH, self.omega_k, DC)

    def set_nbins(self, resolution):
        assert isinstance(resolution, int), "Resolution must be an integer"
        self.__resolution = resolution

    def get_nbins(self):
        return self.__resolution

    def compute_delay_time_distribution(self, *argv, **kwargs):
        """Generates the DTD plot for this ensemble.

        Computes the instantaneous delay-time distribution for this
        ensemble. Returns the histogram generated, however the histogram
        is saved internally in Kea as a matplotlib plot.

        Thus, given an ensemble called ens, one may use

        >>> ens.compute_delay_time_distribution()
        >>> plt.show()

        to render it.

        Note that the binning is logarithmic so bin size does vary
        across the plot.

        Thanks to Max Briel (https://github.com/maxbriel/) for his
        assistance in writing this function.

        Returns:
            hist -- the (kea-generated) histogram object.
        """

        hist = histogram(0, self.tH, self.__resolution)
        culmulative_merge_rate = 0
        edges = hist.getBinEdges()
        bin_widths = np.array([])

        NBins = self.__resolution

        for i in range(0, NBins-1):
            merge_rate_up_to_bin = self.populace.merge_rate(edges[i+1])
            merge_rate_in_bin = merge_rate_up_to_bin - culmulative_merge_rate
            hist.Fill(edges[i], w=merge_rate_in_bin)
            culmulative_merge_rate += merge_rate_in_bin
            bin_widths = np.append(bin_widths, hist.getBinWidth(i))

        # Normalisation
        hist = hist / 1e6 / bin_widths

        return hist

    def get_metallicity(self):
        return self.__z

    def generate_redshift_array(self):
        """Generates an array of redshifts for this universe.

        Generates an array of data (n=n_bins) representing the redshift values
        in this universe. Redshifts will correspond to the interval in
        time space of (0, tH).

        Returns:
            {np.linspace} -- The redshift array
        """
        z_low = self.lookback_to_redshift(0)
        z_high = 6

        return np.linspace(z_low, z_high, self.__resolution)

    def event_rate(self, Z_compute, SFRD_so_far):
        """Generates and plots the event rate distribution for this universe.

        Computes the event rate distribution for this universe. Assumes
        SFRD as given by eqn(5) in Langer & Norman 2006 [1], with
        u = 5.6 (see self.stellar_formation_rate for details). In the case
        of single metallicity evolution, this reduces to the formula given
        by eqn(15) in Madau & Dickinson 2014 [2].

        Returns the given histogram for further manipulation, if required.

        [1] https://arxiv.org/pdf/astro-ph/0512271.pdf

        [2] https://www.annualreviews.org?cid=75#/doi/pdf/10.1146/annurev-astro-081811-125615

        Keyword Arguments:
            SFRD_so_far {kea.hist.histogram} -- A histogram of the
                                                SFRD computed at other
                                                metallicities. Leave as
                                                None if you are only
                                                considering a single-
                                                metallicity universe.
                                                (default: {None})

        Returns:
            dtd_hist {kea.hist.histogram}  -- The DTD of this universe
            SFRD_hist {kea.hist.histogram} -- The (metallicity-dependent)
                                              SFRD of this universe
            events {kea.hist.histogram}    -- The event_rate histogram of
                                              this universe.
        """

        # First, set up some of the histograms we'll be returning
        dtd_hist = histogram(0, self.tH, self.get_nbins())
        edges = dtd_hist.getBinEdges()

        events = histogram(0, self.tH, self.get_nbins())
        ev_edges = events.getBinEdges()

        # This holds the width of each bin, for normalisation reasons later.
        bins = np.array([])

        old_mr = 0

        NBins = self.__resolution

        SFRD_hist = histogram(0, self.tH, self.__resolution)

        # These two lambdas define the SFRD equation given by eqn(5) of
        # Langer & Norman: https://arxiv.org/pdf/astro-ph/0512271.pdf
        fSFRD = lambda z: self.stellar_formation_rate(z=z)
        fGamma = lambda z, Z: gammainc(0.84, Z**2 * 10**(0.3*z)) / gamma(0.84)

        # print(f"{self.get_metallicity()} corresponds to {Z}")

        # Generate a histogram of SFRD at *this* metallicity.
        # This histogram is what is returned by the function.
        for edge in SFRD_hist.getBinEdges():
            z = self.lookback_to_redshift(edge)

            SFRD_at_z = fSFRD(z) * fGamma(z, Z_compute)
            SFRD_hist.Fill(edge, w=SFRD_at_z)

        # Generate a manipulatable histogram. SFRD_hist is the *culmulative*
        # SFRD as given by Langer & Norman, and we will return *that* later.
        SFRD = SFRD_hist.copy()

        # Langer & Norman's formula is *culmulative* in metallicity.
        # So we need to subtract the contributions from individual
        # metallicities we have already considered.
        SFRD._values = SFRD._values - SFRD_so_far._values

        # Iterate over the bins in the histogram.
        for i in range(1, self.__resolution+1):
            # Compute the merge rate of this bin: For use in the DTD
            mr = self.populace.merge_rate(edges[i])
            dtd_bin_width = dtd_hist.getBinWidth(i-1)*1e9
            this_mr = mr - old_mr
            normalised_mr = (this_mr / 1e6) / dtd_bin_width

            dtd_hist.Fill(edges[i-1], w=normalised_mr)
            # Units: # / Msun / yr
            old_mr = mr

            t1 = ev_edges[i-1]
            t2 = ev_edges[i]

            # Compute the mass formed in this time bin
            this_SFR = SFRD.integral(t1, t2) * 1e9 # Units: Msun / Mpc^3

            this_SFR /= (1e-3)**3 # Units: Msun / Gpc^3

            # Convolve the SFH with the DTD to get the event rates
            for j in range(i):
                t1_prime = t2 - ev_edges[j]
                t2_prime = t2 - ev_edges[j+1]

                events_in_bin = dtd_hist.integral(t2_prime, t1_prime) * 1e9
                # Units: # / Msun
                events.Fill(ev_edges[j], events_in_bin * this_SFR)

            width = events.getBinWidth(i-1)*1e9
            bins = np.append(bins, width)

        # events has units: # / Gpc^3
        # Normalise to years:
        events /= bins

        return dtd_hist, SFRD, events, SFRD_hist

    def lookback_to_redshift(self, tL):
        """Internal function to convert a lookback time into a redshift.

        Used by plot_merge_rate in furtherance of computing the SFRD.

        Arguments:
            tL {float} -- A lookback time within the range (0, 14).

        Returns:
            {float} -- The redshift z, corresponding to the lookback time
                       tL
        """

        f = lambda z: np.abs(self.redshift_to_lookback(z) - tL)

        zbest, _, _, _ = fminbound(f, 1e-8, 1000, maxfun=500, full_output=1, xtol=1e-8)

        return zbest

    def redshift_to_lookback(self, z):
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

        This function attempts to infer the columns based on the filename.
        If the dataset is using the StandardJJ prescription, then the
        header is assumed to be the 7 StandardJJ columns (overridable),
        and if the file has _ct appended, it assumes the final column
        is the coalescence time of this system.

        Arguments:
            loader {str} -- a path to the file to load

        Keyword Arguments:
            mass {float} -- the total mass you wish to generate. This
                            generates a total of mass*weight stars for
                            each star type in the dataset.
            name_hints {list} -- a list of column names to pass to the
                                 loader.
        """

        if name_hints == None and "StandardJJ" in loader:
            # Provide column names for the StandardJJ prescription
            name_hints = ['m1','m2','a0','e0']
            name_hints.extend(['weight','evolution_age','rejuvenation_age'])

        if "_ct" in loader:
            # If the filename containts _ct, then we have a file
            # for which the coalescence times have already been computed
            name_hints.append("coalescence_time")

        # Do we load the first n_stars lines, or a random sample of
        # n_stars lines?
        if load_type == 'linear':
            pop = takahe.load.from_file(loader,
                                        name_hints=name_hints,
                                        mass=mass,
                                        n_stars=n_stars)

            self.populace = pop
        elif load_type == 'random':
            self.populace = takahe.load.random_from_file(loader,
                                                         10 * n_stars,
                                                         name_hints=name_hints,
                                                         mass=mass,
                                                         n_stars=n_stars)

        # Extract the metallicity from the filename
        fname = loader.split("/")[-1].split(".")[0].rsplit("_", 1)[0]
        parts = fname.split("-")
        self.__z = None

        for part in parts:
            if part[0] == "z":
                # metallicity term in fname
                if "_" in part:
                    part = part.split("_")[0]
                self.__z = part[1:]
