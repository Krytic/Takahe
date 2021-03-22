import bisect
import warnings

from julia import Main as jl
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound
from scipy.integrate import quad
# import swifter
import takahe

def find_between(a, low, high):
    """Finds all elements of a between low and high.

    Simple binary search algorithm using the bisect module. Searches the
    array a for all elements between low and high.

    Useful for finding mergers that occur between time t1 and t2.

    Arguments:
        a {list} -- The list to search through.
        low {float} -- The lower bound to find
        high {float} -- The upper bound to find.

    Returns:
        list -- The filtered list
.mes

    Raises:
        ValueError -- If the value could not be found.
    """
    i = bisect.bisect_left(a, low)
    g = bisect.bisect_right(a, high)
    if i != len(a) and g != len(a):
        return a[i:g]
    raise ValueError

def memoize(f):
    """Memoizes a function.

    Allows any function to "remember" prior calls. Example:

    >>> @memoize
    >>> def fib(n):
    >>>     return 1 if n <= 1
    >>>     return fib(n-1) + fib(n-2)
    >>> fib(5) # O(2^n) because basic recursive function
    >>> fib(4) # O(1) because it was called during execution of fib(5).

    Arguments:
        f {callable} -- The function you wish to memoize

    Returns:
        {callable} -- The memoized function.
    """
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

def identify(star):
    """Simple helper function to identify a BSS.

    Arguments:
        star {pd.Series} -- the series object representing the BSS.

    Returns:
        {string} -- the type of binary system.
    """
    MASS_NS = takahe.constants.MASS_CUTOFF_NS
    MASS_BH = takahe.constants.MASS_CUTOFF_BH

    if star.m1 < MASS_NS and star.m2 < MASS_NS:
        return 'NSNS'
    if star.m1 < MASS_NS and star.m2 > MASS_BH:
        return 'NSBH'
    if star.m1 > MASS_BH and star.m2 < MASS_NS:
        return 'NSBH'
    if star.m1 > MASS_BH and star.m2 > MASS_BH:
        return 'BHBH'

def format_metallicity(Z):
    r"""Converts a BPASS-formatted metallicity into a "real valued" one.

    Interprets the BPASS-encoded metallicities as *fraction of solar
    metallicity* metallicities. See the BPASS manual [1] for a
    description of the encoding.

    Examples:
        - maps 020 to 1.0
        - maps em5 to 0.0005
        - maps 010 to 0.5
        - maps 0.020 to 1.0

    [1] https://bpass.auckland.ac.nz/8/files/bpassv2_1_manual.pdf

    Arguments:
        Z {string} -- The metallicity value to convert.

    Returns:
        mixed -- The converted metallicity values.
    """
    Z = str(Z)

    if Z == "0":
        div = 0
    elif "." in Z:
        div = float(Z)
    elif Z[:2] == "em":
        div = 1*10**(-int(Z[-1]))
    else:
        div = float("0." + Z)

    return div / takahe.constants.SOLAR_METALLICITY

def find_contours(X, Y, Z, value):
    """Extracts contours from a given dataset.

    Extracts the contours for a given X / Y / Z dataset. X, Y, Z must
    be in a format that can be passed to plt.contour.

    Value can be a list of contours - in which case lists will be
    generated for each value in the list.

    A note about output format
    --------------------------

    This function returns data in the following format:
        paths[contour value][i] = {
            'x': [x coordinates]
            'y': [y coordinates]
        }

    Where i is a counter variable. If the given contour exists only
    once in the dataset, i will be 0 and paths[contour value] will have
    a length of 1. If it exists n times, paths[contour value] will have
    a length of n, and i will range from 0 to n-1.

    Arguments:
        X {np.array} -- A meshgrid object for the X axis.
        Y {np.array} -- A meshgrid object for the Y axis.
        Z {np.matrix} -- The matrix corresponding to the Z-values of the
                         plot.
        value {mixed} -- Either a single value, or a list of values
                         to extract the contours from.

    Returns:
        {list} -- A list of each contour coordinates, in the format
                  described above.
    """

    assert isinstance(X, np.array), "Expected X to be arraylike."
    assert isinstance(Y, np.array), "Expected Y to be arraylike."
    assert isinstance(Z, np.matrix), "Expected Z to be matrixlike."
    assert isinstance(value, [np.float,
                              np.int,
                              np.ndarray,
                              list]), ("Expected value to be a number "
                                       "or listlike.")

    try:
        l = len(value)
    except TypeError:
        l = 1
        value = np.array([value])

    plt.figure()
    cs = plt.contour(X, Y, Z, value)
    plt.close()
    paths = dict()

    for i in range(len(value)):
        path = cs.collections[i].get_paths()

        for j in range(len(path)):
            verts = path[j].vertices
            x = verts[:,0].tolist()
            y = verts[:,1].tolist()
            if j != 0:
                paths[value[i]].append({'x': x, 'y': y})
            else:
                paths[value[i]] = [{'x': x, 'y': y}]

    return paths

def extract_metallicity(filename):
    """Extracts a metallicity from a filename.

    Assumes the filename has individual terms separated by "-" and that
    the metallicity term begins with z. Returns a (formatted) metallicity
    term.

    Examples:
        >>> extract_metallicity("Remnant-Birth-bin-"
                                "imf135_300-z030_StandardJJ.dat")
        1.5

        >>> extract_metallicity("Remnant-Birth-bin-"
                                "imf135_300-z010")
        0.5

    Arguments:
        filename {str} -- The filename you wish to extract Z from.

    Returns:
        float -- The metallicity extracted.
    """
    assert isinstance(filename, str), ("Expected filename to be a string in "
                                       "call to extract_metallicity.")

    fname = filename.split("/")[-1] # Extract filename if it has a
                                    # directory prepended.

    fname = fname.split(".")[0]     # Remove file extension
    parts = fname.split("-")        # Split into parts
    Z = None

    for part in parts:
        if part[0] == "z":          # Detect the term that has the Z
                                    # specification
            if "_" in part:         # We allow z020_StandardJJ for
                                    # example as this usually contains
                                    # the kick specification

                part = part.split("_")[0]

            Z = part[1:]

    return format_metallicity(Z)

@np.vectorize
def compute_period(a, M, m):
    """Computes the period of a BSS

    Uses Kepler's third law to compute the period, in days.

    Decorators:
        np.vectorize

    Arguments:
        a {float} -- The SMA of the BSS (in solar radii)
        M {float} -- The mass of the primary star (in solar masses)
        m {float} -- The mass of the secondary star (in solar masses)

    Returns:
        {float} -- The period in days
    """
    return np.sqrt(4 * np.pi **2 / (takahe.constants.G * (M+m) * takahe.constants.SOLAR_MASS) * (a * takahe.constants.SOLAR_RADIUS)**3) / (60 * 60 * 24)

# Something is fucked here

@np.vectorize
def compute_separation(P, M, m):
    """Computes the separation of a BSS

    Uses Kepler's third law to compute the separation, in days.

    Decorators:
        np.vectorize

    Arguments:
        P {float} -- The period of the binary star (in days)
        M {float} -- The mass of the primary star (in solar masses)
        m {float} -- The mass of the secondary star (in solar masses)

    Returns:
        {float} -- The orbital separation in Solar Radii
    """
    a = ((P * 60 * 60 * 24)**2 * (takahe.constants.G * (M+m) * takahe.constants.SOLAR_MASS) / (4 * np.pi **2))**(1/3) / takahe.constants.SOLAR_RADIUS

    return a

@np.vectorize
@memoize
def lookback_to_redshift(tL):
    """Internal function to convert a lookback time into a redshift.

    Used by plot_merge_rate in furtherance of computing the SFRD.

    Arguments:
        tL {float} -- A lookback time within the range (0, 14).

    Returns:
        {float} -- The redshift z, corresponding to the lookback time
                   tL
    """

    f = lambda z: np.abs(redshift_to_lookback(z) - tL)

    zbest, _, _, _ = fminbound(f, 1e-8, 1000, maxfun=500, full_output=1, xtol=1e-8)

    return zbest

@np.vectorize
def redshift_to_lookback(z):
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
            return np.sqrt(takahe.constants.OMEGA_M * (1+z)**3
                         + takahe.constants.OMEGA_K * (1+z)**2
                         + takahe.constants.OMEGA_L)
        return 1 / ((1+z) * E(z))

    rest, err = quad(integrand, 0, z)

    return takahe.constants.HUBBLE_TIME * rest

def integrate(a0, e0, p):
    """Integrates the System of ODEs that govern binary star evolution
    in period-eccentricity space.

    Uses a custom Julia integrator to integrate the equations of motion
    given by [1]. The integrator constrains the periods and eccentricities
    to the intevals (cutoff_period, P0] and [0, 1] respectively.

    [1] Nyadzani, L. & Razzaque, S. (2019), An Analytical Solution to the Coalescence Time of Compact Binary Systems, Technical report, University of Johannesburg, Johannesburg.

    [2] 64/5 * G**3 * m1 * m2 * (m1 + m2) / c**5

    Raises:
        AssertionError            -- If any parameter is not of an
                                     acceptable type or value.

    Arguments:
        a0 {float}                -- The initial value for the SMA in
                                     solar radii
        e0 {float}                -- The initial value for the
                                     eccentricity
        p {list, np.ndarray}      -- A vector of parameters:
                                        p[0] = beta (defined by [2] above)
                                        p[1] = m1 (in solar masses)
                                        p[2] = m2 (in solar masses)

    Returns:
        {tuple}                   -- A 2-tuple containing the semimajor
                                     axis array and eccentricity array.
    """

    assert isinstance(a0, (float, int, np.int64, np.float)), "Expected a0 to be a float or int type"
    assert isinstance(e0, (float, int, np.int64, np.float)), "Expected e0 to be a float or int type"
    assert isinstance(p,  (np.ndarray, list)), "Expected p to be arraylike"

    assert 0 <= e0 <= 1,                       "e0 outside of range [0, 1]"

    return takahe.integrate_eoms(a0, e0, p)
