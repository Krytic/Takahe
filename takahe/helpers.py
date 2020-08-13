import warnings

import integrator
from numba import njit
import numpy as np
from scipy import LowLevelCallable
from scipy.optimize import fminbound
from scipy.integrate import quad
from scipy.integrate import RK45
import takahe

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

def format_metallicity(Z, as_string=False, rel=True):
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

    Keyword Arguments:
        as_string {bool} -- Whether or not to return the metallicity
                            value as a string. If True, returns it
                            LaTeX-formatted for mathmode (e.g.
                            format_metallicity("020", as_string=True)
                            will return $1.0Z_\odot$.)

                            (default: {False})

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

    if rel:
        res = div / 0.020
    else:
        res = div

    if as_string:
        return rf"${res}Z_\odot$"
    else:
        return res

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
        a {float} -- The SMA of the BSS
        M {float} -- The mass of the primary star
        m {float} -- The mass of the secondary star

    Returns:
        {number} -- The period in days
    """
    return np.sqrt(4 * np.pi **2 / (takahe.constants.G * (M+m) * 1.989e30) * (a * 696340000)**3) / (60 * 60 * 24)

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

@njit
def comoving_vol(DH, omega_k, DC):
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

@njit
def _dadt(t, a, e, beta):
    """Auxiliary function to compute Equation 3.

    Arguments:
        t {ndarray} -- A vector of times.
        e {float}   -- The current eccentricity
        a {float}   -- The current semimajor axis

    Returns:
        {float} -- The quantity da/dt - how the semimajor axis is changing
                   with time.
    """

    initial_term = (-beta / (a**3 * (1-e**2)**(7/2)))

    da = initial_term * (1 + 73/24 * e**2 + 37 / 96 * e ** 4)
    # Units: km/s

    return da

@njit
def _dedt(t, a, e, beta):
    """Auxiliary function to compute Equation 4.

    Arguments:
        t {ndarray} -- A vector of times.
        e {float}   -- The current eccentricity
        a {float}   -- The current semimajor axis

    Returns:
        {float} -- The quantity de/dt - how the eccentricity is changing
                   with time.
    """

    initial_term = (-19/12 * beta / (a**4*(1-e**2)**(5/2)))

    de = initial_term * (e + 121/304 * e ** 3) # Units: s^-1

    return de

@njit
def _coupled_eqs(t, p):
    """Primary workhorse function. Computes the vector [da/dt, de/dt]
    for use in our integrator.

    Arguments:
        t {ndarray} -- A vector of times
        p {list}    -- A list or 2-tuple of arguments. Must take the
                       form [a, e]

    Returns:
        {list} -- A list containing da/dt and de/dt
    """

    return np.array([_dadt(t, p[0], p[1], p[2]),
                     _dedt(t, p[0], p[1], p[2]),
                     p[2]])

def integrate(t_eval, a0, e0, beta, etol=1e-4):
    """Integrates the System of ODEs that govern binary star evolution
    in period-eccentricity space.

    Uses an RKF45 integrator to integrate the equations of motion given
    by [1]. Kills the integrator if:
    - there is a sharp change in da/dt (i.e., da/dt < 0 at step i, and
      da/dt > 0 at step i+1),
    - the eccentricity becomes >= 1
    - the semimajor axis becomes <= 1

    [1] Nyadzani, L. & Razzaque, S. (2019), An Analytical Solution to the Coalescence Time of Compact Binary Systems, Technical report, University of Johannesburg, Johannesburg.

    [2] 64/5 * G**3 * m1 * m2 * (m1 + m2) / c**5

    Decorators:
        njit

    Raises:
        AssertionError            -- If any parameter is not of an
                                     acceptable type or value.

    Arguments:
        t_eval {list, np.ndarray} -- A list of timepoints to perform the
                                     integration at. Used to compute the
                                     step size for the integrator.
        a0 {float}                -- The initial value for the SMA in
                                     solar radii
        e0 {float}                -- The initial value for the
                                     eccentricity
        beta {float}              -- The value beta, defined by [2] above

    Keyword Arguments:
        etol {float}              -- The acceptable tolerance in step
                                     size h (Default: 1e-4)

    Returns:
        {tuple}                   -- A 2-tuple containing the semimajor
                                     axis array and eccentricity array.
    """

    # Convert m^4 / s to Solar Radii^4 / Gyr
    beta = beta / (6.957e+8)**4 * (1e9 * 60 * 60 * 24 * 365.25)

    # Convert m to Solar Radii
    a0 = a0 / 6.957e8

    integrand = LowLevelCallable.from_cython(integrator, 'integrand')

    res = RK45(_coupled_eqs, t_eval[0], [a0, e0, beta], t_eval[1])

    a, e, beta = res.y

    A = [a]
    E = [e]

    for i in range(10000):
        msg = res.step()
        if res.status == 'running':
            A.append(res.y[0] * 6.957e8)
            E.append(res.y[1])
        elif res.status == 'finished':
            break
        elif res.status == 'failed':
            print(msg)
            break

    return np.array(A), np.array(E)

    # assert isinstance(t_eval, (np.ndarray, list)), ("Expected t_eval to"
    #                                                 " be a list-type in"
    #                                                 " call to integrate()")

    # assert len(t_eval >= 2), ("Expected t_eval to be longer than length 2"
    #                           " in call to integrate()")

    # assert isinstance(a0, float), ("Expected a0 to be a float in call to"
    #                                " integrate()")

    # assert isinstance(e0, float), ("Expected e0 to be a float in call to"
    #                                " integrate()")

    # assert 0 <= e0 <= 1, ("Expected e0 to be between 0 and 1 in call to"
    #                       " integrate()")

    # assert isinstance(beta, float), ("Expected beta to be a float in call to"
    #                                " integrate()")

    # h = t_eval[1] - t_eval[0]

    # tolh = etol

    # a, e = a0, e0

    # a_arr = []
    # e_arr = []

    # # Implement the RKF45 algorithm.
    # yk = np.array([a, e])

    # da_last = -np.infty

    # n = 0
    # t = 0

    # while t < t_eval[-1]:
    #     c1 = _coupled_eqs(t, yk, beta)

    #     # sometimes it "kicks up" again - period goes to 0, then suddenly
    #     # trns around. This is unphysical and thus we kill it if there's
    #     # a rapid change in da/dt's sign
    #     da = c1[0]

    #     if da > 0 and da_last < 0:
    #         break

    #     da_last = da

    #     # RKF45 integrator

    #     k1 = h * c1
    #     k2 = h * _coupled_eqs(t + 1/4 * h, yk + 1/4 * k1, beta)

    #     k3 = h * _coupled_eqs(t + 3/8 * h, yk + 3/32 * k1 \
    #                                          + 9/32 * k2, beta)

    #     k4 = h * _coupled_eqs(t+12/13 * h, yk + 1932/2197 * k1 \
    #                                          - 7200/2197 * k2 \
    #                                          + 7293/2197 * k3, beta)

    #     k5 = h * _coupled_eqs(t + h, yk + 439/216 * k1 \
    #                                    - 8*k2 \
    #                                    + 3680/513 * k3
    #                                    - 845/4104*k4, beta)

    #     k6 = h * _coupled_eqs(t + 1/2 * h, yk - 8/27*k1 \
    #                                          + 2*k2 \
    #                                          - 3544/2565*k3 \
    #                                          + 1859/4104 * k4 \
    #                                          - 11/40 * k5, beta)

    #     if yk[1] >= 1:# or yk[0] <= 1:
    #         # runaway integration, we should kill it
    #         # t_eval = (t_eval[0], t_eval[-1], len(e_arr))
    #         break

    #     if np.isnan(yk[1]) or np.isnan(yk[1]):
    #         # runaway integration
    #         break

    #     a_arr.append(yk[0])
    #     e_arr.append(yk[1])

    #     ykplus1 = (yk + 25/216 * k1
    #                   + 1408/2565 * k3
    #                   + 2197/4101 * k4
    #                   - 1/5 * k5)

    #     zkplus1 = (yk + 16/135 * k1
    #                   + 6656/12825*k3
    #                   + 28561/56430 * k4
    #                   - 9/50 *k5
    #                   + 22/5 * k6)

    #     s = (tolh / (2 * np.abs(zkplus1[0] - ykplus1[0]))) ** 0.25

    #     h = s*h

    #     yk = ykplus1

    #     t = t_eval[0] + n * h
    #     n += 1


    # A, E = np.array(a_arr), np.array(e_arr)

    # if len(A) == 0:
    #     A = np.array([a0])
    #     E = np.array([e0])

    # return A, E
