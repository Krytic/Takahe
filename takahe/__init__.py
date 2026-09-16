import pkgutil

import glisten

import takahe.histogram as histogram
import takahe.constants as constants
import takahe.helpers as helpers
import takahe.load as load
import takahe.event_rates as event_rates
import takahe.evolve as evolve
import takahe.exceptions as exceptions
import takahe.SFR as SFR
import takahe.frame as frame

from takahe._metadata import __version__

import numpy as np
from numba import njit, guvectorize, vectorize
from tqdm import tqdm

from inspect import getfullargspec

log = glisten.log.Logger('.log')


class Function(object):
    """Function is a wrap over standard python function."""

    def __init__(self, fn):
        """Wraps a function so it can be registered in the Namespace.

        Arguments:
            fn {callable} -- The function to wrap.
        """
        self.fn = fn

    def __call__(self, *args, **kwargs):
        """Overriding the __call__ function which makes the
        instance callable.
        """
        # fetching the function to be invoked from the virtual namespace
        # through the arguments.
        fn = Namespace.get_instance().get(self.fn, *args)
        if not fn:
            raise Exception("no matching function found.")

        # invoking the wrapped function and returning the value.
        return fn(*args, **kwargs)

    def key(self, args=None):
        """Returns the key that will uniquely identify
        a function (even when it is overloaded).
        """
        # if args not specified, extract the arguments from the
        # function definition
        if args is None:
            args = getfullargspec(self.fn).args

        return tuple([
          self.fn.__module__,
          self.fn.__class__,
          self.fn.__name__,
          len(args or []),
        ])


class Namespace(object):
    """Namespace is the singleton class that is responsible for holding
    all the functions."""

    __instance = None

    def __init__(self):
        """Creates the singleton Namespace instance.

        Raises:
            Exception -- if a Namespace has already been instantiated.
        """
        if self.__instance is None:
            self.function_map = dict()
            Namespace.__instance = self
        else:
            raise Exception("cannot instantiate a virtual Namespace again")

    @staticmethod
    def get_instance():
        """Fetches the singleton Namespace instance.

        Creates the Namespace if it does not already exist.

        Returns:
            {Namespace} -- The singleton Namespace instance.
        """
        if Namespace.__instance is None:
            Namespace()
        return Namespace.__instance

    def register(self, fn):
        """registers the function in the virtual namespace and returns
        an instance of callable Function that wraps the
        function fn.
        """
        func = Function(fn)
        self.function_map[func.key()] = fn
        return func

    def get(self, fn, *args):
        """get returns the matching function from the virtual namespace.

        return None if it did not fund any matching function.
        """
        func = Function(fn)
        return self.function_map.get(func.key(args=args))


def overload(fn):
    """overload is the decorator that wraps the function
    and returns a callable object of type Function.
    """
    return Namespace.get_instance().register(fn)


def debug(msgtype, message, fatal=True):
    """General purpose debug message handler

    Allows us to print to stdout when debugging (developing) and fail
    on production.

    Arguments:
        msgtype {string} -- The message type to throw. Must be 'info', 'warning', or 'error'.

        message {string} -- The message to throw.

    Keyword Arguments:
        fatal {bool} -- Whether or not the message should be a fatal
                        error. Ignored if takahe.constants.DEBUG_MODE
                        is True. (default: {True})

    Raises:
        takahe.TakaheWarning    -- A warning type if we are not in debug
                                   mode and the error should be fatal.
        takahe.TakaheFatalError -- An error type if we are not in debug
                                   mode and the error should be fatal.
    """
    types = log.types()

    if msgtype not in types:
        debug('error', (f'Message type {msgtype} is not recognised.\n'
                        f'Valid types are {"/".join(types)}'))
    else:
        if fatal not in [True, False]:
            debug('error', 'fatal must be True or False.')

        if constants.DEBUG_MODE:
            func = getattr(log, msgtype)
            func(message)
        else:
            if msgtype == 'warning':
                if fatal:
                    raise takahe.exceptions.TakaheWarning(message)
            elif msgtype == 'error':
                if fatal:
                    raise takahe.exceptions.TakaheFatalError(message)
            else:
                func = getattr(log, msgtype)
                func(message)


integrator_initialized = True


def initialize_integrator():
    """Retained for backwards compatibility with older scripts.

    The Julia integrator no longer requires explicit initialization,
    so this function is now a no-op that just emits a deprecation
    notice via takahe.debug().
    """
    debug('info', ("The Julia integrator is broken and has been deprecated. "
                   "The integrator no longer needs to be initialized, you can "
                   "simply call the integration functions."))


# debug('info', "This is Takahe v" + __version__)

# @overload
# def integrate_eoms(m1, m2, p0, e0, evotime):
#     a0 = helpers.compute_separation(p0, m1, m2)
#     return integrate_eoms(a0, e0, [m1, m2, 0, 0, evotime])


def integrate_eoms(a0, e0, p):
    """
    General purpose integrator for Nyadzani & Razzaque eqns for a & e.

    Params:
        a0 - The initial semimajor axis, measured in solar radii
        e0 - The initial eccentricity, dimensionless
        p  - A vector of parameters:
                 p[1] = m1 (units: Solar Mass)
                 p[2] = m2 (units: Solar Mass)
                 p[3] = unused (will be removed in next version)
                 p[4] = unused (will be removed in next version)
                 p[5] = Lifetime (evolution + rejuvenation)

    Returns:
        A  - An array of the semimajor axes of the binary system over time. (Solar Mass)
        E  - An array of the eccentricities of the binary system over time. (no dim.)
    """

    Solar_Mass = 1.989e30       # kg
    Solar_Radius = 696340000.0  # m
    G = 6.67e-11                # m^3 kg^-1 s^-2
    c = 299792458.0             # m/s

    MAX_ATTEMPTS = 150_000

    A = np.array([])
    E = np.array([])
    H = np.array([])

    m1, m2, _, _, evotime = p[0], p[1], p[2], p[3], p[4]

    # number of seconds in a year
    seconds_per_year = 60 * 60 * 24 * 365.25

    ########################
    #      Unit Check      #
    ########################
    a = a0 * Solar_Radius  # Meters
    e = e0                 # Dimensionless
    m1 = m1 * Solar_Mass   # Kilogram
    m2 = m2 * Solar_Mass   # Kilogram
    ########################
    #    End Unit Check    #
    ########################

    # Beta has units m^4 / s
    beta = ((64/5) * G**3 * m1 * m2 * (m1 + m2) / (c**5))

    A = np.append(A, a)
    E = np.append(E, e)
    H = np.append(H, 0.0)

    total_time = 0
    attempts = 0

    # Integrate until past the end of the universe, or a 10km orbit
    with tqdm(total=MAX_ATTEMPTS) as pbar:
        while total_time/seconds_per_year + evotime < 1e11 and a > 1e4 and attempts < MAX_ATTEMPTS:
            # an euler integrator: work out da/dt then times it by dt
            # to get da, which then we can work out as a = a + da/dt * dt.
            initial_da = (- beta / ((a**3) * (1 - e**2)**(7/2)))
            da = initial_da * (1 + (73/24) * e**2 + (37/96) * e**4)

            intial_de = (((-19/12) * beta) / (a**4*(1-e**2)**(5/2)))
            de = intial_de * (e + (121/304) * e**3)
            # Units: s^-1

            timeA = abs(1e-2 * a/da)

            if e > 1e-10:
                timeE = abs(1e-2 * e/de)
            else:
                de = 0
                e = 1e-10
                timeE = timeA * 10

            # maximum timestep is half of the width of the smallest BPASS time bin
            conv_frac = 0.23076752*0.5*seconds_per_year

            dt2 = (evotime + total_time/seconds_per_year)*conv_frac

            # Take a timestep that results in the smallest change: either a
            # change in E, a change in A, 1/2 the smallest BPASS bin.
            dt = min(timeE, timeA, dt2)

            a = a + dt * da
            e = e + dt * de

            A = np.append(A, a)
            E = np.append(E, e)
            H = np.append(H, dt)

            total_time = total_time + dt

            attempts += 1
            pbar.update()

    # Determine why we stopped, so the parent function can tell what went
    # "wrong".
    stop_reason = "flag_not_set"

    if total_time/seconds_per_year + evotime >= 1e11:
        stop_reason = "out_of_time"

    if a <= 1e4:
        stop_reason = "merged"

    if attempts >= MAX_ATTEMPTS:
        stop_reason = 'max_attempts_reached'

    # Solar Radii, Dimensionless
    return A / Solar_Radius, E, H, stop_reason


@vectorize('float64(float64,float64,float64,float64,float64)')
def integrate_timescale(m1, m2, p0, e0, N):
    """
    General purpose integrator for eqn(5.14) of Peters, 1964 [1].

    [1] https://ui.adsabs.harvard.edu/abs/1964PhRv..136.1224P/abstract

    Params:
        m1 -- The mass of the first star  (Solar Mass)
        m2 -- The mass of the second star (Solar Mass)
        p0 -- The initial period          (days)
        e0 -- The initial eccentricity

    Returns:
        tC -- The coalescence time        (seconds)
    """
    Solar_Mass = 1.989e30       # kg
    Solar_Radius = 696340000.0  # m
    G = 6.67e-11                # m^3 kg^-1 s^-2
    c = 299792458.0             # m/s

    # m1 = m1.astype(np.float64)
    # m2 = m2.astype(np.float64)
    # p0 = p0.astype(np.float64)
    # e0 = e0.astype(np.float64)

    ##########################################
    #            BEGIN UNIT CHECK            #
    ##########################################
    m1 = m1 * Solar_Mass    # Kilograms     ##
    m2 = m2 * Solar_Mass    # Kilograms     ##
    p0 = p0 * 24 * 60 * 60  # Seconds       ##
    e0 = e0                 # Dimensionless ##
    ##########################################
    #             END UNIT CHECK             #
    ##########################################

    # Use Kepler's Third Law to compute the semimajor axis, in meters
    a0 = (p0**2.0 * (G * (m1+m2)) / (4.0 * np.pi**2.0))**(1.0/3.0)  # meters

    # Compute the constant beta
    beta = 64.0 / 5.0 * G**3.0 * m1 * m2 * (m1 + m2) / c**5.0
    # meters^4 / s

    # Circular binary coalescence time
    tC = a0**4.0 / (4.0*beta)

    if e0 != 0.0:
        # compute the constant c0
        c0 = a0 * (1.0-e0**2.0) * e0**(-12.0/19.0) * (1.0+(121.0 / 304.0 * e0**2.0))**(-870.0/2299.0)  # meters
        if e0 < 0.01:
            # Low ecc - see eqn after eqn(5.14) of Peters, 1964
            tC = c0**4.0 * e0**(48.0/19.0) / (4.0*beta)
        elif e0 > 0.99:
            # High ecc - see eqn after eqn after eqn(5.14) of Peters, 1964.
            tC = tC * ((768.0 / 425.0) * (1.0-e0**2.0)**3.5)
        else:
            # Medium ecc - see eqn(5.14) of Peters, 1964
            e = 0.0
            de = e0 / N
            summand = 0.0

            while e < e0:
                this_integral = de * e**(29.0/19.0)
                this_integral *= (1.0 + (121.0/304.0) * e**2.0)**(1181.0/2299.0)
                this_integral /= (1.0 - e**2.0)**(1.5)

                summand = summand + this_integral
                e += de

                e = np.float64(e)

            tC = (12.0/19.0) * (c0**4.0 / beta) * summand

    return tC
