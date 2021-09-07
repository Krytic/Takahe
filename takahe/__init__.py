"""
Takahe is a part of my Master of Science thesis at the University of
Auckland, and is in constant, rolling development.

Sample Usage:
>>> import takahe
>>> data = takahe.load.from_directory("../path/to/data.dat")
>>> events = takahe.event_rates.composite_event_rates(data)
>>> events.plot()

"""
import pkgutil

from julia.api import Julia
jl = Julia(compiled_modules=False)

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
    assert msgtype in ['warning', 'error', 'info'], ("Message type "
                                                     f"\"{msgtype}\" is "
                                                     "not recognised.")

    assert fatal == True or fatal == False, "fatal must be a boolean type."

    if constants.DEBUG_MODE:
        if msgtype == 'warning':
            header = "\033[93m\033[1m[WARNING]\033[0m "
        elif msgtype == 'error':
            header = "\033[91m\033[1m[ERROR]\033[0m "
        elif msgtype == 'info':
            header = "\033[96m\033[1m[INFO]\033[0m "

        print(header + str(message))
    else:
        if msgtype == 'warning':
            if fatal:
                raise takahe.exceptions.TakaheWarning(message)
        elif msgtype == 'error':
            if fatal:
                raise takahe.exceptions.TakaheFatalError(message)
        elif msgtype == 'info':
            print(f"\033[96m\033[1m[INFO]\033[0m {message}")

_integration_subroutine = pkgutil.get_data(__name__, "../src/integrator.jl")
_integration_subroutine = _integration_subroutine.decode("utf-8")
integrate_eoms = jl.eval(_integration_subroutine)

_peters_integrator = pkgutil.get_data(__name__, "../src/integrator_peters.jl")
_peters_integrator = _peters_integrator.decode("utf-8")
integrate_timescale = jl.eval(_peters_integrator)

debug('info', "This is Takahe v" + __version__)
