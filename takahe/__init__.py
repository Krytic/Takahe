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

log = glisten.log.Logger('.log')

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

error_message = """Integrator not initialized!
Call takahe.initialize_integrator() first."""

integrate_eoms      = lambda: debug("error", error_message)
integrate_timescale = lambda: debug("error", error_message)

integrator_initialized = False

def initialize_integrator():
    global integrate_timescale, integrate_eoms, integrator_initialized

    from julia.api import Julia
    jl = Julia(compiled_modules=False)

    _integration_subroutine = pkgutil.get_data(__name__, "../src/integrator.jl")
    _integration_subroutine = _integration_subroutine.decode("utf-8")
    integrate_eoms = jl.eval(_integration_subroutine)

    _peters_integrator = pkgutil.get_data(__name__, "../src/integrator_peters.jl")
    _peters_integrator = _peters_integrator.decode("utf-8")
    integrate_timescale = jl.eval(_peters_integrator)

    integrator_initialized = True

debug('info', "This is Takahe v" + __version__)
