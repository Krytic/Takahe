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

from julia import Main as jl
import takahe.histogram as histogram
import takahe.constants as constants
import takahe.helpers as helpers
import takahe.load as load
import takahe.event_rates as event_rates
import takahe.evolve as evolve
import takahe.SFR as SFR
import takahe.frame as frame

from takahe._metadata import __version__

def debug(msgtype, message):
    assert msgtype in ['warning', 'error', 'info']

    if msgtype == 'warning':
        header = "\033[93m\033[1m[WARNING]\033[0m "
    elif msgtype == 'error':
        header = "\033[91m\033[1m[ERROR]\033[0m "
    elif msgtype == 'info':
        header = "\033[96m\033[1m[INFO]\033[0m "

    if constants.DEBUG_MODE:
        print(header + str(message))

_integration_subroutine = pkgutil.get_data(__name__, "../src/integrator.jl")
_integration_subroutine = _integration_subroutine.decode("utf-8")
integrate_eoms = jl.eval(_integration_subroutine)

_peters_integrator = pkgutil.get_data(__name__, "../src/integrator_peters.jl")
_peters_integrator = _peters_integrator.decode("utf-8")
integrate_timescale = jl.eval(_peters_integrator)

debug('info', "This is Takahe v" + __version__)
