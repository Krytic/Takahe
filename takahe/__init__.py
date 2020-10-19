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
import takahe.constants as constantsw
import takahe.helpers as helpers
import takahe.load as load
import takahe.event_rates as event_rates
import takahe.evolve as evolve
import takahe.SFR as SFR

_integration_subroutine = pkgutil.get_data(__name__, "../src/integrator.jl")
_integration_subroutine = _integration_subroutine.decode("utf-8")
integrate_eoms = jl.eval(_integration_subroutine)

_peters_integrator = pkgutil.get_data(__name__, "../src/integrator_peters.jl")
_peters_integrator = _peters_integrator.decode("utf-8")
integrate_timescale = jl.eval(_peters_integrator)
