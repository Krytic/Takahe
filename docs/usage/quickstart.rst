Quick Start Guide
=================

Getting started with Takahe is very straightforward: All you have to do is :code:`import takahe` to get going. Takahe has two main modules you will be interacting with: :code:`takahe.evolve` and :code:`takahe.event_rates`.

Event Rates
-----------

Takahe has a very simple recipe for computing event rates of systems:

.. code-block:: python

    import takahe
    dfs = takahe.load.from_directory('path/to/my/data')

    events = takahe.event_rates.composite_event_rates(dfs, extra_lt=None, transient_type='NSNS', SFRD_function=None):

    events.plot()

The :code:`composite_event_rates` function has a few parameters we can pass it:

1. :code:`extra_lt`: this is a callable which
