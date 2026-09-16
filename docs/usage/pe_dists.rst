Period-Eccentricity Distributions
====================================

As a binary star decays under gravitational-wave emission, its orbital
period shrinks and its eccentricity is damped towards zero (Peters 1964).
Tracking *where* a population of binaries sits in period-eccentricity
space, and *how that changes with time*, is what Takahe calls a
period-eccentricity distribution.

Because the distribution evolves continuously, Takahe represents it not as
a single 2D histogram but as a stack of them - one per timestep - forming a
period x eccentricity x time data cube. The building blocks for this are in
:mod:`takahe.frame` and :mod:`takahe.histogram`:

- :class:`takahe.histogram.histogram_2d` bins a single period-eccentricity
  snapshot.
- :class:`takahe.frame.Frame` wraps one such snapshot together with the
  time it corresponds to.
- :class:`takahe.frame.FrameCollection` is the cube itself: an ordered,
  iterable collection of :class:`~takahe.frame.Frame` objects sharing a
  common period/eccentricity extent, indexable by time via
  :meth:`~takahe.frame.FrameCollection.find`.

Building a cube
------------------

:func:`takahe.evolve.period_eccentricity` drives the whole process for an
ensemble of systems loaded via :mod:`takahe.load`: for every system in the
input data, it numerically evolves the binary with
:func:`takahe.evolve.evolve_system`, converts each integration step's
semimajor axis into a period via :func:`takahe.helpers.compute_period`, and
deposits the system's BPASS weight into the period-eccentricity bin, at the
frame, corresponding to that step's elapsed time:

.. code-block:: python

    import takahe

    df_block = takahe.load.from_directory('Datasets/MyData')
    Z = takahe.helpers.format_metallicity('z020')

    cube, = takahe.evolve.period_eccentricity(df_block[Z], Z)

Each :class:`~takahe.frame.Frame` in the resulting cube holds a 2D array of
accumulated weights, binned by :math:`\log_{10}(P)` (from -2 to 6, i.e.
periods from 0.01 to :math:`10^6` days) against eccentricity (0 to 1).

Reading a cube
-----------------

Once you have a cube, :meth:`~takahe.frame.FrameCollection.find` fetches the
frame nearest a given time (in Gyr):

.. code-block:: python

    frame = cube.find(t=1.5)  # the frame at/after 1.5 Gyr

and :meth:`~takahe.frame.FrameCollection.final_frame` fetches the last
frame, optionally accumulating every prior frame into it:

.. code-block:: python

    final = cube.final_frame(culmulative=True)

:meth:`~takahe.frame.FrameCollection.boundary` gives you the global minimum
and maximum bin values across every frame in the cube, which is useful for
picking consistent colour scales when plotting.

Visualising a cube
----------------------

A cube can be rendered as an animated GIF, one frame per timestep, with
:meth:`takahe.frame.FrameCollection.to_gif`:

.. code-block:: python

    cube.to_gif('output_directory/',
                outname='pe_evolution.gif',
                xlabel=r'$\log_{10}(P / \mathrm{days})$',
                ylabel=r'Eccentricity')

Persisting a cube
--------------------

Cubes can be expensive to compute, so :class:`~takahe.frame.FrameCollection`
supports round-tripping through a pickle file:

.. code-block:: python

    cube.save('my_cube.fr')

    # ... later, or in another script ...
    cube = takahe.frame.load('my_cube.fr')

Isocontours
--------------

Given a precomputed coalescence-time grid, Takahe can also extract
isocontours in period-eccentricity space directly - the loci of systems
that all merge in the same amount of time - via
:func:`takahe.evolve.constant_coalescence_isocontour`, which wraps
:func:`takahe.helpers.find_contours`.

.. note::

   The period-eccentricity machinery is one of the newer, more actively
   developed corners of Takahe. If you hit rough edges, please open an
   issue - see :doc:`../contributing`.
