.. Takahe documentation master file, created by
   sphinx-quickstart on Thu May 28 14:53:28 2020.
   You can adapt this file completely to your liking, but it should at
   least contain the root `toctree` directive.

Takahe
======

.. image:: _static/TakaheLogo.png
   :align: center
   :alt: Takahe logo
   :width: 240px

.. image:: https://readthedocs.org/projects/takahe/badge/?version=latest
   :target: https://takahe.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

Takahe is a Python library for evolving binary star systems in time. Given a
population of binaries - such as one of the `BPASS
<https://bpass.auckland.ac.nz/>`_ model sets - Takahe integrates each system
forward through its orbital decay, tracks how its semimajor axis and
eccentricity evolve, and uses that to compute *when* and *how* systems merge.
That, in turn, lets you compute population-level quantities: merger event
rates as a function of redshift, the period-eccentricity distribution of a
population at a given time, and the star-formation-rate-weighted history
that drives all of it.

Concretely, Takahe helps you:

- Load an ensemble of binary star systems from BPASS-formatted data files
  (:doc:`usage/loading`).
- Numerically evolve individual binaries via gravitational-wave-driven
  orbital decay, using either a fast Julia integrator or a pure-Python
  fallback (:mod:`takahe.evolve`).
- Compute compact-binary merger event rates - per metallicity, or
  composited across an entire population - convolved with a star formation
  rate density (:mod:`takahe.event_rates`, :mod:`takahe.SFR`).
- Bin and manipulate the results with purpose-built 1D and 2D histogram
  classes that track Poissonian uncertainty alongside each bin
  (:mod:`takahe.histogram`).
- Build up period-eccentricity-time data cubes and animate them
  (:doc:`usage/pe_dists`, :mod:`takahe.frame`).

Takahe is developed by the `Auckland Stars Group
<https://github.com/UoA-Stars-And-Supernovae/>`_ at the University of Auckland, and is
released under the MIT License.

Getting started
----------------

New to Takahe? Start with :doc:`usage/installation` to get it set up, then
follow :doc:`usage/quickstart` to compute your first event rate.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   usage/installation
   usage/quickstart
   usage/loading
   usage/pe_dists

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   source/takahe

.. toctree::
   :maxdepth: 1
   :caption: Project

   contributing
   license

Acknowledgements
-----------------

This work makes use of v2.2 of the Binary Population and Spectral Synthesis
(BPASS) models as described in Eldridge, Stanway et al. (2017) and Stanway
& Eldridge et al. (2018).

Citing Takahe
--------------

A paper describing Takahe is forthcoming. In the meantime, please cite it
using the following BibTeX entry:

.. code-block:: bibtex

   @misc{takahe,
       title = {Takahe: Binary Star Systems with BPASS},
       author = {Richards, Sean},
       howpublished = {\url{https://github.com/krytic/takahe}},
       year = {2020}
   }

The name "Takahe"
-------------------

The name Takahe conforms to the naming convention employed by the BPASS releases, which are named after native
creatures from New Zealand - for example Tuatara (the current BPASS
version), named for the native reptile known as the "living fossil", and
Hoki, named for a fish. The takahē itself is a native, endangered,
flightless bird of New Zealand, regarded as taonga (treasure) to Ngāi Tahu,
an iwi (tribe) of the South Island.

Getting help
-------------

If you run into an issue, please open one on the `issue tracker
<https://github.com/Krytic/Takahe/issues>`_. There's no such thing as a
silly question - we were all beginners once, so don't hesitate to ask.

Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
