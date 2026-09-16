Contributing to Takahe
=========================

Takahe is an open-source project released under the :doc:`MIT License
<license>`. That means contributions are welcome, so long as they follow a
few simple guidelines.

Direct code contributions
---------------------------

Issues
^^^^^^^

When making an issue, please detail what you are trying to do (intended
behaviour), what you are getting (main issue), and what you expect to see
(expected output). Be explicit, and where possible provide steps to
reproduce the issue. Even better is to link your code, either in another
GitHub repo or on `Hastebin <https://hastebin.com>`_ or similar.

If you're commenting on an issue, please:

- Be respectful!
- Remember, not everybody possesses the same knowledge as you. Don't
  overcomplicate things, and if someone asks you to explain something,
  don't brush them off. **Just do it.**

Pull requests
^^^^^^^^^^^^^^^

Please don't work directly on ``master``. Master should be for stable code,
but not necessarily feature-complete versions of the whole codebase. For
instance:

- If you are working on a bugfix, please do so on a **dedicated branch**.
- If you are working on a feature, please do so on a **dedicated branch**.
- If your feature is complete, please **open a pull request into master**.

If you include a new feature, please, please, *please* write some docs
about it!

Indirect code contributions
------------------------------

Documentation
^^^^^^^^^^^^^^^

Takahe's documentation is written in reStructuredText and published on
`Read the Docs <https://takahe.readthedocs.io/en/latest/>`_ using Sphinx.
It lives in this repository's ``docs`` directory, alongside the code, and
you're more than welcome to improve it - you're heavily encouraged to, if
you're adding a new feature.

The API reference (:doc:`source/takahe`) is generated automatically from
docstrings in the source via Sphinx's autodoc and Napoleon extensions, so
the most direct way to improve it is to improve a function's or class's
docstring. Takahe's docstrings follow a Google-style convention:

.. code-block:: python

   def compute_period(a, M, m):
       """Computes the period of a BSS.

       Uses Kepler's third law to compute the period, in days.

       Arguments:
           a {float} -- The SMA of the BSS (in solar radii)
           M {float} -- The mass of the primary star (in solar masses)
           m {float} -- The mass of the secondary star (in solar masses)

       Returns:
           {float} -- The period in days
       """

Everything under ``docs/usage`` is hand-written prose describing *how* and
*why* to use Takahe, rather than an exhaustive listing of every function -
that's what the API reference is for. In general, documentation should:

- be friendly,
- be approachable,
- focus on technical details where necessary, but maintain an
  easy-to-approach nature overall.

Building the docs locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the repository root:

.. code-block:: console

   $ pip install -r docs/requirements.txt
   $ sphinx-build -b html docs docs/_build/html

Then open ``docs/_build/html/index.html`` in a browser.
