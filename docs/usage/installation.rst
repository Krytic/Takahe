Installing Takahe
=================

Takahe is hosted on PyPI (link here), though you can also build source from GitHub.

From Source
-----------

Building from source is straightforward:

1. Fork the `Github Repo <https://github.com/Krytic/takahe>`_
2. Locally, run :code:`git clone <your url>`
3. In a terminal, navigate to the directory you installed takahe to
4. Run :code:`pip install -e .` to install it as an editable PIP file
5. Run :code:`f2py3.7 -c src/merge_rate.f95 -m merge_rate` to compile the event rate FORTRAN code.

From PyPI
---------

Takahe is currently *not* on PyPI (yet), but when it is released you will be able to use :code:`pip install takahe` to install it.