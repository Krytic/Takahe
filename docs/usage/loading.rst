.. _loading:

Loading Data
============

Takahe presumes a very strict naming convention when it comes to loading files. The file names contain valuable metadata; a `current project <https://github.com/Krytic/Takahe/issues/10>`_ is refactoring the loader to allow for this.

Takahe's naming convention is :code:`Remnant-Birth-bin-imf135_300-z{metallicity}_StandardJJ.dat` and it presumes that the files are structured as follows:

+----+----+----+----+--------+---------------+------------------+
| m1 | m2 | a0 | e0 | weight | evolution_age | rejuvenation_age |
+----+----+----+----+--------+---------------+------------------+

Finally, if there is a field :code:`_ct` in the filename, Takahe assumes the final field is the coalescence time of the system.

On Metallicity
--------------

It's worth mentioning that Takahe, like BPASS, assumes that solar metallicity is 0.020, and generally measures metallicity as relative to this. So Z = 0.010 represents half-solar metallicity. The metallicity field in the filename can be understood as:

+------+---------+--------------------+
|      |   Z     |  :math:`Z/Z_\odot` |
+------+---------+--------------------+
| zem5 | 0.00001 |       0.0005       |
+------+---------+--------------------+
| zem4 | 0.0001  |        0.005       |
+------+---------+--------------------+
| z001 | 0.001   |        0.05        |
+------+---------+--------------------+
| z002 | 0.002   |        0.10        |
+------+---------+--------------------+
| z003 | 0.003   |        0.15        |
+------+---------+--------------------+
| z004 | 0.004   |        0.20        |
+------+---------+--------------------+
| z006 | 0.006   |        0.30        |
+------+---------+--------------------+
| z008 | 0.008   |        0.40        |
+------+---------+--------------------+
| z010 | 0.010   |        0.50        |
+------+---------+--------------------+
| z014 | 0.014   |        0.70        |
+------+---------+--------------------+
| z020 | 0.020   |        1.00        |
+------+---------+--------------------+
| z030 | 0.030   |        1.50        |
+------+---------+--------------------+
| z040 | 0.040   |        2.00        |
+------+---------+--------------------+

Loading Directories
-------------------

Takahe provides a method to blanket-load an entire directory: :code:`takahe.load.from_directory`. This takes one argument, the directory you want to load.

For instance, the following code loads from a directory :code:`data`:

.. code-block:: python

    import takahe
    dfs = takahe.load.from_directory("../data")

:code:`dfs` is indexed by *relative BPASS metallicity* as a string. For instance, to access the dataframe corresponding to :math:`0.7Z_\odot` (i.e., the :code:`z014` file), we write:

.. code-block:: python

    df = dfs['0.7']
