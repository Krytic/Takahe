Quick Start Guide: Event Rates
==============================

Getting started with Takahe is very straightforward: All you have to do is :code:`import takahe` to get going. Make sure you also have a sample dataset. Takahe assumes that all datasets follow this prescription (a more intelligent loader is being brainstormed):

+----+----+----+----+--------+---------------+------------------+
| m1 | m2 | a0 | e0 | weight | evolution_age | rejuvenation_age |
+----+----+----+----+--------+---------------+------------------+

The simplest way to load in a block of data is to point Takahe towards a directory containing data files. To do so, ensure your directory contains files matching the following naming convention:

- :code:`Remnant-Birth-bin-imf135_300-z{Zi}_StandardJJ.dat`

where :code:`{Zi}` is a BPASS-formatted metallicity (e.g., :code:`020` for solar). This directory must contain files corresponding to each of the 13 BPASS metallicities - Takahe does not fail silently and will complain if files are missing.

To load in your files, run :code:`takahe.load.from_directory(your_path)` where :code:`your_path` is a string pointing to your directory.

This returns a dictionary of Pandas dataframes corresponding to the files. It is indexed by a Takahe-formatted metallicity. Thus, to access the dataframe corresponding to the 0.7 solar metallicity file, one should use::

    df_block = takahe.load.from_directory('Datasets/MyData')
    Z = takahe.helpers.format_metallicity('z014')
    df = df_block[Z]

Now, let's compute the event rate of a given sample. Having run the above code, we can now use::

    event_rate = takahe.event_rates.composite_event_rates(df_block)
    event_rate.plot()

Calling :code:`event_rate.plot()` is a proxy for calling a (modified) version of matplotlib's :code:`plt.plot()` method - so if you want to do anything fancy you may need to run your code before :code:`event_rate.plot()`.