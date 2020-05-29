Quick Start Guide
=================

Getting started with Takahe is very straightforward: All you have to do is :code:`import takahe` to get going. From there, the three main classes you will encounter are:

1. :code:`takahe.BSS`,
2. :code:`takahe.ensemble`, and
3. :code:`takahe.universe`.

To create any of the three classes, just call :code:`.create()` on the relevant object (with the right parameters):

.. code-block:: python
  :linenos:
  
  import takahe
  my_universe = takahe.universe.create('eds')

Each class represents a different level of abstraction for the system. Most simulations will start by creating a Universe, and propagating individual BSS objects in the ensemble through time.

Simulating the Universe
-----------------------

Takahe supports four different models of Universe:

+--------------+----------------------+------------------+------------------------+------------------+--------------+
| Model        | Long Name            | :math:`\Omega_M` | :math:`\Omega_\Lambda` | :math:`\Omega_k` | :math:`H_0`  |
+==============+======================+==================+========================+==================+==============+
| eds          | Einstein-de Sitter   | 1                | 0                      |                0 | customisable |
+--------------+----------------------+------------------+------------------------+------------------+--------------+
| lowdensity   | Low matter density   | 0.05             | 0                      |             0.95 | customisable |
+--------------+----------------------+------------------+------------------------+------------------+--------------+
| highlambda   | High :math:`\Lambda` | 0.2              | 0.8                    |                0 | customisable |
+--------------+----------------------+------------------+------------------------+------------------+--------------+
| lcdm         | Einstein-de Sitter   | 0.3              | 0.7                    |                0 | 70           |
+--------------+----------------------+------------------+------------------------+------------------+--------------+

To create a :math:`\Lambda\text{CDM}` universe, therefore, we can just go:

.. code-block:: python
  :linenos:

  import matplotlib.pyplot as plt
  import takahe
  my_universe = takahe.universe.create('lcdm')

Populating the Universe is also very easy! Just call :code:`.populate()` with the name of the file (and header-hints if you have them):

.. code-block:: python
  :linenos:
  :lineno-start: 4

  my_universe.populate("data/mydatafile.dat")

Takahe sniffs the filename you provide and tries to load the data correctly, though this can all be overridden. Namely, if the filename contains :code:`StandardJJ`, Takahe will assume you are using the StandardJJ prescription, and that the datafile is structured like this:

+----+----+----+----+--------+---------------+------------------+
| m1 | m2 | a0 | e0 | weight | evolution_age | rejuvenation_age |
+----+----+----+----+--------+---------------+------------------+

The full indiosyncracies of the loader are explaineed in :doc:`loading`.

We can now compute and plot the event rate of our universe over its history very easily:

.. code-block:: python
  :linenos:
  :lineno-start: 5

  my_universe.event_rate()
  plt.show()

Which will give you an event rate histogram similar to the one below:

<Todo: insert image>

The full code of this example is below:

.. code-block:: python
  :linenos:

  import matplotlib.pyplot as plt
  import takahe
  my_universe = takahe.universe.create('lcdm')

  my_universe.populate("data/mydatafile.dat")

  my_universe.event_rate()
  plt.show()

Simulating an Ensemble
----------------------

Simulating a single system
--------------------------