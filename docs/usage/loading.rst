.. _loading:

Loading Data
============

Takahe's loader attempts to infer the right headers by sniffing the filename. As such, to receive the best benefits of using the loader, your files should conform to a certain filename structure.

Firstly, Takahe splits filenames based on the underscore character _. We call each part a "field" of the filename. If it sniffs a field beginning with the letter z, Takahe assumes that this field designates the metallicity of the sample.

If Takahe detects "StandardJJ" in the filename, it presumes you are using J.J. Eldridge's StandardJJ structure, where the file is structured as follows:

+----+----+----+----+--------+---------------+------------------+
| m1 | m2 | a0 | e0 | weight | evolution_age | rejuvenation_age |
+----+----+----+----+--------+---------------+------------------+

You cannot currently override the metallicity detection, but you *can* override the inference on the fields -- pass the dictionary :code:`name_hints` to the loader. Passing this will *always* override the inference.

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

Let's examine how Takahe analyses the file :code:`data/Remnant-Birth-bin-imf135_300-z001_StandardJJ_ct.dat`:

1. We strip the file extension and containing directories: :code:`Remnant-Birth-bin-imf135_300-z001_StandardJJ_ct`
2. We split the file from the right based on _: :code:`Remnant-Birth-bin-imf135_300-z001_StandardJJ`, :code:`ct`
3. We split the first part of the filename based on -: :code:`Remnant`, :code:`Birth`, :code:`bin`, :code:`imf135_300`, :code:`z001_StandardJJ`, :code:`ct`
4. We check each field and identify relevant parts: :code:`z001`, :code:`StandardJJ`, :code:`ct`.
5. We confirm that this represents a 5% solar metallicity file, with headers as given by the StandardJJ prescription, containing coalescence times for the ensemble.

Naming Convention
-----------------

Takahe employs a naming convention for data files. In future this will be customisable, but for now it is not.

Files in a data directory must be named according to the convention :code:`Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat`. Optionally, one may suffix this with :code:`.gz`, and one may customise the metallicity away from :code:`z020`. One may also add :code:`_ct` after :code:`StandardJJ` to indicate the presence of an eigth column indicating the coalescence time.