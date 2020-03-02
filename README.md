# Takahe

Takahe is an extension to Heloise Stevance's [hoki code](http://github.com/HeloiseS/hoki), which interfaces with [BPASS](bpass.auckland.ac.nz). It is designed to pick up where hoki leaves off, and allow the user to specify binary star systems from the BPASS library, and propagate those in time to investigate binary star mergers.

## Installation

Installation is handled via pip. Currently the package is not released on pypi but can be installed with `pip install -e .` in the root directory.

## Usage

One installed, you have a couple of options as to creating a binary star system. You can create one the "long way" using the signature `BinaryStarSystem(primary_mass, secondary_mass, a0, e0)`. Alternatively, if you wish to use the BPASS database, you can specify the filename through `BinaryStarSystemLoader(bpass_from)`. Note that `bpass_from` is more or less passed directly to hoki's `load` function -- so what you pass to it must be compatible in and of itself.

## Contributing

Contributions are welcome! Master is considered to be bleeding-edge, and stable releases are tagged.

## Miscellanea

### Citing

A paper will be forthcoming in the future. In the meantime please use the following BibTeX entry:

	@misc{takahe,
		title = {Takahe: Binary Star Systems with BPASS},
		author = {Richards, Sean},
		howpublished = {\url{https://github.com/krytic/takahe}},
		year = {2020}
	}

### The name "Takahe"

The name Takahe conforms to the naming convention employed by Eldridge, Stanaway, Stevance, et al. BPASS releases are named after native creatures from New Zealand (e.g. Tuatara, the current version, is named for the native reptile known as the "living fossil"), as is hoki (being named for a fish). The Takahe is a native endangered flightless bird to New Zealand.