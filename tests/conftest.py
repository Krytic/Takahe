"""Shared pytest fixtures for the Takahe test suite."""
import gzip

import numpy as np
import pytest

import takahe


# One row of synthetic BPASS-style data:
# m1  m2  a0  e0  weight  evolution_age  rejuvenation_age
_SAMPLE_ROWS = [
    (1.4, 1.4, 10.0, 0.1, 12.5, 1.0e9, 0.5e9),
    (8.0, 1.4, 50.0, 0.3, 3.2, 2.0e9, 0.1e9),
]


def _rows_to_text(rows):
    return "\n".join(" ".join(str(v) for v in row) for row in rows) + "\n"


@pytest.fixture
def sample_dat_file(tmp_path):
    """Writes a small, valid takahe/BPASS-formatted .dat file and
    returns its path.
    """
    path = tmp_path / "Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat"
    path.write_text(_rows_to_text(_SAMPLE_ROWS))
    return path


@pytest.fixture
def sample_dat_gz_file(tmp_path):
    """Writes a small, valid gzip-compressed BPASS-formatted .dat.gz
    file and returns its path.
    """
    path = tmp_path / "Remnant-Birth-bin-imf135_300-z020_StandardJJ.dat.gz"
    with gzip.open(path, 'wt') as f:
        f.write(_rows_to_text(_SAMPLE_ROWS))
    return path


@pytest.fixture
def bpass_directory(tmp_path):
    """Builds a directory containing a (tiny, synthetic) data file for
    every BPASS metallicity takahe.load.from_directory() expects, so
    the loader can be exercised without needing a real BPASS dataset.
    """
    for Zi in takahe.constants.BPASS_METALLICITIES:
        fname = f"Remnant-Birth-bin-imf135_300-z{Zi}_StandardJJ.dat"
        (tmp_path / fname).write_text(_rows_to_text(_SAMPLE_ROWS))
    return tmp_path


@pytest.fixture
def small_histogram():
    """A 5-bin histogram spanning [0, 10), useful as a starting point
    for tests that don't care about the specific binning.
    """
    return takahe.histogram.histogram(xlow=0, xup=10, nr_bins=5)


@pytest.fixture
def small_histogram_2d():
    """A 2D histogram built from explicit bin edges: 5 edges on x
    (spanning [0, 10]) and 4 edges on y (spanning [0, 3]).

    Built from explicit edges rather than (range, nr_bins) because
    histogram_2d.__init__() sets nr_bins_x = len(edges_x) and
    nr_bins_y = len(edges_y) when edges are supplied directly - i.e.
    it stores one row/column per *edge*, not per interval between
    edges (unlike the 1D histogram class, which does subtract one).
    Passing edges directly sidesteps that ambiguity and pins the
    resulting _values/_num_hits shape to exactly (5, 4), which the
    tests rely on.
    """
    return takahe.histogram.histogram_2d(
        edges_x=np.array([0, 2.5, 5, 7.5, 10]),
        edges_y=np.array([0, 1, 2, 3]))
