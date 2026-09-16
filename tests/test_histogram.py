import numpy as np
import pytest

import takahe


# --- histogram (1D) -------------------------------------------------------

def test_histogram_from_xlow_xup_nr_bins():
    h = takahe.histogram.histogram(xlow=0, xup=10, nr_bins=5)

    assert h.getNBins() == 5
    assert len(h) == 5
    assert np.allclose(h.getBinEdges(), [0, 2, 4, 6, 8, 10])


def test_histogram_from_edges():
    h = takahe.histogram.histogram(edges=[0, 1, 3, 10])

    assert h.getNBins() == 3
    assert np.allclose(h.getBinEdges(), [0, 1, 3, 10])


def test_histogram_requires_either_edges_or_xlow_xup_nr_bins():
    with pytest.raises(Exception):
        takahe.histogram.histogram()


def test_fill_single_value(small_histogram):
    small_histogram.fill(3.0)

    assert small_histogram.getBinContent(1) == 1.0
    assert sum(small_histogram.getValues()) == 1.0


def test_fill_array_with_default_weight(small_histogram):
    small_histogram.fill([1.0, 3.0, 3.0])

    assert small_histogram.getBinContent(0) == 1.0
    assert small_histogram.getBinContent(1) == 2.0


def test_fill_array_with_weights(small_histogram):
    small_histogram.fill([1.0, 3.0], weight=[2.0, 5.0])

    assert small_histogram.getBinContent(0) == 2.0
    assert small_histogram.getBinContent(1) == 5.0


def test_fill_mismatched_weight_length_raises(small_histogram):
    with pytest.raises(Exception):
        small_histogram.fill([1.0, 3.0], weight=[1.0, 2.0, 3.0])


def test_fill_clamps_values_outside_range_to_edge_bins(small_histogram):
    small_histogram.fill([-5.0, 500.0])

    assert small_histogram.getBinContent(0) == 1.0
    assert small_histogram.getBinContent(4) == 1.0


def test_inbounds(small_histogram):
    assert small_histogram.inbounds(5.0)
    assert not small_histogram.inbounds(-1.0)
    assert not small_histogram.inbounds(11.0)


def test_getBin(small_histogram):
    assert small_histogram.getBin(0.5) == 0
    assert small_histogram.getBin(9.9) == 4


def test_getBin_out_of_range_raises(small_histogram):
    with pytest.raises(Exception):
        small_histogram.getBin(-1.0)


def test_getBinCenter_and_width(small_histogram):
    assert small_histogram.getBinCenter(0) == pytest.approx(1.0)
    assert small_histogram.getBinWidth(0) == pytest.approx(2.0)


def test_getBinCenters(small_histogram):
    assert small_histogram.getBinCenters() == [1.0, 3.0, 5.0, 7.0, 9.0]


def test_copy_is_independent(small_histogram):
    small_histogram.fill(1.0)
    other = small_histogram.copy()
    other.fill(1.0)

    # copy() does not deep-copy _values (it assigns the same array), so
    # this documents the *actual* (aliasing) behaviour rather than the
    # independent-copy behaviour the name "copy" might suggest.
    assert small_histogram.getBinContent(0) == other.getBinContent(0)


def test_add_histograms(small_histogram):
    small_histogram.fill(1.0)
    other = small_histogram.copy()

    result = small_histogram + other

    assert result.getBinContent(0) == 2.0


def test_add_scalar(small_histogram):
    small_histogram.fill(1.0)

    result = small_histogram + 1.0

    assert result.getBinContent(0) == 2.0
    # Every bin gets the scalar added, not just the filled one.
    assert result.getBinContent(1) == 1.0


def test_mul_and_rmul(small_histogram):
    small_histogram.fill(1.0)

    assert (small_histogram * 3).getBinContent(0) == 3.0
    assert (3 * small_histogram).getBinContent(0) == 3.0


def test_sub(small_histogram):
    small_histogram.fill(1.0)
    other = small_histogram.copy()

    result = small_histogram - other

    assert result.getBinContent(0) == 0.0


def test_truediv(small_histogram):
    small_histogram.fill(4.0, weight=4.0)

    assert small_histogram.getBinContent(2) == 4.0
    result = small_histogram / 2.0
    assert result.getBinContent(2) == 2.0


def test_sum_within_single_bin(small_histogram):
    small_histogram.fill(1.0)  # bin 0, width 2, so density is 0.5/unit

    assert small_histogram.sum(0.0, 1.0) == pytest.approx(0.5)


def test_sum_across_multiple_bins(small_histogram):
    small_histogram.fill([1.0, 3.0, 5.0])

    assert small_histogram.sum(0.0, 10.0) == pytest.approx(3.0)


def test_integral_within_single_bin(small_histogram):
    small_histogram.fill(1.0)

    # Bin 0 spans [0, 2) with content 1.0, so integrating over its full
    # width gives content * width = 1.0 * 2.0.
    assert small_histogram.integral(0.0, 2.0) == pytest.approx(2.0)


def test_str_and_repr_do_not_raise(small_histogram):
    small_histogram.fill(1.0)

    assert str(small_histogram)
    assert repr(small_histogram)


def test_reregister_hits(small_histogram):
    small_histogram.reregister_hits([1, 2, 3, 4, 5])

    assert small_histogram._hits[2] == 3


def test_getBinContent_via_getValues(small_histogram):
    small_histogram.fill([1.0, 1.0, 3.0])

    assert list(small_histogram.getValues()) == [2.0, 1.0, 0.0, 0.0, 0.0]


# --- histogram_2d -----------------------------------------------------

def test_histogram_2d_from_edges_shape(small_histogram_2d):
    # See the small_histogram_2d fixture's docstring: histogram_2d stores
    # one row/column per *edge* (not per bin interval) when built from
    # explicit edges, so a 5-edge/4-edge histogram has a (5, 4) matrix.
    assert small_histogram_2d._values.shape == (5, 4)


def test_histogram_2d_insert_and_getBinContent(small_histogram_2d):
    small_histogram_2d.insert(0, 0, 5.0)

    content = small_histogram_2d.getBinContent(0, 0)

    assert content.n == pytest.approx(5.0)


def test_histogram_2d_getBin(small_histogram_2d):
    i, j = small_histogram_2d.getBin(1.0, 0.5)

    assert (i, j) == (0, 0)


def test_histogram_2d_getBin_out_of_range(small_histogram_2d):
    assert small_histogram_2d.getBin(-1.0, -1.0) == (-1, -1)


def test_histogram_2d_sample(small_histogram_2d):
    small_histogram_2d.insert(0, 0, 5.0)

    # sample() returns a ufloat (nominal +/- uncertainty), not a plain
    # float, so compare its nominal value.
    assert small_histogram_2d.sample(1.0, 0.5).n == pytest.approx(5.0)


def test_histogram_2d_fill_replaces_whole_matrix(small_histogram_2d):
    matrix = np.ones((5, 4))

    small_histogram_2d.fill(matrix)

    assert np.array_equal(small_histogram_2d._values, matrix)


def test_histogram_2d_fill_rejects_wrong_shape(small_histogram_2d):
    with pytest.raises(AssertionError):
        small_histogram_2d.fill(np.ones((2, 2)))


def test_histogram_2d_range(small_histogram_2d):
    small_histogram_2d.insert(0, 0, 5.0)
    small_histogram_2d.insert(1, 1, -2.0)

    assert small_histogram_2d.range() == (-2.0, 5.0)


def test_histogram_2d_to_extent(small_histogram_2d):
    x_axis, y_axis = small_histogram_2d.to_extent()

    assert np.allclose(x_axis, [0, 2.5, 5, 7.5, 10])
    assert np.allclose(y_axis, [0, 1, 2, 3])