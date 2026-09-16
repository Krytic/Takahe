import numpy as np
import takahe

import pytest


def test_kepler():
    P = np.linspace(0, 10000, 100000)
    a = takahe.helpers.compute_separation(P, 5, 5)
    P2 = takahe.helpers.compute_period(a, 5, 5)

    assert np.allclose(P, P2)


@pytest.mark.parametrize("Z, expected", [
    ("020", 1.0),
    ("em5", 1e-5 / takahe.constants.SOLAR_METALLICITY),
    ("010", 0.5),
    ("0.020", 1.0),
    ("z020", 1.0),
    ("0", 0.0),
])
def test_format_metallicity(Z, expected):
    assert takahe.helpers.format_metallicity(Z) == pytest.approx(expected)


@pytest.mark.parametrize("filename, expected", [
    ("Remnant-Birth-bin-imf135_300-z030_StandardJJ.dat", 1.5),
    ("Remnant-Birth-bin-imf135_300-z010", 0.5),
])
def test_extract_metallicity(filename, expected):
    assert takahe.helpers.extract_metallicity(
        filename) == pytest.approx(expected)


def test_extract_metallicity_rejects_non_string():
    with pytest.raises(AssertionError):
        takahe.helpers.extract_metallicity(12345)


def test_find_between():
    a = list(range(20))

    assert takahe.helpers.find_between(a, 5, 10) == [5, 6, 7, 8, 9, 10]


def test_find_between_raises_when_not_found():
    a = list(range(20))

    with pytest.raises(ValueError):
        takahe.helpers.find_between(a, 100, 200)


def test_memoize_only_calls_wrapped_function_once_per_argument():
    calls = []

    @takahe.helpers.memoize
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert f(3) == 6
    assert f(4) == 8

    # f(3) should only have actually run the wrapped function once.
    assert calls == [3, 4]


@pytest.mark.parametrize("m1, m2, expected", [
    (1.0, 1.0, 'NSNS'),
    (1.0, 5.0, 'NSBH'),
    (5.0, 1.0, 'NSBH'),
    (5.0, 5.0, 'BHBH'),
])
def test_identify(m1, m2, expected):
    import pandas as pd

    star = pd.Series({'m1': m1, 'm2': m2})

    assert takahe.helpers.identify(star) == expected


def test_redshift_lookback_roundtrip():
    # lookback_to_redshift() is the (numerically inverted) inverse of
    # redshift_to_lookback(), so round-tripping a redshift through both
    # should return (approximately) the original value.
    z = 0.5
    tL = takahe.helpers.redshift_to_lookback(z)
    z2 = takahe.helpers.lookback_to_redshift(tL)

    assert float(z2) == pytest.approx(z, rel=1e-4)
