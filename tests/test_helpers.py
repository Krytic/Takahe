import numpy as np
import takahe

import pytest

def test_kepler():
    P = np.linspace(0, 10000, 100000)
    a = takahe.helpers.compute_separation(P, 5, 5)
    P2 = takahe.helpers.compute_period(a, 5, 5)

    assert np.allclose(P, P2)
