import numpy as np
import takahe

def test_redshift():
    uni = takahe.universe.create('eds')
    for d in range(1, 100):
        z = uni.compute_redshift(d)
        d_c = uni.compute_comoving_distance(z)

        assert np.isclose(d, d_c)
