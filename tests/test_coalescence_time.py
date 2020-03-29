import numpy as np
import takahe

def test_ct():
    cfg = {
        'M1' : 1.33, # Solar masses
        'M2' : 1.35, # Solar masses
        'a0' : 3.28, # Solar radii
        'e0' : 0.274 # 0 <= e < 1
    }

    BSS = takahe.load.from_data(data=cfg)

    assert np.isclose(BSS.coalescence_time(), 2.734, atol=1e-1)
