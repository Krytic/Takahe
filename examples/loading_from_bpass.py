import numpy as np
import matplotlib.pyplot as plt

from takahe import BinaryStarSystemLoader as load

ensemble = load.from_bpass('data/starmass-bin-imf_chab100.z001.dat', 4.0)

print(ensemble.merge_rate(1e17))
