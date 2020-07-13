import numpy as np

BPASS_METALLICITIES = ['em5', 'em4', '001', '002', '003',
                       '004', '006', '008', '010', '014',
                       '020', '030', '040']

BPASS_METALLICITIES_F = np.array([1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040])

MASS_CUTOFF_NS = 2.5 # higher edge
MASS_CUTOFF_BH = 2.5 # lower edge

# Assume standard LCDM cosmology
OMEGA_M = 0.3
OMEGA_K = 0
OMEGA_L = 0.7
HUBBLE_PARAMETER = 70

HUBBLE_TIME = 1 / (HUBBLE_PARAMETER / 3.086e+19 * 1e9 * 60 * 60 * 24 * 365.25) # \simeq 14, in Gyr

G = 6.67430e-11
c = 299792458

LINEAR_BINS = np.linspace(0, 14, 102)
