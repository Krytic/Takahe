import numpy as np

#########################
## BPASS Configuration ##
#########################

BPASS_METALLICITIES   = np.array(['em5', 'em4', '001', '002', '003',
                                  '004', '006', '008', '010', '014',
                                  '020', '030', '040'])

BPASS_METALLICITIES_F = np.array([1e-5 , 1e-4 , 0.001, 0.002, 0.003,
                                  0.004, 0.006, 0.008, 0.010, 0.014,
                                  0.020, 0.030, 0.040])

MASS_CUTOFF_NS = 2.5 # NS have mass LOWER than this
MASS_CUTOFF_BH = 2.5 # BH have mass GREATER than this

######################
## Unit Conversions ##
######################

# Physical Constants in SI units
G = 6.67430e-11
c = 299792458

# Solar Units
SOLAR_RADIUS      = 695500000
SOLAR_MASS        = 1.989e30
SOLAR_METALLICITY = 0.020

# Time conversions
SECONDS_PER_YEAR  = 60 * 60 * 24 * 365.25
SECONDS_PER_GYR   = SECONDS_PER_YEAR * 1e9

# Distance conversions
KILOPARSECS_PER_METER = 3.086e19

#############################
## Cosmological Parameters ##
#############################

# Assume standard LCDM cosmology by default
OMEGA_M = 0.3
OMEGA_K = 0
OMEGA_L = 0.7
HUBBLE_PARAMETER = 70

HUBBLE_TIME = 1 / (HUBBLE_PARAMETER / KILOPARSECS_PER_METER * SECONDS_PER_GYR)
# \simeq 14, in Gyr

#############
## Binning ##
#############

LINEAR_BINS = np.linspace(0, 14, 51)
BINS = lambda n: np.linspace(0, 14, n)

PE_BINS_PER_W = 0.1 # there will be 8 / PE_BINS_PER_W bins
PE_BINS_ECC_W = 0.01 # there will be 1 / PE_BINS_ECC_W bins

################
## Debug only ##
################

DEBUG_MODE        = False
