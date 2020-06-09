import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc
from scipy.integrate import quad
from scipy.optimize import fminbound
import cycler

color = plt.cm.tab20(np.linspace(0.1,0.9,13))

plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

tH = 1 / (70 / 3.086e+19 * 31557600000000000)

@np.vectorize
def estimate_redshift(tL):
    f = lambda z: np.abs(estimate_lookback(z) - tL)

    zbest, _, _, _ = fminbound(f, 1e-8, 1000, maxfun=500, full_output=1, xtol=1e-8)

    return zbest

def estimate_lookback(z):
    def integrand(z):
        def E(z):
            return np.sqrt(0.3 * (1+z)**3
                         + 0 * (1+z)**2
                         + 0.7)
        return 1 / ((1+z) * E(z))

    rest, err = quad(integrand, 0, z)

    return tH*rest

SFRD = lambda z, Z: 0.015 * (1+z)**2.7 / (1+((1+z)/2.9)**5.6) * gammainc(0.84, (Z/0.020) ** 2 * 10**(0.3*z))

tL_array = np.linspace(0, tH, 200)

Z_range = [1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.14, 0.02, 0.03, 0.04]

last_SFR = 0

plt.figure(figsize=(20,40))

for i in range(len(Z_range)):
    Z = Z_range[i]
    all_SFR = SFRD(estimate_redshift(tL_array[1:]), Z)

    # change z_array to x_axis if you want to plot against lookback time
    SFR = all_SFR - last_SFR
    last_SFR = all_SFR

    plt.plot(tL_array[1:], SFR, label=Z)
    print(f"completed z={Z}")

plt.legend()
plt.yscale('log')
plt.show()
