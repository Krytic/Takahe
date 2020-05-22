import time
import multiprocessing as mp
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt
import takahe

n_stars = 1000
#plt.rcParams['figure.figsize'] = (40, 40)

#plt.style.use('krytic')

data_dir = 'data/newdata'

files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]

universe = takahe.universe.create("LCDM")
z = np.linspace(0, universe.tH, universe.get_nbins())
event_rates = np.array([0 for _ in range(len(z))])

start = time.time()

for file in files:
    universe.populate(f"{data_dir}/{file}", n_stars=n_stars)
    events = universe.event_rate()

    event_rates = event_rates + events.getValues()


today = np.round(np.log10(event_rates[0]), 2)
end = time.time()
print(f"Completed in {end-start} seconds.")
plt.xlabel("Lookback Time [Gyr]")
plt.ylabel(r"Event rate [events / yr / Gpc$^{-3}$]")
plt.title(rf"Mixed Metallicity Event Rates, $n={n_stars}, \log(R(z=0))={today}$")
plt.yscale('log')
plt.plot(z, event_rates)
plt.show()
