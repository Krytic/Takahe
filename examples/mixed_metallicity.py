import matplotlib.pyplot as plt
import takahe

universe = takahe.universe.create('lcdm')
universe.populate('data/newdata')

events = universe.event_rate()
events.plot()
plt.title("Event Rate")
plt.yscale('log')
plt.show()
