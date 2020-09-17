import matplotlib.pyplot as plt
import numpy as np

"""

,--.   ,--.,--.,------.     ,--.  ,--. ,-----. ,--------.,--. ,-----.,------.
|  |   |  ||  ||  .--. '    |  ,'.|  |'  .-.  ''--.  .--'|  |'  .--./|  .---'
|  |.'.|  ||  ||  '--' |    |  |' '  ||  | |  |   |  |   |  ||  |    |  `--,
|   ,'.   ||  ||  | --'     |  | `   |'  '-'  '   |  |   |  |'  '--'\|  `---.
'--'   '--'`--'`--'         `--'  `--' `-----'    `--'   `--' `-----'`------'

This package is a WIP - it may change very quickly,
and it may not stay around.
"""

class Frame:
    def __init__(self, x_axis, y_axis):
        self.x = x_axis
        self.y = y_axis

    def plot(self, *args, **kwargs):
        plt.scatter(self.x, self.y, *args, **kwargs)

class FrameCollection:
    def __init__(self):
        self.__i = 0
        self.__frames = []
        self.__size = 0

    def __iter__(self):
        return self

    def __next__(self):
        i = self.__i
        frame = self.__frames[i]
        if self.__i < self.__size:
            self.__i += 1
            return frame
        raise StopIteration

    def plot(self, *args, **kwargs):
        for frame in self:
            frame.plot()

    def __len__(self):
        return self.__size

    def push(self, item):
        if not isinstance(item, Frame):
            raise TypeError("item must be an instance of Frame()")

        self.__frames.append(item)
        self.__size += 1
