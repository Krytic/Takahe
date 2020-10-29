import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
    def __init__(self, time, z):
        self.z    = z
        self.time = time

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time == other.time:
            return  0
        if self.time > other.time:
            return  1

class FrameCollection:
    def __init__(self, extent):
        self.__i = 0
        self.__frames = []
        self.__size = 0
        self.__xaxis = extent[0]
        self.__yaxis = extent[1]

    def __iter__(self):
        return self

    def __next__(self):
        i = self.__i
        if self.__i < self.__size:
            frame = self.__frames[i]
            self.__i += 1
            return frame
        raise StopIteration

    def boundary(self):
        min_so_far = np.infty
        max_so_far = -np.infty
        for frame in self:
            this_min = np.min(frame.z)
            this_max = np.max(frame.z)

            if this_min < min_so_far:
                min_so_far = this_min
            if this_max > max_so_far:
                max_so_far = this_max

        return min_so_far, max_so_far

    def to_gif(self, output_directory, outname="takahegif.gif", duration=1/10):
        i = 0
        L = len(str(len(self)))
        images = []

        vmin, vmax = self.boundary()

        for frame in self.__frames:
            ipad = f"{i}".zfill(L)
            fname = output_directory + f"/frame_{ipad}.png"

            X, Y = np.meshgrid(self.__xaxis, self.__yaxis, indexing='ij')
            plt.figure()
            plt.pcolormesh(X, Y, 1+frame.z,
                           norm=colors.LogNorm(vmin=1+vmin, vmax=1+vmax),
                           vmin=1+vmin, vmax=1+vmax)
            plt.title(rf"$t={frame.time/1e6}$Myr")
            plt.savefig(fname)
            plt.close()

            images.append(imageio.imread(fname))

            i += 1

        imageio.mimsave(outname, images, loop=0, duration=duration)

    def __len__(self):
        return self.__size

    def insert(self, item):
        if not isinstance(item, Frame):
            raise TypeError("item must be an instance of Frame()")

        self.__frames.append(item)
        self.__frames = sorted(self.__frames, key=lambda obj: obj.time)
        self.__size += 1
