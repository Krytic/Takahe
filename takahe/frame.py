import imageio
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import takahe
from tqdm import tqdm

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
    def __init__(self, extent, time=None):
        if time == None:
            self.__dt = 1e6
            self.__timeunit = "Myr"
        else:
            self.__dt = time[0]
            self.__timeunit = time[1]

        self.__i       = 0
        self.__frames  = []
        self.__size    = 0
        self.__xaxis   = extent[0]
        self.__yaxis   = extent[1]
        self.__times   = []

    def __iter__(self):
        return self

    def __next__(self):
        i = self.__i
        if self.__i < self.__size:
            frame = self.__frames[i]
            self.__i += 1
            return frame
        raise StopIteration

    def find(self, time):
        i = np.argwhere(self.__times >= (time * 1e9))
        try:
            i = np.min(i)
        except ValueError:
            i = -1
        return self.__frames[i]

        t = np.ceil(time * 1e9 / self.__dt) * self.__dt
        i = np.argmin(np.abs(self.__times - t))
        return self.__frames[i]

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

    def save(self, outfile):
        contents = {
            'build_params': {
                'xaxis': self.__xaxis,
                'yaxis': self.__yaxis,
                'timestep': self.__dt,
                'timeunit': self.__timeunit
            },
            'frames': [(frame.time, frame.z) for frame in self.__frames],
        }

        with open(outfile, 'wb') as f:
            pickle.dump(contents, f)

    def to_gif(self, output_directory, outname="takahegif.gif", fps=30):
        i = 0
        L = len(str(len(self)))
        images = []

        vmin, vmax = self.boundary()

        dt, unit = self.__dt, self.__timeunit

        Po = np.log10(np.array([4.072, 0.102, 0.421, 0.320, 0.323, 0.206, 0.184, 8.634, 18.779, 1.176, 45.060, 13.638, 2.616, 0.078]))
        Eo = [0.113, 0.088, 0.274, 0.181, 0.617, 0.090, 0.606, 0.249,  0.828, 0.139,  0.399,  0.304, 0.169, 0.064]
        pulsars = ["J0453+1559", "J0737-3039", "B1534+12", "J1756-2251", "B1913+16", "J1913-1102", "J1757-1854", "J1518+4904", "J1811-1736", "J1829+2456", "J1930-1852", "J1753-2240", "J1411+2551", "J1946+2052"]

        with tqdm(total=self.__size) as pbar:
            all_prev = np.ones((len(self.__xaxis), len(self.__yaxis)))
            for frame in self.__frames:
                ipad = f"{i}".zfill(L)
                fname = output_directory + f"/frame_{ipad}.png"

                plot = all_prev + frame.z
                all_prev += frame.z

                X, Y = np.meshgrid(self.__xaxis, self.__yaxis, indexing='ij')
                plt.figure()
                plt.contourf(X, Y, plot,
                             norm=colors.LogNorm(vmin=1+vmin, vmax=1+vmax),
                             vmin=1+vmin, vmax=1+vmax)

                Pulsars = zip(pulsars, Po, Eo)

                for pulsar in Pulsars:
                    plt.plot(pulsar[1], pulsar[2], color='red',
                                                   marker='D')

                plt.colorbar()
                plt.title(rf"$t={frame.time/dt}${unit}")
                plt.xlabel(r"log(Period / days)")
                plt.ylabel(r"Eccentricity [no dim]")
                plt.savefig(fname)
                plt.close()

                images.append(imageio.imread(fname))

                i += 1

                pbar.update(1)

        imageio.mimsave(outname, images, loop=0, fps=fps)

    def __len__(self):
        return self.__size

    def insert(self, item):
        if not isinstance(item, Frame):
            raise TypeError("item must be an instance of Frame()")

        self.__frames.append(item)
        self.__frames = sorted(self.__frames, key=lambda obj: obj.time)
        self.__times.append(item.time)
        self.__times.sort()

        self.__size += 1

class pickledFrameCollection(FrameCollection):
    def __init__(self, infile):
        with open(infile, 'rb') as f:
            contents = pickle.load(f)

            xaxis = contents['build_params']['xaxis']
            yaxis = contents['build_params']['yaxis']
            timestep = contents['build_params']['timestep']
            timeunit = contents['build_params']['timeunit']

            extent = (xaxis, yaxis)
            time = (timestep, timeunit)

            super().__init__(extent, time)

            for frame in contents['frames']:
                frameobj = Frame(frame[0], frame[1])
                self.insert(frameobj)

def load(path):
    return pickledFrameCollection(path)
