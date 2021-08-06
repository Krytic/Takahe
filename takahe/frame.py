import imageio
import pickle
from os import path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import takahe
from tqdm import tqdm

def _dt_from_unit(unit):
    """Infers the timestep based on the unit.


    Arguments:
        unit {string} -- The unit (yr, kyr, Myr, Gyr) under consideration.

    Raises:
        AssertionError -- on malformed input.
    """

    assert unit in ['yr', 'kyr', 'Myr', 'Gyr']

    return {'yr': 1e0,
            'kyr': 1e3,
            'Myr': 1e6,
            'Gyr': 1e9}[unit]

class Frame:
    """Represents a single x-y frame of data."""
    def __init__(self, time, z):
        """Creates the frame.

        Creates a Frame (x-y plane).

        Arguments:
            time {float} -- The time corresponding to the frame.

            z {np.matrix} -- The matrix corresponding to the z-axis.
        """
        self.z    = z
        self.time = time

    def __cmp__(self, other):
        if self.time < other.time:
            return -1
        if self.time == other.time:
            return  0
        if self.time > other.time:
            return  1

class FrameCollectionExtent:
    def __init__(self, x_from, x_to, nr_bins_x, y_from, y_to, nr_bins_y):
        self.__xaxis = np.linspace(x_from, x_to, nr_bins_x)
        self.__yaxis = np.linspace(y_from, y_to, nr_bins_y)

    def fetch(self):
        return (self.__xaxis, self.__yaxis)

class FrameCollection:
    """Represents a collection of Frames - an (x, y, z) cube of data."""
    def __init__(self, extent, time=None):
        """Creates the FrameCollection.

        Creates a cube of x, y, z data. Each Frame in the Collection
        represents a single x-y slice of the data at timestep z.

        Arguments:
            extent {tuple} -- A 2-tuple representing the extent of the
                              axes. extent[0] is a linspace object
                              representing the x-axis ticks, and
                              extent[1] is the same for the y-axis.

        Keyword Arguments:
            time {tuple} -- A 2-tuple representing the time information.
                            time[0] is the timestep between different
                            Frames in the collection, and time[1] is the
                            human-readable name of the step (e.g., Myr).
                            If None, we assume a default timestep
                            of 10^6 years (=1 Myr). (default: {None}).

        Raises:
            AssertionError -- on malformed input.
        """

        assert isinstance(time, tuple) or time == None, ("Expected time "
                                                         "to be either a "
                                                         "tuple or None.")
        if isinstance(time, tuple):
            assert (isinstance(time[0], np.int)
                or  isinstance(time[0], np.float)), ("Expected time[0] to be "
                                                     "a number.")

            assert time[1] in ['yr', 'kyr', 'Myr', 'Gyr'], (f"{time[1]} "
                                                            "is not a valid "
                                                            "timestep.")

        if not isinstance(extent, FrameCollectionExtent):
            assert isinstance(extent, tuple), "Expected extent to be a 2-tuple."

            assert isinstance(extent[0], np.ndarray), ("Expected extent[0] to be "
                                                     "an arraylike object.")

            assert isinstance(extent[1], np.ndarray), ("Expected extent[1] to be "
                                                     "an arraylike object.")

            assert len(extent[0]) > 0 and len(extent[1]) > 0, ("Expected extent "
                                                               "contents to be "
                                                               "non-empty.")
        else:
            extent = extent.fetch()

        if time == None:
            self.__dt = 1e6
            self.__timeunit = "Myr"
        else:
            self.__dt = time[0]
            self.__timeunit = time[1]
            self.__sniffed = _dt_from_unit(time[1])

        self.__i       = 0
        self.__frames  = []
        self.__size    = 0
        self.__xaxis   = extent[0]
        self.__yaxis   = extent[1]
        self.__times   = []

        self.__probability_map = takahe.histogram.histogram_2d(edges_x=self.__xaxis,
                                                               edges_y=self.__yaxis)

    # Iteration methods
    def __iter__(self):
        return self

    def __next__(self):
        i = self.__i
        if self.__i < self.__size:
            frame = self.__frames[i]
            self.__i += 1
            return frame
        raise StopIteration

    def final_frame(self, culmulative):
        """Extracts the final Frame of the FrameCollection.

        Arguments:
            culmulative {bool} -- Whether we should make the final Frame
                                  accumulate all the prior data (True)
                                  or not (False).

        Returns:
            {takahe.frame.Frame} -- The final Frame.
        """
        if not culmulative:
            return self.find(self.__times[-1])
        else:
            takahe.debug("info", "ok extracting frame")
            all_prev = np.ones((len(self.__xaxis), len(self.__yaxis)))
            takahe.debug("info", "all_prev is a " + str(type(all_prev)))
            for frame in self:
                all_prev += frame.z
            takahe.debug('info', "reached end of loop")

            new_frame = Frame(self.__times[-1], all_prev)

            takahe.debug('info', 'made the new frame, shall we return?')
            return new_frame

    def get_grid(self):
        """Extracts the meshgrid corresponding to an individual Frame.

        Returns:
            {np.meshgrid} -- The meshgrid for a single frame.
        """
        return np.meshgrid(self.__xaxis, self.__yaxis, indexing='ij')

    def find(self, time):
        """Finds a specific Frame, by time.

        Finds the Frame corresponding to the given time. If no Frame
        exists, returns the first frame *after* the given time. If the
        time is past the end of the time axis, returns the final Frame.

        Arguments:
            time {float} -- The time to search for (in years).

        Returns:
            {takahe.frame.Frame} -- The Frame found.
        """
        i = np.argwhere(np.array(self.__times) >= (time * 1e9))
        try:
            i = np.min(i)
        except ValueError:
            i = -1

        return self.__frames[i]

    def boundary(self):
        """Finds the global maximum and global minimum of the
        FrameCollection.

        Iterates over the entirety of the FrameCollection to find the
        global minimum and global maximum.

        Returns:
            {tuple} -- A 2-tuple of (global_min, global_max)
        """
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
        """Saves a FrameCollection as a pickle object.

        Arguments:
            outfile {string} -- The filename to save to.

        Raises:
            AssertionError -- on malformed input.
        """
        assert isinstance(outfile, str), "outfile must be a string!"

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

    def to_gif(self,
               output_directory,
               outname="takahegif.gif",
               fps=30,
               Z_Step=0.1,
               overplot_data=None,
               xlabel="",
               ylabel=""
        ):
        """Generates a GIF of the given FrameCollection.

        Uses imageio to stich together multiple images into a single GIF.
        Will generate every frame, and place them in the output_directory.

        Arguments:
            output_directory {str} -- the path to the directory you wish
                                      to generate the individual GIF frames
                                      in.

        Keyword Arguments:
            outname {str}        -- The output name of the GIF
                                    (default: {"takahegif.gif"})
            fps {number}         -- The number of frames to display every
                                    second. (default: {30})
            Z_Step {number}      -- The number of steps to take in log-space
                                    on the z-axis. (default: {0.1})
            overplot_data {dict} -- A dictionary representing any observed
                                    data you wish to overplot on every frame.
                                    Leave None to represent no data.
                                    (default: {None})
            xlabel {str}         -- The label for the x-axis.
                                    (default: {""})
            ylabel {str}         -- The label for the y-axis.
                                    (default: {""})
        """

        assert isinstance(output_directory, str), ("output directory must "
                                                   "be a string!")
        assert path.exists(output_directory), "output directory must exist!"
        assert isinstance(outname, str), "outname must be a string."
        assert isinstance(fps, np.int), "FPS must be an integer."
        assert isinstance(Z_Step, np.float), "Z_Step must be a float."

        valid = (overplot_data == None or isinstance(overplot_data, dict))
        assert valid, "overplot_data must be None or a dictionary type"
        if isinstance(overplot_data, dict):
            keys = ('x' in overplot_data.keys()
                    and 'y' in overplot_data.keys()
                    and 'label' in overplot_data.keys())

            assert keys, ("Incorrect keys passed in overplot_data (expected: "
                          f"x, y, label, got: {list(overplot_data.keys())}).")

        assert isinstance(xlabel, str), "xlabel should be a string."
        assert isinstance(ylabel, str), "ylabel should be a string."

        i = 0
        L = len(str(len(self)))
        images = []

        vmin, vmax = self.boundary()

        dt, unit, unit_dt = self.__dt, self.__timeunit, self.__sniffed

        if overplot_data is not None:
            x, y, label = (overplot_data['x'],
                           overplot_data['y'],
                           overplot_data['label'])

        with tqdm(total=self.__size) as pbar:
            all_prev = np.ones((len(self.__xaxis), len(self.__yaxis)))
            X, Y = np.meshgrid(self.__xaxis, self.__yaxis, indexing='ij')

            for frame in self.__frames:
                i_padded = f"{i}".zfill(L)
                fname = output_directory + f"/frame_{i_padded}.png"

                plot = all_prev + frame.z
                all_prev += frame.z

                plt.figure()
                lev_exp = np.arange(np.log10(vmin+1), np.log10(vmax+1), Z_Step)
                levels = np.power(10, lev_exp)

                plt.contourf(X, Y, plot,
                             locator=ticker.LogLocator(),
                             vmin=1+vmin, vmax=1+vmax,
                             levels=levels)

                if overplot_data is not None:
                    observation_data = zip(label, x, y)

                    for data in observation_data:
                        plt.plot(data[1], data[2], color='red',
                                                   marker='D')

                plt.colorbar(ticks=[10**i for i in range(int(np.ceil(np.log10(vmax))))])
                plt.title(rf"$t={frame.time/unit_dt}${unit}")
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.savefig(fname)
                plt.close()

                images.append(imageio.imread(fname))

                i += 1

                pbar.update(1)

        imageio.mimsave(outname, images, loop=0, fps=fps)

    def __len__(self):
        return self.__size

    def insert(self, item):
        """Inserts a frame into the FrameCollection.

        Arguments:
            item {takahe.frame.Frame} -- The frame to insert.

        Raises:
            TypeError -- on malformed input.
        """
        if not isinstance(item, Frame):
            raise TypeError("item must be an instance of Frame()")

        self.__frames.append(item)
        self.__frames = sorted(self.__frames, key=lambda obj: obj.time)
        self.__times.append(item.time)
        self.__times.sort()

        self.__size += 1

class pickledFrameCollection(FrameCollection):
    """Represents a pickled version of a FrameCollection.

    Extends:
        FrameCollection
    """
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

def load(fname):
    """Loads a pickled FrameCollection.

    Arguments:
        fname {string} -- The file to load.

    Returns:
        {takahe.frame.pickledFrameCollection} -- The FrameCollection
                                                 loaded.
    """
    return pickledFrameCollection(fname)
