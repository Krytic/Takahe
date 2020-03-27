import numpy as np
import takahe
from takahe.constants import *

def create():
    """Creates a BinaryStarSystemEnsemble object (i.e., a collection of
    binary star systems).

    Wrapper for BinaryStarSystemEnsemble()

    Returns:
        BinaryStarSystemEnsemble -- The ensemble object.
    """
    return BinaryStarSystemEnsemble()

class BinaryStarSystemEnsemble:
    """Represents a collection of binary star systems.

    Represents a group of binary star system objects. Will be
    generated if the loader encounters a group of BSS objects (e.g.,
    from BPASS).
    """

    def __init__(self):
        self.__ensemble = []
        self.__count = 0
        self.__pointer = 0
        self.__min_coalescence_time = np.infty
        self.__max_coalescence_time = 0

    def add(self, binary_star):
        """Add a BSS to the current ensemble.

        Arguments:
            binary_star {BinaryStarSystem} -- The BSS to add.

        Raises:
            TypeError -- If the Binary Star System is not an instance of
                         BinaryStarSystem.
        """
        if type(binary_star) != takahe.BSS.BinaryStarSystem:
            raise TypeError("binary_star must be an instance of BinaryStarSystem!")

        ct = binary_star.lifetime()

        if ct > self.__max_coalescence_time:
            self.__max_coalescence_time = ct
        elif ct < self.__min_coalescence_time:
            self.__min_coalescence_time = ct

        self.__ensemble.append(binary_star)
        self.__count += 1

    def average_coalescence_time(self):
        """Computes the average coalescence time for the binary star
        system.

        Returns:
            float -- The average over all the coalescence times in the
                     ensemble.
        """

        running_sum = 0

        for binary_star in self.__ensemble:
            running_sum += binary_star.coalescence_time()

        return running_sum / self.size()

    def merge_rate(self, t_merge, return_as="rel"):
        """Computes the merge rate for this ensemble.

        Computes the merge rate for this system in a given timespan.
        Merge rate is defined as the number of systems that merge in
        some timespan t_merge (possibly relative to the number of
        systems in the ensemble).

        Arguments:
            t_merge {float} -- The timespan under consideration. Must be
                               in gigayears; no conversion is
                               performed before comparison.

        Keyword Arguments:
            return_as {str} -- "abs" or "rel" depending on whether the
                               merge rate should be returned as an
                               absolute count or relative count
                               (to the number of BSS in the ensemble).
                               If defined as relative, then the merge
                               rate is constrained to be in the interval
                               [0, 1]. (default: {"rel"})

        Returns:
            float -- The merge rate of the ensemble.

        Raises:
            ValueError -- if return_as is anything other than "abs" or
                          "rel".
        """
        count = 0

        if return_as.lower() not in ['abs', 'rel']:
            raise ValueError("return_as must be either abs or rel")

        for i in range(self.__count):
            binary_star = self.__ensemble[i]
            valid = (binary_star.lifetime() <= t_merge)
            if valid:
                count += 1

        if return_as == 'abs':
            return count
        elif return_as == 'rel':
            return count / self.__count

    def compute_event_rate_plot(self):
        """Generates the event rate plot for this ensemble.

        VERY INCORRECT CALCULATION!!!!

        This needs serious improvement, it is a "first guess" at the code.

        Returns:
            x_axis -- the x axis (log-binned time array) for the plot
            y_axis -- the y axis (event rate in events per solar mass
                      per year) for the plot.
        """
        from collections import Counter

        time_array = np.linspace(self.__min_coalescence_time,
                                 self.__max_coalescence_time)

        time_array *= 1e9

        x_axis = np.round(np.log10(time_array))
        y_axis = np.array([])

        bin_sizes = Counter(x_axis)

        for bin_i in x_axis:
            mr = self.merge_rate(bin_i, return_as='abs')

            bin_size = bin_sizes[bin_i]

            y_axis = np.append(y_axis, mr / bin_size)

        return [x_axis, y_axis / (1e6 * Solar_Mass)]

    def size(self):
        """Get the size of the ensemble.

        Returns:
            int -- The number of BSS in the ensemble.
        """
        return self.__count

    """
    Magic methods, in order to be able to compute the size of the
    ensemble, and iterate over it, for example. These are uninteresting
    and therefore do not contain any docstrings.
    """

    def __len__(self):
        return self.size()

    def __iter__(self):
        return self

    def __next__(self):
        if self.__pointer >= self.__count:
            raise StopIteration
        else:
            self.__pointer += 1
            return self.__ensemble[self.__pointer - 1]
