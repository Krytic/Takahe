import numpy as np
import takahe
from takahe.constants import *
from kea import hist

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
        self.__min_lifetime = np.infty
        self.__max_lifetime = 0

    def add(self, binary_star):
        """Add a BSS to the current ensemble.

        Arguments:
            binary_star {BinaryStarSystem} -- The BSS to add.

        Raises:
            TypeError -- If the Binary Star System is not an instance of
                         BinaryStarSystem.
        """
        if type(binary_star) != takahe.BSS.BinaryStarSystem:
            raise TypeError("binary_star must be an instance \
                             of BinaryStarSystem!")

        lifetime = binary_star.lifetime()

        if lifetime > self.__max_lifetime:
            self.__max_lifetime = lifetime
        elif lifetime < self.__min_lifetime:
            self.__min_lifetime = lifetime

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

        Returns:
            histogram -- the (kea-generated) histogram object.
        """
        time_array = np.linspace(1, 13.8, 1000)

        mergers = np.array([])

        for t in time_array:
            # compute number of mergers occuring before current time
            mr = self.merge_rate(t, return_as='abs')

            # remove all mergers that occurred before the current time
            # to get just the mergers that occur in this time bin
            if t != time_array[0]:
                mr = mr - np.sum(mergers)

            mergers = np.append(mergers, mr)

        bins = 10*np.log10(time_array)

        nbins = int(np.sqrt(len(time_array)))

        bin_sizes = bins[1:] - bins[0:-1]

        # janky hack to make the bin_sizes array the right length
        bin_sizes = np.append(bin_sizes, bin_sizes[-1])

        histogram = hist.histogram(xlow=bins[0],
                                   xup=bins[-1],
                                   nr_bins=nbins)

        histogram.Fill(bins, mergers / (1e6 * bin_sizes))

        histogram.plot()

        return histogram

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
