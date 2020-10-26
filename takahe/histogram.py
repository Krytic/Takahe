"""

Histogram classes to contain event rate data and allow for easy plotting

Original author: Max Briel
Modified by: Sean Richards

"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

import warnings
from uncertainties import ufloat
from uncertainties.umath import log10 as ulog10

class histogram:
    """A histogram which can contain data and can be manipulated.

    Either **xlow**, **xup**, and **nr_bins** is given or **edges**

    As per any histogram, the upper edges are non inclusive, except for the
    last bin.

    Parameters
    ----------
    xlow : float
        lower bound
    xup : float
        upper bound
    nr_bins : int
        the number of bins
    edges : array
        An array with items defining the edges.

    Attributes
    ----------
    _xlow : float
        lower bound of the histogram
    _xup : float
        upper bound of the histogram
    _nr_bins : int
        the number of bins in the histogram
    _bin_edges : array
        An array of bin edges
    _values : array
        An array of length **_nr_bins** containing the value of each bin
    lower_edges : array
        An array of the lower edges of the bins in the histogram
    upper_edges : array
        An array of the upper edges of the bins in the histogram

    """

    def __init__(self, xlow=None, xup=None, nr_bins=None, edges=None):
        if xlow != None and xup != None and nr_bins != None:
            self._xlow = xlow
            self._xup = xup
            self._nr_bins = nr_bins
            self._bin_edges = np.linspace(xlow, xup, nr_bins+1)

        elif isinstance(edges, type([])) or isinstance(edges, type(np.array([]))):
            self._xlow = edges[0]
            self._xup = edges[-1]
            self._nr_bins = len(edges)-1
            self._bin_edges = np.array(edges)
        else:
            raise Exception("Not given the correct input")

        self._values = np.zeros(self._nr_bins)
        self._hits = np.zeros(self._nr_bins)
        self.lower_edges = self._bin_edges[:-1]
        self.upper_edges = self._bin_edges[1:]

    def __len__(self):
        return len(self._values)

    def __str__(self):
        return str(self._values)

    def __repr__(self):
        return f"The bins: {self._bin_edges}\nThe values: {self._values}"

    def __add__(self, other):
        out = self.copy()

        if isinstance(other, histogram):
            out._values = self._values + other._values
        else:
            out._values = self._values + other

        return out

    def __mul__(self, other):
        out = self.copy()
        out._values = self._values * other
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + -1 * other

    def __div__(self, other):
        out = self.copy()
        out._values = self._values / other
        out._hits = self._hits
        return out

    def __truediv__(self, other):
        out = self.copy()
        out._values = self._values / other
        out._hits = self._hits
        return out

    def inbounds(self, value):
        return self._xlow <= value and self._xup >= value

    def copy(self):
        """ creates a copy of the histogram

        Returns
        -------
        histogram
            An exact (deep) copy of the histogram

        """
        out = histogram(xlow=self._xlow, xup=self._xup, nr_bins=self._nr_bins)
        out._values = self._values
        out._hits   = self._hits
        return out

    def fill(self, x, weight = 1):
        """ Fill the histogram with data.


        Parameters
        ----------
        x : float/array
            Either a single entry or an array of *N* to put into the histogram
        w : float/array
            The weight of the entry of *N* entries to be added to the histogram.
        """
        def _insert(f, g):
            if f >= self._xup:
                 self._values[self._nr_bins-1] += g
                 self._hits[self._nr_bins-1] += 1
            elif f <= self._xlow:
                self._values[0] += g
                self._hits[0] += 1
            else:
                bin_nr = np.where(self.lower_edges <= f)[0][-1]
                self._values[bin_nr] += g
                self._hits[bin_nr] += 1

        if not isinstance(x, type(0.0)):
            if isinstance(weight, type(0)):
                for i in range(0, len(x)):
                    _insert(x[i], weight)
            elif len(x) != len(weight):
                raise Exception(f"Weight needs to be as long as x. (x={len(x)}, weight={len(weight)})")
            else:
                for i in range(0, len(x)):
                    _insert(x[i], weight[i])
        else:
            _insert(x, weight)

        return None

    def plot(self, with_errors=False, *argv, **kwargs):
        """Plot the histogram. matplotlib.pyplot arguments can be passed on too
        """
        entries, edges, _ = plt.hist(self._bin_edges[:-1], self._bin_edges, weights=self._values,histtype=u'step', *argv, **kwargs)

        if with_errors:
            plt.errorbar(self.getBinCenters(), self._values, yerr=np.sqrt(self._hits), fmt='r.')

        return None

    def plotLog(self, with_errors=False, *argv, **kwargs):
        """Plot the histogram. matplotlib.pyplot arguments can be passed on too
        """
        entries, edges, _ = plt.hist(np.log10(self._bin_edges[:-1]), np.log10(self._bin_edges), weights=self._values,histtype=u'step', *argv, **kwargs)

        if with_errors:
            plt.errorbar(self.getBinCenters(), self._values, yerr=np.sqrt(self._hits), fmt='r.')

        return None

    def getBinCenters(self):
        return [self.getBinCenter(i) for i in range(self.getNBins())]

    def reregister_hits(self, hits):
        for i in range(len(self._hits)):
            self._hits[i] = hits[i]

    def getUncertainty(self, bin):
        return np.sqrt(self._hits[bin])

    def get(self, bin):
        return ufloat(self.getBinContent(bin), self.getUncertainty(bin))

    def getLog(self, bin):
        val = self.get(bin)
        val = ulog10(val)
        return val

    def present_value(self, bin, log=False):
        if log:
            val = self.getLog(bin)
        else:
            val = self.get(bin)

        err = val.s * val.n
        nom = val.n

        err_to_1_sf = f"{err:.1g}"
        num_dp = len(str(err_to_1_sf).split('.')[1])

        return_string = rf"${nom:.{num_dp}f} \pm {err_to_1_sf}"

        return return_string

    def to_pickle(self, pickle_path):
        contents = {
            'edges': self._bin_edges,
            'values': self._values,
            'hits': self._hits
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(contents, f)

    def getBinContent(self, bin_nr):
        """Return the value of the given bin

        Parameters
        ----------
        bin_nr : int
            bin number

        Returns
        -------
        float
            The bincontent of the bin
        """

        return self._values[bin_nr]

    def getNBins(self):
        """ Gives the number of bins of the histogram

        Returns
        -------
        float
            Return the number of bins in the histogram
        """
        return self._nr_bins

    def getValues(self):
        """Return the values of the histogram

        Returns
        -------
        array
            The values stored in the histogram

        """
        return self._values

    def getBinWidth(self, i):
        """Returns the width of the given bin

        Parameters
        ----------
        i : int
            Bin number

        Returns
        -------
        float
            The width of the bin

        """
        return self.upper_edges[i] - self.lower_edges[i]

    def getBinCenter(self, i):
        """Returns the center of the bin

        Parameters
        ----------
        i : int
            Bin number

        Returns
        -------
        float
            The center of bin *i*

        """
        return (self.upper_edges[i] + self.lower_edges[i])/2

    def getBin(self, x):
        """Returns the bin number at value **x**

        Parameters
        ----------
        x : float
            value where you want to know the bin number

        Returns
        -------
        int
            The bin number

        """
        if x < self._bin_edges[0] or x > self._bin_edges[-1]:
            raise Exception(f"x={x} outside of range")

        out = np.where(x >= self._bin_edges)[0][-1]

        if out == self._nr_bins:
            out = out-1
        return out

    def getBinEdges(self):
        """Returns the bin edges of the histogram

        Returns
        -------
        array
            An array of all the bin edges

        """
        return self._bin_edges


    def sum(self, x1, x2):
        """Performs a binwise summation between parameters **x1** and **x2**.


        Parameters
        ----------
        x1 : float
            lower bound of the summation
        x2 : float
            upper bound of the summation

        Returns
        -------
        float
            The summation of the bins between **x1** and **x2**

        """
        if x1 >= x2:
            raise Exception("x2 should be larger than x1")
        if x1 < self._bin_edges[0]:
            warnings.warn("Lower limit is below lowest bin edge", )
        if x2 > self._bin_edges[-1]:
            warnings.warn("Higher limit is above the highest bin edge")

        lower_bin = self.getBin(x1)
        upper_bin = self.getBin(x2)

        if lower_bin == upper_bin:
            bin_width = self.getBinWidth(lower_bin)
            return self.getBinContent(lower_bin) * (x2 - x1) / bin_width
        else:
            total = 0
            # get lower bin part
            bin_width = self.getBinWidth(lower_bin)
            total += self.getBinContent(lower_bin) * (self.upper_edges[lower_bin] - x1)/bin_width

            # get upper bin part
            bin_width = self.getBinWidth(upper_bin)
            total += self.getBinContent(upper_bin) * (x2 - self.lower_edges[upper_bin])/bin_width

            # get the parts in between if they are there
            if (lower_bin + 1) != upper_bin:
                for i in range(lower_bin+1, upper_bin):
                    total += self._values[i]

            return total

    def integral(self, x1, x2):
        """Returns the integral of the histogram between **x1** and **x2**.

        Parameters
        ----------
        x1 : float
            lower bound of the integration
        x2 : float
            upper bound of the integration

        Returns
        -------
        float
            The integral between **x1** and **x2**

        """
        if x1 >= x2:
            raise Exception("x2 should be larger than x1")
        if x1 < self._bin_edges[0]:
            warnings.warn("Lower limit is below lowest bin edge")
        if x2 > self._bin_edges[-1]:
            warnings.warn("Higher limit is above the highest bin edge")
        lower_bin = self.getBin(x1)
        upper_bin = self.getBin(x2)
        if lower_bin == upper_bin:
            bin_width = self.getBinWidth(lower_bin)
            return self.getBinContent(lower_bin) * (x2 - x1)
        else:
            total = 0
            # get lower bin part
            bin_width = self.getBinWidth(lower_bin)
            total += self.getBinContent(lower_bin) * (self.upper_edges[lower_bin] - x1)

            # get upper bin part
            bin_width = self.getBinWidth(upper_bin)
            total += self.getBinContent(upper_bin) * (x2 - self.lower_edges[upper_bin])

            # get the parts in between if they are there
            if (lower_bin + 1) != upper_bin:
                for i in range(lower_bin+1, upper_bin):
                    total += self._values[i] * self.getBinWidth(i)

            return total

class histogram_2d:
    def __init__(self, x_range, y_range, nr_bins_x, nr_bins_y):
        xlow, xup         = x_range[0], x_range[1]
        ylow, yup         = y_range[0], y_range[1]

        self._xlow        = xlow
        self._xup         = xup
        self._ylow        = ylow
        self._yup         = yup
        self._num_x       = nr_bins_x
        self._num_y       = nr_bins_y
        self._values      = np.zeros((nr_bins_x, nr_bins_y))
        self._num_hits    = np.zeros((nr_bins_x, nr_bins_y))

        self._bin_edges_x = np.linspace(xlow, xup, nr_bins_x)
        self._bin_edges_y = np.linspace(ylow, yup, nr_bins_y)

    def fill(self, insertion_matrix):
        assert self._values.shape == insertion_matrix.shape

        self._values    = insertion_matrix
        # increment hits by 1 in every cell that contains a value:
        self._num_hits += (insertion_matrix>0).astype(int)

    def insert(self, bin_nr_x, bin_nr_y, value):
        self._values[bin_nr_x][bin_nr_y]   += value
        self._num_hits[bin_nr_x][bin_nr_y] += 1

    def getBin(self, x, y):
        i = np.where(x >= self._bin_edges_x)[0][-1]
        j = np.where(y >= self._bin_edges_y)[0][-1]

        return (i,j)

    def getBinContent(self, bin_nr_x, bin_nr_y):
        val = self._values[bin_nr_x][bin_nr_y]
        err = np.sqrt(self._num_hits[bin_nr_x][bin_nr_y])
        return ufloat(val, err)

    def copy(self):
        x = [self._xlow, self._xup]
        y = [self._ylow, self._yup]

        out = histogram_2d(x, y, self._num_x, self._num_y)
        out._values   = self._values
        out._num_hits = self._num_hits

        return out

    def plot(self, *args, **kwargs):
        x = self._bin_edges_x
        y = self._bin_edges_y
        X, Y = np.meshgrid(x, y, indexing='ij')
        plt.pcolormesh(X, Y, self._values.T, *args, **kwargs)

    def to_pickle(self, pickle_path):
        contents = {
            'build_params': {
                'xlow': self._xlow,
                'xup' : self._xup,
                'ylow': self._ylow,
                'yup' : self._yup,
                'y_nr': self._num_x,
                'x_nr': self._num_y,
            },
            'values': self._values,
            'hits': self._num_hits
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(contents, f)

    def probability_map(self):
        out = self.copy()
        total = np.sum(self._values)
        out._values /= total

        return out

    def likelihood(self, Po, Eo):
        pmap = self.probability_map()
        probabilities = []
        for obj in zip(Po, Eo):
            bin_nr = self.getBin(obj[0], obj[1])
            prob = pmap.getBinContent(bin_nr[0], bin_nr[1])
            probabilities.append(prob)
        print("probability:", probabilities)
        return np.product(probabilities)

    def __add__(self, other):
        assert isinstance(other, histogram_2d)
        assert self._xlow        == other._xlow
        assert self._xup         == other._xup
        assert self._ylow        == other._ylow
        assert self._yup         == other._yup
        assert self._bin_edges_x == other._bin_edges_x
        assert self._bin_edges_y == other._bin_edges_y

        self._values   = self._values + other._values
        self._num_hits = self._num_hits + other._num_hits

class pickledHistogram(histogram):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            contents = pickle.load(f)

            super().__init__(edges=contents['edges'])

            self._values = contents['values']
            self.reregister_hits(contents['hits'])

class pickled2dHistogram(histogram_2d):
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            contents = pickle.load(f)

            xlow = contents['build_params']['xlow']
            xup  = contents['build_params']['xup']
            ylow = contents['build_params']['ylow']
            yup  = contents['build_params']['yup']

            nr_x = contents['build_params']['x_nr']
            nr_y = contents['build_params']['y_nr']

            super().__init__([xlow,xup], [ylow,yup], nr_x, nr_y)

            self._values   = contents['values']
            self._num_hits = contents['hits']

def from_pickle(pickle_path, is_2d=False):
    if not is_2d:
        return pickledHistogram(pickle_path)
    if is_2d:
        return pickled2dHistogram(pickle_path)
