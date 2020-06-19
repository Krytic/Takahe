"""

Histogram classes to contain event rate data and allow for easy plotting

Original author: Max Briel
Modified by: Sean Richards

"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

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
        return out

    def __truediv__(self, other):
        out = self.copy()
        out._values = self._values / other
        return out

    def copy(self):
        """ creates a copy of the histogram

        Returns
        -------
        histogram
            An exact (deep) copy of the histogram

        """
        out = histogram(xlow=self._xlow, xup=self._xup, nr_bins=self._nr_bins)
        out._values = self._values
        return out

    def fill(self, x, w=1):
        """ Fill the histogram with data.


        Parameters
        ----------
        x : float/array
            Either a single entry or an array of *N* to put into the histogram
        w : float/array
            The weight of the entry of *N* entries to be added to the histogram.
        """
        def f(f, g):
            if f >= self._xup:
                 self._values[self._nr_bins-1] += g
            elif f <= self._xlow:
                self._values[0] += g
            else:
                bin_nr = np.where(self.lower_edges <= f)[0][-1]
                self._values[bin_nr] += g

        if not isinstance(x, type(0.0)):
            if isinstance(w, type(0.0)):
                for i in range(0, len(x)):
                    f(x[i], w)
            elif len(x) != len(w):
                raise Exception("Weights needs to be as long as x")
            else:
                for i in range(0, len(x)):
                    f(x[i], w[i])
        else:
            f(x, w)
        return None

    def plot(self, *argv, **kwargs):
        """Plot the histogram. matplotlib.pyplot arguments can be passed on too
        """
        _ = plt.hist(self._bin_edges[:-1], self._bin_edges, weights=self._values,histtype=u'step', *argv, **kwargs)
        return None

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
            raise Exception("x outside of range")

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
            raise Exception("x1 should be larger than x2")
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
            raise Exception("x1 should be larger than x2")
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


class BPASS_hist(histogram):
    """ Container for the BPASS data to reside and make it possible to perform basic
    operations on the enclosed data.

    This class inherits all function of histogram, but overrides a few to allow
    for easier manipulation of linear and logarithmic data.

    The internal binning is linear in Gyr, but plotting can be done in linear and
    logarithmic binning.
    """

    def __init__(self):
        super().__init__(edges=self.getLinEdges())

    def fill(self, x, w=1, ty=None):
        """Adds data to the BPASS histogram. Data values should be in years

        Parameters
        ----------
        x : float/array
            Either a single entry or an array of *N* to put into the histogram
        w : float/array
            The weight of the entry of *N* entries to be added to the histogram.
        ty: string
            The type of input data. Either "log" or "lin".
            If none is given, log is assumed.
        """
        if (ty == "log") or (ty == None):
            x = 10**x /1e9

        histogram.Fill(self, x, w)

    def integral(self, x1, x2):
        """ Returns the integral of the histogram between **x1** and **x2**.
            It return an integral in units :math:`\#events/yr/M_\odot`

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
        return histogram.integral(self, x1, x2) *1e9


    def plotLog(self, *argv, **kwargs):
        """ Plot the histogram on a logirthmic axis
        """
        _ = plt.hist(self.getLogEdges()[:-1], self.getLogEdges(), weights=self._values, histtype=u'step', *argv, **kwargs)
        _ = plt.xlabel("log(age/yr)")
        return None

    def plotLin(self, *argv, **kwargs):
        """Plot the histogram on a linear axis.
        """
        _ = plt.hist(self.getLinEdges()[:-1], self.getLinEdges(), weights=self._values, histtype=u'step', *argv, **kwargs)
        _ = plt.xlabel("age/Gyr")
        return None

    def getLogBins(self):
        """Returns the middle points of all bins, except the first bin, which
        runs from 0 to :math:`10^{6.05}`, but returns 6.

        Returns
        -------
        array
            The middle points of all bins in logarithmic space

        """
        return np.linspace(6,11,51)

    def getLinBins(self):
        """Returns the middle points of all bins, except the first bin, which
        runs from 0 to :math:`10^{6.05}``, but returns :math:`10^6`.

        Returns
        -------
        array
            The middle points of all bins in linear space

        """
        return 10**np.linspace(6, 11, 51)

    def getLogEdges(self):
        """Returns the edges in logarithmic space, except for the first bin,
        where the lowest edge is set to 5.95 instead of 0 for plotting purposes.

        Returns
        -------
        array
            The bin edges in logarithmic space

        """
        return np.linspace(5.95, 11.05, 52)


    def getLinEdges(self):
        """Returns the edges in linear space


        Returns
        -------
        array
            The bin edges in linear space

        """
        x = [0.0]
        x.extend(10**np.linspace(6.05, 11.05, 51)/1e9)
        return x

    def copy(self):
        """Create a copy of the BPASS histogram

        Returns
        -------
        BPASS_hist
            A copy of the BPASS historam.
        """

        out = BPASS_hist()
        out._values = self._values
        return out
