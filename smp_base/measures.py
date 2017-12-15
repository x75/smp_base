"""smp_base.measures

measures: errors, distances, divergences

.. moduleauthor:: 2017 Oswald Berthold

Measures measure things about data like *point-wise errors*,
*statistical moments*, *distances*, *entropy*, *shared entropy*,
*complexity*, and so on.

Information theoretic measures like entropy, shared entropy and its
conditional variants are implemented in the separate
:file:`measures_infth.py`.
"""

import numpy as np

import logging
from smp_base.common import get_module_logger

try:
    from pyemd import emd as pyemd
    from pyemd import emd_with_flow as pyemd_with_flow
    HAVE_PYEMD = True
except ImportError as e:
    print("Couldn't import emd from pyemd with %s, make sure pyemd is installed." % (e, ))
    HAVE_PYEMD = False

try:
    from emd import emd
    HAVE_EMD = True
except ImportError as e:
    print("Couldn't import emd from emd with %s, make sure emd is installed." % (e, ))
    HAVE_EMD = False

loglevel_debug = logging.DEBUG - 0
logger = get_module_logger(modulename = 'measures', loglevel = logging.DEBUG)

def meas_mse(x = None, x_ = None, *args, **kwargs):
    """smp_base.measures.meas_mse

    Compute mean squared error mse = 1/N \sum (x - x_)^2

    Arguments:
     - x(ndarray): matrix of points $x_1$
     - x_(ndarray): matrix of point $x_2$

    Returns:
     - mse(ndarray): $\text{mse} := 1/N \sum_i^N (x_1_i - x_2_i)^2$
    """
    return np.mean(np.power(x - x_, 2), axis = 0, keepdims = True)

def div_kl(h1, h2, *args, **kwargs):
    """naive kullback leibler divergence for histogram, element-wise
    """
    _loglevel = loglevel_debug - 1
    # print "h1", h1, "h2", h2
    # div = np.sum(h1 * np.log(h1/h2))
    # get input sums
    h1_sum = np.sum(h1)
    h2_sum = np.sum(h2)
    logger.log(_loglevel, "h1_sum = %s", type(h1_sum))
    logger.log(_loglevel, "h2_sum = %s", h2_sum)
    # normalize to a density if necessary
    if h1_sum > 1.0: h1 /= h1_sum
    if h2_sum > 1.0: h2 /= h2_sum
    logger.log(_loglevel, "h1 = %s", h1)
    logger.log(_loglevel, "h2 = %s", h2)
    # fix division by zero if h2 contains 0 elements
    if np.any(h2 == 0.0):
        # by adding small amplitude noise
        h2 += np.random.exponential(1e-4, h2.shape)
    # get term 1
    log_h1_h2 = np.log(h1/h2)
    logger.log(_loglevel, "log(h1/h2) = %s", log_h1_h2)
    # sanitize term 1
    log_diff = np.clip(log_h1_h2, -20.0, 7.0)
    logger.log(_loglevel, "log diff = %s", log_diff)
    # get term 2, final
    div = h1 * log_diff
    logger.log(_loglevel, "div = %s/%s", div.shape, div)
    # return element-wise divergence
    logger.debug('div_kl sum(div) = %s, div = %s', np.sum(div), div)
    return np.sum(div), div

def div_chisquare(h1, h2, *args, **kwargs):
    """chi-square divergence of two histograms

    Arguments:
     - h1(ndarray): first histogram
     - h2(ndarray): second histogram

    Returns:
     - div(ndarray): the chi-square divergence
    """
    # if np.sum(h1) > 1.0: h1 /= np.sum(h1)
    # if np.sum(h2) > 1.0: h2 /= np.sum(h2)
    # np.sum()
    div = 1.0 * np.square(h1 - h2)/(h1 + h2 + np.random.uniform(-1e-6, 1e-6, h1.shape))
    logger.debug('div_chisquare sum(div) = %s, div = %s', np.sum(div), div)
    return np.sum(div), div

# earth mover's distance
def div_pyemd_HAVE_PYEMD(h1, h2, *args, **kwargs):
    """earth movers distance using pyemd

    Earth movers distance between two distributions of n-dimensional
    points is a work measure from the product ground distance x mass.

    Pyemd version requires an explicit distance matrix.
    """
    flow = np.zeros((1,1))
    if 'flow' in kwargs and kwargs['flow']:
        div, flow = pyemd_with_flow(h1, h2, args[0])
    else:
        div = pyemd(h1, h2, args[0])
    flow = np.array(flow)
    flow_zero_diag = flow - np.diag(np.diag(flow))
    flow_ = np.sum(flow_zero_diag, axis = 1, keepdims = True).T/2.0
    # logger.debug('div_pyemd_HAVE_PYEMD sum(div) = %s, flow_ = %s, flow = %s', np.sum(div), flow_.shape, flow)
    return div, flow_
    
def div_pyemd_(h1, h2, *args, **kwargs):
    logger.warning('pyemd could not be imported.')
    return -1

if HAVE_PYEMD: div_pyemd = div_pyemd_HAVE_PYEMD
else: div_pyemd = div_pyemd_

def meas_hist(x = None, bins = None, *args, **kwargs):
    """smp_base.measures.meas_hist

    Compute histogram of 'x' with bins 'bins' by wrapping np.histogram.
    """
    return np.histogram(x, bins, **kwargs)

class meas(object):
    def __init__(self):
        pass

    def step(self, x):
        pass

    @staticmethod
    def identity(x, *args, **kwargs):
        return -x
    
    @staticmethod
    def square(x, *args, **kwargs):
        return np.square(x)

    @staticmethod
    def abs(x, *args, **kwargs):
        return np.abs(x)

    @staticmethod
    def abs_sqrt(x, *args, **kwargs):
        return np.sqrt(np.abs(x))

    @staticmethod
    def sum_abs(x, *args, **kwargs):
        return np.ones_like(x) * np.sum(np.abs(x))

    @staticmethod
    def sum_square(x, *args, **kwargs):
        return np.ones_like(x) * np.sum(np.square(x))

    @staticmethod
    def sum_sqrt(x, *args, **kwargs):
        return np.ones_like(x) * np.sum(np.sqrt(meas.abs(x)))

    # accel
    @staticmethod
    def abs_accel(x, *args, **kwargs):
        return np.abs(x)

    @staticmethod
    def perf_accel(err, acc, *args, **kwargs):
        """Simple reward: let body acceleration point into reduced error direction"""
        return np.sign(err) * acc
        # self.perf = np.sign(err) * np.sign(acc) * acc**2

measures = {
    'sub': {'func': np.subtract},
    'mse': {'func': meas_mse},
    'hist': {'func': meas_hist}, # compute histogram
    'kld':  {'func': div_kl},
    'chisq':  {'func': div_chisquare},
}

if HAVE_PYEMD:
    measures['pyemd'] = {'func': div_pyemd}
