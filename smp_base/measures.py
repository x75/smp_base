"""smp_base - smp sensorimotor experiments base functions

measures

2017 Oswald Berthold

Measures measure things about data like statistical moments,
distances, entropy, complexity and so on

Information theoretic measures like entropy, shared entropy and its
conditional variants reside in their own file
:file:`measures_infth.py`
"""

import numpy as np

import logging
from smp_base.common import get_module_logger

loglevel_debug = logging.DEBUG - 0
logger = get_module_logger(modulename = 'measures', loglevel = logging.DEBUG)

def meas_mse(x = None, x_ = None, *args, **kwargs):
    """smp_base.measures.meas_mse

    Compute mean squared error mse = 1/N \sum (x - x_)^2
    """
    return np.mean(np.power(x - x_, 2), axis = 0, keepdims = True)

def div_kl(h1, h2, *args, **kwargs):
    """naive kullback leibler divergence for histogram, element-wise
    """
    _loglevel = loglevel_debug - 1
    # print "h1", h1, "h2", h2
    # div = np.sum(h1 * np.log(h1/h2))
    h1_sum = np.sum(h1)
    h2_sum = np.sum(h2)
    logger.log(_loglevel, "h1_sum = %s", type(h1_sum))
    logger.log(_loglevel, "h2_sum = %s", h2_sum)
    if h1_sum > 1.0: h1 /= h1_sum
    if h2_sum > 1.0: h2 /= h2_sum
    logger.log(_loglevel, "h1 = %s", h1)
    logger.log(_loglevel, "h2 = %s", h2)
    log_h1_h2 = np.log(h1/h2)
    logger.log(_loglevel, "log(h1/h2) = %s", log_h1_h2)
    log_diff = np.clip(log_h1_h2, -20.0, 7.0)
    logger.log(_loglevel, "log diff = %s", log_diff)
    div = h1 * log_diff
    logger.log(_loglevel, "div = %s/%s", div.shape, div)
    return div

def div_chisquare(h1, h2, *args, **kwargs):
    # if np.sum(h1) > 1.0: h1 /= np.sum(h1)
    # if np.sum(h2) > 1.0: h2 /= np.sum(h2)
    # np.sum()
    div = 1.0 * np.square(h1 - h2)/(h1 + h2 + np.random.uniform(-1e-6, 1e-6, h1.shape))
    return div

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
