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

def meas_mse(x = None, x_ = None, *args, **kwargs):
    """smp_base.measures.meas_mse

    Compute mean squared error mse = 1/N \sum (x - x_)^2
    """
    return np.mean(np.power(x - x_, 2), axis = 0, keepdims = True)

def div_kl(h1, h2, *args, **kwargs):
    """naive kullback leibler divergence for histogram, element-wise
    """
    # print "h1", h1, "h2", h2
    # div = np.sum(h1 * np.log(h1/h2))
    if np.sum(h1) > 1.0: h1 /= np.sum(h1)
    if np.sum(h2) > 1.0: h2 /= np.sum(h2)
    # logger.log(loglevel_debug, "h1", h1)
    # logger.log(loglevel_debug, "h2", h2)
    log_diff = np.clip(np.log(h1/h2), -20.0, 7.0)
    # logger.log(loglevel_debug, "log diff", log_diff)
    div = h1 * log_diff
    # logger.log(loglevel_debug, "div", div.shape, div)
    return div

def div_chisquare(h1, h2, *args, **kwargs):
    if np.sum(h1) > 1.0: h1 /= np.sum(h1)
    if np.sum(h2) > 1.0: h2 /= np.sum(h2)
    # np.sum()
    div = 0.5 * np.square(h1 - h2)/(h1 + h2 + np.random.uniform(-1e-6, 1e-6, h1.shape))
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
