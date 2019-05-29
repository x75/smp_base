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

# from smp_base.impl import smpi
# np = smpi('numpy')

import numpy as np
from scipy.stats import wasserstein_distance

import logging
from smp_base.common import get_module_logger, compose
loglevel_debug = logging.DEBUG - 0
logger = get_module_logger(modulename = 'measures', loglevel = logging.DEBUG)

try:
    from pyemd import emd as pyemd
    from pyemd import emd_with_flow as pyemd_with_flow
    from pyemd import emd_samples as pyemd_samples
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

# def meas_mse(x = None, x_2 = None, *args, **kwargs):
def meas_mse(x_1 = None, x_2 = None, *args, **kwargs):
    """smp_base.measures.meas_mse

    Compute mean squared error MSE :math:`= 1/N \sum (x - x_2)^2`

    Args:

    - `x_1` (ndarray): matrix of points :math:`x_1`
    - `x_2` (ndarray): matrix of point :math:`x_2`

    Returns:

    - mse(ndarray): MSE :math:` := 1/N \sum_i^N (x_1_i - x_2_i)^2`
    """
    axis = 0
    keepdims = True
    if 'axis' in kwargs:
        axis = kwargs['axis']
    if 'keepdims' in kwargs:
        keepdims = kwargs['keepdims']
    # legacy argname
    if 'x' in kwargs:
        x_1 = kwargs['x']
        
    mse = np.mean(np.power(x_1 - x_2, 2), axis=axis, keepdims=keepdims)
    return mse

def meas_rmse(x = None, x_2 = None, *args, **kwargs):
    """smp_base.measures.meas_rmse

    Compute root mean squared error $\text{rmse} = \sqrt{1/N \sum(x - x_2)^2$}
    """
    mse = meas_mse(x, x_2, *args, **kwargs)
    return np.sqrt(mse)

def div_wasserstein(h1, h2, w1=None, w2=None):
    # logger.debug('{0}, {1}'.format(h1.shape, h2.shape))
    d = wasserstein_distance(h1, h2, w1, w2)
    return d

def htofloat(h):
    if h.dtype == np.int:
        return h.astype(np.float)
    return h

def div_kl(h1, h2, *args, **kwargs):
    """measures.div_kl

    Discrete kullback leibler divergence for histogram.

    Args:
    - h1(array): histogram 1
    - h2(array): histogram 2

    Returns:
    - sum divergence, element-wise divergences
    """
    _loglevel = loglevel_debug - 1
    logger.log(_loglevel, "h1 = {0}, h2 = {1}".format(h1, h2))

    h1 = htofloat(h1)
    h2 = htofloat(h2)
        
    # div = np.sum(h1 * np.log(h1/h2))
    # get input sums
    h1_sum = np.sum(h1)
    h2_sum = np.sum(h2)
    logger.log(_loglevel, "h1_sum = %s", h1_sum)
    logger.log(_loglevel, "h2_sum = %s", h2_sum)
    # normalize to a density if necessary
    if h1_sum > 1.0:
        h1 /= h1_sum
    if h2_sum > 1.0:
        h2 /= h2_sum
    logger.log(_loglevel, "h1 = %s", h1)
    logger.log(_loglevel, "h2 = %s", h2)
    # fix division by zero if h2 contains 0 elements
    if np.any(h2 == 0.0):
        # by adding small amplitude noise
        h2 += np.random.exponential(1e-4, h2.shape)
        # logger.debug('div_kl h2 == 0')

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
    # logger.debug('div_kl sum(div) = %s, div = %s', np.sum(div), div)
    return np.sum(div), div

def div_chisquare(h1, h2, *args, **kwargs):
    """chi-square divergence of two histograms

    Arguments:
     - h1(ndarray): first histogram
     - h2(ndarray): second histogram

    Returns:
     - div(ndarray): the chi-square divergence
    """
    h1 = htofloat(h1)
    h2 = htofloat(h2)
    # if np.sum(h1) > 1.0: h1 /= np.sum(h1)
    # if np.sum(h2) > 1.0: h2 /= np.sum(h2)
    # np.sum()
    div = 1.0 * np.square(h1 - h2)/(h1 + h2 + np.random.uniform(-1e-6, 1e-6, h1.shape))
    # logger.debug('div_chisquare sum(div) = %s, div = %s', np.sum(div), div)
    return np.sum(div), div

# earth mover's distance 'pyemd'
def div_pyemd_HAVE_PYEMD(h1, h2, *args, **kwargs):
    """earth movers distance using pyemd

    Earth movers distance between two distributions of n-dimensional
    points is a work measure from the product ground distance x mass.

    Pyemd version requires an explicit distance matrix.
    """
    h1 = htofloat(h1)
    h2 = htofloat(h2)
    logger.debug('    len(h1) = {0}'.format(len(h1)))
    
    flow = np.zeros((1,1))
    
    # compute distance matrix (pyemd only?)
    if 'x1_x' in kwargs and 'x2_x' in kwargs:
        distmat = kwargs['x1_x'][None,:] - kwargs['x2_x'][:,None]
        logger.debug('    distmat = %s' % (distmat.shape, ))
    else:
        distmat = np.ones((len(h1), len(h2)))
    
    if 'flow' in kwargs and kwargs['flow']:
        div, flow = pyemd_with_flow(h1, h2, distmat)
    else:
        # div = pyemd(h1, h2, distmat)
        div = pyemd_samples(h1, h2, bins=len(h1))
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

# earth mover's distance 'emd'
def meas_emd_HAVE_EMD(h1, h2, *args, **kwargs):
    """earth movers distance using emd

    Earth movers distance between two distributions of n-dimensional
    points is a work measure from the product ground distance x mass.

    Emd version requires an explicit distance matrix.
    """
    h1 = np.atleast_2d(h1)
    h2 = np.atleast_2d(h2)
    
    logger.debug('meas_emd_HAVE_EMD h1 = {0}, h2 = {1}'.format(h1.shape, h2.shape))

    # flow = np.zeros((1,1))
    # if 'flow' in kwargs and kwargs['flow']:
    #     div, flow = emd.emd(h1, h2, return_flows=True)
    # else:
    
    div = emd.emd(h1, h2)
    logger.debug('meas_emd_HAVE_EMD div {0}'.format(div))
    
    flow = np.array(flow)
    flow_zero_diag = flow - np.diag(np.diag(flow))
    flow_ = np.sum(flow_zero_diag, axis = 1, keepdims = True).T/2.0
    # logger.debug('meas_emd_HAVE_EMD sum(div) = %s, flow_ = %s, flow = %s', np.sum(div), flow_.shape, flow)
    return div, flow_
    
def meas_emd_(h1, h2, *args, **kwargs):
    logger.warning('emd could not be imported.')
    return -1

if HAVE_EMD: meas_emd = meas_emd_HAVE_EMD
else: meas_emd = meas_emd_

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
    'abs': {'func': compose(np.abs, np.subtract)},
    'mse': {'func': meas_mse},
    'rmse': {'func': meas_rmse},
    'hist': {'func': meas_hist}, # compute histogram
    'kld':  {'func': div_kl},
    'chisq':  {'func': div_chisquare},
}

if HAVE_PYEMD:
    measures['pyemd'] = {'func': div_pyemd}

if HAVE_EMD:
    measures['emd'] = {'func': meas_emd}

if __name__ == '__main__':

    from functools import partial
    # run tests

    # fix the seed
    N = 1000
    np.random.seed(19847)
    
    # generate test data
    testdata = {}
    # testdata['t1'] = {
    #     'x1': np.zeros((100, 1)),
    #     'x2': np.ones((100, 1)),
    #     'mse_target': np.ones((1,1)),
    #     'rmse_target': np.ones((1,1)),
    #     'emd_target': (np.array([[72.0]]), np.zeros((N,1))),
    # }
    # testdata['t2'] = {
    #     'x1': np.zeros((100, 1)),
    #     'x2': np.random.uniform(-1, 1, (100, 1)),
    #     'mse_target': np.array([[0.29175371]]),
    #     'rmse_target': np.array([[0.54014231]]),
    # }
    testdata['t3'] = {
        'x1': np.random.uniform(0, 1, (N, 1)),
        'x2': np.random.uniform(2, 3, (N, 1)),
        'mse_target': np.array([[4.1946]]),
        'rmse_target': np.array([[2.0481]]),
        'emd_target': (np.array([[72.0]]), np.zeros((N,1))),
    }

    testfuncs_meas = {
        'mse': meas_mse, 'rmse': meas_rmse,
        # 'wasserstein': meas_wasserstein,
        # 'emd': meas_emd,
    }
    
    for testset in testdata:
        for testfunc in testfuncs_meas:
            testdata[testset][testfunc] = testfuncs_meas[testfunc](testdata[testset]['x1'], testdata[testset]['x2'])
            logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata[testset][testfunc]))
            # assert testdata[testset]['mse_target'] == testdata[testset]['mse'], 'Test {0} failed for {1}: {2} {3}'.format(testset, 'mse', testdata[testset]['mse_target'], testdata[testset]['mse'])
            error = np.sum(np.abs(testdata[testset]['{0}_target'.format(testfunc)] - testdata[testset][testfunc]))
            assert error <= 0.01, 'Test {0} failed for {1}: {2} {3}'.format(testset, testfunc, testdata[testset]['{0}_target'.format(testfunc)], testdata[testset][testfunc])
            logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata[testset][testfunc]))

    # divergence
    testdata_div = {}
    testdata_div['t1'] = {
        'x1': np.histogram(np.random.uniform(0, 1, (N, 1)))[0],
        'x2': np.histogram(np.random.uniform(0, 1, (N, 1)))[0],
        'kld_target': (np.array([[0.0118]]), np.zeros((N,1))),
        'chisq_target': (np.array([[10.8331]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[4.1999]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[59.0]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[3.99]]),
    }
    testdata_div['t2'] = {
        'x1': np.histogram(np.random.uniform(0, 1, (N, 1)))[0],
        'x2': np.histogram(np.random.normal(0, 1, (N, 1)))[0],
        'kld_target': (np.array([[0.6278]]), np.zeros((N,1))),
        'chisq_target': (np.array([[406.1133]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[69.66]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[390.0]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[70.6]]),
    }
    testdata_div['t3'] = {
        'x1': np.histogram(np.random.uniform(0, 1, (N, 1)), density=True)[0],
        'x2': np.histogram(np.random.normal(0, 1, (N, 1)), density=True)[0],
        'kld_target': (np.array([[0.8630]]), np.zeros((N,1))),
        'chisq_target': (np.array([[0.5057]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[0.0768]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[0.4439]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[0.0771]]),
    }

    testfuncs_div = {
        'kld': div_kl, 'chisq': div_chisquare,
        'pyemd': div_pyemd, 'pyemd_flow': partial(div_pyemd, flow=True),
        'wasserstein': div_wasserstein
    }
            
    for testset in testdata_div:
        for testfunc in testfuncs_div:
            testdata_div[testset][testfunc] = testfuncs_div[testfunc](testdata_div[testset]['x1'], testdata_div[testset]['x2'])
            # assert testdata_div[testset]['mse_target'] == testdata_div[testset]['mse'], 'Test {0} failed for {1}: {2} {3}'.format(testset, 'mse', testdata_div[testset]['mse_target'], testdata_div[testset]['mse'])
            # error = np.sum(np.abs(testdata_div[testset]['{0}_target'.format(testfunc)] - testdata_div[testset][testfunc]))
            ret = testdata_div[testset][testfunc]
            if type(ret) is tuple:
                logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata_div[testset][testfunc][0]))
                error = np.sum(np.abs(testdata_div[testset]['{0}_target'.format(testfunc)][0] - testdata_div[testset][testfunc][0]))
                assert error <= 0.01, 'Test {0} failed for {1}: {2} {3}'.format(testset, testfunc, testdata_div[testset]['{0}_target'.format(testfunc)][0], testdata_div[testset][testfunc][0])
                logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata_div[testset][testfunc][0]))
            else:
                error = np.sum(np.abs(testdata_div[testset]['{0}_target'.format(testfunc)] - testdata_div[testset][testfunc]))
                assert error <= 0.01, 'Test {0} failed for {1}: {2} {3}'.format(testset, testfunc, testdata_div[testset]['{0}_target'.format(testfunc)], testdata_div[testset][testfunc])
                logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata_div[testset][testfunc]))
