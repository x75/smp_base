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

try:
    from cbemd import cbemd
    HAVE_CBEMD = True
except ImportError as e:
    print("Couldn't import cbemd from cbemd with %s, make sure cbemd is installed." % (e, ))
    HAVE_CBEMD = False

# TODO: cbemd, wasserstein

def htofloat(h):
    if h.dtype == np.int:
        return h.astype(np.float)
    return h

################################################################################
# distance measures

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

################################################################################
# divergence measures

def div_kl(h1, h2, x1=None, x2=None, *args, **kwargs):
    """measures.div_kl

    Discrete kullback leibler divergence for histogram.

    .. TODO:: fix KLD for arbitrary support?

    Args:
    - h1(np.ndarray): histogram 1
    - h2(np.ndarray): histogram 2
    - x1(np.ndarray): support values 1
    - x2(np.ndarray): support values 2

    Returns:
    - ret(tuple): sum divergence, element-wise divergences
    """
    _loglevel = loglevel_debug - 1
    logger.log(_loglevel, "div_kld: h1 = {0}, h2 = {1}".format(h1, h2))

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

def div_chisquare(h1, h2, x1=None, x2=None, *args, **kwargs):
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
def div_pyemd_HAVE_PYEMD(h1, h2, x1=None, x2=None, *args, **kwargs):
    """earth movers distance using pyemd

    Earth movers distance between two distributions of n-dimensional
    points is a work measure from the product ground distance x mass.

    Pyemd version requires an explicit distance matrix.
    """
    h1 = htofloat(h1)[:]
    h2 = htofloat(h2)[:]

    # logger.debug('    len(h1) = {0}'.format(len(h1)))
    # logger.debug('div_pyemd_HAVE_PYEMD: h1 = {0}, h2 = {1}'.format(h1, h2))
    # logger.debug('div_pyemd_HAVE_PYEMD: x1 = {0}, x2 = {1}'.format(x1, x2))
    
    flow = np.zeros((1,1))
    
    # compute distance matrix (pyemd only?)
    if 'x1_x' in kwargs and 'x2_x' in kwargs:
        x1 = kwargs['x1_x']
        x2 = kwargs['x2_x']
    # else:
    #     distmat = np.ones((len(h1), len(h2)))
    
    distmat = x1[None,:] - x2[:,None]
    # logger.debug('    distmat = %s' % (distmat.shape, ))
        
    if 'flow' in kwargs and kwargs['flow']:
        div, flow = pyemd_with_flow(h1, h2, distmat)
    else:
        div = pyemd(h1, h2, distmat)
        
    # div = pyemd_samples(h1, h2, bins=len(h1))
    flow = np.array(flow)
    flow_zero_diag = flow - np.diag(np.diag(flow))
    flow_ = np.sum(flow_zero_diag, axis = 1, keepdims = True).T/2.0
    # logger.debug('div_pyemd_HAVE_PYEMD sum(div) = %s, flow_ = %s, flow = %s', np.sum(div), flow_.shape, flow)
    return div, flow_
    
def div_pyemd_(h1, h2, x1=None, x2=None, *args, **kwargs):
    logger.warning('pyemd could not be imported.')
    return -1, None

if HAVE_PYEMD: div_pyemd = div_pyemd_HAVE_PYEMD
else: div_pyemd = div_pyemd_

# earth mover's distance 'emd'
def div_emd_HAVE_EMD(h1, h2, x1=None, x2=None, *args, **kwargs):
    """earth movers distance using emd

    Earth movers distance between two distributions of n-dimensional
    points is a work measure from the product ground distance x mass.

    emd.emd requires original observation, thus measure
    """
    h1 = np.atleast_2d(h1)
    h2 = np.atleast_2d(h2)
    
    # logger.debug('div_emd_HAVE_EMD h1 = {0}, h2 = {1}'.format(h1, h2))
    # logger.debug('div_emd_HAVE_EMD x1 = {0}, x2 = {1}'.format(x1, x2))

    # flow = np.zeros((1,1))
    # if 'flow' in kwargs and kwargs['flow']:
    #     div, flow = emd.emd(h1, h2, return_flows=True)
    # else:
    
    div = emd.emd(h1, h2)
    logger.debug('div_emd_HAVE_EMD div {0}'.format(div))
    
    flow = np.array(flow)
    flow_zero_diag = flow - np.diag(np.diag(flow))
    flow_ = np.sum(flow_zero_diag, axis = 1, keepdims = True).T/2.0
    # logger.debug('div_emd_HAVE_EMD sum(div) = %s, flow_ = %s, flow = %s', np.sum(div), flow_.shape, flow)
    return div, flow_
    
def div_emd_(h1, h2, *args, **kwargs):
    logger.warning('emd could not be imported.')
    return -1

if HAVE_EMD: div_emd = div_emd_HAVE_EMD
else: div_emd = div_emd_

# earth mover's distance 'cbemd'
def div_cbemd_HAVE_CBEMD(h1, h2, x1=None, x2=None, *args, **kwargs):
    """earth movers distance using cbemd

    Earth movers distance between two distributions of n-dimensional
    points is a work measure from the product ground distance x mass.

    cbemd.cbemd requires original observation, thus measure
    """
    logger.debug('div_cbemd_HAVE_CBEMD h1 = {0}, h2 = {1}'.format(h1.shape, h2.shape))
    logger.debug('div_cbemd_HAVE_CBEMD h1 = {0}, h2 = {1}'.format(h1.dtype, h2.dtype))
    logger.debug('div_cbemd_HAVE_CBEMD x1 = {0}, x2 = {1}'.format(x1.shape, x2.shape))
    logger.debug('div_cbemd_HAVE_CBEMD x1 = {0}, x2 = {1}'.format(x1.dtype, x2.dtype))

    # flow = np.zeros((1,1))
    # if 'flow' in kwargs and kwargs['flow']:
    #     div, flow = cbemd.cbemd(h1, h2, return_flows=True)
    # else:
    
    div = cbemd(x1, x2, h1.tolist(), h1.tolist())
    logger.debug('div_cbemd_HAVE_CBEMD div {0}'.format(div))
    
    # flow = np.array(flow)
    # flow_zero_diag = flow - np.diag(np.diag(flow))
    # flow_ = np.sum(flow_zero_diag, axis = 1, keepdims = True).T/2.0
    # logger.debug('div_cbemd_HAVE_CBEMD sum(div) = %s, flow_ = %s, flow = %s', np.sum(div), flow_.shape, flow)
    return div, None
    
def div_cbemd_(h1, h2, *args, **kwargs):
    logger.warning('cbemd could not be imported.')
    return -1

if HAVE_CBEMD: div_cbemd = div_cbemd_HAVE_CBEMD
else: div_cbemd = div_cbemd_

def div_wasserstein(h1, h2, x1=None, x2=None, *args, **kwargs):

    if 'x1_x' in kwargs:
        x1 = kwargs['x1_x']
    if 'x2_x' in kwargs:
        x2 = kwargs['x2_x']

    if 'x1w' in kwargs:
        x1 = kwargs['x1w']
    if 'x2w' in kwargs:
        x2 = kwargs['x2w']

    if 'w1' in kwargs:
        x1 = kwargs['w1']
    if 'w2' in kwargs:
        x2 = kwargs['w2']

    if x1 is None:
        x1 = np.linspace(0, 1, h1.shape[0])
    if x2 is None:
        x2 = np.linspace(0, 1, h2.shape[0])

    # logger.debug('weights: h1 {0}, h2 {1}'.format(h1, h2))
    # logger.debug(' values: x1 {0}, x2 {1}'.format(x1, x2))
        
    # u,v values, u,v weights
    d = wasserstein_distance(x1, x2, h1.tolist(), h2.tolist())
    return d, np.random.uniform(size=h1.shape)

def meas_hist(x = None, bins = None, *args, **kwargs):
    """smp_base.measures.meas_hist

    Compute histogram of 'x' with bins 'bins' by wrapping np.histogram.
    """
    return np.histogram(x, bins, **kwargs)

# meas class approach
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

def test_measures(*args, **kwargs):
    # generate test data
    testdata = {}
    testdata['t1'] = {
        'x1': np.zeros((100, 1)),
        'x2': np.ones((100, 1)),
        'mse_target': np.ones((1,1)),
        'rmse_target': np.ones((1,1)),
        'emd_target': (np.array([[72.0]]), np.zeros((N,1))),
    }
    testdata['t2'] = {
        'x1': np.zeros((100, 1)),
        'x2': np.random.uniform(-1, 1, (100, 1)),
        'mse_target': np.array([[0.29175371]]),
        'rmse_target': np.array([[0.54014231]]),
    }
    testdata['t3'] = {
        'x1': np.random.uniform(0, 1, (N, 1)),
        'x2': np.random.uniform(2, 3, (N, 1)),
        'mse_target': np.array([[4.1823]]),
        'rmse_target': np.array([[2.0481]]),
        # 'emd_target': (np.array([[72.0]]), np.zeros((N,1))),
    }

    testfuncs_meas = {
        'mse': meas_mse, 'rmse': meas_rmse,
        # 'wasserstein': meas_wasserstein,
        # 'emd': meas_emd,
    }
    
    logger.info('#### testing distance measures')
    for testset in testdata:
        for testfunc in testfuncs_meas:
            testdata[testset][testfunc] = testfuncs_meas[testfunc](testdata[testset]['x1'], testdata[testset]['x2'])
            logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata[testset][testfunc]))
            # assert testdata[testset]['mse_target'] == testdata[testset]['mse'], 'Test {0} failed for {1}: {2} {3}'.format(testset, 'mse', testdata[testset]['mse_target'], testdata[testset]['mse'])
            error = np.sum(np.abs(testdata[testset]['{0}_target'.format(testfunc)] - testdata[testset][testfunc]))
            assert error <= 0.01, 'Test {0} failed for {1}: {2} {3}'.format(testset, testfunc, testdata[testset]['{0}_target'.format(testfunc)], testdata[testset][testfunc])
            logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata[testset][testfunc]))

def test_divergence_1(*args, **kwargs):
    # divergence
    testdata_div = {}
    testdata_div['t1'] = {
        'x1': np.histogram(np.random.uniform(0, 1, (N, 1)))[0],
        'x2': np.histogram(np.random.uniform(0, 1, (N, 1)))[0],
        'kld_target': (np.array([[0.0118]]), np.zeros((N,1))),
        'chisq_target': (np.array([[14.9508]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[3.84]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[63.0]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[0.01322]]),
    }
    testdata_div['t2'] = {
        'x1': np.histogram(np.random.uniform(0, 1, (N, 1)))[0],
        'x2': np.histogram(np.random.normal(0, 1, (N, 1)))[0],
        'kld_target': (np.array([[0.6022]]), np.zeros((N,1))),
        'chisq_target': (np.array([[396.472]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[71.28]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[381.0]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[0.1375]]),
    }
    testdata_div['t3'] = {
        'x1': np.histogram(np.random.uniform(0, 1, (N, 1)), density=True)[0],
        'x2': np.histogram(np.random.normal(0, 1, (N, 1)), density=True)[0],
        'kld_target': (np.array([[0.8630]]), np.zeros((N,1))),
        'chisq_target': (np.array([[0.4863]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[0.0768]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[0.43]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[0.15533333333333335]]),
    }

    testfuncs_div = {
        'kld': div_kl, 'chisq': div_chisquare,
        'pyemd': div_pyemd, 'pyemd_flow': partial(div_pyemd, flow=True),
        'wasserstein': div_wasserstein
    }
            
    logger.info('#### testing divergence measures')
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

def test_divergence_2(*args, **kwargs):
    numbins = 10
    # divergence
    testdata_div = {}
    x1 = np.random.uniform(0, 1, (N, 1))
    x2 = np.random.uniform(0, 1, (N, 1))
    x1h = np.histogram(x1, density=True, bins=numbins)
    x2h = np.histogram(x2, density=True, bins=numbins)
    testdata_div['t1'] = {
        'h1': x1h[0],
        'h2': x2h[0],
        'x1': x1h[1][:-1],
        'x2': x2h[1][:-1],
        'kld_target': (np.array([[0.0092]]), np.zeros((N,1))),
        'chisq_target': (np.array([[0.0092]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[0.0051]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[0.0052]]), np.zeros((N,1))),
        'cbemd_target': (np.array([[0.00125]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[0.0123]]),
    }
    
    x1 = np.random.uniform(0, 1, (N, 1))
    x2 = np.random.normal(0, 1, (N, 1))
    x1h = np.histogram(x1, density=True, bins=numbins)
    x2h = np.histogram(x2, density=True, bins=numbins)
    testdata_div['t2'] = {
        'h1': x1h[0],
        'h2': x2h[0],
        'x1': x1h[1][:-1],
        'x2': x2h[1][:-1],
        'kld_target': (np.array([[0.565]]), np.zeros((N,1))),
        'chisq_target': (np.array([[0.394]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[0.176]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[0.176]]), np.zeros((N,1))),
        'cbemd_target': (np.array([[1.2524]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[0.769]]),
    }

    x1 = np.random.uniform(0, 1, (N, 1))
    x2 = np.random.normal(0, 1, (N, 1))
    x1h = np.histogram(x1, density=True, bins=numbins)
    x2h = np.histogram(x2, density=True, bins=numbins)
    testdata_div['t3'] = {
        'h1': x1h[0],
        'h2': x2h[0],
        'x1': x1h[1][:-1],
        'x2': x2h[1][:-1],
        'kld_target': (np.array([[0.561]]), np.zeros((N,1))),
        'chisq_target': (np.array([[0.372]]), np.zeros((N,1))),
        'pyemd_target': (np.array([[0.1215]]), np.zeros((N,1))),
        'pyemd_flow_target': (np.array([[0.1215]]), np.zeros((N,1))),
        'cbemd_target': (np.array([[1.32]]), np.zeros((N,1))),
        'wasserstein_target': np.array([[0.868]]),
    }

    testfuncs_div = {
        'kld': div_kl,
        'chisq': div_chisquare,
        'pyemd': div_pyemd,
        'pyemd_flow': partial(div_pyemd, flow=True),
        # 'emd': div_emd,
        'cbemd': div_cbemd,
        'wasserstein': div_wasserstein
    }
            
    logger.info('#' * 80)
    logger.info('# testing emd')
    for testset in testdata_div:
        for testfunc in testfuncs_div:
            testdata_div[testset][testfunc] = testfuncs_div[testfunc](
                testdata_div[testset]['h1'],
                testdata_div[testset]['h2'],
                testdata_div[testset]['x1'],
                testdata_div[testset]['x2'],
            )
            # assert testdata_div[testset]['mse_target'] == testdata_div[testset]['mse'], 'Test {0} failed for {1}: {2} {3}'.format(testset, 'mse', testdata_div[testset]['mse_target'], testdata_div[testset]['mse'])
            # error = np.sum(np.abs(testdata_div[testset]['{0}_target'.format(testfunc)] - testdata_div[testset][testfunc]))
            ret = testdata_div[testset][testfunc]
            if type(ret) is tuple:
                # logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata_div[testset][testfunc][0]))
                error = np.sum(np.abs(testdata_div[testset]['{0}_target'.format(testfunc)][0] - testdata_div[testset][testfunc][0]))
                assert error <= 0.01, 'Test {0} failed for {1}: {2} {3}'.format(testset, testfunc, testdata_div[testset]['{0}_target'.format(testfunc)][0], testdata_div[testset][testfunc][0])
                logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata_div[testset][testfunc][0]))
            else:
                error = np.sum(np.abs(testdata_div[testset]['{0}_target'.format(testfunc)] - testdata_div[testset][testfunc]))
                assert error <= 0.01, 'Test {0} failed for {1}: {2} {3}'.format(testset, testfunc, testdata_div[testset]['{0}_target'.format(testfunc)], testdata_div[testset][testfunc])
                logger.info('{0}, {1} = {2}'.format(testset, testfunc, testdata_div[testset][testfunc]))
                
        
measures = {
    'sub': {'func': np.subtract},
    'abs': {'func': compose(np.abs, np.subtract)},
    'mse': {'func': meas_mse},
    'rmse': {'func': meas_rmse},
    'hist': {'func': meas_hist}, # compute histogram
    'kld':  {'func': div_kl},
    'chisq':  {'func': div_chisquare},
    'wasserstein': {'func': div_wasserstein},
}

if HAVE_PYEMD:
    measures['pyemd'] = {'func': div_pyemd}

if HAVE_EMD:
    measures['emd'] = {'func': div_emd}
    
if HAVE_CBEMD:
    measures['cbemd'] = {'func': div_cbemd}
    
if __name__ == '__main__':

    from functools import partial
    # run tests

    # fix the seed
    N = 1000
    np.random.seed(19847)
    
    # # meas test
    # test_measures(N)
    
    # # old div test
    # test_divergence_1(N)

    # new div test
    test_divergence_2(N)
                
