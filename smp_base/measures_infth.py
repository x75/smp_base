"""smp_base.measures_infth

.. moduleauthor:: Oswald Berthold, 2017

Information theoretic measures: entropy, mutual information,
conditional mutual information, and derived aggregate measures based
on the java information dynamics toolkit (jidt) by Joe Lizier [1].

Current implementation consists of thin wrappers around jidt calls as
functions without a class.

.. TODO:: sift, sort and clean-up input from: smp/smp/infth.py,
   smp/playground/infth_feature_relevance.py, smp/sequence/\*.py,
   smp_sphero (was smp_infth), evoplast/ep3.py, smp/infth,
   smp/infth/infth_homeokinesis_analysis_cont.py,
   smp/infth/infth_playground, smp/infth/infth_explore.py,
   smp/infth/infth_pointwise_plot.py, smp/infth/infth_measures.py:
   unfinished, smp/infth/infth_playground.py,
   smp/infth/infth_EH-2D.py, smp/infth/infth_EH-2D_clean.py

[1] https://github.com/jlizier/jidt
"""
import sys, os
from pprint import pformat
import numpy as np


import smp_base.config as config
# from smp_base.measures import meas
from smp_base.measures import meas

import logging
from smp_base.common import get_module_logger

logger = get_module_logger(modulename = 'measures_infth', loglevel = logging.DEBUG)

try:
    from jpype import getDefaultJVMPath, isJVMStarted, startJVM, attachThreadToJVM, isThreadAttachedToJVM
    from jpype import JPackage
    HAVE_JPYPE = True
except ImportError as e:
    print("Couldn't import jpype, %s" % e)
    HAVE_JPYPE = False
    # sys.exit(1)

def init_jpype(jarloc=None, jvmpath=None):
    if not HAVE_JPYPE:
        print("Cannot initialize jpype because it couldn't be imported. Make sure jpype is installed")
        return

    jarloc = jarloc or config.__dict__.get(
        'JARLOC',
        '%s/infodynamics/infodynamics.jar' % os.path.dirname(os.__file__)
    )
    assert os.path.exists(jarloc), "Jar file %s doesn't exist" % jarloc

    jvmpath = jvmpath or config.__dict__.get('JVMPATH', getDefaultJVMPath())

    logger.debug("measures_infth.init_jpype: Setting jpype jvmpath = {0}".format(jvmpath))
    logger.debug("measures_infth.init_jpype: setting JIDT jar location = {0}".format(jarloc))

    # startJVM(getDefaultJVMPath(), "-ea", "-Xmx2048M", "-Djava.class.path=" + jarLocation)
    if not isJVMStarted():
        logger.debug("Starting JVM")
        startJVM(jvmpath, "-ea", "-Xmx8192M", "-Djava.class.path=" + jarloc)
    else:
        logger.debug("Attaching JVM")
        attachThreadToJVM()

# call init_jpype with global effects
init_jpype()

# obsolete 20180201
# # infth classes
# class measH(meas):
#     """!@brief Measure entropy"""
#     def __init__(self):
#         meas.__init__(self)

#     def step(self, x):
#         """Assume observations in rows, variables in columns"""
#         # print "%s step: x = %s" % (self.__class__.__name__, x.shape)
#         # return compute_entropy(src = x)
#         return infth_mi_multivariate({'X': x, 'Y': x})


# class measMI(meas):
#     def __init__(self):
#         meas.__init__(self)

#     def step(self, x, y):
#         """Assume observations in rows, variables in columns"""
#         # print "%s step: x = %s, y = %s" % (self.__class__.__name__, x.shape, y.shape)
#         return compute_mutual_information(src = x, dst = y)

class dec_compute_infth_soft(object):
    """wrap infth calls and fail softly
    """
    def __call__(self, f):
        def wrap(*args, **kwargs):
            # assert HAVE_JPYPE
            # assert isJVMStarted() and isThreadAttachedToJVM(), "Either JVM not started or thread not attached. Hm."
            if HAVE_JPYPE:
                return f(*args, **kwargs)
            else:
                return None

        return wrap

class dec_compute_infth(object):
    """wrap infth calls and fail hard
    """
    def __call__(self, f):
        def wrap(*args, **kwargs):
            assert HAVE_JPYPE
            assert isJVMStarted() and isThreadAttachedToJVM(), "Either JVM not started or thread not attached. Hm."
            return f(*args, **kwargs)

        return wrap

def prepare_data_and_attributes(data, check_shape = False): # False
    """smp_base.prepare_data_and_attributes

    Take a dict with keys 'X','Y' and copy each into src, dst
    variables respectively.

    Args:

    - data(dict): dictionary with 'X' and 'Y' np.ndarray

    Returns:

    - tuple: src, dst
    """
    # prepare data and attributes
    src = np.atleast_2d(data["X"])
    dst = np.atleast_2d(data["Y"])
    # condition
    if 'C' in data:
        cond = np.atleast_2d(data["C"])
    else:
        cond = np.zeros((1,1))
            
    # check orientation
    if check_shape:
        if src.shape[0] < src.shape[1]:
            src = src.T
        if dst.shape[0] < dst.shape[1]:
            dst = dst.T
        if cond.shape[0] < cond.shape[1]:
            cond = cond.T

    if 'C' in data:
        return src, dst, cond
    
    return src, dst

################################################################################
# wrap these into a thin class
# @dec_compute_infth()
def compute_entropy(src):
    """compute entropy

    Compute the global average entropy of the `src` data

    Wrapper for `compute_entropy_univariate` or
    `compute_entropy_multivariate`

    Args:

    - src(np.ndarray, tuple): entropy source data

    Returns:

    - h(float): the average entropy over the `src` episode

    """
    if src.shape[1] > 1:
        return compute_entropy_multivariate(src)
    else:
        return compute_entropy_multivariate(src)

# @dec_compute_infth()
def compute_entropy_univariate(src):
    ent_class = JPackage('infodynamics.measures.continuous.kernel').EntropyCalculatorKernel
    ent = ent_class()
    ent.setProperty("NORMALISE", "true")
    # what's that?
    ent.initialise(0.1)
    ent.setObservations(src)
    h = ent.computeAverageLocalOfObservations()
    return h

# @dec_compute_infth()
def compute_entropy_multivariate(src, delay = 0):
    """compute_entropy_multivariate

    Compute the joint entropy as the self-information I(src;src|delay=delay).
    """
    # concatenate all arrays in tuple
    if type(src) is tuple:
        randvars = (rv for rv in src if rv is not None)
        src = np.hstack(tuple(randvars))
    # otherwise: array already

    # ent_class = JPackage('infodynamics.measures.continuous.kernel').EntropyCalculatorMultiVariateKernel
    # ent_class = JPackage('infodynamics.measures.continuous.gaussian').EntropyCalculatorMultiVariateGaussian
    # ent_class = JPackage('infodynamics.measures.continuous.kozachenko').EntropyCalculatorMultiVariateKozachenko

    # ent_class = JPackage('infodynamics.measures.continuous.kraskov1').MutualInfoCalculatorMultiVariateKraskov1
    # ent = ent_class()
    # ent.setProperty("NORMALISE", "true")
    # # ent.initialise(src.shape[1], 0.1)
    # ent.initialise(src.shape[1], src.shape[1])
    # ent.setObservations(src, src)
    # h = ent.computeAverageLocalOfObservations()

    return infth_mi_multivariate(data = {'X': src, 'Y': src}, delay = delay)

# def compute_mi_multivariate(*args, **kwargs):
#     return infth_mi_multivariate(data = kwargs['data'], estimator = kwargs['estimator'], normalize = kwargs['normalize'])

def compute_mi_multivariate(data = {}, estimator = "kraskov1", normalize = True, delay = 0):
    return infth_mi_multivariate(data = data, estimator = estimator, normalize = normalize, delay = delay)

@dec_compute_infth()
def infth_mi_multivariate(data = {}, estimator = "kraskov1", normalize = True, delay = 0):
    """infth_mi_multivariate

    Compute the total (scalar) multivariate mutual information
    
    see also playground/infth_feature_relevance
    """
    # print "infth_mi_multivariate estimator = %s" % estimator
    # init class and instance
    if estimator == 'kraskov1':
        mimvCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    elif estimator == 'kraskov2':
        mimvCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    elif estimator == 'kernel':
        mimvCalcClass = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel

    # instantiate
    mimvCalc      = mimvCalcClass()

    # set properties
    mimvCalc.setProperty("NORMALISE", str(normalize).lower())
    # mimvCalc.setProperty("PROP_TIME_DIFF", str(delay))

    # print "measures_infth: infth_mi_multivariate: mimvCalc.timeDiff = %d" % (mimvCalc.timeDiff)

    mimvCalc.timeDiff = delay

    # print "measures_infth: infth_mi_multivariate: mimvCalc.timeDiff = %d" % (mimvCalc.timeDiff)

    # prepare data and attributes
    src, dst = prepare_data_and_attributes(data)
    # src_ = src.copy()
    # src = dst.copy()

    # pl.hist(src[0], bins=255)
    # pl.show()


    # print "infth_mi_multivariate src/dst shapes", src.shape, dst.shape
    # print "infth_mi_multivariate src/dst dtypes", src.dtype, dst.dtype

    dim_src, dim_dst = src.shape[1], dst.shape[1]

    # compute stuff
    # mimvCalc.initialise()
    mimvCalc.initialise(dim_src, dim_dst)
    mimvCalc.setObservations(src, dst)
    # the average global MI between all source channels and all destination channels
    try:
        mimv_avg = mimvCalc.computeAverageLocalOfObservations()
    except Exception as e:
        mimv_avg = np.random.uniform(0, 1e-5, (1,1)) # np.zeros((1,1))
        logger.error("Error occured in mimv calc, %s. Setting default mimv_avg = %s" % (e, mimv_avg))
    return mimv_avg

@dec_compute_infth()
def compute_cond_mi_multivariate(data = {}, estimator = "kraskov1", normalize = True, delay = 0):
    """infth_mi_multivariate

    Compute the total (scalar) multivariate conditional mutual information
    """

    # init class and instance
    if estimator == 'kraskov1':
        calcClass = JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov1
    elif estimator == 'kraskov2':
        calcClass = JPackage("infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov2
    elif estimator == 'kernel':
        calcClass = JPackage("infodynamics.measures.continuous.kernel").ConditionalMutualInfoCalculatorMultiVariateKernel

    # instantiate
    calc = calcClass()

    # set properties
    calc.setProperty("NORMALISE", str(normalize).lower())
    # calc.setProperty("PROP_TIME_DIFF", str(delay))

    # print "measures_infth: infth_mi_multivariate: calc.timeDiff = %d" % (calc.timeDiff)

    if hasattr(calc, 'timeDiff'):
        calc.timeDiff = delay

    # print "measures_infth: infth_mi_multivariate: calc.timeDiff = %d" % (calc.timeDiff)

    # prepare data and attributes
    assert 'C' in data, 'No condition passed via data, %s' % (list(data.keys()))
    src, dst, cond = prepare_data_and_attributes(data)
    # src_ = src.copy()
    # src = dst.copy()

    # print('src = %s, dst = %s, cond = %s' % (src.shape, dst.shape, cond.shape))
    
    # pl.hist(src[0], bins=255)
    # pl.show()


    # print "infth_mi_multivariate src/dst shapes", src.shape, dst.shape
    # print "infth_mi_multivariate src/dst dtypes", src.dtype, dst.dtype

    # expecting: observations on rows, variables on columns
    dim_src, dim_dst, dim_cond = src.shape[1], dst.shape[1], cond.shape[1]

    # compute stuff
    # calc.initialise()
    calc.initialise(dim_src, dim_dst, dim_cond)
    calc.setObservations(src, dst, cond)
    # the average global MI between all source channels and all destination channels
    try:
        mimv_avg = calc.computeAverageLocalOfObservations()
    except Exception as e:
        mimv_avg = np.random.uniform(0, 1e-5, (1,1)) # np.zeros((1,1))
        logger.error("Error occured in mimv calc, %s. Setting default mimv_avg = %s" % (e, mimv_avg))
    return mimv_avg

@dec_compute_infth()
def compute_transfer_entropy_multivariate(
        src, dst, delay = 0,
        k = 1, k_tau = 1, l = 1, l_tau = 1):
    """measures_infth: compute the multivariate transfer entropy from src to dst"""
    temvCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov
    temvCalc = temvCalcClass()

    srcdim = src.shape[1]
    dstdim = dst.shape[1]
    # k = 10
    # k_tau = 1
    # l = 10
    # l_tau = 1
    # delay = 1 # param u in TE equations

    temvCalc.initialise(srcdim, dstdim, k, k_tau, l, l_tau, delay)
    # print "measures_infth: compute_transfer_entropy_multivariate: temvCalc.timeDiff = %d" % (temvCalc.delay)
    temvCalc.setObservations(src, dst)
    temv = temvCalc.computeAverageLocalOfObservations()
    return temv

@dec_compute_infth()
def compute_conditional_transfer_entropy_multivariate(src, dst, cond, delay = 0):
    """measures_infth: compute the multivariate conditional transfer entropy from src to dst"""
    logger.debug("This doesn't exist in JIDT yet""")
    return -1

# FIXME: use this one from infth_feature_relevance
# def infth_mi_multivariate(self, data, estimator = "kraskov1", normalize = True):
@dec_compute_infth()
def compute_mutual_information(src, dst, k = 0, tau = 1, delay = 0, norm_in = True, norm_out = None):
    """compute_mutual_information

    Compute the matrix of pairwise mutual information (MI) for all
    pairs of (src_i, dst_j)

    ..note::
    
    taken from smp/im/im_quadrotor_plot.py
    """

    # src - dest is symmetric for MI but hey ...
    # from jpype import startJVM, isJVMStarted, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble, attachThreadToJVM
    # from smp.infth import init_jpype, ComplexityMeas

    # init_jpype()
    assert len(src.shape) == 2 and len(dst.shape) == 2, "src %s, dst %s" % (src.shape, dst.shape)
    # rows observations, columns variables
    numsrcvars, numdestvars = (src.shape[1], dst.shape[1])

    # miCalcClassC = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
    # miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    # miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
    miCalcC = miCalcClassC()
    miCalcC.setProperty("NORMALISE", str(norm_in).lower())
    miCalcC.setProperty(miCalcC.PROP_TIME_DIFF, str(delay))

    # print "measures_infth: compute_mutual_information: miCalcC.timeDiff = %d" % (miCalcC.timeDiff)

    measmat  = np.zeros((numdestvars, numsrcvars))

    if norm_out is not None:
        # assert norm_out is type(float), "Normalization constant needed (float)"
        norm_out_ = 1.0/norm_out
    else:
        norm_out_ = 1.0

    for m in range(numdestvars):
        for s in range(numsrcvars):
            print("compute_mutual_information dst[%d], src[%d]" % (m, s))

            # logger.debug("ha", m, motor[:,[m]])
            miCalcC.initialise() # sensor.shape[1], motor.shape[1])
            # print "measures_infth: compute_mutual_information: miCalcC.timeDiff = %d" % (miCalcC.timeDiff)
            # miCalcC.setObservations(src[:,s], dst[:,m])
            # print "compute_mutual_information src[%s] = %s, dst[%s] = %s" % (s, src[:,[s]].shape, m, dst[:,[m]].shape)
            logger.debug('isnan(src) = %s, isnan(dst) = %s' % (np.isnan(src[:,[s]]).sum(), np.isnan(dst[:,[m]]).sum()))
            logger.info('var(src) = %s, var(dst) = %s' % (np.var(src[:,[s]]), np.var(dst[:,[m]])))
            miCalcC.setObservations(src[:,[s]], dst[:,[m]])
            mi = miCalcC.computeAverageLocalOfObservations()
            print(f'mi = {mi}')
            measmat[m,s] = mi

    return measmat * norm_out_

@dec_compute_infth()
def compute_information_distance(src, dst, delay = 0, normalize = 1.0):
    """check how 1 - mi = infodist via joint H
    """
    mi = compute_mutual_information(src, dst, delay = delay, norm_in = True)
    # print "compute_information_distance mi  = %s" % (mi, )
    return 1 - (mi * normalize)

@dec_compute_infth()
def compute_transfer_entropy(src, dst, delay = 0):
    """taken from smp/im/im_quadrotor_plot.py"""
    # from jpype import startJVM, isJVMStarted, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble, attachThreadToJVM
    # from smp.infth import init_jpype, ComplexityMeas

    # init_jpype()

    numsrcvars, numdstvars = (src.shape[1], dst.shape[1])

    # teCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov
    teCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    # teCalcClassC = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalcC = teCalcClassC()
    teCalcC.setProperty("NORMALISE", "true")
    # k is destination embedding length
    # teCalcC.setProperty(teCalcC.K_PROP_NAME, "1")
    # teCalcC.setProperty("k", "100")
    # l is source embedding length
    # teCalcC.setProperty(teCalcC.L_PROP_NAME, "1")
    # teCalcC.setProperty(teCalcC.DELAY_PROP_NAME, "0")
    # teCalcC.setProperty(teCalcC.PROP_AUTO_EMBED_METHOD, "AUTO_EMBED_METHOD_NONE")
    # logger.debug("teCalcClassC", teCalcClassC, "teCalcC", teCalcC)

    # matrix of measures
    measmat  = np.zeros((numdstvars, numsrcvars))

    # temporal embedding params
    k = 1
    k_tau = 1
    l = 1
    l_tau = 1
    # delay = 0 # param u in TE equations

    # loop over all combinations
    for m in range(numdstvars):
        for s in range(numsrcvars):
            # logger.debug("m,s", m, s)
            # teCalcC.initialise()
            teCalcC.initialise(k, k_tau, l, l_tau, delay)
            # teCalcC.initialise(1, 1, 1, 1, 1, 1, 0)
            teCalcC.setObservations(src[:,s], dst[:,m])
            te = teCalcC.computeAverageLocalOfObservations()
            # tes = teCalcC.computeSignificance(10)
            # logger.debug("te", te)
            measmat[m,s] = te

    return measmat

@dec_compute_infth()
def compute_conditional_transfer_entropy(src, dst, cond, delay = 0, xcond = False):
    """conditional transfer entropy (CTE)

    Compute the conditional transfer entropy using jidt

    Args:

     - src(ndarray): source variables
     - dst(ndarray): destination variables
     - cond(ndarray): conditioning vars
     - delay: delay u between src/dst
     - xcond: do cross conditional assuming src and cond are the same vector
    """

    numsrcvars, numdstvars, numcondvars = (src.shape[1], dst.shape[1], cond.shape[1])

    cteCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").ConditionalTransferEntropyCalculatorKraskov
    # teCalcClassC = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    cteCalcC = cteCalcClassC()
    cteCalcC.setProperty("NORMALISE", "true")
    # k is destination embedding length
    # cteCalcC.setProperty(cteCalcC.K_PROP_NAME, "1")
    # teCalcC.setProperty("k", "100")
    # l is source embedding length
    # cteCalcC.setProperty(cteCalcC.L_PROP_NAME, "1")
    # cteCalcC.setProperty(cteCalcC.DELAY_PROP_NAME, "0")
    # teCalcC.setProperty(teCalcC.PROP_AUTO_EMBED_METHOD, "AUTO_EMBED_METHOD_NONE")
    # logger.debug("teCalcClassC", teCalcClassC, "teCalcC", teCalcC)

    # init return container
    measmat  = np.zeros((numdstvars, numsrcvars))

    # init calc params
    k = 1
    k_tau = 1
    l = 1
    l_tau = 1
    cond_emb_len = 1
    cond_emb_tau = 1
    cond_emb_delay = 0

    # loop over all combinations
    for m in range(numdstvars):
        for s in range(numsrcvars):
            # logger.debug("m,s", m, s)
            # cteCalcC.initialise(1, 1, 1, 1, 0, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
            # k, k_tau, l, l_tau, delay,
            # cteCalcC.initialise(1, 1, 1, 1, delay, [1] * numcondvars, [1] * numcondvars, [0] * numcondvars)

            condsl = list(range(numcondvars))
            numcondvars_ = numcondvars
            # cross-condition with src/cond being the same vector, condition on all vector elements besides s
            if xcond:
                del condsl[s]
                numcondvars_ -= 1

            # print "numsrcvars = %d, numdstvars = %d, numcondvars = %d, numcondvars_ = %d" % (numsrcvars, numdstvars, numcondvars, numcondvars_)
            # print "condsl = %s" % (condsl, )

            cond_emb_lens = [cond_emb_len] * numcondvars_
            cond_emb_taus = [cond_emb_tau] * numcondvars_
            cond_emb_delays = [cond_emb_delay] * numcondvars_

            # re-initialise calc
            cteCalcC.initialise(k, k_tau, l, l_tau, delay,
                                cond_emb_lens,
                                cond_emb_taus,
                                cond_emb_delays)
            # load the data
            cteCalcC.setObservations(src[:,s], dst[:,m], cond[:,condsl])
            # compute the measures
            cte = cteCalcC.computeAverageLocalOfObservations()
            # tes = teCalcC.computeSignificance(10)
            # logger.debug("cte", cte)
            measmat[m,s] = cte

    return measmat


def test_compute_mutual_information():
    N = 1000
    src = np.arange(N)
    dst = np.sin(2.0 * src / float(N))
    src = np.atleast_2d(src).T
    dst = np.atleast_2d(dst).T
    # stack
    src = np.hstack((src, dst, dst))
    dst = src.copy()
    logger.info("src.sh = %s, dst.sh = %s" % (src.shape, dst.shape))
    jh  = infth_mi_multivariate({'X': src, 'Y': dst})
    result = compute_mutual_information(src, dst)
    logger.info("result = %s/%s" % (result,result/jh))
    logger.info("result = %s" % (result,))

if __name__ == '__main__':
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

    print('in main, running test_compute_mutual_information')
    test_compute_mutual_information()
