"""smp_base.measures_infth

..moduleauthor:: Oswald Berthold, 2017

Information theoretic measures measure things related to the
multivariate entropy of some data.

TODO
sift, sort and clean-up input from
- smp/smp/infth.py
- smp/playground/infth_feature_relevance.py
- smp/sequence/\*.py
- smp_sphero (was smp_infth)
- evoplast/ep3.py
- smp/infth
- smp/infth/infth_homeokinesis_analysis_cont.py
- smp/infth/infth_playground
- smp/infth/infth_explore.py
- smp/infth/infth_pointwise_plot.py
- smp/infth/infth_measures.py: unfinished
- smp/infth/infth_playground.py
- smp/infth/infth_EH-2D.py
- smp/infth/infth_EH-2D_clean.py
"""
import sys, os
import numpy as np
import config

# from smp_base.measures import meas
from smp_base.measures import meas

import logging
from smp_base.common import get_module_logger

logger = get_module_logger(modulename = 'measures_infth', loglevel = logging.INFO) # .DEBUG)

try:
    from jpype import getDefaultJVMPath, isJVMStarted, startJVM, attachThreadToJVM, isThreadAttachedToJVM
    from jpype import JPackage
    HAVE_JPYPE = True
except ImportError, e:
    print "Couldn't import jpype, %s" % e
    HAVE_JPYPE = False
    # sys.exit(1)

def init_jpype(jarloc=None, jvmpath=None):
    if not HAVE_JPYPE:
        print "Cannot initialize jpype because it couldn't be imported. Make sure jpype is installed"
        return

    jarloc = jarloc or config.__dict__.get(
        'JARLOC',
        '%s/infodynamics/infodynamics.jar' % os.path.dirname(os.__file__)
    )
    assert os.path.exists(jarloc), "Jar file %s doesn't exist" % jarloc

    jvmpath = jvmpath or config.__dict__.get('JVMPATH', getDefaultJVMPath())

    print("infth.init_jpype: Set jidt jar location to %s" % jarloc)
    print("infth.init_jpype: Set jidt jvmpath      to %s" % jvmpath)

    # startJVM(getDefaultJVMPath(), "-ea", "-Xmx2048M", "-Djava.class.path=" + jarLocation)
    if not isJVMStarted():
        print("Starting JVM")
        startJVM(jvmpath, "-ea", "-Xmx8192M", "-Djava.class.path=" + jarloc)
    else:
        print("Attaching JVM")
        attachThreadToJVM()

# call init_jpype with global effects
init_jpype()

# infth classes
class measH(meas):
    """!@brief Measure entropy"""
    def __init__(self):
        meas.__init__(self)

    def step(self, x):
        """Assume observations in rows, variables in columns"""
        # print "%s step: x = %s" % (self.__class__.__name__, x.shape)
        # return compute_entropy(src = x)
        return infth_mi_multivariate({'X': x, 'Y': x})


class measMI(meas):
    def __init__(self):
        meas.__init__(self)

    def step(self, x, y):
        """Assume observations in rows, variables in columns"""
        # print "%s step: x = %s, y = %s" % (self.__class__.__name__, x.shape, y.shape)
        return compute_mutual_information(src = x, dst = y)

class dec_compute_infth_soft(object):
    """wrap infth calls and fail softly"""
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
    """wrap infth calls and fail hard"""
    def __call__(self, f):
        def wrap(*args, **kwargs):
            assert HAVE_JPYPE
            assert isJVMStarted() and isThreadAttachedToJVM(), "Either JVM not started or thread not attached. Hm."
            return f(*args, **kwargs)

        return wrap

def prepare_data_and_attributes(data, check_shape = False): # False
    """take dict with keys 'X','Y' and copy the into src,dst variables respectively"""
    # prepare data and attributes
    src = np.atleast_2d(data["X"])
    dst = np.atleast_2d(data["Y"])
    # check orientation
    if check_shape:
        if src.shape[0] < src.shape[1]:
            src = src.T
        if dst.shape[0] < dst.shape[1]:
            dst = dst.T
    return src, dst

################################################################################
# wrap these into a thin class
@dec_compute_infth()
def compute_entropy(src):
    if src.shape[1] > 1:
        return compute_entropy_multivariate(src)
    else:
        return compute_entropy_univariate(src)

@dec_compute_infth()
def compute_entropy_univariate(src):
    ent_class = JPackage('infodynamics.measures.continuous.kernel').EntropyCalculatorKernel
    ent = ent_class()
    ent.setProperty("NORMALISE", "true")
    ent.initialise(0.1)
    ent.setObservations(src)
    h = ent.computeAverageLocalOfObservations()
    return h

@dec_compute_infth()
def compute_entropy_multivariate(src, delay = 0):
    """compute_entropy_multivariate

    compute the joint entropy as self-information
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

    return compute_mi_multivariate(data = {'X': src, 'Y': src}, delay = delay)

# def compute_mi_multivariate(*args, **kwargs):
#     return infth_mi_multivariate(data = kwargs['data'], estimator = kwargs['estimator'], normalize = kwargs['normalize'])

def compute_mi_multivariate(data = {}, estimator = "kraskov1", normalize = True, delay = 0):
    return infth_mi_multivariate(data = data, estimator = estimator, normalize = normalize, delay = delay)

@dec_compute_infth()
def infth_mi_multivariate(data = {}, estimator = "kraskov1", normalize = True, delay = 0):
    """compute total scalar MI multivariate
    (from playground/infth_feature_relevance)"""
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
    except Exception, e:
        mimv_avg = np.random.uniform(0, 1e-5, (1,1)) # np.zeros((1,1))
        logger.error("Error occured in mimv calc, %s. Setting default mimv_avg = %s" % (e, mimv_avg))
    return mimv_avg

@dec_compute_infth()
def compute_transfer_entropy_multivariate(src, dst, delay = 0):
    """measures_infth: compute the multivariate transfer entropy from src to dst"""
    temvCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov
    temvCalc = temvCalcClass()

    srcdim = src.shape[1]
    dstdim = dst.shape[1]
    k = 1
    k_tau = 1
    l = 1
    l_tau = 1
    # delay = 1 # param u in TE equations

    temvCalc.initialise(srcdim, dstdim, k, k_tau, l, l_tau, delay)
    # print "measures_infth: compute_transfer_entropy_multivariate: temvCalc.timeDiff = %d" % (temvCalc.delay)
    temvCalc.setObservations(src, dst)
    temv = temvCalc.computeAverageLocalOfObservations()
    return temv

@dec_compute_infth()
def compute_conditional_transfer_entropy_multivariate(src, dst, cond, delay = 0):
    """measures_infth: compute the multivariate conditional transfer entropy from src to dst"""
    print "This doesn't exist in JIDT yet"""
    return -1

# FIXME: use this one from infth_feature_relevance
# def infth_mi_multivariate(self, data, estimator = "kraskov1", normalize = True):
@dec_compute_infth()
def compute_mutual_information(src, dst, k = 0, tau = 1, delay = 0, norm_in = True, norm_out = None):
    """taken from smp/im/im_quadrotor_plot.py

    computes a matrix of pairwise MI for all pairs of src_i,dst_j
    (elementwise)
    """

    # src - dest is symmetric for MI but hey ...
    # from jpype import startJVM, isJVMStarted, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble, attachThreadToJVM
    # from smp.infth import init_jpype, ComplexityMeas

    # init_jpype()
    assert len(src.shape) == 2 and len(dst.shape) == 2, "src %s, dst %s" % (src.shape, dst.shape)
    numsrcvars, numdestvars = (src.shape[1], dst.shape[1])

    # miCalcClassC = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
    miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    # miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    # miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
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
            # print "compute_mutual_information dst[%d], src[%d]" % (m, s)

            # print("ha", m, motor[:,[m]])
            miCalcC.initialise() # sensor.shape[1], motor.shape[1])
            # print "measures_infth: compute_mutual_information: miCalcC.timeDiff = %d" % (miCalcC.timeDiff)
            # miCalcC.setObservations(src[:,s], dst[:,m])
            # print "compute_mutual_information src[%s] = %s, dst[%s] = %s" % (s, src[:,[s]].shape, m, dst[:,[m]].shape)
            # logger.debug('isnan(src) = %s, isnan(dst) = %s' % (np.isnan(src[:,[s]]), np.isnan(dst[:,[m]])))
            logger.debug('var(src) = %s, var(dst) = %s' % (np.var(src[:,[s]]), np.var(dst[:,[m]])))
            miCalcC.setObservations(src[:,[s]], dst[:,[m]])
            mi = miCalcC.computeAverageLocalOfObservations()
            # print "mi", mi
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
    # print("teCalcClassC", teCalcClassC, "teCalcC", teCalcC)

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
            # print("m,s", m, s)
            # teCalcC.initialise()
            teCalcC.initialise(k, k_tau, l, l_tau, delay)
            # teCalcC.initialise(1, 1, 1, 1, 1, 1, 0)
            teCalcC.setObservations(src[:,s], dst[:,m])
            te = teCalcC.computeAverageLocalOfObservations()
            # tes = teCalcC.computeSignificance(10)
            # print("te", te)
            measmat[m,s] = te

    return measmat

@dec_compute_infth()
def compute_conditional_transfer_entropy(src, dst, cond, delay = 0, xcond = False):
    """!@breif compute the conditional transfer entropy using jidt

params
src: source variables
dst: destination variables
cond: conditioning vars
delay: delay u between src/dst
xcond: do cross conditional assuming y and cond are the same vector
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
    # print("teCalcClassC", teCalcClassC, "teCalcC", teCalcC)

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
            # print("m,s", m, s)
            # cteCalcC.initialise(1, 1, 1, 1, 0, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
            # k, k_tau, l, l_tau, delay,
            # cteCalcC.initialise(1, 1, 1, 1, delay, [1] * numcondvars, [1] * numcondvars, [0] * numcondvars)

            condsl = range(numcondvars)
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
            # print("cte", cte)
            measmat[m,s] = cte

    return measmat


def test_compute_mutual_information():
    N = 1000
    src = np.arange(N)
    dst = np.sin(2.0 * src / float(N))
    src = np.atleast_2d(src).T
    dst = np.atleast_2d(dst).T
    # stack
    src = np.hstack((src, dst))
    dst = src.copy()
    print "src.sh = %s, dst.sh = %s" % (src.shape, dst.shape)
    jh  = infth_mi_multivariate({'X': src, 'Y': dst})
    result = compute_mutual_information(src, dst)
    print "result = %s/%s" % (result,result/jh)
    print "result = %s" % (result,)

if __name__ == '__main__':
    import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument()

    test_compute_mutual_information()
