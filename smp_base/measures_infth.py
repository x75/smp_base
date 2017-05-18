"""smp_base - smp sensorimotor experiments base functions

measures_infth

2017 Oswald Berthold

information theoretic measures measure things related to the multivariate entropy of some data

TODO
# sift, sort and clean-up input from
#  - smp/smp/infth.py
#  - smp/playground/infth_feature_relevance.py
#  - smp/sequence/*.py
#  - smp_sphero (was smp_infth)
#  - evoplast/ep3.py
#  - smp/infth
#  - smp/infth/infth_homeokinesis_analysis_cont.py
#  - smp/infth/infth_playground
#  - smp/infth/infth_explore.py
#  - smp/infth/infth_pointwise_plot.py
#  - smp/infth/infth_measures.py: unfinished
#  - smp/infth/infth_playground.py
#  - smp/infth/infth_EH-2D.py
#  - smp/infth/infth_EH-2D_clean.py
"""

import sys
import numpy as np

from smp_base.measures import meas


try:
    from jpype import getDefaultJVMPath, isJVMStarted, startJVM, attachThreadToJVM, isThreadAttachedToJVM
    from jpype import JPackage
    HAVE_JPYPE = True
except ImportError, e:
    print "Couldn't import jpype, %s" % (e,)
    HAVE_JPYPE = False
    # sys.exit(1)

def init_jpype(jarloc = None, jvmpath = None):
    if not HAVE_JPYPE:
        print "Cannot initialize jpype because it couldn't be imported. Make sure jpype is installed"
        return
    
    if jarloc is None:
        jarloc = "/home/src/QK/infodynamics-dist/infodynamics.jar"

    if jvmpath is None:
        jvmpath = getDefaultJVMPath()

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
    def __call__(self, f):
        def wrap(*args, **kwargs):
            assert HAVE_JPYPE
            assert isJVMStarted() and isThreadAttachedToJVM(), "Either JVM not started or thread not attached. Hm."
            return f(*args, **kwargs)
            
        return wrap

def prepare_data_and_attributes(data, check_shape = False): # False
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
def compute_entropy_multivariate(src):
    # ent_class = JPackage('infodynamics.measures.continuous.kernel').EntropyCalculatorMultiVariateKernel
    # ent_class = JPackage('infodynamics.measures.continuous.gaussian').EntropyCalculatorMultiVariateGaussian
    # ent_class = JPackage('infodynamics.measures.continuous.kozachenko').EntropyCalculatorMultiVariateKozachenko
    ent_class = JPackage('infodynamics.measures.continuous.kraskov1').MutualInfoCalculatorMultiVariateKraskov1
    ent = ent_class()
    # ent.setProperty("NORMALISE", "true")
    # ent.initialise(src.shape[1], 0.1)
    ent.initialise(src.shape[1], src.shape[1])
    ent.setObservations(src, src)
    h = ent.computeAverageLocalOfObservations()
    return h

# def compute_mi_multivariate(*args, **kwargs):
#     return infth_mi_multivariate(data = kwargs['data'], estimator = kwargs['estimator'], normalize = kwargs['normalize'])

def compute_mi_multivariate(data = {}, estimator = "kraskov1", normalize = True):
    return infth_mi_multivariate(data = data, estimator = estimator, normalize = normalize)

@dec_compute_infth()
def infth_mi_multivariate(data = {}, estimator = "kraskov1", normalize = True):
    """compute total scalar MI multivariate
    (from playground/infth_feature_relevance)"""
    print "infth_mi_multivariate estimator = %s" % estimator
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
    # mimvCalc.setProperty("PROP_TIME_DIFF", 0)

    # prepare data and attributes
    src, dst = prepare_data_and_attributes(data)
    # src_ = src.copy()
    # src = dst.copy()

    # pl.hist(src[0], bins=255)
    # pl.show()
        
        
    print "infth_mi_multivariate src/dst shapes", src.shape, dst.shape
    print "infth_mi_multivariate src/dst dtypes", src.dtype, dst.dtype
    dim_src, dim_dst = src.shape[1], dst.shape[1]
        
    # compute stuff
    # mimvCalc.initialise()
    mimvCalc.initialise(dim_src, dim_dst)
    mimvCalc.setObservations(src, dst)
    # the average global MI between all source channels and all destination channels
    mimv_avg = mimvCalc.computeAverageLocalOfObservations()
    return mimv_avg

@dec_compute_infth()
def compute_transfer_entropy_multivariate(src, dst, delay = 0):
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
    temvCalc.setObservations(src, dst)
    temv = temvCalc.computeAverageLocalOfObservations()
    return temv
    
# FIXME: use this one from infth_feature_relevance
# def infth_mi_multivariate(self, data, estimator = "kraskov1", normalize = True):
@dec_compute_infth()
def compute_mutual_information(src, dst, k = 0, tau = 1):
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
    # miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
    miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    # miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
    miCalcC = miCalcClassC()
    miCalcC.setProperty("NORMALISE", "true")
    # miCalcC.setProperty(miCalcC.PROP_TIME_DIFF, "0")

    measmat  = np.zeros((numdestvars, numsrcvars))

    for m in range(numdestvars):
        for s in range(numsrcvars):
            # print "compute_mutual_information dst[%d], src[%d]" % (m, s)

            # print("ha", m, motor[:,[m]])
            miCalcC.initialise() # sensor.shape[1], motor.shape[1])
            # miCalcC.setObservations(src[:,s], dst[:,m])
            print "compute_mutual_information src[%s] = %s, dst[%s] = %s" % (s, src[:,[s]].shape, m, dst[:,[m]].shape)
            miCalcC.setObservations(src[:,[s]], dst[:,[m]])
            mi = miCalcC.computeAverageLocalOfObservations()
            # print("mi", mi)
            measmat[m,s] = mi

    return measmat

@dec_compute_infth()
def compute_information_distance(src, dst):
    """check how 1 - mi = infodist via joint H"""
    mi = compute_mutual_information(src, dst)
    return 1 - (mi / infth_mi_multivariate(data = {'X': src, 'Y': dst}))

@dec_compute_infth()
def compute_transfer_entropy(src, dst):
    """taken from smp/im/im_quadrotor_plot.py"""
    # from jpype import startJVM, isJVMStarted, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble, attachThreadToJVM
    # from smp.infth import init_jpype, ComplexityMeas

    # init_jpype()

    numsrcvars, numdstvars = (src.shape[1], dst.shape[1])
    
    # teCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorMultiVariateKraskov
    teCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    # teCalcClassC = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalcC = teCalcClassC()
    # teCalcC.setProperty("NORMALISE", "true")
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
    k = 10
    k_tau = 1
    l = 10
    l_tau = 1
    delay = 1 # param u in TE equations
    
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
def compute_conditional_transfer_entropy(src, dst, cond):
    """compute the conditional transfer entropy using jidt"""

    numsrcvars, numdstvars, numcondvars = (src.shape[1], dst.shape[1], cond.shape[1])

    cteCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").ConditionalTransferEntropyCalculatorKraskov
    # teCalcClassC = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    cteCalcC = cteCalcClassC()
    cteCalcC.setProperty("NORMALISE", "true")
    # k is destination embedding length
    cteCalcC.setProperty(cteCalcC.K_PROP_NAME, "1")
    # teCalcC.setProperty("k", "100")
    # l is source embedding length
    cteCalcC.setProperty(cteCalcC.L_PROP_NAME, "1")
    cteCalcC.setProperty(cteCalcC.DELAY_PROP_NAME, "0")
    # teCalcC.setProperty(teCalcC.PROP_AUTO_EMBED_METHOD, "AUTO_EMBED_METHOD_NONE")
    # print("teCalcClassC", teCalcClassC, "teCalcC", teCalcC)

    measmat  = np.zeros((numdstvars, numsrcvars))

    for m in range(numdstvars):
        for s in range(numsrcvars):
            # print("m,s", m, s)
            # cteCalcC.initialise(1, 1, 1, 1, 0, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0])
            cteCalcC.initialise(1, 1, 1, 1, 0, [1] * numcondvars, [1] * numcondvars, [0] * numcondvars)
            cteCalcC.setObservations(src[:,s], dst[:,m], cond)
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
