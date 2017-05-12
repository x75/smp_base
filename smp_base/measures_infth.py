"""smp_base - smp sensorimotor experiments base functions

measures_infth

2017 Oswald Berthold

information theoretic measures measure things related to the multivariate entropy of some data

"""

import numpy as np

from smp_base.measures import meas

from jpype import getDefaultJVMPath, isJVMStarted, startJVM, attachThreadToJVM
from jpype import JPackage

def init_jpype(jarloc = None, jvmpath = None):
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
        return compute_entropy(src = x)


class measMI(meas):
    def __init__(self):
        meas.__init__(self)

    def step(self, x, y):
        """Assume observations in rows, variables in columns"""
        # print "%s step: x = %s, y = %s" % (self.__class__.__name__, x.shape, y.shape)
        return compute_mutual_information(src = x, dest = y)

def compute_entropy(src):
    if src.shape[1] > 1:
        return compute_entropy_multivariate(src)
    else:
        return compute_entropy_univariate(src)

def compute_entropy_univariate(src):
    ent_class = JPackage('infodynamics.measures.continuous.kernel').EntropyCalculatorKernel
    ent = ent_class()
    ent.setProperty("NORMALISE", "true")
    ent.initialise(0.1)
    ent.setObservations(src)
    h = ent.computeAverageLocalOfObservations()
    return h

def compute_entropy_multivariate(src):
    # ent_class = JPackage('infodynamics.measures.continuous.kernel').EntropyCalculatorMultiVariateKernel
    ent_class = JPackage('infodynamics.measures.continuous.gaussian').EntropyCalculatorMultiVariateGaussian
    # ent_class = JPackage('infodynamics.measures.continuous.kozachenko').EntropyCalculatorMultiVariateKozachenko
    ent = ent_class()
    # ent.setProperty("NORMALISE", "true")
    # ent.initialise(src.shape[1], 0.1)
    ent.initialise(src.shape[1])
    ent.setObservations(src)
    h = ent.computeAverageLocalOfObservations()
    return h
    
def compute_mutual_information(src, dest):
    """taken from smp/im/im_quadrotor_plot.py

    computes a matrix of pairwise MI for all pairs of src_i,dst_j
    """
    
    # src - dest is symmetric for MI but hey ...
    # from jpype import startJVM, isJVMStarted, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble, attachThreadToJVM
    # from smp.infth import init_jpype, ComplexityMeas

    # init_jpype()
    
    numsrcvars, numdestvars = (src.shape[1], dest.shape[1])
    
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
            # print("m,s", m, s)

            # print("ha", m, motor[:,[m]])
            miCalcC.initialise() # sensor.shape[1], motor.shape[1])
            # miCalcC.setObservations(src[:,s], dest[:,m])
            miCalcC.setObservations(src[:,[s]], dest[:,[m]])
            mi = miCalcC.computeAverageLocalOfObservations()
            # print("mi", mi)
            measmat[m,s] = mi

    return measmat
        
