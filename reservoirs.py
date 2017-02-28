"""A leaky integrator rate-coded reservoir class"""

# TODO
# - switch to matrix data types?
# - simplify and clean up
# - FIXME: intrinsic plasticity: input layer, reservoir layer
# - FIXME: correlated exploration noise
# - FIXME: multiple timescales / tau
# - FIXME: pooling mechanism / competition

# Authors
# - Oswald Berthold, Aleke Nolte

import sys, time, argparse
from jpype import startJVM, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble

import numpy as np
import numpy.linalg as LA
import scipy.sparse as spa
import matplotlib.pyplot as pl
import rlspy

# if "/home/src/QK/smp/neural" not in sys.path:
#     sys.path.insert(0, "/home/src/QK/smp/neural")
from learners import GHA


############################################################
# utility functions
def create_matrix_reservoir(N, p):
    """Create an NxN reservoir recurrence matrix with density p"""
    M = spa.rand(N, N, p)
    M = M.todense()
    tmp_idx = M != 0
    tmp = M[tmp_idx]
    tmp_r = np.random.normal(0, 1, size=(tmp.shape[1],))
    M[tmp_idx] = tmp_r
    # print "type(M)", type(M)
    # print "M.shape", M.shape
    # M = np.array(M * self.g * self.scale)
    return np.array(M).copy()
    # return spa.bsr_matrix(M)

def normalize_spectral_radius(M, g):
    """Normalize the spectral radius of matrix M to g"""
    # compute eigenvalues
    [w,v] = LA.eig(M)
    # get maximum absolute eigenvalue
    lae = np.max(np.abs(w))
    # print "lae pre", lae
    # normalize matrix
    M /= lae
    # scale to desired spectral radius
    M *= g
    # print "type(M)", type(M)
    # print "M.shape", M.shape
    # # check for scaling
    # [w,v] = LA.eig(self.M)
    # lae = np.max(np.abs(w))
    # print "lae post", lae

################################################################################
# input matrix creation
def res_input_matrix_random_sparse(idim = 1, odim = 1, sparsity=0.1):
    """create a sparse input matrix"""
    p_wi = sparsity
    wi_ = spa.rand(odim, idim, p_wi)
    # print "sparse wi", wi_
    wi = wi_.todense()
    tmp_idx = wi != 0
    tmp = wi[tmp_idx]
    # tmp_r = np.random.normal(0, 1, size=(tmp.shape[1],))
    tmp_r = np.random.uniform(-1, 1, size=(tmp.shape[1],))
    wi[tmp_idx] = tmp_r
    return np.asarray(wi)

def res_input_matrix_disjunct_proj(idim = 1, odim = 1):
    """create an input matrix that projects inputs onto disjunct regions of hidden space"""
    # all matrices tensor
    wi_ = np.zeros((idim, odim, idim))
    numhiddenperinput = odim / idim
    # for each input create a matrix
    # sum all matrices up
    for i in range(idim):
        # print wi_[i]
        offset = (i * numhiddenperinput)
        wi_[i,offset:offset+numhiddenperinput,i] = np.random.uniform(-1, 1, (numhiddenperinput, ))
    wi = np.sum(wi_, axis=0)
    return wi

class LearningRules(object):
    def __init__(self):
        pass

    ############################################################
    # learning rule: FORCE
    # David Sussillo, L.F. Abbott, Generating Coherent Patterns of
    # Activity from Chaotic Neural Networks, Neuron, Volume 63, Issue
    # 4, 27 August 2009, Pages 544-557, ISSN 0896-6273,
    # http://dx.doi.org/10.1016/j.neuron.2009.07.018. (http://www.sciencedirect.com/science/article/pii/S0896627309005479)
    # Keywords: SYSNEURO
    def learnFORCE_update_P(self, P, r):
        k = np.dot(P, r)
        rPr = np.dot(r.T, k)
        c = 1.0/(1.0 + rPr)
        # print "r.shape", self.r.shape
        # print "k.shape", k.shape, "P.shape", self.P.shape, "rPr.shape", rPr.shape, "c.shape", c.shape
        P = P - np.dot(k, (k.T*c))
        return (P, k, c)
        
    def learnFORCE(self, target, P, k, c, r, z, channel):
        """FORCE learning rule for reservoir online supervised learning"""
        # required arguments: 
        # use FORCE to calc dW
        # use EH to calc dW
        # scalar !!!
        e = z - target
        dw = -e * k * c
        # self.wo[:,i] = self.wo[:,i] + dw[:,0]
        # self.wo[:,i] += dw[:]
        # print "FORCE", LA.norm(self.wo, 2)
        # return 0
        return dw
    
        
class Reservoir(object):
    def __init__(self, N=100, p = 0.1, g = 1.2, alpha = 1.0, tau = 0.1,
                 input_num=1, output_num=1, input_scale = 0.05,
                 feedback_scale = 0.01, bias_scale = 0.,
                 eta_init=1e-5, theta = 1e-1, theta_state = 1e-2,
                 nonlin_func=np.tanh,
                 sparse=True,
                 ip=False,
                 coeff_a = 0.2,
                 mtau=False):
        # reservoir size
        self.N = N
        # connection density
        self.p = p
        # reservoir weight gain
        self.g = g
        # leak rate
        self.mtau = mtau
        if self.mtau:
            # self.tau = np.exp(np.random.uniform(-10, -2, (self.N, 1)))
            self.tau = np.exp(np.random.uniform(-8, -0.5, (self.N, 1)))
        else:
            self.tau = tau
        self.alpha = alpha
        # scale gain to spectral radius lambda
        self.scale = 1.0/np.sqrt(self.p*self.N)
        # reservoir connection matrix
        self.M = create_matrix_reservoir(self.N, self.p)

        normalize_spectral_radius(self.M, self.g)
        self.M = spa.csr_matrix(self.M)
        
        # inputs and input weight matrix
        self.input_num = input_num
        self.wi_amp = input_scale
        # self.wi = np.random.uniform(-self.wi_amp, self.wi_amp, (self.N, self.input_num))
        self.wi = np.random.normal(0, self.wi_amp, (self.N, self.input_num))
        
        # readout feedback term
        self.output_num = output_num
        self.wf_amp = feedback_scale
        self.wf = np.random.uniform(-self.wf_amp, self.wf_amp, (self.N, self.output_num))
        # outputs and readout weight matrix
        self.wo = np.zeros((self.N, self.output_num));

        # initialize states to zeros
        self.u = np.zeros(shape=(self.input_num, 1))
        self.x = np.zeros(shape=(self.N, 1))
        self.r = np.zeros(shape=(self.N, 1))
        self.z = np.zeros(shape=(self.output_num, 1))
        self.zn = np.zeros((self.output_num, 1))
        # initialize states randomly
        # self.init_states_random()
        # bias
        self.bias = np.random.uniform(-1., 1., size=self.x.shape)
        self.bias_scale = bias_scale
        
        # rewards
        self.perf = np.zeros((self.output_num, 1))
        # self.perf_t = np.zeros((self.output_num, self.simtime_len))
        self.mdltr = np.zeros((self.output_num, 1))
        # self.mdltr_t = np.zeros((self.output_num, self.simtime_len))
        self.zn_lp = np.zeros((self.output_num, 1))
        # self.zn_lp_t = np.zeros((self.output_num, self.simtime_len))
        self.perf_lp = np.zeros((self.output_num, 1))
        # self.perf_lp_t = np.zeros((self.output_num, self.simtime_len))
        # self.coeff_a = 0.2
        self.coeff_a = coeff_a

        # initial learning rate and decay time constant
        self.eta_init = eta_init
        self.eta_tau = 200000.

        # exploration noise
        self.set_theta(theta)
        # noise filtering
        self.nu_ = np.zeros_like(self.z)
        # self.theta = theta # 1/10.
        # # different exploration noise amplitudes for different output dimensions
        # self.theta_amps = np.ones(shape=(self.output_num, 1)) * self.theta
        # state noise amplitude
        # self.theta_state = 0 #1/1000.
        self.theta_state = theta_state # 1/20.

        # FORCE stuff
        self.P = (1.0/self.alpha)*np.eye(self.N)
    
        # nonlinearity
        self.nonlin_func = nonlin_func


        # intrinsic plasticity
        self.ip = ip
        self.ip_a = np.random.uniform(-1e-1, 1e-1, (self.input_num, 1))
        self.ip_b = np.random.uniform(-1e-1, 1e-1, (self.input_num, 1))
        self.ip_eta = np.ones((self.input_num,1)) * 1e-5
        self.ip_mu  = np.zeros((self.input_num, 1))
        self.ip_var = np.ones((self.input_num, 1)) * 0.01
        # for whitening
        # self.u_mean = np.zeros_like(self.u)
        # self.u_var  = np.zeros_like(self.u)
        self.u_mean = np.random.uniform(-0.1, 0.1, self.u.shape)
        self.u_var  = np.random.uniform(-0.1, 0.1, self.u.shape)

    # setters
    def set_theta(self, theta):
        """Set exploration noise amplitude (theta) and theta_amps"""
        self.theta = theta
        self.theta_amps = np.ones(shape=(self.output_num, 1)) * self.theta
        # print "res: theta_amps", self.theta_amps
        


    ############################################################
    # save network
    def save(self, filename=""):
        from cPickle import Pickler
        if filename == "":
            timestamp = time.strftime("%Y-%m-%d-%H%M%S")
            filename = "reservoir-%s.bin" % timestamp
        f = open(filename,'wb')
        p = Pickler(f, 2)
        p.dump(self.__dict__)
        f.close()
        return
        

    ############################################################
    # load network (restore from file)
    # @classmethod
    def load(self, filename):
        from cPickle import Unpickler
        print "reservoirs.py: loading %s" % filename
        f = open(filename,'rb')
        u = Unpickler(f)
        tmp_dict = u.load()
        f.close()
        self.__dict__.update(tmp_dict)

    ############################################################
    # initialize states randomly
    def init_states_random(self):
        self.x = 0.5 * np.random.normal(size=(self.N, 1))
        self.z = 0.5 * np.random.normal(size=(self.output_num, 1))

    ############################################################
    # reset states to zero
    def reset_states(self):
        self.x = np.zeros(shape=(self.N, 1))
        self.z = np.zeros(shape=(self.output_num, 1))


    ############################################################
    # initialize output weights randomly
    def init_wo_random(self, mu=0., std=0.001):
        self.wo = np.random.normal(mu, std, (self.N, self.output_num))

    ############################################################
    # initialize output weights to zero
    def init_wo_zero(self):
        self.wo = np.zeros((self.N, self.output_num))

    ############################################################
    # initialize input weights to I
    def init_wi_identity(self):
        # self.wo = np.eye((self.N, self.output_num))
        self.wi = np.eye((self.input_num)) 
        
    ############################################################
    # initialize input weights to ones
    def init_wi_ones(self):
        # self.wo = np.eye((self.N, self.output_num))
        self.wi = np.ones((self.N, self.input_num)) * self.wi_amp

    def ip_input_layer(self, u):
        # FIXME: self.ip_a * u ?
        u = u.reshape((self.input_num, 1))
        # print self.ip_a.shape, u.shape
        u_tmp = np.tanh(self.ip_a * u + self.ip_b)
        # print u_tmp.shape
        db = -self.ip_eta * ((-self.ip_mu/self.ip_var) + (u_tmp / self.ip_var) * (2 * self.ip_var + 1 - u_tmp**2 + self.ip_mu * u_tmp))
        # print "db", db.shape, db
        da = (self.ip_eta/self.ip_a) + (db * u)
        # print "da", da.shape, da
        self.ip_b += db
        self.ip_a += da
        # print "da,db norm", LA.norm(self.ip_a, 2), LA.norm(self.ip_b, 2), LA.norm(da, 2), LA.norm(db, 2)
        # print u_tmp.shape
        return u_tmp

    def ip_input_layer_whiten(self, u):
        self.u_mean = 0.999 * self.u_mean + 0.001 * u
        self.u_var = 0.999 * self.u_var + 0.001 * np.sqrt(np.square(u - self.u_mean))
        # print "u_mean", self.u_mean
        # print "u_var", self.u_var
        # print np.linalg.norm(self.u_mean)
        # print np.linalg.norm(self.u_var)
        # print np.linalg.norm(u - self.u_mean)
        u_out = (u - self.u_mean)/self.u_var
        # print np.linalg.norm(u_out)
        return u_out
        # return (0.5 * u - 0.5 * self.u_mean) # / self.u_var

    ############################################################
    # execute Reservoir
    def execute(self, u):
        # FIXME: check for proper shape of elements
        # FIXME: guard for proper shapes (atleast_..., ...)
        # collect the terms
        zeta_state = np.random.uniform(-self.theta_state, self.theta_state, size=self.x.shape)
        # state
        x_tp1 = (1.0 - self.tau) * self.x
        # reservoir
        # r_tp1 = np.dot(self.M, self.r)
        r_tp1 = self.M.dot(self.r)
        # r_tp1 = self.r
        # r_tp1 = np.dot(self.M, self.r)
        # readout feedback
        f_tp1 = np.dot(self.wf, self.zn)
        # f_tp1 = np.dot(self.wf, self.zn)
        # print x_tp1, r_tp1, f_tp1
        # input
        if self.ip:
            self.u = self.ip_input_layer(u)
            # print self.u
        else:
            self.u = u
        u_tp1 = np.reshape(np.dot(self.wi, self.u), (self.x.shape))
        # bias
        b_tp1 = self.bias * self.bias_scale
        # print "x_tp1", x_tp1.shape
        # print "r_tp1", r_tp1.shape
        # print "f_tp1", f_tp1.shape
        # print "u_tp1", u_tp1.shape
        # print "zn_tp1", self.zn.shape

        # update state
        self.x = x_tp1 + self.tau * (r_tp1 + f_tp1 + u_tp1 + b_tp1)
        # self.r = self.tau * np.tanh(r_tp1 + f_tp1 + u_tp1 + b_tp1)
        # self.x = x_tp1 + self.r
        # self.r = np.tanh(self.x) + zeta_state
        self.r = self.nonlin_func(self.x) + zeta_state
        # print "self.x", self.x.shape
        # print "self.r", self.r, self.r.shape

        # print "shapes", self.wo.shape, self.r.shape
        self.z = np.dot(self.wo.T, self.r)
        # self.zn = self.z + np.random.normal(0, self.theta, size=(self.z.shape))
        # generate standard normal noise per output dimension
        nu_tmp = np.random.normal(0, 1.0, size=(self.z.shape))
        # nu_tmp = np.random.pareto(1.2, size=(self.z.shape))
        # nu_tmp = ((np.random.binomial(1, 0.5) - 0.5) * 2) * np.random.pareto(1.5, self.z.shape) #  * self.sigma_expl
        # multiply with noise amplitudes
        nu = nu_tmp * self.theta_amps
        self.nu_ = 0.8 * self.nu_ + 0.2 * nu
        # print "nu", nu
        # apply to noisy readout
        self.zn = self.z + nu
        # self.zn = self.z + self.nu_
        
        self.zn_lp = self.zn_lp * (1-self.coeff_a) + self.zn * self.coeff_a
        # print "Wo", self.wo.T.shape
        # print "self.r", self.r.shape
        # print "z", self.z.shape, self.z, self.zn.shape, self.zn
        return self.z

    # ############################################################
    # # learn: RLS
    # # FIXME: rlspy
    # # function [th,p] = rolsf(x,y,p,th,lam)
    # # % function [th,p] = rolsf(x,y,p,th,lam)
    # # %    Recursive ordinary least squares for single output case,
    # # %       including the forgetting factor, lambda.
    # # %    Enter with x = input, y = output, p = covariance, th = estimate, 
    # # lam = forgetting factor
    # # %
    # #      a=p*x;
    # #      g=1/(x'*a+lam);
    # #      k=g*a;
    # #      e=y-x'*th;
    # #      th=th+k*e;
    # #      p=(p-g*a*a')/lam;
    # def learnRLS(self, target):
    #     lam = 0.98
    #     a = np.dot(self.P, self.r)
    #     g = 1 / (np.dot(self.r, a) + lam)
    #     k = np.dot(g, a)
    #     e = target - self.z[:,0] # np.dot(self.r, self.wo)
    #     dw = np.dot(k, e)
    #     self.wo += dw
    #     self.P = (self.P-np.dot(g, np.dot(a, a.T)))/lam

    # def dwFORCE(self, P, r, z, target):

    ############################################################
    # learn: FORCE, EH, recursive regression (RLS)?
    def learnFORCE(self, target):
        # get some target
        # use FORCE to calc dW
        # use EH to calc dW
        k = np.dot(self.P, self.r)
        rPr = np.dot(self.r.T, k)
        c = 1.0/(1.0 + rPr)
        # print "r.shape", self.r.shape
        # print "k.shape", k.shape, "P.shape", self.P.shape, "rPr.shape", rPr.shape, "c.shape", c.shape
        self.P = self.P - np.dot(k, (k.T*c))
        
        for i in range(self.output_num):
            # print "self.P", self.P
            # print "target.shape", target.shape
            e = self.z[i,0] - target[i,0]
            # print "error e =", e, self.z[i,0]

            # print "err", e, "k", k, "c", c
            # print "e.shape", e.shape, "k.shape", k.shape, "c.shape", c.shape
            # dw = np.zeros_like(self.wo)
            dw = -e * k * c
            # dw = -e * np.dot(k, c)
            # print "dw", dw.shape
            # print "shapes", self.wo.shape, dw.shape # .reshape((self.N, 1))
            # print i
            # print "shapes", self.wo[:,0].shape, dw[:,0].shape # .reshape((self.N, 1))
            # print "types", type(self.wo), type(dw)
            self.wo[:,i] += dw[:,0]
            # self.wo[:,i] = self.wo[:,i] + dw[:,0]
            # self.wo[:,i] += dw[:]
        # print "FORCE", LA.norm(self.wo, 2)
        # return 0

    def learnRLSsetup(self,wo_init, P0_init):
        # Attention! not using function parameters
         
        #self.rls_E = rlspy.data_matrix.Estimator(np.zeros(shape=(self.N, 1)) ,(1.0/self.alpha)*np.eye(self.N))
        #self.rls_E = rlspy.data_matrix.Estimator(np.random.uniform(0, 0.0001, size=(self.N, 1)) , np.eye(self.N))
        if wo_init==None  and   P0_init==None :
          print ("using random initialization for RLS setup ")
          # self.rls_E = rlspy.data_matrix.Estimator(np.random.uniform(0, 0.1, size=(self.N, 1)) , np.eye(self.N))
          self.rls_E = rlspy.data_matrix.Estimator(np.random.uniform(0, 0.01, size=(self.N, 1)) , np.eye(self.N))
          # self.wo = np.random.uniform(-1e-4,1e-4, size=(self.N, self.output_num))
          self.wo = np.zeros((self.N, self.output_num))
        else:
          print ('taking arguments as initialization for RLS setup')
          self.wo = wo_init
          self.rls_E = rlspy.data_matrix.Estimator(P0_init[0], P0_init[1])

        #self.wo = np.random.uniform(-1e-3,1e-3, size=(self.N, self.output_num))
        #self.wo = np.random.uniform(0,1, size=(self.N, self.output_num))

    def learnRLS(self, target):
        self.rls_E.update(self.r.T, target, self.theta_state) 
        self.wo = self.rls_E.x
        
    def learnEH(self, target):
        eta = self.eta_init # 0.0003
        # eta = self.eta_init / (1 + ()) # 0.0003
        self.perf = -np.square(self.zn - target)
        # self.perf = self.zn - target
        # print (self.zn, target)
        # print self.perf.shape, self.perf_lp
        for i in range(self.output_num):
            if self.perf[i,0] > self.perf_lp[i,0]:
                # print "mdltr", self.perf[i,0], self.perf_lp[i,0]
                self.mdltr[i,0] = 1
            else:
                self.mdltr[i,0] = 0
            dW = eta * (self.zn[i,0] - self.zn_lp[i,0]) * self.mdltr[i,0] * self.r
            # dW = eta * (self.zn[i,0] - self.zn_lp[i,0]) * -self.perf * self.r
            # print dW
            # print dW.shape, self.x.shape, self.wo[:,i].shape
            # print np.reshape(dW, (self.N, )).shape
            self.wo[:,i] += dW[:,0]
            # self.wo[:,i] += np.reshape(dW, (self.N, ))
            
        self.perf_lp = ((1 - self.coeff_a) * self.perf_lp) + (self.coeff_a * self.perf)
        
        # return 0

    def learnPISetup(self):
        
        self.piCalcClassD = JPackage("infodynamics.measures.discrete").PredictiveInformationCalculatorDiscrete
        self.piCalcClass = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
        self.entCalcClass = JPackage("infodynamics.measures.continuous.kernel").EntropyCalculatorKernel
        self.base = 20
        self.piCalcD = self.piCalcClassD(self.base,3)
        # self.aisCalcD = self.aisCalcClassD(self.base,5)
        self.piCalc = self.piCalcClass();
        self.entCalc = self.entCalcClass();


    def perfVar(self, x):
        # self.eta_init = 1.2e-3
        # self.eta_init = 2e-3

        # em.plot_pi_discrete_cont()
        # em.calc_pi_cont_windowed()
        # em.plot_pi_cont_windowed()
        # em.calc_global_entropy()
        # em.calc_local_entropy()
        # em.calc_global_entropy_cont()
        # em.plot_measures()
        # em.calc_pi_discrete()
        # pi = np.mean(em.pi)
        # em.calc_pi_discrete_avg()
        # pi = em.pi
        perf = np.var(x)
        
        self.perf = np.array([perf]).reshape((self.output_num, 1))
        
    def learnEHPerf(self):
        eta = self.eta_init
        # print "perf", self.perf
        for i in range(self.output_num):
            if self.perf[i,0] > self.perf_lp[i,0]:
                # print "mdltr", self.perf[i,0], self.perf_lp[i,0]
                self.mdltr[i,0] = 1
            else:
                self.mdltr[i,0] = 0
            dW = eta * (self.zn[i,0] - self.zn_lp[i,0]) * self.mdltr[i,0] * self.r
            # dW = eta * (self.zn[i,0] - self.zn_lp[i,0]) * -self.perf * self.r
            # print dW
            # print dW.shape, self.x.shape, self.wo[:,i].shape
            # print np.reshape(dW, (self.N, )).shape
            self.wo[:,i] += dW[:,0]
            # self.wo[:,i] += np.reshape(dW, (self.N, ))
            
        self.perf_lp = ((1 - self.coeff_a) * self.perf_lp) + (self.coeff_a * self.perf)
        
                        
    def learnPI(self, x):
        eta = self.eta_init
        
        # dmax = np.max(x)
        # dmin = np.min(x)
        # # print self.dmax, self.dmin
        # # bins = np.arange(self.dmin, self.dmax, 0.1)
        # bins = np.linspace(dmin, dmax, 20) # FIXME: determine binnum
        # # print "x.shape", x.shape
        # x_d = np.digitize(x[0,:], bins).reshape(x.shape)
        # # print "x_d.shape", x_d.shape
        # # pi = list(self.piCalcD.computeLocal(x_d))
        # pi = self.piCalcD.computeAverageLocal(x_d)
        # # print pi
        # # base = np.max(x_d)+1 # 1000
        
        # # compute PI
        # self.piCalc.setProperty("NORMALISE", "true"); # Normalise the individual variables
        # self.piCalc.initialise(1, 1, 0.25); # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
        # # print "x", x.shape
        # src = np.atleast_2d(x[0,0:-100]).T # start to end - 1
        # dst = np.atleast_2d(x[0,100:]).T # 1 to end
        # # print "src, dst", src, dst
        # # print "src, dst", src.shape, dst.shape
        # self.piCalc.setObservations(src, dst)
        # pi = self.piCalc.computeAverageLocalOfObservations()
        
        
        # compute differential entropy
        # self.entCalc.setProperty("NORMALISE", "true"); # Normalise the individual variables
        self.entCalc.initialise(0.5); # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
        # print "x", x.shape
        # src = np.atleast_2d(x[0,0:-10]).T # start to end - 1
        # dst = np.atleast_2d(x[0,10:]).T # 1 to end
        # print "src, dst", src, dst
        # print "src, dst", src.shape, dst.shape
        self.entCalc.setObservations(JArray(JDouble, 1)(x.T))
        pi = self.entCalc.computeAverageLocalOfObservations()
        
        self.perf = np.array([pi]).reshape((self.output_num, 1))
        
        # print "perf", self.perf
        for i in range(self.output_num):
            if self.perf[i,0] > self.perf_lp[i,0]:
                # print "mdltr", self.perf[i,0], self.perf_lp[i,0]
                self.mdltr[i,0] = 1
            else:
                self.mdltr[i,0] = 0
            dW = eta * (self.zn[i,0] - self.zn_lp[i,0]) * self.mdltr[i,0] * self.r
            # dW = eta * (self.zn[i,0] - self.zn_lp[i,0]) * -self.perf * self.r
            # print dW
            # print dW.shape, self.x.shape, self.wo[:,i].shape
            # print np.reshape(dW, (self.N, )).shape
            self.wo[:,i] += dW[:,0]
            # self.wo[:,i] += np.reshape(dW, (self.N, ))
            
        self.perf_lp = ((1 - self.coeff_a) * self.perf_lp) + (self.coeff_a * self.perf)
    
    def learnPCA_init(self, eta=1e-4):
        # self.ro_dim = 2
        eta_gha = eta
        self.gha = GHA(eta = eta_gha, ndims = self.N, pdims = self.output_num)
        self.gha.w *= 0.1
        self.wo = self.gha.w

    def learnPCA(self):
        # resin
        # self.res.execute(resin)
        y_gha = self.gha.update(self.r)
        self.wo = self.gha.w
        self.z = y_gha
        # self.zn = self.z + np.random.normal(0, self.theta, size=(self.z.shape))
        self.zn = self.z + (np.random.normal(0, 1.0, size=(self.z.shape)) * self.theta)
        return self.z
    
    # various getters
    def get_reservoir_matrix(self):
        return self.M

    def get_input_matrix(self):
        return self.wi

    def get_feedback_matrix(self):
        return self.wf

    def get_readout_matrix(self):
        return self.wo

    def get_reservoir_size(self):
        return self.N

class Reservoir2(Reservoir):
    def __init__(self, N=100, p = 0.1, g = 1.2, alpha = 1.0, tau = 0.1,
                 input_num=1, output_num=1, input_scale=0.05,
                 feedback_scale=0.01, bias_scale=0., sparse=False,
                 eta_init=1e-3, theta = 1e-1, theta_state = 1e-2,
                 nonlin_func=np.tanh, ip=False, coeff_a=0.2):
        super(Reservoir2, self).__init__(N=N, p=p, g=g, alpha=alpha, tau=tau,
                                        input_num=input_num, output_num=output_num,
                                        input_scale=input_scale, feedback_scale=feedback_scale,
                                        bias_scale=bias_scale, eta_init=eta_init, theta = theta,
                                        theta_state = theta_state,
                                        nonlin_func=np.tanh,
                                        ip=ip, coeff_a=coeff_a)
        ############################################################
        # reservoir connection matrix
        if self.g == 0.:
            sparse = False
            print ("sparse False (g = %f)" % self.g)
        # initialize reservoir matrix with weights drawn from N(0,1)
        if sparse:
            # reservoir connection matrix: sparse version
            self.M = spa.rand(self.N, self.N, self.p)
            self.M = self.M.todense()
            tmp_idx = self.M != 0
            tmp = self.M[tmp_idx]
            tmp_r = np.random.normal(0, 1, size=(tmp.shape[1],))
            self.M[tmp_idx] = tmp_r
            # self.M = np.array(self.M * self.g * self.scale)
            # print "scale", self.scale.shape, "M", self.M.shape
            # self.M = np.array(self.M * self.g)
            self.M = np.array(self.M)
            print "M", self.M.shape
            
            # input matrix
            p_wi = 0.5
            self.wi_ = spa.rand(self.N, self.input_num, p_wi)
            # print "sparse wi", self.wi_
            self.wi = self.wi_.todense()
            # print "dense wi", self.wi
            tmp_idx = self.wi != 0
            tmp = self.wi[tmp_idx]
            # tmp_r = np.random.normal(0, 1, size=(tmp.shape[1],))
            tmp_r = np.random.uniform(-self.wi_amp, self.wi_amp, size=(tmp.shape[1],))
            self.wi[tmp_idx] = tmp_r
            # print "type(self.wi)", type(self.wi)
            self.wi = np.array(self.wi) # make sure we're numpy.ndarray and not
                                        # numpy.matrixlib.defmatrix.matrix
            # print "dense wi", self.wi
            # print "type(self.wi)", type(self.wi)
        else:
            self.M = np.random.normal(0, 1., (self.N, self.N))
            # self.wi = np.random.uniform(-self.wi_amp, self.wi_amp, (self.N, self.input_num))

        # compute eigenvalues
        [w,v] = LA.eig(self.M)
        # get maximum absolute eigenvalue
        lae = np.max(np.abs(w))
        print "lae pre", lae
        # normalize matrix
        self.M /= lae
        # scale to desired spectral radius
        self.M *= self.g

        # check for scaling
        # [w,v] = LA.eig(self.M)
        # lae = np.max(np.abs(w))
        # print "lae post", lae

        # nonlinearity
        self.nonlin_func = nonlin_func

        print "ip", self.ip        
        # # inputs and input weight matrix
        # self.wi = np.random.normal(0, self.wi_amp, (self.N, self.input_num))
        # # readout feedback term
        # self.wf = np.random.normal(0, self.wf_amp, (self.N, self.output_num))
        self.theta_amps = np.ones(shape=(self.output_num, 1)) * self.theta
        
    ############################################################
    # execute Reservoir2
    # use model from waegemann et al 2012
    def execute(self, u):
        # collect the terms
        zeta_state = np.random.uniform(-self.theta_state, self.theta_state, size=self.x.shape)
        # state
        x_tp1 = (1.0 - self.tau) * self.x
        # reservoir
        r_tp1 = np.dot(self.M, self.x)
        # r_tp1 = np.dot(self.M, self.r)
        # readout feedback
        # f_tp1 = np.dot(self.wf, self.z)
        f_tp1 = np.dot(self.wf, self.zn)
        # print x_tp1, r_tp1, f_tp1
        # input
        if self.ip:
            # print u
            self.u = self.ip_input_layer(u)
            # self.u = self.ip_input_layer_whiten(u)
            # print u
        else:
            self.u = u
        u_tp1 = np.reshape(np.dot(self.wi, self.u), (self.x.shape))
        # bias
        b_tp1 = self.bias * self.bias_scale
        # print "x_tp1", x_tp1.shape
        # print "r_tp1", r_tp1.shape
        # print "f_tp1", f_tp1.shape
        # print "u_tp1", u_tp1.shape

        # update state
        # self.x = x_tp1 + self.tau * np.tanh(r_tp1 + f_tp1 + u_tp1 + b_tp1)
        # self.x = x_tp1 + (self.tau * np.tanh(r_tp1 + f_tp1 + u_tp1 + b_tp1)) + zeta_state
        self.x = x_tp1 + (self.tau * self.nonlin_func(r_tp1 + f_tp1 + u_tp1 + b_tp1)) + zeta_state
        # self.r = self.x + zeta_state
        self.r = self.x
        # self.r = self.tau * np.tanh(r_tp1 + f_tp1 + u_tp1 + b_tp1)
        # self.x = x_tp1 + self.r
        # self.r = np.tanh(self.x) + zeta_state
        # print "self.x", self.x.shape
        # print "self.r", self.r.shape

        # print "shapes", self.wo.shape, self.r.shape
        self.z = np.dot(self.wo.T, self.r)
        # nu = np.random.normal(0, self.theta, size=(self.z.shape))
        nu = np.random.normal(0, 1.0, size=(self.z.shape))
        nu = nu * self.theta_amps
        self.zn = self.z + nu
        self.zn_lp = self.zn_lp * (1-self.coeff_a) + self.zn * self.coeff_a
        # print "Wo", self.wo.T.shape
        # print "self.r", self.r.shape
        # print "z", self.z.shape, self.z, self.zn.shape, self.zn
        return self.z

    # save network
    def save(self, filename):
        super(Reservoir2, self).save(filename)

    # load network
    def load(self, filename):
        super(Reservoir2, self).load(filename)
        
def get_ds_MSO(elen, outdim, mode="MSO_s1"):
    """Get target data from a datasource"""
    # t_dur = 1
    # t_s = 1000
    # t_numsteps = t_dur * t_s + 1
    # t = np.linspace(0, t_dur, t_numsteps)
    # t = np.linspace(0, 100, elen)
    # t = np.arange(0, elen)
    t = np.linspace(0, int(elen), elen)
    # print t

    d_sig_freqs = np.array([1.]) * 1.0
    if mode == "MSO_s1":
        # simple
        d_sig_freqs = np.array([0.1, 0.2]) * 0.01
        # d_sig_freqs = np.array([0.13, 0.26]) * 0.1
    elif mode == "MSO_s2":
        # simple
        d_sig_freqs = np.array([0.1, 0.2, 0.31]) * 0.01
        # d_sig_freqs = np.array([0.13, 0.26]) * 0.1
    elif mode == "MSO_s3":
        # simple
        d_sig_freqs = np.array([0.1, 0.2,  0.3]) * 0.02
        # d_sig_freqs = np.array([0.13, 0.26]) * 0.1
    elif mode == "MSO_s4":
        # simple
        # f(t)=(1.3/1.5)*sin(2*pi*t)+(1.3/3)*sin(4*pi*t)+(1.3/9)*sin(6*pi*t)+(1.3/3)*sin(8*pi*t)        
        d_sig_freqs = np.array([0.1, 0.2,  0.3]) * 0.02
        # d_sig_freqs = np.array([0.13, 0.26]) * 0.1
        # 400 seconds learning
        t = t/1000.0
        print "t", t
        data = (1.3/1.5) * np.sin(2*np.pi*t) + (1.3/3) * np.sin(4*np.pi*t) + (1.3/9) * np.sin(6*np.pi*t) + (1.3/3) * np.sin(8*np.pi*t)
        data = data.reshape((1, -1)) * 0.7
        print "data.shape", data.shape
        return data
    elif mode == "MSO_c1":
        # d_sig_freqs = np.array([0.0085, 0.0174]) * 0.1
        d_sig_freqs = np.array([0.0085, 0.011]) * 0.2
        # d_sig_freqs = np.array([0.13, 0.27, 0.53, 1.077]) * 0.3
        # d_sig_freqs = [1.002, 2.004, 3.006, 4.008]
        # d_sig_freqs = np.array([0.1, 0.2, 0.3, 0.4]) * 1.
        # d_sig_freqs = np.array([0.01, 1.01, 1.02, 1.03]) * 0.1
    elif mode == "MSO_c2":
        d_sig_freqs = np.array([0.0085, 0.0174, 0.0257, 0.0343]) * 0.1
        # d_sig_freqs = np.array([0.13, 0.27, 0.53, 1.077]) * 0.3
        # d_sig_freqs = [1.002, 2.004, 3.006, 4.008]
        # d_sig_freqs = np.array([0.1, 0.2, 0.3, 0.4]) * 1.
        # d_sig_freqs = np.array([0.01, 1.01, 1.02, 1.03]) * 0.1
    elif mode == "MG": # Mackey-Glass
        import Oger
        import scipy.signal as ss
        import scipy.interpolate as si
        ds = Oger.datasets.mackey_glass(sample_len=elen/20, n_samples=outdim)
        # ds = Oger.datasets.mackey_glass(sample_len=elen, n_samples=1)
        ds_n = []
        for i in range(outdim):
        # for i in range(1):
            # ds1 = ds[0][0]
            # ds2 = ds[1][0]
            # rsmp = ss.resample(ds[i][0].T, elen)
            # ds_n.append(rsmp)
            f = si.interp1d(np.arange(0, elen/20), ds[i][0].T)
            tt = np.linspace(0, elen/20-1, elen)
            print f(tt)
            ds_n.append(f(tt))
            # ds_n.append(ds[i][0].T)
        # 2 rows, n cols
        # ds_real = np.vstack([ds1.T, ds2.T])
        #    print len(ds_n)
        # sys.exit()
        ds_real = np.vstack(ds_n)
        return ds_real
    elif mode == "wav":
        from scipy.io import wavfile
        # from smp.datasets import wavdataset
        # from models import SeqData
        rate, data = wavfile.read("notype_mono_short.wav")
        # rate, data = wavfile.read("drinksonus_mono_short.wav")
        offset = np.random.randint(0, data.shape[0] - elen)
        print data.dtype, offset
        data = data.astype(np.float)
        data = data[offset:offset+elen].reshape((1, -1))
        data /= np.max(np.abs(data))
        # print data.shape
        # sys.exit()
        return data
    
    d_sig_a = np.zeros(shape=(elen, len(d_sig_freqs)))
    for j in range(len(d_sig_freqs)):
        d_sig_a[:,j] = np.sin(2*np.pi*d_sig_freqs[j]*t)

    d_sig = np.sum(d_sig_a, axis=1)/len(d_sig_freqs)
    d_sig = d_sig.reshape(1, (len(d_sig)))
    return d_sig

def test_ip(args):
    idim = 1
    odim = 1
    saveplot = True
    res = Reservoir2(N=100, input_num=idim, output_num=odim, g = 0.,
                     tau = 1., feedback_scale=0., input_scale=1.0,
                     eta_init=1e-3, sparse=True, ip=True)
    # custom IP eta?
    res.ip_eta = np.ones((idim,1)) * 4e-4

    u_    = np.zeros((args.length, idim))
    for i in range(idim):
        u_[:,i]    = np.random.uniform(np.random.uniform(-5., .0),
                                       np.random.uniform(0., 5.),
                                       size=(args.length,))
        # u_[:,i]    = np.random.beta(np.random.uniform(0, 1.),
        #                                    np.random.uniform(0, 1.),
        #                                    size=(args.length,))
        # u_[:,i]    = np.random.exponential(np.random.uniform(0, 1.),
        #                                    size=(args.length,))
    u_ip_ = np.zeros((args.length, idim))
    z_    = np.zeros((args.length, odim))
    for i in range(args.length):
        # print u_[i]
        z_[i] = res.execute(u_[i])
        # print res.u
        u_ip_[i] = res.u.reshape((idim,))


    sl1 = slice(0, args.length / 3)
    sl2 = slice(args.length / 3, 2 * args.length / 3)
    sl3 = slice(2 * args.length / 3, None)
    pl.subplot(241)
    # pl.plot(u_, "k-", lw=0.3)
    pl.plot(u_ + [i * 10 for i in range(idim)], "k-", lw=0.3)
    pl.ylabel("Neuron input $x$")
    pl.subplot(242)
    pl.hist(u_[sl1], bins=50, normed=True, orientation="horizontal")
    pl.subplot(243)
    pl.hist(u_[sl2], bins=50, normed=True, orientation="horizontal")
    pl.subplot(244)
    pl.hist(u_[sl3], bins=50, normed=True, orientation="horizontal")
    pl.subplot(245)
    # pl.plot(u_ip_, "k-", lw=0.3)
    pl.plot(u_ip_, "k-", lw=0.3)
    pl.ylabel("Neuron output $y$")
    pl.subplot(246)
    pl.hist(u_ip_[sl1], bins=50, normed=True, orientation="horizontal")
    pl.subplot(247)
    pl.hist(u_ip_[sl2], bins=50, normed=True, orientation="horizontal")
    pl.subplot(248)
    pl.hist(u_ip_[sl3], bins=50, normed=True, orientation="horizontal")

    if saveplot:
        pl.gcf().set_size_inches((18,10))
        pl.gcf().savefig("reservoir_test_ip.pdf", dpi=300, bbox_inches="tight")
    pl.show()

class ReservoirTest(object):
    modes = {"ol_rls": 0, "ol_force": 1, "ol_eh": 2, "ip": 3, "fwd": 4, "ol_pi": 5}
    targets = {"MSO_s1": 0, "MSO_s2": 1, "MSO_s3": 2, "MSO_c1": 3, "MSO_c2": 4, "MG": 5, "wav": 6}

if __name__ == "__main__":
    import jpype
    jarLocation = "/home/src/QK/infodynamics-dist/infodynamics.jar"
    # print jarLocation
    # print startJVM
    if not jpype.isJVMStarted():
        print "Starting JVM"
        startJVM(getDefaultJVMPath(), "-ea", "-Xmx8192M", "-Djava.class.path=" + jarLocation)
    else:
        print "Attaching JVM"
        jpype.attachThreadToJVM()
    
    parser = argparse.ArgumentParser(description="Reservoir library: call main for testing \
\
 - example: python reservoirs.py -t MSO_s1 -m ol_eh -rs 500 -l 100000")
    parser.add_argument("-m", "--mode", help="Mode, one of " + str(ReservoirTest.modes.keys()),
                        default = "ol_rls")
    parser.add_argument("-l", "--length", help="Episode length", default=30000, type=int)
    parser.add_argument("-lr", "--learning_ratio", help="ratio of learning to episode len", default=0.8, type=float)
    parser.add_argument("-mt", "--multitau", dest="multitau", help="use multiple random time constants in reservoir, doesn't seem to work so well with EH", action="store_true")
    parser.add_argument("-rs", "--ressize", help="Reservoir (hidden layer) size", default=300, type=int)
    parser.add_argument("-s", "--seed", help="RNG seed", default=101, type=int)
    parser.add_argument("-t", "--target", help="Target, one of " + str(ReservoirTest.targets.keys()),
                        default = "MSO_s1")
    
    args = parser.parse_args()
    
    if ReservoirTest.modes[args.mode] == ReservoirTest.modes["ip"]:
        test_ip(args)
        sys.exit()

    # FIXME: define different tasks
    #  - signal generation: simple MSO, non-integer relation MSO, Mackey-Glass
    #  - compare with batch learning for signal generation
    #  - simple prediction tasks
    #  - simultaneous learning of inverse and forward model, modulate forward learning based on variance
    # FIXME:
    #  - ascii progress bar
    #  - on the fly plotting
    #  - modularize learning rules: use Reward class?

    # print args.length
    # for i in [10, 100, 1000]:
    np.random.seed(args.seed)
    # episode_len = 100000
    # episode_len = 10000
    episode_len = args.length
    washout_ratio = 0.1
    washout = washout_ratio * episode_len
    learning_ratio = args.learning_ratio
    testing = learning_ratio * episode_len
    insize = 1
    outsize = 1
    # for feedback
    percent_factor = 1./(episode_len/100.)

    eta_init_ = 5e-4
    
    # get training target
    # print args.target
    ds_real = get_ds_MSO(episode_len+1, outsize, args.target)
    ds_real2 = get_ds_MSO(episode_len+1, outsize, "MSO_s3")
    print "ds_real.shape", ds_real.shape

    # # plot target
    # pl.plot(ds_real.T)
    # pl.plot(ds_real2.T)
    # pl.show()
    
    print (ds_real.shape)
    # sys.exit()

    # print ds.shape
    # print ds
    # sys.exit()
    # ressize = 300
    for i in [args.ressize]:
        out_t = np.zeros(shape=(outsize, episode_len))
        out_t_n = np.zeros(shape=(outsize, episode_len))
        r_t = np.zeros(shape=(i, episode_len))
        perf_t = np.zeros(shape=(outsize, episode_len))
        wo_t = np.zeros(shape=(i, outsize, episode_len))
        wo_t_norm = np.zeros((outsize, episode_len))
        # test save and restore
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        filename = "reservoir-%s.bin" % timestamp
        # res = Reservoir(N=10)
        # for larger reservoirs: decrease eta, increase input scale
        # res = Reservoir(N = i, input_num = insize, output_num = outsize, g = 1.5, tau = 0.01,
        #                 feedback_scale=0.1, input_scale=2.0, bias_scale=0.2, eta_init=1e-3,
        #                 theta = 1e-1, theta_state = 1e-2, coeff_a = 0.2, mtau=args.multitau)
        res = Reservoir(N = i, input_num = insize, output_num = outsize, g = 1.5, tau = 0.01,
                        feedback_scale=0.1, input_scale=2.0, bias_scale=0.01, eta_init=eta_init_,
                        theta = 5e-1, theta_state = 5e-2, coeff_a = 0.2, mtau=args.multitau)
        
        # res.save(filename)
        
        # print ("first", res.M)
        
        # res = Reservoir2(N=i, input_num=insize, output_num=outsize, g = 1.5, tau = 0.1,
        #                 feedback_scale=0.0, input_scale=1.0, bias_scale=0.0, eta_init=1e-3,
        #                 sparse=True, coeff_a = 0.2,
        #                 ip=False, theta = 1e-5, theta_state = 1e-2)

        lr = LearningRules()
        
        # 2015-01-14: try eta = 1e-4, nice but still has the phase drift
        # res.theta = 1e-1
        # res.theta_state = 1e-3
        # pl.title("res.wo")
        # pl.plot(res.wo)
        # pl.show()
        # print ("second", res.M)
        
        # res.load(filename)
        
        # print "third", res.M
        
        # print "res size", res.get_reservoir_size()
        # print "res matrix", res.get_reservoir_matrix()
        # print "res input matrix", res.get_input_matrix()
        # print "res feedback matrix", res.get_feedback_matrix()
        # print "res readout matrix", res.get_readout_matrix()

        if ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_rls"]:
            res.learnRLSsetup(None, None)
        elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_pi"]:
            # sys.path.insert(0, "/home/src/QK/smp/infth")
            # from infth_playground import Meas
            # res.learnPISetup()
            # em = Meas(length=10000)
            # print "shape", nlzr.X["x"][-2000:]
            print "ol_pi in development, not working yet"
            time.sleep(3)
            res.init_wo_random(0., 1e-5)
            pass

        pl.ion()
        pl.plot(out_t.T)
        pl.draw()
        
        for j in range(episode_len):
            # res.execute(np.random.uniform(size=(10, 1)))
            # out_t[:,j] = res.execute(np.random.uniform(size=(insize, 1)))[:,0]

            # if j < testing:
            #     inputs = ds_real[:,j-1].reshape((insize, 1))
            #     # inputs = ds_real[:,j].reshape((insize, 1))
            # else:
            #     inputs = out_t[:,j-1].reshape((insize, 1))
            #     # inputs = out_t[:,j].reshape((insize, 1))
            # print out_t.shape
            # inputs = out_t[:,j-1].reshape((insize, 1))
            inputs = out_t[0,j-1].reshape((insize, 1))

            # inputs = ds_real[:,j-1].reshape((insize, 1))
            # inputs = out_t[:,j-1].reshape((insize, 1))
            
            # inputs = np.random.uniform(size=(insize, 1))
            # print "inputs", inputs.shape, inputs
            out_t[:,j] = res.execute(inputs)[:,0]
            # res.execute(inputs)[:,0]
            out_t_n[:,j] = res.zn
            # print "zn", j, res.zn, out_t[:,j]
            # print (r_t[:,j].shape, res.r[:,0].shape)
            r_t[:,j] = res.r[:,0].reshape((args.ressize,))
            # if j % 2 == 0:
            # start testing / freerunning mode
            if j < testing and j > washout:
                if ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_rls"]:
                    res.learnRLS(ds_real[:,j].reshape((outsize, 1)))
                elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_force"]:
                    # old: reservoir module learning rule
                    # res.learnFORCE(ds_real[:,j].reshape((outsize, 1)))
                    # experimental: decouple learning rule from core reservoir module
                    # first readout
                    (res.P, k, c) = lr.learnFORCE_update_P(res.P, res.r)
                    dw = lr.learnFORCE(ds_real[:,j].reshape((1, 1)),
                                  res.P, k, c, res.r, res.z[0,0], 0)
                    res.wo[:,0] += dw[:,0]
                    # second readout
                    if outsize > 1:
                        dw = lr.learnFORCE(ds_real2[:,j].reshape((1, 1)),
                                    res.P, k, c, res.r, res.z[1,0], 1)
                        res.wo[:,1] += dw[:,0]

                    # print np.linalg.norm(res.wo, 2)
                elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_eh"]:
                    j_ = j - washout
                    res.eta_init = eta_init_ / (1 + (j_/20000.0))
                    res.learnEH(ds_real[:,j].reshape((outsize, 1)))
                elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_pi"]:
                    # print "out", out_t[:,j-1000:j]
                    if j > 100:
                        # em.x = out_t_n[:,j-1000:j].T # nlzr.X["x"][-10000:,0].reshape((10000,1))
                        # em.preprocess()
                        # em.discretize_data()
                        # res.learnPI(out_t_n[:,j-1000:j])
                        # res.perfVar(out_t_n[:,j-1000:j])
                        # res.perfVar(out_t[:,j-100:j])
                        res.perfVar(out_t)
                        # print res.perf
                        res.learnEHPerf()
            perf_t[0,j] = res.perf[0,0]
            wo_t[:,:,j] = res.wo
            for k in range(outsize):
                wo_t_norm[k,j] = LA.norm(wo_t[:,k,j])
            
            # print "state", res.x
            if j > washout and j % 1000 == 0:
                # # old style
                print "eta", res.eta_init
                # print("%d of %d" % (j,  episode_len))
                # new style
                progress = j * percent_factor
                # print '\r[{0}] {1}%'.format('#'*(progress/10), progress)
                sys.stdout.write( '\r[{0}] {1}%'.format('#'*int(progress), progress))
                sys.stdout.flush()
                
                pl.subplot(311)
                pl.gca().clear()
                # backlog = 200
                backlog = 1000
                pl.plot(ds_real[:,(j-backlog):j].T, lw=0.5)
                # pl.plot(ds_real2[:,(j-backlog):j].T, lw=0.5)
                pl.plot(out_t[:,(j-backlog):j].T, lw=0.5)
                pl.subplot(312)
                pl.gca().clear()
                pl.plot(wo_t_norm.T)
                pl.subplot(313)
                pl.gca().clear()
                pl.plot(perf_t.T)
                pl.draw()
                pl.pause(1e-9)

    pl.ioff()
    # print (wo_t.shape)
    # wo_t_norm = np.zeros((outsize, episode_len))
    for j in range(episode_len):
        for k in range(outsize):
            wo_t_norm[k,j] = LA.norm(wo_t[:,k,j])
            
    pl.subplot(411)
    pl.title("Target")
    pl.plot(ds_real.T)
    pl.subplot(412)
    pl.title("Target and output")
    pl.plot(ds_real.T)
    pl.plot(out_t.T)
    # pl.axvline(testing)
    pl.axvspan(testing, episode_len, alpha=0.1)
    # pl.plot(ds_real.T - out_t.T)
    pl.subplot(413)
    pl.title("reservoir traces")
    selsize = 20
    rindex = np.random.randint(i, size=selsize)
    pl.plot(r_t.T[:,rindex])
    pl.subplot(414)
    pl.title("weight norm")
    pl.plot(wo_t_norm.T)
    pl.show()

    from scipy.io import wavfile
    # wav_out = (out_t.T * 32767).astype(np.int16)
    out_t /= np.max(np.abs(out_t))
    wav_out = (out_t.T * 32767).astype(np.int16)
    wavfile.write("res_out.wav", 44100, wav_out)
