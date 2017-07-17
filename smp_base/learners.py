"""smp_base: learner class as robot specific model integrating supervised online prediction learners
"""

import sys, time

import numpy as np
import numpy.linalg as LA
import scipy.linalg as sLA

import ConfigParser, ast


from .eligibility import Eligibility

try:
    from .measures_infth import init_jpype, dec_compute_infth_soft
    from jpype import JPackage
    init_jpype()
    HAVE_JPYPE = True
except ImportError, e:
    print "Couldn't import init_jpype from measures_infth, make sure jpype is installed", e
    HAVE_JPYPE = False
    
# TODO
# - make proper test case and compare with batch PCA
# - implement APEX algorithm?

TWOPI_SQRT = np.sqrt(2*np.pi)

def gaussian(m, s, x):
    return 1/(s*TWOPI_SQRT) * np.exp(-0.5*np.square((m-x)/s))

class learnerConf():
    """Common parameters for exploratory Hebbian learners"""
    def __init__(self, cfgfile="default.cfg"):
        """learnerConf init"""
        self.cfgfile = cfgfile
        self.cfg = ConfigParser.ConfigParser()

    def set_cfgfile(self, cfgfile="default.cfg"):
        """set the config file path"""
        self.cfgfile = cfgfile
        
    def read(self):
        """read config from file"""
        print("opening %s" % self.cfgfile)
        self.cfg.read(self.cfgfile)
        # print("tau = %f" % (self.cfg.getfloat("learner", "tau")))

    def write(self):
        """write config to file"""
        self.cfg.write(self.cfgfile)
        
    def cfgget(self):
        """move config variables into members"""
        # FIXME: fill in reasonable defaults for diverging config files
        # FIXME: automate / param definition via params dictionary
        self.mode                 = self.cfg.get("learner", "mode")
        self.N                    = self.cfg.getint("learner", "network_size")
        self.idim                 = self.cfg.getint("learner", "network_idim")
        self.odim                 = self.cfg.getint("learner", "network_odim")
        self.tau                  = self.cfg.getfloat("learner", "tau")
        self.g                    = self.cfg.getfloat("learner", "g")
        self.p                    = self.cfg.getfloat("learner", "p")
        self.lag                  = self.cfg.getint("learner", "lag")
        self.eta_EH               = self.cfg.getfloat("learner", "eta_EH")
        self.res_theta            = self.cfg.getfloat("learner", "res_theta")
        self.res_theta_state      = self.cfg.getfloat("learner", "network_theta_state")
        self.res_input_scaling    = self.cfg.getfloat("learner", "res_input_scaling")
        self.res_feedback_scaling = self.cfg.getfloat("learner", "res_feedback_scaling")
        self.res_bias_scaling     = self.cfg.getfloat("learner", "res_bias_scaling")
        self.res_output_scaling   = self.cfg.getfloat("learner", "res_output_scaling")
        self.target               = self.cfg.getfloat("learner", "target")
        self.use_ip               = self.cfg.getint("learner", "use_ip")
        self.use_et               = self.cfg.getint("learner", "use_et")
        self.use_anneal           = self.cfg.getint("learner", "use_anneal")
        self.use_mt               = self.cfg.getint("learner", "use_mt")
        self.use_wb               = self.cfg.getint("learner", "use_wb")
        self.anneal_const         = self.cfg.getfloat("learner", "anneal_const")
        self.et_winsize           = self.cfg.getint("learner", "et_winsize")
        self.wb_thr               = self.cfg.getfloat("learner", "wb_thr")
        self.do_savelogs          = self.cfg.getint("learner", "do_savelogs")
        self.coeff_a              = self.cfg.getfloat("learner", "coeff_a")
        self.pm_mass              = self.cfg.getfloat("learner", "pm_mass")
        # predictor foo
        self.use_pre              = self.cfg.getint("learner", "use_pre")
        self.pre_inputs           = ast.literal_eval(self.cfg.get("learner", "pre_inputs"))
        # print "pre_inputs", self.pre_inputs
        self.pre_delay            = ast.literal_eval(self.cfg.get("learner", "pre_delay"))
        # print "pre_delay", self.pre_delay
        # density estimation foo
        self.use_density          = self.cfg.getint("learner", "use_density")
        self.density_mode         = self.cfg.getint("learner", "density_mode")

        # complex configuration variables which need some kind of evaluation
        # input coupling matrix
        self.input_coupling_mtx_spec = ast.literal_eval(self.cfg.get("learner", "input_coupling_mtx"))
        # print ("input coupling matrix:", self.input_coupling_mtx_spec)
        # nonlinearity / neuron activation function
        # self.nonlin_func             = self.cfg.get("learner", "nonlin_func")
        # self.nonlin_func             = ast.literal_eval(self.cfg.get("learner", "nonlin_func"))
        self.nonlin_func             = eval(self.cfg.get("learner", "nonlin_func"))
        # \print "cfgget: nonlin", self.nonlin_func
        # perturbation types
        self.tp_perturbation_spec = ast.literal_eval(self.cfg.get("experiment", "tp_perturbation"))
        # print ("perturbation types:", self.tp_perturbation_spec)
        # target type
        self.tp_target_spec  = ast.literal_eval(self.cfg.get("experiment", "tp_target"))
        # print ("target types:", self.tp_target_spec)
        self.target_interval = self.cfg.getint("experiment", "target_interval")
        print ("config params: tau = %f, g = %f, lag = %d, eta = %f, theta = %f, target = %f" % (self.tau, self.g, self.lag, self.eta_EH, self.res_theta, self.target))

        self.len_episode          = self.cfg.getint("experiment", "len_episode")
        self.len_washout          = self.cfg.getint("experiment", "len_washout")
        self.ratio_testing        = self.cfg.getfloat("experiment", "ratio_testing")

class learnerIOS(object):
    """Input, Output and State container structure"""
    def __init__(self, idim=1, odim=1):
        self.idim = idim
        self.odim = odim
        # input
        self.x_raw = np.zeros((self.idim, 1)) # raw input
        self.x     = np.zeros((self.idim, 1)) # scaled input
        # output
        self.z      = np.zeros((self.odim, 1)) # network output / readout
        self.zn     = np.zeros((self.odim, 1)) # network output / readout / noisy / exploration
        
        # extended performance measures
        self.e = np.zeros((self.odim, 1)) # error
        self.t = np.zeros((self.idim + self.odim, 1)) # target
        self.mae = np.zeros((self.odim, 1)) # absolute error
        self.mse = np.zeros((self.odim, 1)) # mean square error
        self.rmse = np.zeros((self.odim, 1)) # root mean squared error
        self.itae = np.zeros((self.odim, 1)) # integral of time-weighted absolute error

class learnerIOSMem(learnerIOS):
    """Input, Output and State memory container structure"""
    def __init__(self, idim=1, odim=1, sdim=100, memlen=1000):
        learnerIOS.__init__(self, idim, odim)
        self.sdim = sdim # network state dimension
        self.len = memlen # history / memory length
        # history / synchronous logging
        self.x_     = np.zeros((self.len, self.idim))
        self.x_raw_ = np.zeros((self.len, self.idim))
        self.z_     = np.zeros((self.len, self.odim))
        self.zn_    = np.zeros((self.len, self.odim))
        self.zn_lp_ = np.zeros((self.len, self.odim))
        self.r_     = np.zeros((self.len, self.sdim))
        self.w_     = np.zeros((self.len, self.sdim, self.odim))
        
        # extended performance measures
        self.t_ = np.zeros((self.len, self.idim + self.odim)) # target
        self.e_ = np.zeros((self.len, self.odim)) # error
        self.mae_ = np.zeros((self.len, self.odim)) # mean absolute error
        self.mse_ = np.zeros((self.len, self.odim)) # mean square error
        self.rmse_ = np.zeros((self.len, self.odim)) # root mean squared error
        self.itae_ = np.zeros((self.len, self.odim)) # integral of time-weighted absolute error
        
class learnerReward(object):
    # infth stuff
    
    # discretization base
    base = 1000
    basehalf = base/2

    if HAVE_JPYPE:
        # calculation classes
        piCalcClass = JPackage("infodynamics.measures.discrete").PredictiveInformationCalculatorDiscrete
        piCalcD = piCalcClass(base,1)

        # ais
        aisCalcClassD = JPackage("infodynamics.measures.discrete").ActiveInformationCalculatorDiscrete
        aisCalcD = aisCalcClassD(base,1)

        # contiuous estimation
        # piCalcClassC = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
        piCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").PredictiveInfoCalculatorKraskov
        piCalcC = piCalcClassC();
        # print dir(piCalcC)
        piCalcC.setProperty("NORMALISE", "true"); # Normalise the individual variables
    
        # active information storage
        aisCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
        aisCalcC = aisCalcClassC()
        aisCalcC.setProperty("NORMALISE", "false"); # Normalise the individual variables
    
        # FIXME: do a shutdownJVM after being finished, use isJVMStarted?, attachtoJVM
    
    """Learner reward data"""
    def __init__(self, idim=1, odim=1, memlen=1000, coeff_a = 0.2):
        """Init learnerReward
        
        idim: input dimensionality (default: 1) \n
        odim: output dimensionality (default: 1) \n
        memlen: length of memory to be kept, in steps (default: 1000)"""
        
        self.idim = idim
        self.odim = odim
        # print "learnerReward", self.idim, self.odim
        self.len = memlen
        # reward
        self.perf     = np.zeros((self.odim, 1))
        self.perf_lp  = np.zeros((self.odim, 1))
        self.perf_    = np.zeros((self.len, self.odim)) # FIXME: transpose this
        self.perf_lp_ = np.zeros((self.len, self.odim))
        self.coeff_a  = coeff_a
        # self.coeff_a  = 0.05
        self.mdltr    = np.zeros((self.odim, 1))
        
        # print "learnerReward", self.perf.shape
    def discretize(self, x, llim=None, ulim=None):
        if llim == None and ulim == None:
            bins = np.linspace(np.min(x), np.max(x), learnerReward.base-1)
        else:
            bins = np.linspace(llim, ulim, learnerReward.base-1)
        return np.digitize(x, bins)

    def perf_accel(self, err, acc):
        """Simple reward: let body acceleration point into reduced error direction"""
        self.perf[0:self.odim,0] = np.sign(err) * acc
        # self.perf = np.sign(err) * np.sign(acc) * acc**2
        
    def perf_pos(self, err, acc):
        """Simple reward: let body acceleration point into reduced error direction"""
        self.perf = err

    def perf_gauss_double(self, mean=0., sigma=1.0, accel=0.):
        """Double gaussian for forcing output to be in a given range"""
        g1 = gaussian(mean,  sigma, accel) 
        g2 = gaussian(-mean, sigma, accel)
        self.perf = g1 + g2 # .reshape((1,2))

    @dec_compute_infth_soft()
    def perf_pi_discrete(self, x, avg=False):
        if avg:
            # compute average PI
            return learnerReward.piCalcD.computeAverageLocal(x)
        else:
            # compute list of momentary PI estimates
            pi = learnerReward.piCalcD.computeLocal(x)
            # return last element of list
            return list(pi)[-1]

    @dec_compute_infth_soft()
    def perf_ais_discrete(self, x, avg=False):
        if avg:
            # compute average PI
            return learnerReward.aisCalcD.computeAverageLocal(x)
        else:
            # compute list of momentary PI estimates
            pi = learnerReward.aisCalcD.computeLocal(x)
            # return last element of list
            return list(pi)[-1]

        
    @dec_compute_infth_soft()
    def perf_pi_continuous(self, x):
        # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
        # learnerReward.piCalcC.initialise(40, 1, 0.5);
        # print "perf_pi_continuous", x
        learnerReward.piCalcC.initialise(100, 1)
        # src = np.atleast_2d(x[0:-1]).T # start to end - 1
        # dst = np.atleast_2d(x[1:]).T # 1 to end
        # learnerReward.piCalcC.setObservations(src, dst)
        # src = np.atleast_2d(x).T # start to end - 1
        # learnerReward.piCalcC.setObservations(src.reshape((src.shape[0],)))
        learnerReward.piCalcC.setObservations(x)
        # print type(src), type(dst)
        # print src.shape, dst.shape
        return learnerReward.piCalcC.computeAverageLocalOfObservations()# * -1

    @dec_compute_infth_soft()
    def perf_ais_continuous(self, x):
        # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
        # learnerReward.piCalcC.initialise(40, 1, 0.5);
        # print "perf_pi_continuous", x
        learnerReward.aisCalcC.initialise(100, 1);
        # src = np.atleast_2d(x[0:-1]).T # start to end - 1
        # dst = np.atleast_2d(x[1:]).T # 1 to end
        # learnerReward.piCalcC.setObservations(src, dst)
        # src = np.atleast_2d(x).T # start to end - 1
        # learnerReward.piCalcC.setObservations(src.reshape((src.shape[0],)))
        learnerReward.aisCalcC.setObservations(x)
        # print type(src), type(dst)
        # print src.shape, dst.shape
        return learnerReward.aisCalcC.computeAverageLocalOfObservations()# * -1

class DataLoader():
    def __init__(self, prefix="."):
        self.prefix = prefix
        self.data_names = []
        self.data = 0
        self.delim = "\t"
        # self.selection = []
        self.sel_start = 0
        self.sel_end = 0
        
    def load(self, filename):
        filepath = self.prefix + filename
        try:
            f = open(filepath, "r")
        except IOError as e:
            print ("IOError", e)
            sys.exit(1)
        
        self.data_names = f.readline().rstrip("\n").split(self.delim)
        self.data = np.genfromtxt(f, delimiter=self.delim, skip_header=1)
        # self.selection = range(self.data.shape[0]) # entire file
        self.sel_start = 0
        self.sel_end = self.data.shape[0]
        f.close()

    def get_data(self, cols=[0]):
        # print "self.selection", self.selection
        return self.data[self.sel_start:self.sel_end,cols]

    def set_selection(self, start, end):
        self.selection = selection

class Predictor(object):
    def __init__(self):
        pass

class PredictorReservoir(Predictor):
    def __init__(self, pre_inputs = [0], pre_delay = [10, 50, 100], episode_len=10000, network_size=200):
        Predictor.__init__(self)
        
        # forward model / predictor array
        # predict which input
        self.pre_inputs = pre_inputs
        # predict which temporal distance
        self.pre_delay = pre_delay
        self.pre_num = len(self.pre_inputs) * len(self.pre_delay)
        self.pre_z = np.zeros((self.pre_num, episode_len+1))
        self.pre_err = np.zeros((self.pre_num, 1))
        # print self.cfg.N
        self.pre_w = np.zeros((self.pre_num, network_size, episode_len+1))
        pre_alpha = 1.
        self.pre_P = (1.0/pre_alpha)*np.eye(network_size)
        self.pre_P = np.tile(self.pre_P, len(self.pre_delay))
        self.pre_P = self.pre_P.T.reshape((len(self.pre_delay), network_size, network_size))
        
class learner():
    """Generic learner class"""
    def __init__(self):
        self.name = "learner"

    def step(self):
        pass
        
    def reset(self):
        pass

    def update(self, x, y):
        pass

    def update_batch(self, X, Y):
        pass

    def anneal(self, eta_init=1., cnt=0, anneal_const=1e5):
        """Return anneal value for index/step i"""
        eta = eta_init / (1 + (cnt/anneal_const))
        return eta

    def save(self):
        pass

class learnerEH(learner):
    """Basic exploratory Hebbian learner ingredients"""
    def __init__(self, args):
        learner.__init__(self)
        
        # parameter configuration
        self.cfgprefix = args.datadir # "esn-reward-UF-EH-MORSE-altitude-vision/"
        # self.cfgtype = "new"
        # the config file name
        self.cfgfile = args.cfgfilename # the config file name
        # print ("cfgfile", self.cfgfile)
        self.cfg = learnerConf() # the configuration itself
        # self.cfg.set_cfgfile(self.cfgprefix + mode + ".cfg")
        self.cfg.set_cfgfile(self.cfgprefix + "/" + self.cfgfile)
        self.cfg.read()
        # read all parameters
        self.cfg.cfgget()

        # FIXME: need to put this here to avoid circular import
        from reservoirs import Reservoir, Reservoir2
        # print "g =", self.cfg.g
        self.res = Reservoir(N=self.cfg.N,
                             p = self.cfg.p,
                             input_num=self.cfg.idim,
                             output_num=self.cfg.odim,
                             g = self.cfg.g,
                             tau = self.cfg.tau,
                             eta_init = 0,
                             feedback_scale = self.cfg.res_feedback_scaling,
                             input_scale = self.cfg.res_input_scaling,
                             bias_scale = self.cfg.res_bias_scaling,
                             nonlin_func = self.cfg.nonlin_func, # np.tanh, # lambda x: x,
                             sparse = True, ip=bool(self.cfg.use_ip),
                             theta = self.cfg.res_theta,
                             theta_state = self.cfg.res_theta_state,
                             coeff_a = self.cfg.coeff_a
                             )
        
        # self.res = Reservoir2(N = self.cfg.N,
        #                       p = self.cfg.p,
        #                       input_num = self.cfg.idim,
        #                       output_num = self.cfg.odim,
        #                       g = self.cfg.g,
        #                       tau = self.cfg.tau,
        #                       eta_init = 0.,
        #                       feedback_scale = self.cfg.res_feedback_scaling,
        #                       input_scale = self.cfg.res_input_scaling,
        #                       bias_scale = self.cfg.res_bias_scaling,
        #                       nonlin_func = self.cfg.nonlin_func, # np.tanh, # lambda x: x,
        #                       sparse = True, ip=bool(self.cfg.use_ip),
        #                       theta = self.cfg.res_theta,
        #                       theta_state = self.cfg.res_theta_state,
        #                       coeff_a = self.cfg.coeff_a)
        # # myidentity, np.tanh
                              
        # counting
        self.cnt_main = 0

        # input / output / state variables and memory for synchronous logging
        self.iosm = learnerIOSMem(self.cfg.idim, self.cfg.odim, self.cfg.N,
                                  memlen = self.cfg.len_episode)
        # reward
        self.rew = learnerReward(self.cfg.idim, self.cfg.odim, memlen = self.cfg.len_episode,
                                 coeff_a = self.cfg.coeff_a)

        # FIXME: parameter configuration post-processing
        # expand input coupling matrix from specification
        self.use_icm = True
        self.input_coupling_mtx = np.zeros_like(self.iosm.x)
        for k,v in self.cfg.input_coupling_mtx_spec.items():
            self.input_coupling_mtx[k] = v
        # print ("input coupling matrix", self.input_coupling_mtx)

        # eligibility traces
        self.ewin_off = 0
        self.ewin = self.cfg.et_winsize
        # print "ewin", self.ewin
        self.ewin_inv = 1./self.ewin
        funcindex = 0 # rectangular
        # funcindex = 3 # double_exponential
        self.etf = Eligibility(self.ewin, funcindex)
        self.et_corr = np.zeros((1, self.cfg.et_winsize))

        # predictors
        if self.cfg.use_pre:
            self.pre = PredictorReservoir(self.cfg.pre_inputs,
                                        self.cfg.pre_delay,
                                        self.cfg.len_episode,
                                        self.cfg.N)

        # use weight bounding
        if self.cfg.use_wb:
            self.bound_weight_fit(self.cfg.wb_thr)
        # density estimators

        # other measures
    def __del__(self):
        del self.res
        del self.iosm
        del self.rew

    def bound_weight_fit(self, thr):
        p_x = np.array([0., 0.5, 0.8, 1.]) * thr
        p_y = np.array([1., 1.0,  0.70, 0.0])
        pdeg = 5

        z = np.polyfit(p_x, p_y, pdeg)

        print "learner polyfit", type(z), z

        self.bound_weight_poly = np.poly1d(z)

    def bound_weight(self, x):
        ret = np.clip(self.bound_weight_poly(x), 0., 1.)
        print "bw", ret
        return ret
    
    def predict_and_learn(self, ti):
        ############################################################
        # forward model
        # compute time delayed predictions
        pre_r = np.zeros((len(self.pre.pre_delay), self.cfg.N, 1))
        e = np.zeros((self.pre.pre_num, 1))
        for idx,pre_delay in enumerate(self.pre.pre_delay):
            # print pre_delay
            pre_r[idx] = self.iosm.r_[ti-pre_delay,:].reshape((self.cfg.N, 1))
            self.pre.pre_z[idx,ti] = np.dot(self.pre.pre_w[idx,:,ti].reshape((1, self.cfg.N)), pre_r[idx])
        # print "r", pre_r
        # print "z_fwd", self.pre_z[:,ti]
        # print ti
        # if ti == 5000:
        #     self.ip2d.mass = 2.
        #     self.cfg.lag = 20
        #     self.ip2d.alag = self.cfg.lag-1
                
        # learn weights for time delayed prediction (with FORCE rule)
        # FIXME: add Soh's SOESGP learning here
        for idx,pre_delay in enumerate(self.pre.pre_delay):
            e[idx,0] = self.pre.pre_z[idx,ti] - self.ip2d.v[ti,0]
            self.pre.pre_err[idx,0] = (0.8 * self.pre.pre_err[idx,0]) + (0.2 * np.abs(e[idx,0]))
            if ti > 100 and ti < 30000:
                k = np.dot(self.pre.pre_P[idx], pre_r[idx])
                rPr = np.dot(pre_r[idx].T, k)
                c = 1.0/(1.0 + rPr)
                # print "r.shape", self.r.shape
                # print "k.shape", k.shape, "P.shape", self.P.shape, "rPr.shape", rPr.shape, "c.shape", c.shape
                self.pre.pre_P[idx] = self.pre.pre_P[idx] - np.dot(k, (k.T*c))
                # print self.pre_P
                # error
                # e[idx,0] = self.pre_z[idx,ti] - self.ip2d.v[ti,0]
                # self.pre_err[idx,0] = 0.95 * self.pre_err[idx,0] + 0.05 * e[idx,0]
                # print "e", e
                dw = -e[idx,0] * k * c
                # print dw.shape
                # print self.pre_w[:,ti].shape
                self.pre.pre_w[idx,:,ti+1] = self.pre.pre_w[idx,:,ti] + dw[:,0]
            else:
                self.pre.pre_w[idx,:,ti+1] = self.pre.pre_w[idx,:,ti]
        
    def learnEH(self, channel=0):
        """Apply modulator with Hebbian LR"""
        now = self.cnt_main
        
        bi = (now - self.cfg.lag) # backindex
        # for experimentation
        # bi = (now - self.cfg.lag) - 1 # backindex
        # print now, bi
        # bi = now - 1 # test lag information dependence
        
        for i in range(self.cfg.odim):
            if self.rew.perf[i,0] > self.rew.perf_lp[i,0]:
                self.rew.mdltr[i,0] = 1
            else:
                self.rew.mdltr[i,0] = 0
            # this needs to be indented one level up
            dw = np.zeros_like(self.res.wo)
            if self.cfg.use_anneal:
                eta = self.anneal(self.cfg.eta_EH, self.cnt_main, self.cfg.anneal_const)
                theta = self.anneal(self.cfg.res_theta, self.cnt_main, self.cfg.anneal_const)
                self.res.set_theta(theta)
            else:
                eta = self.cfg.eta_EH

            if self.cfg.use_wb:
                eta *= self.bound_weight(np.linalg.norm(self.res.wo[:,i], 2))
            # print "eta", eta, "theta", self.res.theta
            # print "learnEH: zn_lp", bi, self.iosm.zn_lp_[bi,i]
            # dw = eta * self.iosm.r_[bi] * (self.iosm.zn_[bi,i] - self.iosm.zn_lp_[bi,i]) * self.rew.mdltr[i,0]
            # dw = eta * self.iosm.r_[bi] * (self.iosm.zn_[bi,i] - self.iosm.zn_lp_[bi,i]) * (self.rew.perf[i,0] - self.rew.perf_lp[i,0])
            # print "mse", self.iosm.mse.shape
            if self.iosm.mse[i,0] == 0.:
                print "WARNING: mse = 0"
            # this should be abs(mse)
            dw = eta * self.iosm.r_[bi] * (self.iosm.zn_[bi,i] - self.iosm.zn_lp_[bi,i]) * self.rew.mdltr[i,0] # * self.iosm.mse[i,0]
            # print np.linalg.norm(dw)
            # print self.iosm.zn_[bi]
            # print self.iosm.r_[bi]
            # print "dw", dw.shape
            # print "wo", self.res.wo[:,i].shape
            
            # if not washout or testing
            if self.cnt_main > self.cfg.len_washout and self.cnt_main < (self.cfg.len_episode * self.cfg.ratio_testing):
                self.res.wo[:,i] += dw[:]

    def learnEHE(self):
        """Apply modulator with Hebbian LR using eligibility traces"""
        # current time step
        now = self.cnt_main
        # print "learnEHE ----"
        # eta = self.cfg.eta_EH * self.ewin_inv
        if self.cfg.use_anneal:
            eta = self.anneal(self.cfg.eta_EH, self.cnt_main, self.cfg.anneal_const)
            theta = self.anneal(self.cfg.res_theta, self.cnt_main, self.cfg.anneal_const)
            self.res.set_theta(theta)
        else:
            eta = self.cfg.eta_EH
        print "eta", eta
        print "theta", self.res.theta
        #  rescale for eligibility window
        # eta *= self.ewin_inv
        # eta = eta_EH * anneal factor

        for i in range(self.cfg.odim):
            if self.rew.perf[i,0] > self.rew.perf_lp[i,0]:
                self.rew.mdltr[i,0] = 1
            else:
                self.rew.mdltr[i,0] = 0
            # this needs to be indented one level up
            dw = np.zeros_like(self.res.wo).T
            # print "dw.shape", dw.shape
            mdltr = self.rew.mdltr[i,0]
            # print "self.etf", self.etf
            # always offset at least = 1
            # this loop is a convolution
            for j in range(self.ewin_off+1, self.ewin+self.ewin_off):
                bi = now - j # backindex
                # print "learnEHE: ", bi
                # eta = self.cfg.eta_EH * self.etf.efunc(j)#  * 0.01
                eta_eff = eta * self.etf.efunc(j)#  * 0.01
                # print "eta", j, eta
                # eta = self.cfg.eta_EH * self.ewin_inv
                # derivative postsynaptic activation, resp. motor output
                postsyn = self.iosm.zn_[bi,i] - self.iosm.zn_lp_[bi,i]
                # presynaptic activation, also: state
                presyn = self.iosm.r_[bi]
                # print "hebbian terms eta, post, mod, pre", eta, postsyn, mdltr, presyn
                # print "learnEHE: zn_lp", bi, self.iosm.zn_lp_[bi,i], self.iosm.zn_[bi,i]
                ddw = eta_eff * postsyn * presyn * mdltr # * et
                # self.et_corr[0,j] += np.abs(np.sum(ddw))
                self.et_corr[0,j] += np.sum(ddw)
                dw += ddw

            print "et corr argmax:", self.et_corr.argmax()
            print "et corr argmin:", self.et_corr.argmin()
            # print self.iosm.zn_[bi]
            # print self.iosm.r_[bi]
            # print "dw", dw.shape, dw

            # if not washout or testing
            if self.cnt_main > self.cfg.len_washout and self.cnt_main < (self.cfg.len_episode * self.cfg.ratio_testing):
                self.res.wo[:,i] += dw[0]

    def learnEHEAvg(self):
        """Average over eligibility window and make a single update"""
        now = self.cnt_main
        # print "learnEHE ----"
        for i in range(self.cfg.odim):
            if self.rew.perf[i,0] > self.rew.perf_lp[i,0]:
                self.rew.mdltr[i,0] = 1
            else:
                self.rew.mdltr[i,0] = 0
            # this needs to be indented one level up
            dw = np.zeros_like(self.res.wo).T
            # print "dw.shape", dw.shape
            mdltr = self.rew.mdltr[i,0]
            # print "self.etf", self.etf
            # eta = eta_EH * anneal factor
            # for j in range(20):
            # always offset at least = 1

            sl = slice(now-self.ewin,now-1)
            postsyn = self.iosm.zn_[sl,i] - self.iosm.zn_lp_[sl,i]
            presyn = self.iosm.r_[sl]
            # this doesn't work
            hebbian = np.mean(postsyn * presyn) # component-wise product
            # don't need reduced eta because 1/N is included in mean
            eta = self.cfg.eta_EH # * self.ewin_inv
                        
            dw = eta * hebbian * mdltr # * et
            # self.et_corr[0,j] += np.abs(np.sum(ddw))
            # self.et_corr[0,j] += np.sum(ddw)
            # dw += ddw

            # print "et corr argmax:", self.et_corr.argmax()
            # print "et corr argmin:", self.et_corr.argmin()
            # print self.iosm.zn_[bi]
            # print self.iosm.r_[bi]
            # print "dw", dw.shape, dw

            # if not washout or testing
            if self.cnt_main > self.cfg.len_washout and self.cnt_main < (self.cfg.len_episode * self.cfg.ratio_testing):
                self.res.wo[:,i] += dw[0]
        


    

    def memory_pushback(self):
        # print "pushing"
        # print self.iosm.x.shape
        self.iosm.x_[self.cnt_main,:] = self.iosm.x.reshape((self.cfg.idim, ))
        self.iosm.x_raw_[self.cnt_main,:] = self.iosm.x_raw.reshape((self.cfg.idim, ))
        self.iosm.z_[self.cnt_main,:] = self.iosm.z.reshape((self.cfg.odim, ))
        self.iosm.zn_[self.cnt_main,:] = self.iosm.zn.reshape((self.cfg.odim, ))
        # 20150210: changed that to iosm.z
        # self.iosm.z_[self.cnt_main,:] = self.res.z.reshape((self.cfg.odim, ))
        # self.iosm.zn_[self.cnt_main,:] = self.res.zn.reshape((self.cfg.odim, ))
        self.iosm.zn_lp_[self.cnt_main,:] = self.res.zn_lp.reshape((self.cfg.odim, ))
        self.iosm.r_[self.cnt_main,:] = self.res.r.reshape((1, self.cfg.N))
        self.iosm.w_[self.cnt_main,:] = self.res.wo.reshape((1, self.cfg.N, self.cfg.odim))
        self.rew.perf_[self.cnt_main,:] = self.rew.perf.reshape((1, self.cfg.odim))
        self.rew.perf_lp_[self.cnt_main,:] = self.rew.perf_lp.reshape((1, self.cfg.odim))
        # performance
        self.iosm.e_[self.cnt_main,:]   = self.iosm.e.reshape((1, self.cfg.odim))
        self.iosm.t_[self.cnt_main,:]   = self.iosm.t.reshape((1, self.cfg.idim + self.cfg.odim))
        self.iosm.mse_[self.cnt_main,:] = self.iosm.mse.reshape((1, self.cfg.odim))

    def savenetwork(self, filename):
        # from cPickle import Pickler
        # this is from picloud/cloud because built-in pickle won't do the job
        from cloud.serialization.cloudpickle import dumps
        f = file(filename, "wb")
        # p = Pickler(f)
        # p.dump(self.res)
        f.write(dumps(self.res))
        f.close()
        # self.res.save(filename)

    def loadnetwork(self, filename):
        # from cPickle import Unpickler
        from pickle import Unpickler
        f = file(filename, "rb")
        u = Unpickler(f)
        self.res = u.load()
        # tmp = u.load()
        # print tmp
        f.close()
        # self.res.save(filename)
                
    def savelogs(self, ts=None, saveres=True, filename=None):
        # FIXME: consider HDF5
        if ts == None:
            ts = time.strftime("%Y%m%d-%H%M%S")

        # np.save("%s/log-x-%s" % (self.cfgprefix, ts), self.iosm.x_)
        # np.save("%s/log-x_raw-%s" % (self.cfgprefix, ts), self.iosm.x_raw_)
        # np.save("%s/log-z-%s" % (self.cfgprefix, ts), self.iosm.z_)
        # np.save("%s/log-zn-%s" % (self.cfgprefix, ts), self.iosm.zn_)
        # np.save("%s/log-zn_lp-%s" % (self.cfgprefix, ts), self.iosm.zn_lp_)
        # np.save("%s/log-r-%s" % (self.cfgprefix, ts), self.iosm.r_)
        # np.save("%s/log-w-%s" % (self.cfgprefix, ts), self.iosm.w_)
        # network data, pickling reservoir, input weights, output weights
        # self.res.save("%s/log-%s-res-%s.bin" % (self.cfgprefix, self.cfgprefix, ts))

        if filename == None:
            logfile = "%s/log-learner-%s" % (self.cfgprefix, ts)
        else:
            logfile = filename
        if saveres:
            np.savez_compressed(logfile, x = self.iosm.x_,
                            x_raw = self.iosm.x_raw_, z = self.iosm.z_, zn = self.iosm.zn_,
                            zn_lp = self.iosm.zn_lp_, r = self.iosm.r_, w = self.iosm.w_, e = self.iosm.e_,
                            t = self.iosm.t_, mse = self.iosm.mse_)
        else:
            np.savez_compressed(logfile, x = self.iosm.x_,
                            x_raw = self.iosm.x_raw_, z = self.iosm.z_, zn = self.iosm.zn_,
                            zn_lp = self.iosm.zn_lp_, w = self.iosm.w_, e = self.iosm.e_,
                            t = self.iosm.t_,
                            mse = self.iosm.mse_)
        print "logs saved to %s" % logfile
        return logfile
        
        
class GHA(learner):
    """Generalized Hebbian Algorithm for learning PCA"""
    def __init__(self, eta=1e-3, ndims=2, pdims=1):
        # super(GHA, self).__init__(self)
        learner.__init__(self)
        self.name = "GHA"
        self.eta = eta
        self.ndims = ndims
        self.pdims = pdims
        self.w = 0.1 * np.random.randn(self.ndims, self.pdims)
        self.cnt = 0

    def reset(self):
        self.cnt = 0
        
    def update(self, x):
        """Single step learning update"""
        # print x.shape
        x = x.reshape((self.ndims, 1))
        y = np.dot(self.w.T, x)
        # GHA rule in matrix form
        d_w = self.anneal(self.cnt) * self.eta * (np.dot(x, y.T) - np.dot(self.w, np.tril(np.dot(y, y.T))))
        self.w += d_w
        self.cnt += 1
        return y

    def anneal(self, i):
        return 1
        # return np.exp(-i/10000.)


class dataPCA():
    def __init__(self, l=1000, ndims = 2, seed=0, ):
        np.random.seed(seed)
        self.l = l
        self.ndims = ndims
        
    def generate(self, type="2d"):
        if type == "2d":
            M = np.random.rand(self.ndims, self.ndims)
            print (M)
            M = sLA.orth(M)
            print (M)
            S = np.dot(np.diag([0, .25]), np.random.randn(self.ndims, self.l))
            print ("S.shape", S.shape)
            print (S)
            A = np.dot(M, S)
            print ("A.shape", A.shape)
            # print A
            return(A, S, M)
            
        elif type == "close":
            S = 2 * (np.random.rand(self.ndims, self.l) - 0.5)
            A = S
            print (A.shape)
            # A(2:end,:) = A(2:end,:) + A(1:end-1, :)/2;
            A[1:-1,:] = A[1:-1,:] + A[0:-2, :]/2.
            return (A, S, np.zeros((1,1)))

        elif type == "noisysinewave":
            t = np.linspace(0, 2 * np.pi, self.l)
            sine = np.sin(t * 10)
            # sine = 1.2 * (np.random.rand(1, self.l) - 0.5)
            # nu = 0.1 * (np.random.rand(1, self.l) - 0.5)
            # sine = 2.3 * np.random.randn(1, self.l)
            nu = 0.7 * np.random.randn(1, self.l)
            c1 = (2.7 * sine) + (2 * nu)
            c2 = (1.1 * sine) + (1.2 * nu)
            A = np.vstack((c1, c2))
            print (A.shape)
            # A(2:end,:) = A(2:end,:) + A(1:end-1, :)/2;
            # A[1:-1,:] = A[1:-1,:] + A[0:-2, :]/2.
            return (A, np.zeros((2, self.l)), np.zeros((1,1)))

if __name__ == "__main__":
    try:
        import argparse
    except ImportError, e:
        print "Importing argparse failed: %s" % e
        sys.exit(1)

    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--mode", dest="mode", default="test")
    parser.add_argument("-m", "--mode", dest="mode", default="other")

    args = parser.parse_args()
    
    if args.mode == "test":
        try:
            # unit testing
            import unittest
            from smptests import TestLearner
        except ImportError, e:
            print "Importing unittest failed: %s" % (e)
            sys.exit(1)

        # proceed with testing
        tl = TestLearner()
        print tl
        unittest.main()

    else:
        import matplotlib.pylab as pl
        elen = 1000
        ndims = 2
        pdims = 2
        eta = 1e-2
        niter = 1
        # learner instance
        gha = GHA(eta = eta, ndims = ndims, pdims = pdims)
        # data instance
        datagen = dataPCA(l = elen, ndims=ndims, seed = 123456789)
        # generate test data
        # (A, S, M) = datagen.generate(type = "close")
        (A, S, M) = datagen.generate(type = "noisysinewave")
         # debug plot
        pl.subplot(211)
        pl.title("Matrix S")
        pl.plot(S.T)
        pl.subplot(212)
        pl.title("Matrix A")
        pl.plot(A.T)
        pl.show()

        # allocate for plotting
        y_n = np.zeros((pdims, elen))
        w_norm_n = np.zeros((1, elen))
        
        # iterate globally
        for ni in range(niter):
            # learn single steps
            for i, Ai in enumerate(A.T):
                # print i, Ai
                y = gha.update(Ai)
                # print y.shape
                y_n[:,i] = y.reshape((pdims,))
                w_norm_n[0,i] = LA.norm(gha.w, 2)
                
        pl.subplot(211)
        pl.scatter(y_n[0,:], y_n[1,:])
        pl.subplot(212)
        pl.scatter(A[0,:], A[1,:])
        pl.show()

        pl.subplot(311)
        pl.plot(y_n.T)
        pl.subplot(312)
        pl.plot(A.T)
        pl.subplot(313)
        pl.plot(w_norm_n.T)
        pl.show()
