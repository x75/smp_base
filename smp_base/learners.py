"""smp_base/learners.py

FIXME: models_res

learner class using the reservoirs.py model

interface to the robot sensorimotor loop for doing
 - reward modulated hebbian learning with different noise models (incremental online reinforcement learning)
 - supervised online prediction learning

"""

import sys, time

import numpy as np
import numpy.linalg as LA
import scipy.linalg as sLA

from matplotlib.pyplot import figure

import configparser, ast

from smp_base.eligibility import Eligibility
from smp_base.models import smpModelInit, smpModelStep, smpModel
from smp_base.models import make_figure, make_gridspec
from smp_base.models import iir_fo
from smp_base.reservoirs import Reservoir, LearningRules

try:
    from smp_base.measures_infth import init_jpype, dec_compute_infth_soft
    from jpype import JPackage
    init_jpype()
    HAVE_JPYPE = True
except ImportError as e:
    print("Couldn't import init_jpype from measures_infth, make sure jpype is installed", e)
    HAVE_JPYPE = False
    
# TODO
# - make proper test case and compare with batch PCA
# - implement APEX algorithm?

TWOPI_SQRT = np.sqrt(2*np.pi)

def gaussian(m, s, x):
    return 1/(s*TWOPI_SQRT) * np.exp(-0.5*np.square((m-x)/s))

class smpSHL(smpModel):
    """smpSHL class

    Single Hidden Layer neural network model class.

    This class implements networks in the spectrum from reservoir
    networks (echo state network, liquid state machines) to extreme
    learning machines. Networks are trained by fitting output weights
    to approximate a target from the single hidden layer
    activations. Current learning algorithms are recursive least
    squares variants (rls, force) and reward modulated hebbian
    learning (eh).

    Because this is a stateful model (it has memory) the fit step
    needs to know the sensorimotor tapping context to be able to
    correlate past activation states and outputs with delayed error
    feedback or rewards.
    """
    
    defaults = {
        'idim': 1,
        'odim': 1,
        'memory': 1,
        'tau': 1.0,
        'multitau': False,
        'modelsize': 100,
        'density': 0.1,
        'spectral_radius': 0.0,
        'w_input': 0.66,
        'w_feedback': 0.0,
        'w_bias': 1.0,
        'nonlin_func': np.tanh,
        'sparse': True,
        'ip': False,
        'theta': 0.01,
        'theta_state': 0.05,
        'coeff_a': 0.2,
        'visualize': True,
        'alpha': 10.0,
        'lrname': 'FORCE',
        'mixcomps': 6,
        'sigma_mu': 5e-2,
        'sigma_sig': 5e-2,
        'sigma_pi': 5e-2,
        'eta_init': 1e-4,
        'oversampling': 1,
        'input_coupling': 'normal',
        'wgt_thr': 1.0,
        'lag_past': (-2, -1),
        'lag_future': (-1, -0),
        }

    conf_fwd_perf = {
        'idim': 6,
        'odim': 2,
        'memory': 1,
        'tau': 0.1, # 1.0,
        'multitau': False,
        'modelsize': 200,
        'density': 0.1,
        'spectral_radius': 0.99,
        'w_input': 1.0,
        'w_feedback': 0.0,
        'w_bias': 0.1,
        'nonlin_func': np.tanh,
        'sparse': True,
        'ip': False,
        'theta': 0.01,
        'theta_state': 0.01,
        'coeff_a': 0.2,
        'visualize': False,
        'alpha': 0.5,
        'lrname': 'FORCE',
        'mixcomps': 3,
        'eta_init': 1e-4,
        'oversampling': 1,
        'input_coupling': 'sparse_normal',
        }
        
    @smpModelInit()
    def __init__(self, conf):
        """smpSHL.init

        Use conf dictionary to select, configure and load an adaptive
        model from smp_base.
        """
        # base init
        smpModel.__init__(self, conf)

        # debugging
        for v in ['lrname', 'theta', 'input_coupling', 'visualize']:
            print("%s.%s = %s, conf[%s] = %s" % (
                self.__class__.__name__, v, getattr(self, v), v, conf[v]))

        # representation specific pre config
        if self.lrname == 'FORCEmdn':
            # self.odim_real = self.odim * self.mixcomps * 3
            # wo initialization
            self.num_mu = self.odim * self.mixcomps
            # self.num_sig = self.odim ** 2 * self.mixcomps
            self.num_sig = ((self.odim ** 2 - self.odim)/2 + self.odim) * self.mixcomps
            self.num_pi = self.mixcomps
            self.odim_real = self.num_mu + self.num_sig + self.num_pi
            # self.alpha = 10.0
            self.tau = 1.0 # 0.025
            print("self.alpha", self.alpha)
        else:
            self.odim_real = self.odim

        #
        if self.lrname in ['eh', 'EH']:
            # algorithm variables
            # FIXME: these should all go into the learning rule class
            # tapping spec
            self.laglen_past = self.lag_past[1] - self.lag_past[0]
            self.laglen_future = self.lag_future[1] - self.lag_future[0]
            self.idim_single = self.idim / self.laglen_past
            self.odim_single = self.odim_real / self.laglen_future
            # adjust eta for multiple updates
            self.eta = self.eta / float(self.laglen_past)
            self.eta2 = self.eta

            # forward models: output
            self.y_model = iir_fo(a = self.coeff_a, dim = self.odim_real)
            # forward models: performance (reward prediction)
            if self.perf_model_type == 'lowpass':
                self.perf_model = iir_fo(a = self.coeff_a, dim = self.odim_real, y_init = -4.)
            else:
                # fancy predictors
                # models: soesgp, resrls, resforce, knn, i/gmm, hebbsom
                # inputs: perf, meas, pre_l0
                # outputs: perf_hat
                conf_fwd_perf = smpSHL.conf_fwd_perf
                conf_fwd_perf.update({'idim': 2 + 2 + 2 + 0, 'odim': 2})
                # conf_fwd_perf.update({'lag_past': self.lag_past, 'lag_future': self.lag_future})
                conf_fwd_perf.update({'memory': self.memory})
                self.perf_model = smpSHL(conf = smpSHL.conf_fwd_perf)

            # output variables
            # self.y     = np.zeros((self.odim_real, 1))   # output
            self.y_lp  = np.zeros((self.odim_real, 1))   # output prediction
            # self.perf    = np.zeros((self.odim_real, 1)) # performance (-|error|)
            # self.perf_lp = np.zeros((self.odim_real, 1)) # performance prediction
            self.perf    = np.ones((self.odim_real, 1)) * -4 # performance (-|error|)
            self.perf_lp = np.ones((self.odim_real, 1)) * -4 # performance prediction
        
            # explicit short term memory needed for tapping, lambda and gamma
            self.y_lp_ = np.zeros((self.odim_real, self.memory))
            self.perf_ = np.zeros((self.odim_real, self.memory))
            self.perf_lp_ = np.zeros((self.odim_real, self.memory))

        self.y     = np.zeros((self.odim_real, 1))   # output
        # smpSHL learning rule init
        self.lr = LearningRules(ndim_out = self.odim_real, dim = self.odim)
        
        # smpSHL reservoir init
        self.model = Reservoir(
            N = self.modelsize,
            p = self.density,
            input_num = self.idim,
            output_num = self.odim_real,
            g = self.spectral_radius,
            tau = self.tau,
            mtau = self.multitau,
            eta_init = self.eta_init,
            feedback_scale = self.w_feedback,
            input_scale = self.w_input,
            bias_scale = self.w_bias,
            nonlin_func = self.nonlin_func, # np.tanh, # lambda x: x,
            sparse = True, ip = self.ip,
            theta = self.theta,
            theta_state = self.theta_state,
            coeff_a = self.coeff_a,
            alpha = self.alpha,
            input_coupling = self.input_coupling,
        )

        # memory
        self.r_ = np.zeros((self.modelsize, self.memory))
        self.y_ = np.zeros((self.odim_real, self.memory))
            
        # learning rule specific initializations
        if self.lrname == 'FORCEmdn':
            sigmas = [self.sigma_mu] * self.num_mu + [self.sigma_sig] * self.num_sig + [self.sigma_pi] * self.num_pi
            # sigmas = [1e-1] * self.odim_real
            print("%s.init sigmas = %s" % (self.__class__.__name__, sigmas))
            # output weight initialization
            self.model.init_wo_random(
                np.zeros((1, self.odim_real)),
                np.array(sigmas)
            )
            # print "self.model.wo", self.model.wo.shape
            # argh, multivariate output
            self.lr.learnFORCEmdn_setup(mixcomps = self.mixcomps)
        elif self.lrname == 'FORCE':
            self.lr.learnFORCEsetup(self.alpha, self.modelsize)
        elif self.lrname == 'RLS':
            self.lr.learnRLSsetup(x0 = self.model.wo, P0 = np.eye(self.model.N))

        # everybody counts
        self.cnt_step = 0
        
    def visualize_model_init(self):
        """smpSHL.visualize_model_init

        Init model visualization
        """

        self.Ridx  = np.random.choice(self.modelsize, min(30, int(self.modelsize * 0.1)))
        self.Rhist = []
        self.losshist = []
        self.Whist = []
        
        fig = make_figure()
        # print "fig", fig
        self.figs.append(fig)
        gs = make_gridspec(5, 1)
        for subplot in gs:
            self.figs[0].add_subplot(subplot)
        
    def visualize_model(self):
        """smpSHL.visualize_model

        Visualize model state
        """
        plotdata = []

        # print "Yhist", self.Yhist
        if len(self.Xhist) == 0: return

        plottitles = ['X', 'Y', 'r', 'loss', '|W|']
        
        plotdata.append(np.vstack(self.Xhist))
        plotdata.append(np.vstack(self.Yhist))
        plotdata.append(np.hstack(self.Rhist)[self.Ridx].T)
        # plotdata.append(np.hstack(self.losshist)/self.cnt_step)
        plotdata.append(np.hstack(self.losshist))
        plotdata.append(np.hstack(self.Whist))

        print(plotdata[-1].shape)

        lentotal = len(self.Xhist)
        backwin = 10000

        sl = slice(max(0, lentotal - backwin), lentotal)
        
        # plotting
        for i, item in enumerate(plotdata):
            ax = self.figs[0].axes[i]
            ax.clear()
            ax.plot(item[sl])
            ax.set_title(plottitles[i])

    def learnEH_prepare(self, perf = None):
        """smpSHL.learnEH_prepare

        Prepare variables for learning rule input which are not
        covered by step()'s calling pattern.
        """
        # make sure shape agrees with output
        # print "perf", perf.shape
        if np.isscalar(perf) or len(perf.shape) < 1:
            perf_ = -np.ones_like(self.y) * perf
        else:
            perf_ = -np.ones_like(self.y) * perf.reshape(self.odim_real, 1)
        self.perf = perf_
        # copy perf to lr loss
        self.lr.loss = self.perf
        
    @smpModelStep()
    def step(self, X, Y, update = True, rollback = False, *args, **kwargs):
        """smpSHL.step

        Step the model: fit (to most recent state or specified as lag), predict
        """
        # # oversample reservoir: clamp inputs and step the network
        # for i in range(self.oversampling):
        #     _ = self.model.execute(X.T)
        
        # fit (maximization)
        if Y is not None:
            
            # handle different learning rules
            if self.lrname in ['FORCE']:
                dw = np.zeros_like(self.model.wo)
                
                for i in range(self.lag_past[0], self.lag_past[1]):
                    # print "fetching i = %d from r_" % (i,)
                    r = self.r_[...,[i+1]]
                    # print "r", r.T
                    pred = self.y_[...,[i+1]] # Y,
                    # modular learning rule (ugly call)
                    # (self.model.P, k, c) = self.lr.learnFORCE_update_P(self.model.P, self.model.r)
                    (self.model.P, k, c) = self.lr.learnFORCE_update_P(self.model.P, r)
                    # print "self.model.P, k, c", self.model.P, k, c
                    # dw = self.lr.learnFORCE(Y.T, self.model.P, k, c, self.model.r, self.model.z, 0)
                    # dw += self.lr.learnFORCE(
                    #     target = Y.T, P = self.model.P,
                    #     k = k, c = c, r = self.model.r, z = self.model.z, channel = 0)
                    dw += self.lr.learnFORCE(
                        target = Y.T, P = self.model.P,
                        k = k, c = c, r = r, z = pred, channel = 0)
                # print "|dw|", np.linalg.norm(dw, 2)
                self.model.wo += dw
                # dw_t_norm[k,j] = LA.norm(dw[:,k])
                self.model.perf = self.lr.e # mdn_loss_val
                # y_ = self.model.z # self.model.zn.T
            elif self.lrname == 'RLS':
                # print "RLS", self.cnt_step
                dw = np.zeros_like(self.model.wo)
                for i in range(self.lag_past[0], self.lag_past[1]):
                    r = self.r_[...,[i+1]]
                    # print "r", r.T
                    pred = self.y_[...,[i+1]] # Y,
                    # dw = self.lr.learnRLS(target = Y.T, r = self.model.r)
                    # dw = self.lr.learnRLS(target = Y.T, r = r, noise = 1e-1)
                    dw = self.lr.learnRLS(target = Y.T, r = r, noise = 1e-2)
                self.model.wo += dw
                self.model.perf = self.lr.rls_estimator.y
                # for k in range(outsize_):
                #     dw_t_norm[k,j] = LA.norm(dw[:,k])
                
            elif self.lrname == 'FORCEmdn':
                # modular learning rule (ugly call)
                (self.model.P, k, c) = self.lr.learnFORCE_update_P(self.model.P, self.model.r)
                dw = self.lr.learnFORCEmdn(Y.T, self.model.P, k, c, self.model.r, self.model.z, 0, X.T)
                # self.model.wo += (1e-1 * dw)
                # print "dw.shape", dw
                # when using delta rule
                # leta = (1/np.log((self.cnt_step * 0.1)+2)) * 1e-4
                # self.model.wo = self.model.wo + (leta * dw)
                # leta = 1.0
                self.model.wo += dw
                # for k in range(outsize_):
                #     # wo_t_norm[k,j] = LA.norm(wo_t[:,k,j])
                #     dw_t_norm[k,j] = LA.norm(dw[:,k])

                # loss_t[0,j] = self.lr.loss
                if self.cnt_step % 100 == 0:
                    print("|wo| = %s" % np.linalg.norm(self.model.wo)) # mdn_loss_val
                self.model.perf = self.lr.e # mdn_loss_val
            elif self.lrname in ['EH', 'eh']:
                """exploratory hebbian rule"""
                # outside: lag info, X, Y, perf
                # inside: r, Y_bar, perf_bar

                # shorthand
                lag = self.minlag
                # print "    %s.step learnEH lag = %d" % (self.__class__.__name__, lag)

                # compute dw
                dw = np.zeros_like(self.model.wo)
                # print "        dw", dw.shape
                
                # commit to memory
                self.perf_[...,[-1]] = self.perf.copy()
                self.perf_lp_[...,[-1]] = self.perf_lp.copy()

                # loop over embedding
                for i in range(self.lag_past[0], self.lag_past[1]):
                    # print "        lag_past = %d, " % (i, )
                    # for j in range(self.lag_future[0], self.lag_future[1]):
                    # print "            lag_future = %d, " % (j, )

                    # FORCE says i+1
                    lag_past = i+1
                    # lag_future = j
                    dw += self.lr.learnEH(
                        target = None,
                        r = self.r_[...,[lag_past]],
                        pred = self.y_[...,[lag_past]], # Y,
                        pred_lp = self.y_lp_[...,[lag_past]],
                        perf = self.perf,
                        perf_lp = self.perf_lp,
                        eta = self.eta,
                        mdltr_type = self.mdltr_type,
                        mdltr_thr = self.mdltr_thr,
                    )
                        
                # dw = self.lr.learnEH(
                #     target = None,
                #     r = self.r_[...,[-lag]],
                #     pred = self.y_[...,[-lag]], # Y,
                #     pred_lp = self.y_lp_[...,[-lag]],
                #     perf = self.perf,
                #     perf_lp = self.perf_lp,
                #     eta = self.eta,
                #     )
                # apply dw
                if self.cnt_step >= 100:
                    # if np.all(self.perf_lp <= 0.05) and np.all(self.perf <= 0.05):
                    self.model.wo += dw

            # weight decay
            thr = self.wgt_thr # 0.8
            if np.linalg.norm(self.model.wo, 2) > thr:
                # self.model.wo *= 0.95
                self.model.wo /= np.linalg.norm(self.model.wo, 2)
                self.model.wo *= thr
                # else:
                #     print "setting theta low"
                #     # self.theta = 1e-3
            self.model.perf = self.lr.perf # hm?
            # self.model.perf_lp = ((1 - self.model.coeff_a) * self.model.perf_lp) + (self.model.coeff_a * self.model.perf)

        if not update:
            # print "returning", self.cnt_step
            return self.y.T
            
        # prediction (expectation)

        # save current state for rollback
        if rollback:
            self.r_current = self.model.r.copy()
            self.x_current = self.model.x.copy()
        # oversample reservoir: clamp inputs and step the network
        # print "smpSHL.step X", X.shape
        
        for i in range(self.oversampling):
            _ = self.model.execute(X.T)

        # roll memory buffer
        for mem in ['r_', 'y_']:
            # print "smpSHL.step %s pre roll = %s" % (mem, getattr(self, mem))
            setattr(self, mem, np.roll(getattr(self, mem), shift = -1, axis = 1))
            # print "smpSHL.step %s post roll = %s" % (mem, getattr(self, mem))

            
        # prepare output
        if self.lrname in ['RLS', 'FORCE']:
            self.y = self.model.z
        elif self.lrname in ['eh', 'EH']:
            self.y = self.model.zn
            
            # roll memory buffer
            for mem in ['y_lp_']:
                # print "smpSHL.step %s pre roll = %s" % (mem, getattr(self, mem))
                setattr(self, mem, np.roll(getattr(self, mem), shift = -1, axis = 1))
                # print "smpSHL.step %s post roll = %s" % (mem, getattr(self, mem))

            # commit to memory
            self.y_lp_[...,[-1]] = self.y_lp.copy()

            # update output and performance prediction
            self.y_lp = self.y_model.predict(self.y)
            if self.perf_model_type == 'lowpass':
                self.perf_lp = self.perf_model.predict(self.perf)
            elif self.perf_model_type in ['resforce', 'resrls']:
                # FIXME clean up .Ts?
                self.perf_lp = self.perf_model.step(X = X, Y = self.perf.T).T
            
        elif self.lrname == 'FORCEmdn':
            if self.odim < 2:
                self.y = self.lr.mixture(
                    self.model.z[:self.mixcomps,0],
                    np.exp(self.model.z[self.mixcomps:(2*self.mixcomps),0]),
                    self.lr.softmax(self.model.z[(2*self.mixcomps):,0])
                )
            else:
                self.y = self.lr.mixtureMV(
                    self.model.z[:self.num_mu],
                    np.exp(self.model.z[self.num_mu:self.num_mu + self.num_sig]),
                    self.lr.softmax(self.model.z[self.num_mu + self.num_sig:])
                ).reshape((1, self.odim))
                # print "mixtureMV", self.y.shape
        else:
            # self.y = _.T
            self.y = self.model.zn
            # self.y = y.T

        # print "smpSHL.step(X = %s, Y = %s, y_ = %s)" % ( X.shape, Y, self.y)

        # tidy up
        if rollback:
            self.model.r = self.r_current.copy()
            self.model.x = self.x_current.copy()
            
        # commit to memory
        self.r_[...,[-1]] = self.model.r.copy()
        self.y_[...,[-1]] = self.y.copy()
            
        # keep counting, it's important
        self.cnt_step += 1
        
        # return new prediction
        # print "Y", Y.shape, "y_", self.y.shape
        return self.y.T
        
    def predict(self, X, rollback = False):
        """smpSHL.predict

        Predict from input and state
        """
        if X.shape[0] > 1: # batch input
            ret = np.zeros((X.shape[0], self.odim))
            for i in range(X.shape[0]):
                a = self.step(X[i], None, rollback = rollback)
                ret[i] = a
            return ret
        else:
            # X_ = X.flatten().tolist()
            # return self.predict_step(X_)
            return self.step(X, Y = None, update = True, rollback = rollback)
        
    def fit(self, X, Y = None, y = None, update = False):
        """smpSHL.fit

        Fit current or lagged state to target
        """
        if Y is None and y is not None:
            Y = y
            
        if X.shape[0] > 1: # batch input
            ret = np.zeros((X.shape[0], self.odim))
            for i in range(X.shape[0]):
                ret[i] = self.step(X[i], Y[i], update = True)
            return ret
        else:
            # X_ = X.flatten().tolist()
            # return self.predict_step(X_)
            return self.step(X, Y, update = update)
        
class learnerConf():
    """Common parameters for exploratory Hebbian learners"""
    def __init__(self, cfgfile="default.cfg"):
        """learnerConf init"""
        self.cfgfile = cfgfile
        self.cfg = configparser.ConfigParser()

    def set_cfgfile(self, cfgfile="default.cfg"):
        """set the config file path"""
        self.cfgfile = cfgfile
        
    def read(self):
        """read config from file"""
        print(("opening %s" % self.cfgfile))
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
        print(("config params: tau = %f, g = %f, lag = %d, eta = %f, theta = %f, target = %f" % (self.tau, self.g, self.lag, self.eta_EH, self.res_theta, self.target)))

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
    """learnerReward class

    This class holds reward functions for use in learnerEH and elsewhere.

    TODO
    - use measures and measures_infth.py from smp_base or merge and delete
    """

    ################################################################################
    # infth setup stuff
    
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
        """learnerReward.init
        
        idim: input dimensionality (default: 1) \n
        odim: output dimensionality (default: 1) \n
        memlen: length of memory to be kept, in steps (default: 1000)
        """
        
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
        
    def perf_accel_sum(self, err, acc):
        """Simple reward: let body acceleration point into reduced error direction"""
        self.perf[0:self.odim,0] = -np.sum(np.square(err)) # np.sign(err) * acc
        # self.perf = np.sign(err) * np.sign(acc) * acc**2
        
    def perf_pos(self, err, acc):
        """perf_pos

        Position error, 0th order, return err
        """
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
        # learnerReward.piCalcC.initialise(1, 1, 0.5);
        # src = np.atleast_2d(x[0:-1]).T # start to end - 1
        # dst = np.atleast_2d(x[1:]).T # 1 to end
        # learnerReward.piCalcC.setObservations(src, dst)
        
        # print "perf_pi_continuous", x
        # learnerReward.piCalcC.initialise(100, 1);
        # learnerReward.piCalcC.initialise(50, 1);
        learnerReward.piCalcC.initialise(10, 1);
        # src = np.atleast_2d(x).T # start to end - 1
        # learnerReward.piCalcC.setObservations(src.reshape((src.shape[0],)))
        # print "x", x.shape
        learnerReward.piCalcC.setObservations(x)
        # print type(src), type(dst)
        # print src.shape, dst.shape
        return learnerReward.piCalcC.computeAverageLocalOfObservations()# * -1

    @dec_compute_infth_soft()
    def perf_ais_continuous(self, x):
        # Use history length 1 (Schreiber k=1), kernel width of 0.5 normalised units
        # learnerReward.piCalcC.initialise(40, 1, 0.5);
        # print "perf_pi_continuous", x
        learnerReward.aisCalcC.initialise(10, 1);
        # learnerReward.aisCalcC.initialise(10, 10);
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
            print(("IOError", e))
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
        from .reservoirs import Reservoir, Reservoir2
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
        for k,v in list(self.cfg.input_coupling_mtx_spec.items()):
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

        print("learner polyfit", type(z), z)

        self.bound_weight_poly = np.poly1d(z)

    def bound_weight(self, x):
        ret = np.clip(self.bound_weight_poly(x), 0., 1.)
        print("bw", ret)
        return ret
    
    def predict_and_learn(self, ti):
        """learnerEH.predict_and_learn

        Train an additional multi-step forward model alongside the
        inverse one.
        """
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
        """learnerEH.learnEH

        Apply modulator with Hebbian LR
        """
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
                print("WARNING: mse = 0")
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
        """learnerEH.learnEHE

        Apply modulator with Hebbian LR using eligibility traces
        """
        # current time step
        now = self.cnt_main

        # eta = self.cfg.eta_EH * self.ewin_inv
        if self.cfg.use_anneal:
            eta = self.anneal(self.cfg.eta_EH, self.cnt_main, self.cfg.anneal_const)
            theta = self.anneal(self.cfg.res_theta, self.cnt_main, self.cfg.anneal_const)
            self.res.set_theta(theta)
        else:
            eta = self.cfg.eta_EH
            
        print("%s.learnEHE eta = %s" % (self.__class__.__name__, eta))
        print("%s.learnEHE theta = %s, self.res.theta = %s" % (
            self.__class__.__name__, theta, self.res.theta))
        
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
                # accumulate dw for each slot in the eligibility window
                dw += ddw

            print("et corr argmax:", self.et_corr.argmax())
            print("et corr argmin:", self.et_corr.argmin())
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
        print("logs saved to %s" % logfile)
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
            print(("S.shape", S.shape))
            print (S)
            A = np.dot(M, S)
            print(("A.shape", A.shape))
            # print A
            return(A, S, M)
            
        elif type == "close":
            S = 2 * (np.random.rand(self.ndims, self.l) - 0.5)
            A = S
            print((A.shape))
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
            print((A.shape))
            # A(2:end,:) = A(2:end,:) + A(1:end-1, :)/2;
            # A[1:-1,:] = A[1:-1,:] + A[0:-2, :]/2.
            return (A, np.zeros((2, self.l)), np.zeros((1,1)))

if __name__ == "__main__":
    try:
        import argparse
    except ImportError as e:
        print("Importing argparse failed: %s" % e)
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
        except ImportError as e:
            print("Importing unittest failed: %s" % (e))
            sys.exit(1)

        # proceed with testing
        tl = TestLearner()
        print(tl)
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
