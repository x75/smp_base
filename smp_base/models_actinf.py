"""smp_base.models_actinf

..moduleauthor:: Oswald Berthold, 2016-2017

Active inference models based on :mod:`smp.actinf` project code.

This file contains the models_learners which can be used as adaptive models
of sensorimotor contexts designed for an active inference
approach. Currently implemented models are
- k nearest neighbours (knn)
- sparse online gaussian process models powered by Harold Soh's OTL library (soesgp, storkgp)
- gaussian mixture model based on pypr's gmm (gmm)
- hebbian connected SOM via bruno lara, guido schillaci (hebbsom)
- incremental gaussian mixtures (igmm via juan acevedo-valle)
- SOMs connected with hebbian associative links

TODO:
- consolidate calling convention / api for all model types
-- init with single argument config dictionary
-- predict, fit, sample, conditionals, visualize
-- common test code

- implement missing models
   - missing: single hidden layer networks: linear/elm/res with RLS/FORCE/MDN/EH, merge with otl
   - missing: imol/models.py
   - missing: im/models.py
   - missing: smp/models_seq.py
   - missing: smp/models_karpmdn.py
   - MDN model: florens, karpathy, hardmaru, amjad, cbonnett, edward

   - including 'predict_naive' and 'predict_full' methods that would capture returning confidences about the current prediction
   - other variables that might be used by the context to modulate exploration, learning and behaviour
   - disambiguate static and dynamic (conditional inference types) idim/odim
   - consistent sampling from probabilistic models (gmm, hebbsom, ...): sample from prior, stick with last sample's vicinity


- model visualization
   - def visualize for all models
   - plot current / final som configuration
   - plot densities

- hebbsom
   - som track residual error from map training
   - som use residual for adjusting rbf width
   - som extend sampling to sample actual prediction from gaussian with unit's mu and sigma

"""




import numpy as np
import scipy.sparse as sparse
import scipy.stats  as stats
import pylab as pl
import matplotlib.gridspec as gridspec
import pickle
from functools import partial

from .models import smpModelInit, smpModel
from .models import savefig, plot_nodes_over_data_1d_components_fig, plot_nodes_over_data_1d_components

# KNN
from sklearn.neighbors import KNeighborsRegressor

# Online Gaussian Processes
try:
    from otl_oesgp import OESGP
    from otl_storkgp import STORKGP
    HAVE_SOESGP = True
except ImportError as e:
    print("couldn't import online GP models:", e)
    HAVE_SOESGP = False

# Gaussian mixtures PyPR
try:
    import pypr.clustering.gmm as gmm
except ImportError as e:
    print("Couldn't import pypr.clustering.gmm", e)

# hebbsom
try:
    from kohonen.kohonen import Map, Parameters, ExponentialTimeseries, ConstantTimeseries
    from kohonen.kohonen import Gas, GrowingGas, GrowingGasParameters, Filter
    from kohonen.kohonen import argsample
except ImportError as e:
    print("Couldn't import lmjohns3's kohonon SOM lib", e)

# IGMM
try:
    from igmm_cond import IGMM_COND
except ImportError as e:
    print("Couldn't import IGMM lib", e)

# requirements: otl, kohonen, pypr, igmm
    
from smp_base.models_reservoirs import LearningRules

import logging
from smp_base.common import get_module_logger
logger = get_module_logger(modulename = 'models_actinf', loglevel = logging.DEBUG)

saveplot = False # True
model_classes = ["KNN", "SOESGP", "STORKGP", "GMM", "HebbSOM", ",IGMM", "all"]
        
class smpKNN(smpModel):
    """smpKNN

    k-NN function approximator smpmodel originally used for the active
    inference developmental model but generally reusable.
    """
    defaults = {
        'idim': 1,
        'odim': 1,
        'n_neighbors': 5,
        'prior': 'random', # ['random', 'linear']
        'prior_width': 0.01,
    }
    @smpModelInit()
    def __init__(self, conf):
        """smpKNN.__init__

        init
        """
        smpModel.__init__(self, conf)

        # comply
        if not hasattr(self, 'modelsize'):
            self.modelsize = 1000 # self.n_neighbors

        # the scikit base model
        self.fwd = KNeighborsRegressor(n_neighbors = self.n_neighbors)

        # the data store
        self.X_ = []
        self.y_ = []
        
        self.hidden_dist = np.zeros((1, self.n_neighbors))
        self.hidden_dist_sum = np.zeros((1, 1))
        self.hidden_dist_sum_avg = np.zeros((1, 1))
        self.hidden_idx = np.zeros((1, self.n_neighbors))
        # bootstrap the model with prior
        self.bootstrap()

    def get_params(self, *args, **kwargs):
        if 'param' in kwargs:
            if 'w_norm' in kwargs['param']:
                # return np.tile(np.array([(len(self.X_) + len(self.y_))/2.0]), (self.odim, 1))
                return np.tile(np.array([len(self.y_)]), (self.odim, 1))
        
        return self.fwd.get_params()
        
    def visualize(self):
        pass
        
    def bootstrap(self):
        """smpKNN.bootstrap

        Bootstrap the model with some initial dummy samples to prepare it for inference after init
        """
        # bootstrap model
        self.n_samples_bootstrap = max(10, self.n_neighbors)
        logger.info("%s.bootstrapping with %s prior" % (self.__class__.__name__, self.prior))
        if self.prior == 'random':
            for i in range(self.n_samples_bootstrap):
                if self.idim == self.odim:
                    self.X_.append(np.ones((self.idim, )) * i * 0.1)
                    self.y_.append(np.ones((self.odim, )) * i * 0.1)
                else:
                    noise_amp = self.prior_width
                    self.X_.append(np.random.uniform(
                        -noise_amp, noise_amp, (self.idim,)))
                    self.y_.append(np.random.uniform(
                        -noise_amp, noise_amp, (self.odim,)))

        elif self.prior == 'linear':
            for i in range(self.n_samples_bootstrap):
                p_ = -self.prior_width/2.0 + float(i)/self.n_samples_bootstrap
                X = np.ones((self.idim, )) * p_ + np.random.uniform(-0.01, 0.01)
                y = np.ones((self.odim, )) * p_ + np.random.uniform(-0.01, 0.01)
                self.X_.append(X)
                self.y_.append(y)

        # print(self.X_, self.y_)
        self.fwd.fit(self.X_, self.y_)
            
    def predict(self, X):
        """smpKNN.predict

        Predict Y using X on the current model state
        """
        # FIXME: change scikit to store intermediate query results
        #    or: fully local predict def
        self.hidden_dist, self.hidden_idx = self.fwd.kneighbors(X)
        self.hidden_dist_sum = np.mean(self.hidden_dist)
        self.hidden_dist_sum_avg = 0.1 * self.hidden_dist_sum + 0.9 * self.hidden_dist_sum_avg
        # self.hidden_idx_norm = self.hidden_idx.astype(np.float) * self.hidden_dist_sum_avg/1000.0
        self.hidden_idx_norm = self.hidden_idx.astype(np.float) * 1e-3
        # logger.debug('hidden dist = %s, idx = %s', self.hidden_dist, self.hidden_idx)
        return self.fwd.predict(X)

    def fit(self, X, y):
        """smpKNN.fit

        Single fit Y to X step. If the input is a batch of data, fit
        that entire batch and forgetting existing data in X_ and
        Y_. If the input is a single data point, append to X_ and Y_
        and refit the model to that new data.
        """
        if X.shape[0] > 1: # batch of data
            # self.modelsize = X.shape[0]
            return self.fit_batch(X, y)

        # logger.debug("%s.fit[%d] len(X_) = %d, len(y_) = %d, modelsize = %d", self.__class__.__name__, self.cnt, len(self.X_), len(self.y_), self.modelsize)
        self.cnt += 1
        # if len(self.X_) > self.modelsize: return
        
        self.X_.append(X[0,:])
        # self.y_.append(self.m[0,:])
        # self.y_.append(self.goal[0,:])
        self.y_.append(y[0,:])
        
        self.fwd.fit(self.X_, self.y_)

    def fit_batch(self, X, y):
        """smpKNN.fit

        Batch fit Y to X
        """
        self.X_ = X.tolist()
        self.y_ = y.tolist()
        self.fwd.fit(self.X_, self.y_)
        
################################################################################
# ActiveInference OTL library based model, base class implementing predict,
# predict_step (otl can't handle batches), fit, save and load methods
class smpOTLModel(smpModel):
    """smpOTLModel

    Sparse online echo state gaussian process function approximator
    for active inference
    """
    defaults = {
        'idim': 1,
        'odim': 1,
        'otlmodel_type': 'soesgp',
        'otlmodel': None,
        'memory': 1,
        'lag_off': 1,
        }
        
    @smpModelInit()
    def __init__(self, conf):
        # if conf is None: conf = self.defaults
    
        smpModel.__init__(self, conf)

        # self.otlmodel_type = "soesgp"
        # self.otlmodel = None
        # introspection
        self.cnt = 0
        
        # explicit short term memory needed for tapping across lag gaps
        self.r_l = []
        print( "otlmodel.memory", self.memory)
        self.r_ = np.zeros((self.modelsize, self.memory))
        # self.r_ = np.random.uniform(-1, 1, (self.modelsize, self.memory)) * 1.0
        
        # output variables arrays
        self.pred = np.zeros((self.odim, 1))
        self.var = np.zeros((self.odim, 1))
        # output variables lists
        self.pred_l = []
        self.var_l  = []

    def update(self, X_):
        # update state
        self.otlmodel.update(X_)
        # store state
        self.r_ = np.roll(self.r_, shift = -1, axis = -1)
        self.otlmodel.getState(self.r_l)
        tmp = np.array([self.r_l]).T
        # print("%s r_ = %s, r[...,[-1] = %s, tmp = %s" % (self.__class__.__name__, self.r_.shape, self.r_[...,[-1]].shape, tmp.shape))
        self.r_[...,[-1]] = tmp.copy()

    def predict(self, X,rollback = False):
        # row vector input
        if X.shape[0] > 1: # batch input
            ret = np.zeros((X.shape[0], self.odim))
            for i in range(X.shape[0]):
                ret[i] = self.predict_step(X[i].flatten().tolist(), rollback = rollback)
            return ret
        else:
            X_ = X.flatten().tolist()
            return self.predict_step(X_, rollback = rollback)

    def predict_step(self, X_, rollback = False):
        # update state and store it
        self.update(X_)
        # predict output variables from state
        self.otlmodel.predict(self.pred_l, self.var_l)
        # return np.zeros((1, self.odim))
        # set prediction variables
        self.pred = np.array(self.pred_l)
        self.var = np.abs(np.array(self.var_l))

        # roll back the reservoir state if rollback on
        if rollback:
            self.r_ = np.roll(self.r_, shift = 1, axis = -1)
            self.otlmodel.setState(self.r_[...,[-1]].copy().flatten().tolist())

        self.cnt += 1
        return self.pred.reshape((1, self.odim))
        
    def fit(self, X, y, update = True):
        """smpOTLModel.fit

        Fit model to data X, y
        """
        if self.cnt < self.memory: return
        
        if X.shape[0] > 1: # batch of data
            return self.fit_batch(X, y)

        if update:
            X_ = X.flatten().tolist()
            self.update(X_)
            # print("X.shape", X.shape, len(X_), X_)
            # self.otlmodel.update(X_)
            # copy state into predefined structure
            # self.otlmodel.getState(self.r)

        # consider lag and restore respective state
        # print("otlmodel.fit lag_off", self.lag_off)
        r_lagged = self.r_[...,[-self.lag_off]]
        # print ("r_lagged", r_lagged.shape)
        self.otlmodel.setState(r_lagged.flatten().tolist())
            
        # prepare target and fit
        # print("soesgp.fit y", type(y))
        y_ = y.flatten().tolist()
        self.otlmodel.train(y_)

        # restore chronologically most recent state
        r_lagged = self.r_[...,[-1]]
        self.otlmodel.setState(r_lagged.flatten().tolist())
        
    def fit_batch(self, X, y):
        for i in range(X.shape[0]):
            self.fit(X[[i]], y[[i]])
        
    def save(self, filename):
        otlmodel_ = self.otlmodel
        self.otlmodel.save(filename + "_%s_model" % self.otlmodel_type)
        print("otlmodel", otlmodel_)
        self.otlmodel = None
        print("otlmodel", otlmodel_)       
        pickle.dump(self, open(filename, "wb"))
        self.otlmodel = otlmodel_
        print("otlmodel", self.otlmodel)

    @classmethod
    def load(cls, filename):
        # otlmodel_ = cls.otlmodel
        otlmodel_wrap = pickle.load(open(filename, "rb"))
        print("%s.load cls.otlmodel filename = %s, otlmodel_wrap.otlmodel_type = %s" % (cls.__name__, filename, otlmodel_wrap.otlmodel_type))
        if otlmodel_wrap.otlmodel_type == "soesgp":
            otlmodel_cls = OESGP
        elif otlmodel_wrap.otlmodel_type == "storkgp":
            otlmodel_cls = STORKGP
        else:
            otlmodel_cls = OESGP
            
        otlmodel_wrap.otlmodel = otlmodel_cls()
        print("otlmodel_wrap.otlmodel", otlmodel_wrap.otlmodel)
        otlmodel_wrap.otlmodel.load(filename + "_%s_model" % otlmodel_wrap.otlmodel_type)
        # print("otlmodel_wrap.otlmodel", dir(otlmodel_wrap.otlmodel))
        # cls.bootstrap(otlmodel_wrap)
        # otlmodel_wrap.otlmodel = otlmodel_
        return otlmodel_wrap

################################################################################
# Sparse Online Echo State Gaussian Process (SOESGP) OTL library model
class smpSOESGP(smpOTLModel):
    """smpSOESGP

    Sparse online echo state gaussian process function approximator
    for active inference
    """
    # # for input modulation style
    # defaults = {
    #     'idim': 1,
    #     'odim': 1,
    #     'otlmodel_type': 'soesgp',
    #     'otlmodel': None,
    #     'modelsize': 300,
    #     'input_weight': 2.0,
    #     'output_feedback_weight': 0.0,
    #     'activation_function': 1,
    #     'leak_rate': 0.8, # 0.9,
    #     'connectivity': 0.1,
    #     'spectral_radius': 0.99, # 0.999,
    #     # 'kernel_params': [10.0, 10.0], # [2.0, 2.0],
    #     # 'noise': 0.01,
    #     # 'kernel_params': [10.0, 10.0], # [2.0, 2.0],
    #     # 'noise': 1.0, # 0.01,
    #     'kernel_params': [2.0, 2.0], # [2.0, 2.0],
    #     'noise': 5e-2, # 0.01,
    #     'epsilon': 1e-3,
    #     'capacity': 100,  # 10
    #     'random_seed': 101,
    #     'visualize': False,
    # }

    # for self-sampling style
    defaults = {
        'idim': 1,
        'odim': 1,
        'otlmodel_type': 'soesgp',
        'otlmodel': None,
        'memory': 1,
        'lag_off': 1,
        'modelsize': 200,
        'output_feedback_weight': 0.0,
        'use_inputs_in_state': False,
        'activation_function': 0,
        'connectivity': 0.1,
        # 'kernel_params': [10.0, 10.0], # [2.0, 2.0],
        # 'noise': 0.01,
        # 'kernel_params': [10.0, 10.0], # [2.0, 2.0],
        # 'noise': 1.0, # 0.01,
        # pointmass
        'input_weight': 1.0,
        'kernel_params': [10.0, 1.5],
        'noise': 5e-3, #8e-2, # 0.01,
        'leak_rate': 0.1, # 0.9,
        'spectral_radius': 0.9,
        # # barrel
        # 'input_weight': 1.0,
        # 'kernel_params': [1.2, 1.2], # [2.0, 2.0],
        # 'noise': 1e-2,
        # 'leak_rate': 0.9, # 0.9,
        # 'spectral_radius': 0.99, # 0.999,
        'epsilon': 1e-4,
        'capacity': 200,  # 10
        'random_seed': 106,
        'visualize': False,
    }
        
    @smpModelInit()
    def __init__(self, conf):
        smpOTLModel.__init__(self, conf = conf)

        # self.otlmodel_type = "soesgp"
        self.otlmodel = OESGP()

        # self.res_size = 100 # 20
        # self.input_weight = 1.0 # 1.0
        
        # self.output_feedback_weight = 0.0
        # self.activation_function = 1
        # # leak_rate: x <= (1-lr) * input + lr * x
        # self.leak_rate = 0.96 # 0.05 # 0.0 # 0.1 # 0.3
        # self.connectivity = 0.1
        # self.spectral_radius = 0.99

        # # covariances
        # self.kernel_params = [2.0, 2.0]
        # # self.kernel_params = [1.0, 1.0]
        # # self.kernel_params = [0.1, 0.1]
        # self.noise = 0.05
        # self.epsilon = 1e-3
        # self.capacity = 100
        # self.random_seed = 100 # FIXME: constant?

        # self.X_ = []
        # self.y_ = []

        self.bootstrap()
    
    def bootstrap(self):
        from .models_reservoirs import res_input_matrix_random_sparse
        self.otlmodel.init(self.idim, self.odim, self.modelsize, self.input_weight,
                    self.output_feedback_weight, self.activation_function,
                    self.leak_rate, self.connectivity, self.spectral_radius,
                    False, self.kernel_params, self.noise, self.epsilon,
                    self.capacity, self.random_seed)
        im = res_input_matrix_random_sparse(self.idim, self.modelsize, 0.2) * self.input_weight
        # print("im", type(im))
        self.otlmodel.setInputWeights(im.tolist())

################################################################################
# StorkGP OTL based model
class smpSTORKGP(smpOTLModel):
    """smpSTORKGP

    Sparse online echo state gaussian process function approximator
    for active inference
    """
    defaults = {
        'idim': 1,
        'odim': 1,
        'otlmodel_type': 'storkgp',
        'otlmodel': None,
        'modelsize': 50,
        'memory': 1,
        'lag_off': 1,
        'input_weight': 1.0,
        'output_feedback_weight': 0.0,
        'activation_function': 1,
        'leak_rate': 0.96,
        'connectivity': 0.1,
        'spectral_radius': 0.99,
        'kernel_params': [2.0, 2.0],
        'noise': 0.05,
        'epsilon': 1e-3,
        'capacity': 100,
        'random_seed': 100,
        'visualize': False,
    }
        
    @smpModelInit()
    def __init__(self, conf):
        smpOTLModel.__init__(self, conf = conf)
        
        # self.otlmodel_type = "storkgp"
        self.otlmodel = STORKGP()

        # self.res_size = self.modelsize # 100 # 20
        
        self.bootstrap()
    
    def bootstrap(self):

        self.otlmodel.init(
            self.idim, self.odim,
            self.modelsize, # window size
            0, # kernel type
            [0.5, 0.99, 1.0, self.idim],
            1e-4,
            1e-4,
            100 # seed
        )
        
        self.otlmodel.getState(self.r_l)
        # print("|self.r_l| = ", len(self.r_l))
        self.r_ = np.zeros((len(self.r_l), self.memory))

################################################################################
# inference type multivalued models: GMM, SOMHebb, MDN
# these are somewhat different in operation than the models above
# - fit vs. fit_batch
# - can create conditional submodels

# GMM - gaussian mixture model
class smpGMM(smpModel):
    """smpGMM

    Gaussian mixture model based on PyPR's gmm
    """
    defaults = {
        'idim': 1, 'odim': 1, 'K': 10, 'fit_interval': 100,
        'numepisodes': 10, 'visualize': False, 'em_max_iter': 1000}
    
    @smpModelInit()
    def __init__(self, conf):
        """smpGMM.__init__
        """
        smpModel.__init__(self, conf)

        self.cdim = self.idim + self.odim

        # data
        self.Xy_ = []
        self.X_  = []
        self.y_  = []
        self.Xy = np.zeros((1, self.cdim))
        # fitting configuration
        # self.fit_interval = 100
        self.fitted =  False

        # number of mixture components
        # self.K = K
        # list of K component idim x 1    centroid vectors
        # self.cen_lst = []
        self.cen_lst = [] # np.random.uniform(-1, 1, (self.K,)).tolist()
        # list of K component idim x idim covariances
        self.cov_lst = [] # [np.eye(self.cdim) * 0.1 for _ in range(self.K)]
        # K mixture coeffs
        # self.p_k = None
        self.p_k = None # [1.0/self.K for _ in range(self.K)]
        # log loss after training
        self.logL = 0
        
        print("%s.__init__, idim = %d, odim = %d" % (self.__class__.__name__, self.idim, self.odim))

    def fit(self, X, y):
        """smpGMM.fit

        Single step fit: X, y are single patterns
        """
        # print("%s.fit" % (self.__class__.__name__), X.shape, y.shape)
        if X.shape[0] == 1:
            # single step update, add to internal data and refit if length matches update intervale
            self.Xy_.append(np.hstack((X[0], y[0])))
            self.X_.append(X[0])
            self.y_.append(y[0])
            if len(self.Xy_) % self.fit_interval == 0:
                # print("len(Xy_)", len(self.Xy_), self.Xy_[99])
                # pl.plot(self.Xy_)
                # pl.show()
                # self.fit_batch(self.Xy)
                self.fit_batch(self.X_, self.y_)
        else:
            # batch fit, just fit model to the input data batch
            self.Xy_ += np.hstack((X, y)).tolist()
            # self.X_  += X.tolist()
            # self.y_  += y.tolist()
            # self.Xy = np.hstack((X, y))
            # self.Xy  = np.asarray(self.Xy_)
            # print("X_, y_", self.X_, self.y_)
            self.fit_batch(X, y)
        
    def fit_batch(self, X, y):
        """smpGMM.fit_batch

        Fit the GMM model with batch data
        """
        # print("%s.fit X.shape = %s, y.shape = %s" % (self.__class__.__name__, X.shape, y.shape))
        # self.Xy = np.hstack((X[:,3:], y[:,:]))
        # self.Xy = np.hstack((X, y))
        # self.Xy = np.asarray(self.Xy_)
        # self.Xy = Xy
        # X = np.asarray(X_)
        # y = np.asarray(y_)
        self.Xy = np.hstack((X, y))
        # self.Xy  = np.asarray(self.Xy_)
        print("%s.fit_batch self.Xy.shape = %s" % (self.__class__.__name__, self.Xy.shape))
        # fit gmm
        # max_iter = 10
        try:
            self.cen_lst, self.cov_lst, self.p_k, self.logL = gmm.em_gm(
                self.Xy, K = self.K, max_iter = self.em_max_iter,
                verbose = False, iter_call = None)
            self.fitted =  True
        except Exception as e:
            print( "%s.fit_batch fit failed with %s" % (self.__class__.__name__,  e.args ,))
            # sys.exit()
        print("%s.fit_batch Log likelihood (how well the data fits the model) = %f" % (self.__class__.__name__, self.logL))

    def predict(self, X, rollback = False):
        """smpGMM.predict

        Predict Y from X by forwarding to default sample call
        """
        return self.sample(X, rollback = rollback)

    def sample(self, X, rollback = False):
        """smpGMM.sample

        Default sample function

        Assumes the input is X with dims = idim located in
        the first part of the conditional inference combined input vector

        This method constructs the corresponding conditioning input from the reduced input
        """
        print("%s.sample: X.shape = %s, idim = %d" % (self.__class__.__name__, X.shape, self.idim))
        assert X.shape[1] == self.idim

        # cond = np.zeros((, self.cdim))
        uncond    = np.empty((X.shape[0], self.odim))
        uncond[:] = np.nan
        # print("%s.sample: uncond.shape = %s" % (self.__class__.__name__, uncond.shape))
        # np.array([np.nan for i in range(self.odim)])
        cond = np.hstack((X, uncond))
        # cond[:self.idim] = X.copy()
        # cond[self.idim:] = np.nan
        # print("%s.sample: cond.shape = %s" % (self.__class__.__name__, cond.shape))
        if X.shape[0] > 1: # batch
            return self.sample_batch(cond)
        return self.sample_cond(cond)
    
    def sample_cond(self, X):
        """smpGMM.sample_cond

        Single sample from the GMM model with conditioning on single input pattern X

        TODO: function conditional_dist, make predict/sample comply with sklearn and use the lowlevel
              cond_dist for advanced uses like dynamic conditioning
        """
    
        # gmm.cond_dist want's a (n, ) shape, not (1, n)
        if len(X.shape) > 1:
            cond = X[0]
        else:
            cond = X

        # print("%s.sample_cond: cond.shape = %s" % (self.__class__.__name__, cond.shape))
        if not self.fitted:
            # return np.zeros((3,1))
            # model has not been bootstrapped, return random goal
            cond_sample = np.random.uniform(-1.0, 1.0, (1, self.odim)) # FIXME hardcoded shape
            # cen_con = self.cen_lst
            # cov_con = self.cov_lst
            # new_p_k = self.p_k
        else:
            (cen_con, cov_con, new_p_k) = gmm.cond_dist(cond, self.cen_lst, self.cov_lst, self.p_k)
            # print( "cen_con", cen_con, "cov_con", cov_con, "p_k", new_p_k)
            cond_sample = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
        # print("%s.sample_cond: cond_sample.shape = %s" % (self.__class__.__name__, cond_sample.shape))
        return cond_sample

    def sample_batch(self, X):
        """smpGMM.sample_batch

        If X has more than one rows, return batch of samples for
        every condition row in X
        """
        samples = np.zeros((X.shape[0], self.odim))
        for i in range(X.shape[0]):
            samples[i] = self.sample_cond(X[i])
        return samples
    
    # def sample_batch_legacy(self, X, cond_dims = [0], out_dims = [1], resample_interval = 1):
    #     """smpGMM.sample_batch_legacy

    #     Sample from gmm model with conditioning batch input X legacy function
    #     """
    #     # compute conditional
    #     sampmax = 20
    #     numsamplesteps = X.shape[0]
    #     odim = len(out_dims) # self.idim - X.shape[1]
    #     self.y_sample_  = np.zeros((odim,))
    #     self.y_sample   = np.zeros((odim,))
    #     self.y_samples_ = np.zeros((sampmax, numsamplesteps, odim))
    #     self.y_samples  = np.zeros((numsamplesteps, odim))
    #     self.cond       = np.zeros_like(X[0])

    #     print("%s.sample_batch: y_samples_.shape = %s" % (self.__class__.__name__, self.y_samples_.shape))
        
    #     for i in range(numsamplesteps):
    #         # if i % 100 == 0:
    #         if i % resample_interval == 0:
    #             # print("%s.sample_batch: sampling gmm cond prob at step %d" % (self.__class__.__name__, i))
    #             ref_interval = 1
    #             # self.cond = self.logs["EP"][(i+ref_interval) % self.logs["EP"].shape[0]] # self.X__[i,:3]
    #             self.cond = X[(i+ref_interval) % numsamplesteps] # self.X__[i,:3]
    #             # self.cond = np.array()
    #             # self.cond[:2] = X_
    #             # print(self.cond, out_dims, X.shape)
    #             self.cond[out_dims] = np.nan
    #             (self.cen_con, self.cov_con, self.new_p_k) = gmm.cond_dist(self.cond, self.cen_lst, self.cov_lst, self.p_k)
    #             # print "run_hook_e2p_sample gmm.cond_dist:", np.array(self.cen_con).shape, np.array(self.cov_con).shape, self.new_p_k.shape
    #             samperr = 1e6
    #             j = 0
    #             while samperr > 0.1 and j < sampmax:
    #                 self.y_sample = gmm.sample_gaussian_mixture(self.cen_con, self.cov_con, self.new_p_k, samples = 1)
    #                 self.y_samples_[j,i] = self.y_sample
    #                 samperr_ = np.linalg.norm(self.y_sample - X[(i+1) % numsamplesteps,:odim], 2)
    #                 if samperr_ < samperr:
    #                     samperr = samperr_
    #                     self.y_sample_ = self.y_sample
    #                 j += 1
    #                 # print "sample/real err", samperr
    #             print("sampled", j, "times")
    #         else:
    #             # retain samples from last sampling interval boundary
    #             self.y_samples_[:,i] = self.y_samples_[:,i-1]
    #         # return sample array
    #         self.y_samples[i] = self.y_sample_
            
    #     return self.y_samples, self.y_samples_

# IGMM - incremental gaussian mixture model, from juan
class smpIGMM(smpModel):
    """smpIGMM

    Gaussian mixture model based on PyPR's gmm
    """
    defaults = {'idim': 1, 'odim': 1, 'K': 10, 'numepisodes': 10, 'visualize': False}
    
    @smpModelInit()
    def __init__(self, conf):
        """smpIGMM.__init__
        """
        smpModel.__init__(self, conf)

        self.cdim = self.idim + self.odim

        # number of mixture components
        # self.K = K
        
        # list of K component idim x 1    centroid vectors
        self.cen_lst = []
        # list of K component idim x idim covariances
        self.cov_lst = []
        # K mixture coeffs
        self.p_k = None

        self.cen_lst = np.random.uniform(-1, 1, (self.K,)).tolist()
        # list of K component idim x idim covariances
        self.cov_lst = [np.eye(self.cdim) * 0.1 for _ in range(self.K)]
        # K mixture coeffs
        # self.p_k = None
        self.p_k = [1.0/self.K for _ in range(self.K)]
        
        # log loss after training
        self.logL = 0

        # data
        self.Xy_ = []
        self.X_  = []
        self.y_  = []
        self.Xy = np.zeros((1, self.cdim))
        # fitting configuration
        self.fit_interval = 100
        self.fitted =  False

        self.model = IGMM_COND(min_components=3, forgetting_factor=0.5)
        
        # print("%s.__init__, idim = %d, odim = %d" % (self.__class__.__name__, self.idim, self.odim))

    def fit(self, X, y):
        """smpIGMM.fit

        Single step fit: X, y are single patterns
        """
        # print("%s.fit" % (self.__class__.__name__), X.shape, y.shape)
        if X.shape[0] == 1:
            # single step update, add to internal data and refit if length matches update intervale
            self.Xy_.append(np.hstack((X[0], y[0])))
            self.X_.append(X[0])
            self.y_.append(y[0])
            if len(self.Xy_) % self.fit_interval == 0:
                # print("len(Xy_)", len(self.Xy_), self.Xy_[99])
                # pl.plot(self.Xy_)
                # pl.show()
                # self.fit_batch(self.Xy)
                self.fit_batch(self.X_, self.y_)
                self.Xy_ = []
                self.X_ = []
                self.y_ = []
        else:
            # batch fit, just fit model to the input data batch
            self.Xy_ += np.hstack((X, y)).tolist()
            # self.X_  += X.tolist()
            # self.y_  += y.tolist()
            # self.Xy = np.hstack((X, y))
            # self.Xy  = np.asarray(self.Xy_)
            # print("X_, y_", self.X_, self.y_)
            self.fit_batch(X, y)
        
    def fit_batch(self, X, y):
        """smpIGMM.fit_batch

        Fit the IGMM model with batch data
        """
        # print("%s.fit X.shape = %s, y.shape = %s" % (self.__class__.__name__, X.shape, y.shape))
        # self.Xy = np.hstack((X[:,3:], y[:,:]))
        # self.Xy = np.hstack((X, y))
        # self.Xy = np.asarray(self.Xy_)
        # self.Xy = Xy
        # X = np.asarray(X_)
        # y = np.asarray(y_)
        self.Xy = np.hstack((X, y))
        # self.Xy  = np.asarray(self.Xy_)
        print("%s.fit_batch self.Xy.shape = %s" % (self.__class__.__name__, self.Xy.shape))
        # fit gmm
        # self.cen_lst, self.cov_lst, self.p_k, self.logL = gmm.em_gm(self.Xy, K = self.K, max_iter = 1000,
        #                                                             verbose = False, iter_call = None)
        self.model.train(self.Xy)
        self.fitted =  True
        # print("%s.fit_batch Log likelihood (how well the data fits the model) = %f" % (self.__class__.__name__, self.logL))

    def predict(self, X):
        """smpIGMM.predict

        Predict Y from X by forwarding to default sample call
        """
        # print("IGMM.predict X.shape", X.shape, X)
        return self.sample(X)
    
    def sample(self, X):
        """smpIGMM.sample

        Default sample function

        Assumes the input is X with dims = idim located in
        the first part of the conditional inference combined input vector

        This method constructs the corresponding conditioning input from the reduced input
        """
        # print("%s.sample: X.shape = %s, idim = %d" % (self.__class__.__name__, X.shape, self.idim))
        assert X.shape[1] == self.idim

        # cond = np.zeros((, self.cdim))
        uncond    = np.empty((X.shape[0], self.odim))
        uncond[:] = np.nan
        # print("%s.sample: uncond.shape = %s, %s" % (self.__class__.__name__, uncond.shape, uncond))
        
        cond = np.hstack((X, uncond))
        # cond[:self.idim] = X.copy()
        # cond[self.idim:] = np.nan
        # print("%s.sample: cond.shape = %s, %s" % (self.__class__.__name__, cond.shape, cond))
        
        if X.shape[0] > 1: # batch
            return self.sample_batch(cond)
        sample = self.sample_cond(cond)
        # print("%s.sample sample = %s, X = %s" % (self.__class__.__name__, sample.shape, X.shape))
        # FIXME: fix that inference configuration
        if sample.shape[1] == self.odim:
            return sample
        else:
            return sample[...,X.shape[1]:]
    
    def sample_cond(self, X):
        """smpIGMM.sample_cond

        Single sample from the IGMM model with conditioning on single input pattern X

        TODO: function conditional_dist, make predict/sample comply with sklearn and use the lowlevel
              cond_dist for advanced uses like dynamic conditioning
        """
        if not self.fitted:
            # return np.zeros((3,1))
            # model has not been bootstrapped, return random prediction
            return np.random.uniform(-0.1, 0.1, (1, self.odim)) # FIXME hardcoded shape
    
        # gmm.cond_dist want's a (n, ) shape, not (1, n)
        if len(X.shape) > 1:
            cond = X[0]
        else:
            cond = X

        # print("%s.sample_cond: cond.shape = %s" % (self.__class__.__name__, cond.shape))
        # (cen_con, cov_con, new_p_k) = gmm.cond_dist(cond, self.cen_lst, self.cov_lst, self.p_k)
        # cond_sample = gmm.sample_gaussian_mixture(cen_con, cov_con, new_p_k, samples = 1)
        cond_sample = self.model.sample_cond_dist(cond, 1)
        # print("%s.sample_cond: cond_sample.shape = %s, %s" % (self.__class__.__name__, cond_sample.shape, cond_sample))
        return cond_sample

    def sample_batch(self, X):
        """smpIGMM.sample_batch

        If X has more than one rows, return batch of samples for
        every condition row in X
        """
        samples = np.zeros((X.shape[0], self.odim))
        for i in range(X.shape[0]):
            samples[i] = self.sample_cond(X[i])
        return samples
    
################################################################################
# Hebbian SOM model: connect to SOMs with hebbian links
class smpHebbianSOM(smpModel):
    """smpHebbianSOM class

    Hebbian SOM model

    FIXME: conf: kohonen/map.Map init distribution and scaling
    FIXME: conf: fit_hebb onset delay
    FIXME: conf: sampling mode (weights, gaussian(wgts, sigmas), ...
    """
    defaults = {
        'idim': 1, 'odim': 1, 'numepisodes': 100, 'visualize': False, 'mapsize_e': 10, 'mapsize_p': 10, 'som_lr': 1e-0,
        'som_nhs': 3, 'init_range': (-1.0, 1.0)}
    @smpModelInit()
    def __init__(self, conf):
        """smpHebbianSOM

        Two SOM's coding the input and output space connected by associative Hebbian links
        """
        smpModel.__init__(self, conf)

        # SOMs training self assessment
        self.cnt_fit     = 0
        self.cnt_predict = 0
        self.fitted = False
        self.soms_cnt_fit     = 0
        self.soms_cnt_predict = 0
        self.soms_fitted = False
        self.hebb_cnt_fit     = 0
        self.hebb_cnt_predict = 0
        self.hebb_fitted = False
        self.decay_const = -1e-5
        
        # learning rate proxy
        self.ET = ExponentialTimeseries
        self.CT = ConstantTimeseries
        
        self.mapsize = 10 ** 2 # 100
        # self.mapsize_e = mapsize_e # 100 # int(np.sqrt(self.mapsize)) # max(10, self.idim * 3)
        # self.mapsize_p = mapsize_p # 150 # int(np.sqrt(self.mapsize)) # max(10, self.odim * 3)
        self.numepisodes_som  = self.numepisodes
        self.numepisodes_hebb = self.numepisodes
        # FIXME: make neighborhood_size decrease with time

        # som_lr = som_lr # 1e0
        # som_lr = 1e-1 # Haykin, p475
        # som_lr = 5e-1
        # som_lr = 5e-4
        # self.som_nhs = 3 # 1.5

        maptype = "som"
        # maptype = "gas"
                
        # SOM exteroceptive stimuli 2D input
        if maptype == "som":
            if self.idim == 1:
                mapshape_e = (self.mapsize_e, )
            else:
                mapshape_e = (self.mapsize_e, self.mapsize_e)
            # 1D better?
            # mapshape_e = (self.mapsize_e, )
            self.kw_e = self.kwargs(
                shape = mapshape_e, dimension = self.idim, lr_init = self.som_lr,
                neighborhood_size = self.som_nhs, init_variance = 1.0) #, z = 0.001)
            # self.kw_e = self.kwargs(shape = (self.mapsize_e, self.mapsize_e), dimension = self.idim, lr_init = 0.5, neighborhood_size = 0.6)
            self.som_e = Map(Parameters(**self.kw_e))
        elif maptype == "gas":
            self.kw_e = self.kwargs_gas(shape = (self.mapsize_e ** 2, ), dimension = self.idim, lr_init = self.som_lr, neighborhood_size = 0.5)
            self.som_e = Gas(Parameters(**self.kw_e))

        # SOM proprioceptive stimuli 3D input
        if maptype == "som":
            if self.idim == 1:
                mapshape_p = (self.mapsize_p, )
            else:
                mapshape_p = (int(self.mapsize_p), int(self.mapsize_p))
            # 1D better?
            mapshape_p = (self.mapsize_p, )
            self.kw_p = self.kwargs(shape = mapshape_p, dimension = self.odim, lr_init = self.som_lr,
                                    neighborhood_size = self.som_nhs, init_variance = 0.2) #, z = 0.001)
            # self.kw_p = self.kwargs(shape = (int(self.mapsize_p * 1.5), int(self.mapsize_p * 1.5)), dimension = self.odim, lr_init = 0.5, neighborhood_size = 0.7)
            self.som_p = Map(Parameters(**self.kw_p))
        elif maptype == "gas":
            self.kw_p = self.kwargs_gas(shape = (self.mapsize_p ** 2, ), dimension = self.odim, lr_init = self.som_lr, neighborhood_size = 0.5)
            self.som_p = Gas(Parameters(**self.kw_p))

        print("HebbianSOM mapsize_e,p", self.mapsize_e, self.mapsize_p)
            
        # FIXME: there was a nice trick for node distribution init in _some_ recently added paper

        # create "filter" using existing SOM_e, filter computes activation on distance
        self.filter_e = Filter(self.som_e, history=lambda: 0.0)
        # print("neurons_e", self.filter_e.map.neurons)
        self.filter_e.reset()
        # print("neurons_e", self.filter_e.map.neurons)
        self.filter_e_lr = self.filter_e.map._learning_rate

        # kw_f_p = kwargs(shape = (mapsize * 3, mapsize * 3), dimension = 3, neighborhood_size = 0.5, lr_init = 0.1)
        # filter_p = Filter(Map(Parameters(**kw_f_p)), history=lambda: 0.01)
        
        # create "filter" using existing SOM_p, filter computes activation on distance
        self.filter_p = Filter(self.som_p, history=lambda: 0.0)
        self.filter_p.reset()
        self.filter_p_lr = self.filter_p.map._learning_rate

        # Hebbian links
        # hebblink_som    = np.random.uniform(-1e-4, 1e-4, (np.prod(som_e._shape), np.prod(som_p._shape)))
        # hebblink_filter = np.random.uniform(-1e-4, 1e-4, (np.prod(filter_e.map._shape), np.prod(filter_p.map._shape)))
        self.hebblink_som    = np.zeros((np.prod(self.som_e._shape), np.prod(self.som_p._shape)))
        # self.hebblink_filter = np.zeros((np.prod(self.filter_e.map._shape), np.prod(self.filter_p.map._shape)))
        self.hebblink_filter = np.random.normal(0, 1e-6, (np.prod(self.filter_e.map._shape), np.prod(self.filter_p.map._shape)))

        # # sparse hebblink
        # self.hebblink_filter = sparse.rand(m = np.prod(self.filter_e.map._shape),
        #                                    n = np.prod(self.filter_p.map._shape)) * 1e-3
        
        self.hebblink_use_activity = True # use activation or distance
        
        # Hebbian learning rate
        if self.hebblink_use_activity:
            # self.hebblink_et = ExponentialTimeseries(self.decay_const, 1e-0, 0)
            self.hebblink_et = ConstantTimeseries(1e-0)
            # self.hebblink_et = ConstantTimeseries(0.0)
        else:
            self.hebblink_et = ConstantTimeseries(1e-12)

        # visualization
        if self.visualize:
            self.figs.append(plot_nodes_over_data_1d_components_fig(title = self.__class__.__name__, numplots = self.idim + self.odim))
            
    # SOM argument dict
    def kwargs(self, shape=(10, 10), z=0.001, dimension=2, lr_init = 1.0, neighborhood_size = 1, init_variance = 1.0):
        """smpHebbianSOM params function for Map"""
        return dict(
            dimension = dimension,
            shape = shape,
            neighborhood_size = self.ET(self.decay_const, neighborhood_size, 0.1), # 1.0),
            learning_rate=self.ET(self.decay_const, lr_init, 0.0),
            # learning_rate=self.CT(lr_init),
            noise_variance=z,
            init_variance = init_variance)

    def kwargs_gas(self, shape=(100,), z=0.001, dimension=3, lr_init = 1.0, neighborhood_size = 1):
        """smpHebbianSOM params function for Gas"""
        return dict(
            dimension=dimension,
            shape=shape,
            neighborhood_size = self.ET(self.decay_const, neighborhood_size, 1.0),
            learning_rate=self.ET(self.decay_const, lr_init, 0.0),
            noise_variance=z)

    def visualize_model(self):
        """smpHebbianSOM.visualize_model

        Plot the model state visualization
        """
        e_nodes, p_nodes = hebbsom_get_map_nodes(self, self.idim, self.odim)
        e_nodes_cov = np.tile(np.eye(self.idim) * 0.05, e_nodes.shape[0]).T.reshape((e_nodes.shape[0], self.idim, self.idim))
        p_nodes_cov = np.tile(np.eye(self.odim) * 0.05, p_nodes.shape[0]).T.reshape((p_nodes.shape[0], self.odim, self.odim))

        X = np.vstack(self.Xhist)
        Y = np.vstack(self.Yhist)
        
        # print(X.shape)
        
        plot_nodes_over_data_1d_components(
            fig = self.figs[0], X = X, Y = Y, mdl = self,
            e_nodes = e_nodes, p_nodes = p_nodes, e_nodes_cov = e_nodes_cov, p_nodes_cov = p_nodes_cov,
            saveplot = False
        )
    
    def set_learning_rate_constant(self, c = 0.0):
        # print("fit_hebb", self.filter_e.map._learning_rate)
        self.filter_e.map._learning_rate = self.CT(c)
        self.filter_p.map._learning_rate = self.CT(c)
        # fix the SOMs with learning rate constant 0
        self.filter_e_lr = self.filter_e.map._learning_rate
        self.filter_p_lr = self.filter_p.map._learning_rate

    def fit_soms(self, X, y):
        """smpHebbianSOM"""
        # print("%s.fit_soms fitting X = %s, y = %s" % (self.__class__.__name__, X.shape, y.shape))
        # if X.shape[0] != 1, r
        # e = EP[i,:dim_e]
        # p = EP[i,dim_e:]
        
        self.filter_e.map._learning_rate = self.filter_e_lr
        self.filter_p.map._learning_rate = self.filter_p_lr

        # don't learn twice
        # som_e.learn(e)
        # som_p.learn(p)
        # TODO for j in numepisodes
        if X.shape[0] > 1:
            numepisodes = self.numepisodes_som
        else:
            numepisodes = 1
        if X.shape[0] > 100:
            print("%s.fit_soms batch fitting of size %d" % (self.__class__.__name__, X.shape[0]))
        i = 0
        j = 0
        eps_convergence = 0.01
        # eps_convergence = 0.005
        dWnorm_e_ = 1  # short horizon
        dWnorm_p_ = 1
        dWnorm_e__ = dWnorm_e_ + 2 * eps_convergence # long horizon
        dWnorm_p__ = dWnorm_p_ + 2 * eps_convergence

        idx_shuffle = np.arange(X.shape[0])
                
        # for j in range(numepisodes):
        # (dWnorm_e_ == 0 and dWnorm_p_ == 0) or 
        # while (dWnorm_e_ > 0.05 and dWnorm_p_ > 0.05):
        do_convergence = True
        while (do_convergence) and (np.abs(dWnorm_e__ - dWnorm_e_) > eps_convergence and np.abs(dWnorm_p__ - dWnorm_p_) > eps_convergence): # and j < 10:
            if j > 0 and j % 10 == 0:
                print("%s.fit_soms episode %d / %d" % (self.__class__.__name__, j, numepisodes))
            if X.shape[0] == 1:
                # print("no convergence")
                do_convergence = False
            dWnorm_e = 0
            dWnorm_p = 0
            
            np.random.shuffle(idx_shuffle)
            # print("neurons_e 1", self.filter_e.map.neurons.flatten())
            for i in range(X.shape[0]):
                # lidx = idx_shuffle[i]
                lidx = i
                self.filter_e.learn(X[lidx])
                dWnorm_e += np.linalg.norm(self.filter_e.map.delta)
                self.filter_p.learn(y[lidx])
                dWnorm_p += np.linalg.norm(self.filter_p.map.delta)
            # print("neurons_e 2", self.filter_e.map.neurons.flatten(), X, X[lidx])
            dWnorm_e /= X.shape[0]
            dWnorm_e /= self.filter_e.map.numunits
            dWnorm_p /= X.shape[0]
            dWnorm_p /= self.filter_p.map.numunits
            # short
            dWnorm_e_ = 0.8 * dWnorm_e_ + 0.2 * dWnorm_e
            dWnorm_p_ = 0.8 * dWnorm_p_ + 0.2 * dWnorm_p
            # long
            dWnorm_e__ = 0.83 * dWnorm_e__ + 0.17 * dWnorm_e_
            dWnorm_p__ = 0.83 * dWnorm_p__ + 0.17 * dWnorm_p_
            # print("%s.fit_soms batch e |dW| = %f, %f, %f" % (self.__class__.__name__, dWnorm_e, dWnorm_e_, dWnorm_e__))
            # print("%s.fit_soms batch p |dW| = %f, %f, %f" % (self.__class__.__name__, dWnorm_p, dWnorm_p_, dWnorm_p__))
            j += 1

        if True and self.soms_cnt_fit % 100 == 0:
            print("%s.fit_soms batch e mean error = %f, min = %f, max = %f" % (
                self.__class__.__name__,
                np.asarray(self.filter_e.distances_).mean(),
                np.asarray(self.filter_e.distances_[-1]).min(),
                np.asarray(self.filter_e.distances_).max() ))
            print("%s.fit_soms batch p mean error = %f, min = %f, max = %f" % (
                self.__class__.__name__,
                np.asarray(self.filter_p.distances_).mean(),
                np.asarray(self.filter_p.distances_[-1]).min(),
                np.asarray(self.filter_p.distances_).max() ))
        # print np.argmin(som_e.distances(e)) # , som_e.distances(e)
        
        self.soms_cnt_fit += 1

    def fit_hebb(self, X, y):
        """smpHebbianSOM"""
        # print("%s.fit_hebb fitting X = %s, y = %s" % (self.__class__.__name__, X.shape, y.shape))
        if X.shape[0] == 1 and self.soms_cnt_fit < 200: # 200: # 1500:
            return
        # numepisodes_hebb = 1
        if X.shape[0] > 100:
            print("%s.fit_hebb batch fitting of size %d" % (self.__class__.__name__, X.shape[0]))
        numsteps = X.shape[0]
        ################################################################################
        # fix the SOMs with learning rate constant 0
        self.filter_e_lr = self.filter_e.map._learning_rate
        self.filter_p_lr = self.filter_p.map._learning_rate
        # print("fit_hebb", self.filter_e.map._learning_rate)
        self.filter_e.map._learning_rate = self.CT(0.0)
        self.filter_p.map._learning_rate = self.CT(0.0)

        e_shape = (np.prod(self.filter_e.map._shape), 1)
        p_shape = (np.prod(self.filter_p.map._shape), 1)

        eps_convergence = 0.05
        z_err_coef_1 = 0.8
        z_err_coef_2 = 0.83
        z_err_norm_ = 1 # fast
        z_err_norm__ = z_err_norm_ + 2 * eps_convergence # slow
        Z_err_norm  = np.zeros((self.numepisodes_hebb*numsteps,1))
        Z_err_norm_ = np.zeros((self.numepisodes_hebb*numsteps,1))
        W_norm      = np.zeros((self.numepisodes_hebb*numsteps,1))

        # # plotting
        # pl.ion()
        # fig = pl.figure()
        # fig2 = pl.figure()
                    
        # TODO for j in numepisodes
        # j = 0
        if X.shape[0] > 1:
            numepisodes = self.numepisodes_hebb
        else:
            numepisodes = 1
        i = 0
        dWnorm_ = 10.0
        j = 0
        # for j in range(numepisodes):
        do_convergence = True
        while do_convergence and z_err_norm_ > eps_convergence and np.abs(z_err_norm__ - z_err_norm_) > eps_convergence: #  and j < 20:
            if j > 0 and j % 10 == 0:
                print("%s.fit_hebb episode %d / %d" % (self.__class__.__name__, j, numepisodes))
            if X.shape[0] == 1:
                # print("no convergence")
                do_convergence = False
            for i in range(X.shape[0]):
                # just activate
                self.filter_e.learn(X[i])
                self.filter_p.learn(y[i])
        
                # fetch data induced activity
                if self.hebblink_use_activity:
                    p_    = self.filter_p.activity.reshape(p_shape)
                    # print(p_.shape)
                else:
                    p_    = self.filter_p.distances(p).flatten().reshape(p_shape)
                p__ = p_.copy()
                # p_ = p_ ** 2
                p_ = (p_ == np.max(p_)) * 1.0
        
                e_ = self.filter_e.activity.reshape(e_shape) # flatten()
                e__ = e_.copy()
                # e_ = e_ ** 2
                e_ = (e_ == np.max(e_)) * 1.0
                
                # compute prediction for p using e activation and hebbian weights
                if self.hebblink_use_activity:
                    # print(self.hebblink_filter.T.shape, self.filter_e.activity.reshape(e_shape).shape)
                    # p_bar = np.dot(self.hebblink_filter.T, self.filter_e.activity.reshape(e_shape))
                    # e_act = e_.reshape(e_shape)
                    # e_act
                    p_bar = np.dot(self.hebblink_filter.T, e_.reshape(e_shape))
                    # # sparse
                    # p_bar = self.hebblink_filter.T.dot(e_.reshape(e_shape))
                    # print("p_bar", type(p_bar))
                else:
                    p_bar = np.dot(self.hebblink_filter.T, self.filter_e.distances(e).flatten().reshape(e_shape))
                p_bar_ = p_bar.copy()
                p_bar = (p_bar == np.max(p_bar)) * 1.0
                    
                # print("p_bar", type(p_bar), type(p_bar_))

                # # plotting
                # ax1 = fig.add_subplot(411)
                # ax1.cla()
                # ax1.plot(e_ * np.max(e__))
                # ax1.plot(e__)
                # ax2 = fig.add_subplot(412)
                # ax2.cla()
                # ax2.plot(p_ * np.max(p_bar_))
                # ax2.plot(p__)
                # ax2.plot(p_bar * np.max(p_bar_))
                # ax2.plot(p_bar_)
                # ax3 = fig.add_subplot(413)
                # ax3.cla()
                # ax3.plot(self.filter_e.distances_[-1])
                # ax4 = fig.add_subplot(414)
                # ax4.cla()
                # ax4.plot(self.filter_p.distances_[-1])
                # pl.pause(0.001)
                # pl.draw()
                    
                # inject activity prediction
                p_bar_sum = p_bar.sum()
                if p_bar_sum > 0:
                    p_bar_normed = p_bar / p_bar_sum
                else:
                    p_bar_normed = np.zeros(p_bar.shape)
            
                # compute prediction error: data induced activity - prediction
                # print("p_", np.linalg.norm(p_))
                # print("p_bar", np.linalg.norm(p_bar))
                z_err = p_ - p_bar
                idx = np.argmax(p_bar_)
                # print("sum E", np.sum(z_err))
                # print("idx", p_bar_, idx, z_err[idx])
                # z_err = (p_[idx] - p_bar[idx]) * np.ones_like(p_)
                # z_err = np.ones_like(p_) * 
                # print("z_err", z_err)
                # z_err = p_bar - p_
                # z_err_norm = np.linalg.norm(z_err, 2)
                z_err_norm = np.sum(np.abs(z_err))
                # if j == 0 and i == 0:
                #     z_err_norm_ = z_err_norm
                # else:
                z_err_norm_  = z_err_coef_1 * z_err_norm_  + (1 - z_err_coef_1) * z_err_norm
                z_err_norm__ = z_err_coef_2 * z_err_norm__ + (1 - z_err_coef_2) * z_err_norm
        
                w_norm = np.linalg.norm(self.hebblink_filter)

                
                # logidx = (j*numsteps) + i
                # Z_err_norm [logidx] = z_err_norm
                # Z_err_norm_[logidx] = z_err_norm_
                # W_norm     [logidx] = w_norm
            
                # z_err = p_bar - self.filter_p.activity.reshape(p_bar.shape)
                # print "p_bar.shape", p_bar.shape
                # print "self.filter_p.activity.flatten().shape", self.filter_p.activity.flatten().shape
                
                # if i % 100 == 0:
                #     print("%s.fit_hebb: iter %d/%d: z_err.shape = %s, |z_err| = %f, |W| = %f, |p_bar_normed| = %f" % (self.__class__.__name__, logidx, (self.numepisodes_hebb*numsteps), z_err.shape, z_err_norm_, w_norm, np.linalg.norm(p_bar_normed)))
            
                # d_hebblink_filter = et() * np.outer(self.filter_e.activity.flatten(), self.filter_p.activity.flatten())
                eta = self.hebblink_et()
                if eta > 0.0:
                    if False and self.hebblink_use_activity:
                        # eta = 5e-4
                        # outer = np.outer(self.filter_e.activity.flatten(), np.clip(z_err, 0, 1))
                        # outer = np.outer(e_, np.clip(z_err, 0, 1))
                        # outer = np.outer(e_, p_)
                        # outer = np.outer(e_, p__ * np.clip(z_err, 0, 1))
                        # FIXME: this can be optimized with sparsity
                        # print("e_", e_, e__, p_)
                        outer = np.outer(e_ * e__, p_)
                        
                        # print(outer.shape, self.hebblink_filter.shape)
                        # print("outer", outer)
                        # print("modulator", z_err[idx])
                        # d_hebblink_filter = eta * outer * (-1e-3 - z_err[idx])
                        # d_hebblink_filter = eta * np.outer(z_err, self.filter_e.activity.flatten()).T
                        # d_hebblink_filter = eta * outer * np.abs((z_err_norm_ - z_err_norm))
                        # d_hebblink_filter = eta * outer * (z_err_norm - z_err_norm_)
                        d_hebblink_filter = eta * outer
                    
                        # # plotting
                        # f2ax1 = fig2.add_subplot(111)
                        # f2ax1.imshow(self.hebblink_filter.T, interpolation="none")
                        # # im = f2ax1.imshow(outer, interpolation="none")
                        # # f2ax2 = pl.colorbar(im, ax=f2ax1)
                        # pl.pause(1e-5)
                        # pl.draw()
                    elif self.hebblink_use_activity:
                        e_idx = np.argmax(e_)
                        p_idx = np.argmax(p_)
                        # print("e_", e_idx, "p_", p_idx)
                        d_hebblink_filter = np.zeros_like(self.hebblink_filter)
                    else:
                        d_hebblink_filter = eta * np.outer(self.filter_e.distances(e), z_err)

                    # does what?
                    self.hebblink_filter[e_idx, p_idx] += eta * e__[e_idx]
                    
                    dWnorm = np.linalg.norm(d_hebblink_filter)
                    dWnorm_ = 0.8 * dWnorm_ + 0.2 * dWnorm
                    # print ("dWnorm", dWnorm)
                    # self.hebblink_filter += d_hebblink_filter
                
            # print("hebblink_filter type", type(self.hebblink_filter))
            # print("np.linalg.norm(self.hebblink_filter, 2)", np.linalg.norm(self.hebblink_filter, 2))
            
            self.hebblink_filter /= np.linalg.norm(self.hebblink_filter, 2)
            
            j += 1

        if False and self.hebb_cnt_fit % 100 == 0:
            # print("hebblink_filter type", type(self.hebblink_filter))
            # print(Z_err_norm)
            # print("%s.fit_hebb error p/p_bar %f" % (self.__class__.__name__, np.array(Z_err_norm)[:logidx].mean()))
            print("%s.fit_hebb |dW| = %f, |W| = %f, mean err = %f / %f" % (self.__class__.__name__, dWnorm_, w_norm, np.min(z_err), np.max(z_err)))
            # z_err_norm_, z_err_norm__))
            # print("%s.fit_hebb |W|  = %f" % (self.__class__.__name__, w_norm))
        self.hebb_cnt_fit += 1
            
    def fit(self, X, y):
        """smpHebbianSOM

        Fit model to data
        """
        # print("%s.fit fitting X = %s, y = %s" % (self.__class__.__name__, X, y))
        # if X,y have more than one row, train do batch training on SOMs and links
        # otherwise do single step update on both or just the latter?        
        self.fit_soms(X, y)
        self.fit_hebb(X, y)
        self.fitted = True

        # if self.visualize:
        #     self.Xhist.append(X)
        #     self.Yhist.append(y)
            
        #     if self.cnt_fit % 100 == 0:
        #         self.visualize_model()
            
        self.cnt_fit += 1
            
    def predict(self, X):
        """smpHebbianSOM"""
        return self.sample(X)

    def sample(self, X):
        """smpHebbianSOM.sample"""
        # print("%s.sample X.shape = %s, %d" % (self.__class__.__name__, X.shape, 0))
        if len(X.shape) == 2 and X.shape[0] > 1: # batch
            return self.sample_batch(X)
        return self.sample_cond(X)

    def sample_cond(self, X):
        """smpHebbianSOM.sample_cond: draw single sample from model conditioned on X"""
        # print("%s.sample_cond X.shape = %s, %d" % (self.__class__.__name__, X.shape, 0))

        # fix the SOMs with learning rate constant 0
        self.filter_e_lr = self.filter_e.map._learning_rate
        self.filter_p_lr = self.filter_p.map._learning_rate
        # print("fit_hebb", self.filter_e.map._learning_rate)
        self.filter_e.map._learning_rate = self.CT(0.0)
        self.filter_p.map._learning_rate = self.CT(0.0)
        
        e_shape = (np.prod(self.filter_e.map._shape), 1)
        p_shape = (np.prod(self.filter_p.map._shape), 1)

        # activate input network
        self.filter_e.learn(X)

        # pl.plot(self.filter_e.
        
        # propagate activation via hebbian associative links
        if self.hebblink_use_activity:
            e_ = self.filter_e.activity.reshape((np.prod(self.filter_e.map._shape), 1))
            e_ = (e_ == np.max(e_)) * 1.0
            e2p_activation = np.dot(self.hebblink_filter.T, e_)
            # print("e2p_activation", e2p_activation)
            self.filter_p.activity = np.clip((e2p_activation / (np.sum(e2p_activation) + 1e-9)).reshape(self.filter_p.map._shape), 0, np.inf)
        else:
            e2p_activation = np.dot(self.hebblink_filter.T, self.filter_e.distances(e).flatten().reshape(e_shape))

        # sample the output network with
        sidxs = self.filter_p.sample(100)
        # print("sidxs", stats.mode(sidxs)[0], sidxs)
        # sidx = self.filter_p.sample(1)[0]
        # find the mode (most frequent realization) of distribution
        sidx = stats.mode(sidxs)[0][0]
        e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(sidx))
        # e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(np.argmax(self.filter_p.activity)))
        
        # ret = np.random.normal(e2p_w_p_weights, self.filter_p.sigmas[sidx], (1, self.odim))
        ret = np.random.normal(e2p_w_p_weights, np.sqrt(self.filter_p.sigmas[sidx]), (1, self.odim))
        # ret = np.random.normal(e2p_w_p_weights, 0.01, (1, self.odim))
        # print("hebbsom sample", sidx, e2p_w_p_weights) # , sidxs) # , self.filter_p.sigmas[sidx])
        # ret = e2p_w_p_weights.reshape((1, self.odim))
        return ret

    def sample_prior(self):
        """smpHebbianSOM.sample_prior

        Sample from input map prior distribution
        """
        # print("pr")
        # pass
        # print("prior", self.filter_e.map.prior)
        # sidxs = argsample(self.filter_e.map.prior, n = 1)
        sidxs = argsample(np.sum(self.filter_e.sigmas, axis = 1), n = 1)
        prior_sample_mu = self.filter_e.neuron(self.filter_e.flat_to_coords(sidxs[0]))
        # print ('prior_sample_mu', prior_sample_mu.shape, self.filter_e.sigmas[sidxs[0]].shape)
        # prior_sample = np.random.normal(prior_sample_mu, self.filter_e.sigmas[sidxs[0]]).reshape((self.idim, 1))
        prior_sample = prior_sample_mu.reshape((self.idim, 1))
        # print("prior_sample", prior_sample)
        return prior_sample
        
    # def sample_cond_legacy(self, X):
    #     """smpHebbianSOM.sample_cond: sample from model conditioned on X"""
    #     sampling_search_num = 100

    #     e_shape = (np.prod(self.filter_e.map._shape), 1)
    #     p_shape = (np.prod(self.filter_p.map._shape), 1)

    #     # P_ = np.zeros((X.shape[0], self.odim))
    #     # E_ = np.zeros((X.shape[0], self.idim))
        
    #     e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(self.filter_p.sample(1)[0]))
    #     for i in range(X.shape[0]):
    #         # e = EP[i,:dim_e]
    #         # p = EP[i,dim_e:]
    #         e = X[i]
    #         # print np.argmin(som_e.distances(e)), som_e.distances(e)
    #         self.filter_e.learn(e)
    #         # print "self.filter_e.winner(e)", self.filter_e.winner(e)
    #         # filter_p.learn(p)
    #         # print "self.filter_e.activity.shape", self.filter_e.activity.shape
    #         # import pdb; pdb.set_trace()
    #         if self.hebblink_use_activity:
    #             e2p_activation = np.dot(self.hebblink_filter.T, self.filter_e.activity.reshape((np.prod(self.filter_e.map._shape), 1)))
    #             self.filter_p.activity = np.clip((e2p_activation / np.sum(e2p_activation)).reshape(self.filter_p.map._shape), 0, np.inf)
    #         else:
    #             e2p_activation = np.dot(self.hebblink_filter.T, self.filter_e.distances(e).flatten().reshape(e_shape))
    #         # print "e2p_activation.shape, np.sum(e2p_activation)", e2p_activation.shape, np.sum(e2p_activation)
    #         # print "self.filter_p.activity.shape", self.filter_p.activity.shape
    #         # print "np.sum(self.filter_p.activity)", np.sum(self.filter_p.activity), (self.filter_p.activity >= 0).all()
        
    #         # self.filter_p.learn(p)
    #         # emodes: 0, 1, 2
    #         emode = 0 #
    #         if i % 1 == 0:
    #             if emode == 0:
    #                 e2p_w_p_weights_ = []
    #                 for k in range(sampling_search_num):
    #                     # filter.sample return the index of the sampled unit
    #                     e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(self.filter_p.sample(1)[0]))
    #                     e2p_w_p_weights_.append(e2p_w_p_weights)
    #                 pred = np.array(e2p_w_p_weights_)
    #                 # print "pred", pred

    #                 # # if we can compare against something
    #                 # pred_err = np.linalg.norm(pred - p, 2, axis=1)
    #                 # # print "np.linalg.norm(e2p_w_p_weights - p, 2)", np.linalg.norm(e2p_w_p_weights - p, 2)
    #                 # e2p_w_p = np.argmin(pred_err)

    #                 # if not pick any
    #                 e2p_w_p = np.random.choice(pred.shape[0])
                    
    #                 # print("pred_err", e2p_w_p, pred_err[e2p_w_p])
    #                 e2p_w_p_weights = e2p_w_p_weights_[e2p_w_p]
    #             elif emode == 1:
    #                 if self.hebblink_use_activity:
    #                     e2p_w_p = np.argmax(e2p_activation)
    #                 else:
    #                     e2p_w_p = np.argmin(e2p_activation)
    #                 e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(e2p_w_p))
                        
    #             elif emode == 2:
    #                 e2p_w_p = self.filter_p.winner(p)
    #                 e2p_w_p_weights = self.filter_p.neuron(self.filter_p.flat_to_coords(e2p_w_p))
    #         # P_[i] = e2p_w_p_weights
    #         # E_[i] = environment.compute_sensori_effect(P_[i])
    #         # print("e2p shape", e2p_w_p_weights.shape)
    #         return e2p_w_p_weights.reshape((1, self.odim))
        
        
    def sample_batch(self, X):
        """smpHebbianSOM.sample_batch: If X has more than one rows, return batch of samples for
        every condition row in X"""
        samples = np.zeros((X.shape[0], self.odim))
        for i in range(X.shape[0]):
            samples[i] = self.sample_cond(X[i])
        return samples
    
    def sample_batch_legacy(self, X, cond_dims = [0], out_dims = [1], resample_interval = 1):
        """smpHebbianSOM"""
        print("%s.sample_batch_legacy data X = %s" % (self.__class__.__name__, X))
        sampmax = 20
        numsamplesteps = X.shape[0]
        odim = len(out_dims) # self.idim - X.shape[1]
        self.y_sample_  = np.zeros((odim,))
        self.y_sample   = np.zeros((odim,))
        self.y_samples_ = np.zeros((sampmax, numsamplesteps, odim))
        self.y_samples  = np.zeros((numsamplesteps, odim))
        self.cond       = np.zeros_like(X[0])
        
        return self.y_samples, self.y_samples_
    
################################################################################
# models_actinf: model testing and plotting code
################################################################################

def hebbsom_get_map_nodes(mdl, idim, odim):
    """hebbsom_get_map_nodes

    Get all the nodes of the coupled SOM maps
    """
    e_nodes = mdl.filter_e.map.neurons
    p_nodes = mdl.filter_p.map.neurons
    # print("e_nodes", e_nodes.shape, "p_nodes", p_nodes.shape)
    e_nodes = e_nodes.reshape((-1,idim))
    p_nodes = p_nodes.reshape((-1,odim))
    # print("e_nodes", e_nodes.shape, "p_nodes", p_nodes.shape)
    return (e_nodes, p_nodes)

def hebbsom_predict_full(X, Y, mdl):
    """hebbsom_predict_full

    Predict using a HebbSOM and return full internal activations as tuple
    - (predictions (samples), distances (SOM distance func), activiations (distances after act. func))
    """
    distances = []
    activities = []
    predictions = np.zeros_like(Y)
    # have to loop over single steps until we generalize predict function to also yield distances and activities
    for h in range(X.shape[0]):
        # X_ = (Y[h]).reshape((1, odim))
        X_ = X[h]
        # print("X_", X_.shape, X_)
        # predict proprio 3D from extero 2D
        predictions[h] = mdl.predict(X_)
        # print("X_.shape = %s, %d" % (X_.shape, 0))
        # print("prediction.shape = %s, %d" % (prediction.shape, 0))
        distances.append(mdl.filter_e.distances(X_).flatten())
        activities.append(mdl.filter_e.activity.flatten())
        activities_sorted = activities[-1].argsort()
        # print("Y[h]", h, Y[h].shape, prediction.shape)
    return (predictions, distances, activities)
    
def plot_nodes_over_data_scattermatrix(X, Y, mdl, e_nodes, p_nodes, e_nodes_cov, p_nodes_cov, saveplot = False):
    """plot_nodes_over_data_scattermatrix

    Plot SOM node locations over input data as scattermatrix all X
    comps over all Y comps.
    """
    
    import pandas as pd
    from pandas.tools.plotting import scatter_matrix

    idim = X.shape[1]
    odim = Y.shape[1]
    numplots = idim + odim

    # e_nodes, p_nodes = hebbsom_get_map_nodes(mdl, idim, odim)
    
    dfcols = []
    dfcols += ["e_%d" % i for i in range(idim)]
    dfcols += ["p_%d" % i for i in range(odim)]

    # X_plus_e_nodes = np.vstack((X, e_nodes))
    # Y_plus_p_nodes = np.vstack((Y, p_nodes))

    # df = pd.DataFrame(np.hstack((X_plus_e_nodes, Y_plus_p_nodes)), columns=dfcols)
    df = pd.DataFrame(np.hstack((X, Y)), columns=dfcols)
    sm = scatter_matrix(df, alpha=0.2, figsize=(5,5), diagonal="hist")
    # print("sm = %s" % (sm))
    # loop over i/o components
    idims = list(range(idim))
    odims = list(range(idim, idim+odim))

        
    for i in range(numplots):
        for j in range(numplots):
            if i != j and i in idims and j in idims:
                # center = np.array()
                # x1, x2 = gmm.gauss_ellipse_2d(centroids[i], ccov[i])
                
                sm[i,j].plot(e_nodes[:,j], e_nodes[:,i], "ro", alpha=0.5, markersize=8)
            if i != j and i in odims and j in odims:
                sm[i,j].plot(p_nodes[:,j-idim], p_nodes[:,i-idim], "ro", alpha=0.5, markersize=8)
            
            # if i != j and i in idims and j in odims:
            #     sm[i,j].plot(p_nodes[:,j-idim], e_nodes[:,i], "go", alpha=0.5, markersize=8)
            # if i != j and i in odims and j in idims:
            #     sm[i,j].plot(e_nodes[:,j], p_nodes[:,i-idim], "go", alpha=0.5, markersize=8)

    # get figure reference from axis and show
    fig = sm[0,0].get_figure()
    fig.suptitle("Predictions over data scattermatrix (%s)" % (mdl.__class__.__name__))
    if saveplot:
        filename = "plot_nodes_over_data_scattermatrix_%s.jpg" % (mdl.__class__.__name__,)
        savefig(fig, filename)
    fig.show()

def plot_nodes_over_data_scattermatrix_hexbin(X, Y, mdl, predictions, distances, activities, saveplot = False):
    """models_actinf.plot_nodes_over_data_scattermatrix_hexbin

    Plot models nodes (if applicable) over the hexbinned data
    expanding dimensions as a scattermatrix.
    """
    
    idim = X.shape[1]
    odim = Y.shape[1]
    numplots = idim * odim + 2
    fig = pl.figure()
    fig.suptitle("Predictions over data xy scattermatrix/hexbin (%s)" % (mdl.__class__.__name__))
    gs = gridspec.GridSpec(idim, odim)
    figaxes = []
    for i in range(idim):
        figaxes.append([])
        for o in range(odim):
            figaxes[i].append(fig.add_subplot(gs[i,o]))
    err = 0

    # colsa = ["k", "r", "g", "c", "m", "y"]
    # colsb = ["k", "r", "g", "c", "m", "y"]
    colsa = ["k" for col in range(idim)]
    colsb = ["r" for col in range(odim)]
    for i in range(odim): # odim * 2
        for j in range(idim):
            # pl.subplot(numplots, 1, (i*idim)+j+1)
            ax = figaxes[j][i]
            # target = Y[h,i]
            # X__ = X_[j] # X[h,j]
            # err += np.sum(np.square(target - prediction))
            # ax.plot(X__, [target], colsa[j] + ".", alpha=0.25, label="target_%d" % i)
            # ax.plot(X__, [prediction[0,i]], colsb[j] + "o", alpha=0.25, label="pred_%d" % i)
            # ax.plot(X[:,j], Y[:,i], colsa[j] + ".", alpha=0.25, label="target_%d" % i)
            ax.hexbin(X[:,j], Y[:,i], gridsize = 20, alpha=0.75, cmap=pl.get_cmap("gray"))
            ax.plot(X[:,j], predictions[:,i], colsb[j] + "o", alpha=0.15, label="pred_%d" % i, markersize=8)
            # pred1 = mdl.filter_e.neuron(mdl.filter_e.flat_to_coords(activities_sorted[-1]))
            # ax.plot(X__, [pred1], "ro", alpha=0.5)
            # pred2 = mdl.filter_e.neuron(mdl.filter_e.flat_to_coords(activities_sorted[-2]))
            # ax.plot(X__, [pred2], "ro", alpha=0.25)
    # print("accum total err = %f" % (err / X.shape[0] / (idim * odim)))
    if saveplot:
        filename = "plot_nodes_over_data_scattermatrix_hexbin_%s.jpg" % (mdl.__class__.__name__,)
        savefig(fig, filename)
    fig.show()

def plot_hebbsom_links_distances_activations(X, Y, mdl, predictions, distances, activities, saveplot = False):
    """plot the hebbian link matrix, and all node distances and activities for all inputs"""
    

    hebblink_log = np.log(mdl.hebblink_filter.T + 1.0)
    
    fig = pl.figure()
    fig.suptitle("Debugging SOM: hebbian links, distances, activities (%s)" % (mdl.__class__.__name__))
    gs = gridspec.GridSpec(4, 1)
    # pl.plot(X, Y, "k.", alpha=0.5)
    # pl.subplot(numplots, 1, numplots-1)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('hebbian associative links')
    # im1 = ax1.imshow(mdl.hebblink_filter, interpolation="none", cmap=pl.get_cmap("gray"))
    im1 = ax1.pcolormesh(hebblink_log, cmap=pl.get_cmap("gray"))
    ax1.set_xlabel("in (e)")
    ax1.set_ylabel("out (p)")
    cbar = fig.colorbar(mappable = im1, ax=ax1, orientation="horizontal")
    
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('distances over time')

    distarray = np.array(distances)
    # print("distarray.shape", distarray.shape)
    pcm = ax2.pcolormesh(distarray.T)
    cbar = fig.colorbar(mappable = pcm, ax=ax2, orientation="horizontal")
    
    # pl.subplot(numplots, 1, numplots)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title('activations propagated via hebbian links')
    actarray = np.array(activities)
    # print("actarray.shape", actarray.shape)
    pcm = ax3.pcolormesh(actarray.T)
    cbar = fig.colorbar(mappable = pcm, ax=ax3, orientation="horizontal")

    ax4 = fig.add_subplot(gs[3])
    ax4.set_title('flattened link table')
    ax4.plot(hebblink_log.flatten())

    # print("hebblink_log", hebblink_log)
    
    if saveplot:
        filename = "plot_hebbsom_links_distances_activations_%s.jpg" % (mdl.__class__.__name__,)
        savefig(fig, filename)
    fig.show()

def plot_mdn_mues_over_data_scan(X, Y, mdl, saveplot = False):
    mues = []
    sigs = []
    pis = []

    print("plot_mdn_mues_over_data_scan: X", X.shape)
    fig = pl.figure()
    gs = gridspec.GridSpec(2, 2)
    dim = Y.shape[1]
    xscan = np.linspace(-np.pi, np.pi, 101).reshape((-1, 1))

    num_mu = mdl.mixcomps * dim
    # num_sig = mixcomps * d ** 2
    num_sig = ((dim ** 2 - dim)/2 + dim) * mdl.mixcomps
    num_pi = mdl.mixcomps
    
    if X.shape[1] > 1:
        xscan = np.hstack((xscan, xscan))
        print("xscan", xscan.shape)
    xscan = X[:100]
    for xs in xscan:
        # print("xs", xs)
        xs = np.atleast_2d(xs)
        print("xs", xs)
        y = mdl.predict(xs)
        # mues.append(mdl.model.z[:mdl.mixcomps,0])
        # sigs.append(np.exp(mdl.model.z[mdl.mixcomps:(2*mdl.mixcomps),0]))
        # pis.append(mdl.lr.softmax(mdl.model.z[(2*mdl.mixcomps):,0]))
        mues.append(mdl.model.z[:num_mu])
        sigs.append(np.exp(mdl.model.z[num_mu:num_mu + num_sig]))
        pis.append(mdl.lr.softmax(mdl.model.z[-num_pi:]))
        # print("xs", xs, "ys", y)
    # print("mues", mues)
    numpoints = xscan.shape[0]
    mues = np.vstack(mues).reshape((numpoints, mdl.mixcomps, dim))
    sigs = np.vstack(sigs).reshape((numpoints, mdl.mixcomps, num_sig / mdl.mixcomps))
    pis = np.vstack(pis).reshape((numpoints, mdl.mixcomps))

    print("mues", mues.shape)
    print("sigs", sigs.shape)
    print("pis", pis.shape)


    colors = ['r', 'g', 'b', 'c', 'y', 'm']
    for h in range(dim):
        # ax = fig.add_subplot(dim, 2, h + 1)
        ax = fig.add_subplot(gs[h,0])
        for i in range(mdl.mixcomps):
            for j in range(xscan.shape[0]):
                # print("mues", mues[[j],[i]], "pis", pis[j,i])
                ax.plot(
                    xscan[[j]], mues[[j],[i],[h]],
                    marker = 'o', markerfacecolor = colors[i % len(colors)],
                    markeredgecolor = colors[i % len(colors)],
                    alpha = pis[j,i])
                # ax.plot(xscan[[j]], mues[[j],[i],[h]] - sigs[[j],[i],[h]], "bo", alpha = pis[j,i], markersize = 2.5)
                # ax.plot(xscan[[j]], mues[[j],[i],[h]] + sigs[[j],[i],[h]], "bo", alpha = pis[j,i], markersize = 2.5)
                
    ax = fig.add_subplot(gs[0,1])
    if dim == 1:
        plot_predictions_over_data(X, Y, mdl, saveplot, ax = ax, datalim = 1000)
    else:
        plot_predictions_over_data_2D(X, Y, mdl, saveplot, ax = ax, datalim = 1000)

        for i in range(mdl.mixcomps):
            ax.plot(mues[:,i,0], mues[:,i,1], linestyle = "none", marker = 'o', markerfacecolor = colors[i % len(colors)], alpha = np.mean(pis[:,i]))
    # ax.plot(xscan, mues - sigs, "bo", alpha = 0.5, markersize = 2.0)
    # ax.plot(xscan, mues + sigs, "bo", alpha = 0.5, markersize = 2.0)
    # ax.plot(xscan, mues, "ro", alpha = 0.5)
    # ax.plot(mues, xscan, "ro", alpha = 0.5)


    if saveplot:
        filename = "plot_mdn_mues_over_data_scan_%s.jpg" % (mdl.__class__.__name__,)
        savefig(fig, filename)
        
    fig.show()
    
def plot_predictions_over_data(X, Y, mdl, saveplot = False, ax = None, datalim = 1000):
    do_hexbin = False
    if X.shape[0] > 4000:
        do_hexbin = False # True
        X = X[-4000:]
        Y = Y[-4000:]
    # plot prediction
    idim = X.shape[1]
    odim = Y.shape[1]
    numsamples = 1 # 2
    Y_samples = []
    for i in range(numsamples):
        Y_samples.append(mdl.predict(X))
    # print("Y_samples[0]", Y_samples[0])

    fig = pl.figure()
    fig.suptitle("Predictions over data xy (numsamples = %d, (%s)" % (numsamples, mdl.__class__.__name__))
    gs = gridspec.GridSpec(odim, 1)
    
    for i in range(odim):
        ax = fig.add_subplot(gs[i])
        target     = Y[:,i]
        
        if do_hexbin:
            ax.hexbin(X, Y, gridsize = 20, alpha=1.0, cmap=pl.get_cmap("gray"))
        else:
            ax.plot(X, target, "k.", label="Y_", alpha=0.5)
        for j in range(numsamples):
            prediction = Y_samples[j][:,i]
            # print("X", X.shape, "prediction", prediction.shape)
            # print("X", X, "prediction", prediction)
            if do_hexbin:
                ax.hexbin(X[:,i], prediction, gridsize = 30, alpha=0.6, cmap=pl.get_cmap("Reds"))
            else:
                ax.plot(X[:,i], prediction, "r.", label="Y_", alpha=0.25)
                
        # get limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        error = target - prediction
        mse   = np.mean(np.square(error))
        mae   = np.mean(np.abs(error))
        xran = xlim[1] - xlim[0]
        yran = ylim[1] - ylim[0]
        ax.text(xlim[0] + xran * 0.1, ylim[0] + yran * 0.3, "mse = %f" % mse)
        ax.text(xlim[0] + xran * 0.1, ylim[0] + yran * 0.5, "mae = %f" % mae)
        
    if saveplot:
        filename = "plot_predictions_over_data_%s.jpg" % (mdl.__class__.__name__,)
        savefig(fig, filename)
        
    fig.show()

def plot_predictions_over_data_2D(X, Y, mdl, saveplot = False, ax = None, datalim = 1000):
    do_hexbin = False
    if X.shape[0] > datalim:
        do_hexbin = False # True
        X = X[-datalim:]
        Y = Y[-datalim:]
    # plot prediction
    idim = X.shape[1]
    odim = Y.shape[1]
    numsamples = 1 # 2
    Y_samples = []
    for i in range(numsamples):
        Y_samples.append(mdl.predict(X))
    # print("Y_samples[0]", Y_samples[0].shape)
    # Y_samples

    if ax is None:
        fig = pl.figure()
        fig.suptitle("Predictions over data xy (numsamples = %d, (%s)" % (numsamples, mdl.__class__.__name__))
        gs = gridspec.GridSpec(1, 1)

        ax = fig.add_subplot(gs[0])
    else:
        fig = None
        
    ax.plot(Y[:,0], Y[:,1], 'ko', alpha = 0.1)
    ax.plot(Y_samples[0][:,0], Y_samples[0][:,1], 'r.', alpha = 0.1)
    ax.set_aspect(1)
    
    # for i in range(odim):
    #     ax = fig.add_subplot(gs[i])
    #     target     = Y[:,i]
        
    #     if do_hexbin:
    #         ax.hexbin(X, Y, gridsize = 20, alpha=1.0, cmap=pl.get_cmap("gray"))
    #     else:
    #         ax.plot(X, target, "k.", label="Y_", alpha=0.5)
    #     for j in range(numsamples):
    #         prediction = Y_samples[j][:,i]
    #         # print("X", X.shape, "prediction", prediction.shape)
    #         # print("X", X, "prediction", prediction)
    #         if do_hexbin:
    #             ax.hexbin(X[:,i], prediction, gridsize = 30, alpha=0.6, cmap=pl.get_cmap("Reds"))
    #         else:
    #             ax.plot(X[:,i], prediction, "r.", label="Y_", alpha=0.25)
                
    #     # get limits
    #     xlim = ax.get_xlim()
    #     ylim = ax.get_ylim()
    #     error = target - prediction
    #     mse   = np.mean(np.square(error))
    #     mae   = np.mean(np.abs(error))
    #     xran = xlim[1] - xlim[0]
    #     yran = ylim[1] - ylim[0]
    #     ax.text(xlim[0] + xran * 0.1, ylim[0] + yran * 0.3, "mse = %f" % mse)
    #     ax.text(xlim[0] + xran * 0.1, ylim[0] + yran * 0.5, "mae = %f" % mae)

    if fig is not None:
        if saveplot:
            filename = "plot_predictions_over_data_%s.jpg" % (mdl.__class__.__name__,)
            savefig(fig, filename)
        
        fig.show()
    
def plot_predictions_over_data_ts(X, Y, mdl, saveplot = False):
    # plot prediction
    idim = X.shape[1]
    odim = Y.shape[1]
    numsamples = 2
    Y_samples = []
    print("Xxx", X.shape)
    for i in range(numsamples):
        Y_samples.append(mdl.predict(X))
    print("Y_samples[0]", Y_samples[0])

    fig = pl.figure()
    fig.suptitle("Predictions over data timeseries (numsamples = %d), (%s)" % (numsamples, mdl.__class__.__name__))
    gs = gridspec.GridSpec(odim, 1)
    
    for i in range(odim):
        # pl.subplot(odim, 2, (i*2)+1)
        ax = fig.add_subplot(gs[i])
        target     = Y[:,i]
        ax.plot(target, "k.", label="Y_", alpha=0.5)

        # pl.subplot(odim, 2, (i*2)+2)
        # prediction = Y_[:,i]
        
        # pl.plot(target, "k.", label="Y")
        mses = []
        maes = []
        errors = []
        for j in range(numsamples):
            prediction = Y_samples[j][:,i]
            error = target - prediction
            errors.append(error)
            mse   = np.mean(np.square(error))
            mae   = np.mean(np.abs(error))
            mses.append(mse)
            maes.append(mae)
            # pl.plot(prediction, target, "r.", label="Y_", alpha=0.25)
            ax.plot(prediction, "r.", label="Y_", alpha=0.25)

        errors = np.asarray(errors)
        # print("errors.shape", errors.shape)
        aes = np.min(np.abs(errors), axis=0)
        ses = np.min(np.square(errors), axis=0)
        mae = np.mean(aes)
        mse = np.mean(ses)
                
        # get limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xran = xlim[1] - xlim[0]
        yran = ylim[1] - ylim[0]
        ax.text(xlim[0] + xran * 0.1, ylim[0] + yran * 0.3, "mse = %f" % mse)
        ax.text(xlim[0] + xran * 0.1, ylim[0] + yran * 0.5, "mae = %f" % mae)
        # pl.plot(X[:,i], Y[:,i], "k.", alpha=0.25)
    if saveplot:
        filename = "plot_predictions_over_data_ts_%s.jpg" % (mdl.__class__.__name__,)
        savefig(fig, filename)
    fig.show()
        
def get_class_from_name(name = "KNN"):
    """models_actinf.get_class_from_name

    Get a class by a common name string.
    """
    if name == "KNN":
        cls = smpKNN
    elif name == "SOESGP":
        cls = smpSOESGP
    elif name == "STORKGP":
        cls = smpSTORKGP
    elif name == "GMM":
        cls = partial(smpGMM, K = 20)
    elif name == "IGMM":
        cls = partial(smpIGMM, K = 20)
    elif name == "HebbSOM":
        cls = smpHebbianSOM
    elif name == 'resRLS':
        from smp_base.models_learners import smpSHL
        cls = smpSHL
    else:
        cls = smpKNN
    return cls

def generate_inverted_sinewave_dataset(N = 1000, f = 1.0, p = 0.0, a1 = 1.0, a2 = 0.3):
    """models_actinf.generate_inverted_sinewave_dataset

    Generate the inverted sine dataset used in Bishop's (Bishop96)
    mixture density paper

    Returns:
    - matrices X, Y
    """
    X = np.linspace(0,1,N)
    # FIXME: include phase p
    Y = a1 * X + a2 * np.sin(f * (2 * 3.1415926) * X) + np.random.uniform(-0.1, 0.1, N)
    X,Y = Y[:,np.newaxis],X[:,np.newaxis]
    
    # pl.subplot(211)
    # pl.plot(Y, X, "ko", alpha=0.25)
    # pl.subplot(212)
    # pl.plot(X, Y, "ko", alpha=0.25)
    # pl.show()
    
    return X,Y

def generate_2devensimpler_component(x):
    """models_actinf.generate_2devensimpler_component

    Generate a two-dimensional correspondence dataset to test
    covariance learning of the multivariate mixture density learning
    rule.

    Returns:
    - matrix X
    """
    y1_1 = np.sin(x * 10.0) * 0.5 + x * 0.3 + x ** 2 * 0.05
    y1_1 += np.random.normal(0, np.abs(x - np.mean(x)) * 0.3)
    y1_2 = np.sin(x * 5.0) * 0.3 + x * 0.5 - x ** 2 * 0.2
    y1_2 += np.random.normal(0, np.abs(x - np.mean(x)) * 0.3)
    print(y1_1.shape, y1_2.shape)
    return np.vstack((y1_1, y1_2)).T

def test_model(args):
    """actinf_models.test_model

    Test the model type given in args.modelclass on data
    """
    
    # import pylab as pl
    from sklearn.utils import shuffle
    
    # get last component of datafile, the actual filename
    datafilepath_comps = args.datafile.split("/")
    if datafilepath_comps[-1].startswith("EP"):
        idim = 2
        odim = 3
        EP = np.load(args.datafile)
        sl = slice(0, args.numsteps)
        X = EP[sl,:idim]
        Y = EP[sl,idim:]
        # print("X_std.shape", X_std.shape)
    elif datafilepath_comps[-1].startswith("NAO_EP"):
        idim = 4
        odim = 4
        EP = np.load(args.datafile)
        sl = slice(0, args.numsteps)
        X = EP[sl,:idim]
        Y = EP[sl,idim:]
    elif args.datafile.startswith("inverted"):
        idim = 1
        odim = 1
        X,Y = generate_inverted_sinewave_dataset(N = args.numsteps)
        idx = list(range(args.numsteps))
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]
    elif args.datafile.startswith("2dinverted"):
        idim = 2
        odim = 2
        X1,Y1 = generate_inverted_sinewave_dataset(N = args.numsteps, f = 1.0, a1 = 0.0)
        # X2,Y2 = generate_inverted_sinewave_dataset(N = args.numsteps, f = 2.0, a1 = -0.5, a2 = 0.5)
        # X2,Y2 = generate_inverted_sinewave_dataset(N = args.numsteps, f = 1.5, a1 = 1.0, a2 = 0.4)
        X2,Y2 = generate_inverted_sinewave_dataset(N = args.numsteps, f = 0.25, p = np.pi/2.0, a1 = 0.0, a2 = 0.3)
        idx = list(range(args.numsteps))
        np.random.shuffle(idx)
        print("X1.shape", X1.shape, X1[idx].shape)
        # print("idx", idx)
        
        X = np.tile(X1[idx], (1, 2))
        Y = np.tile(Y1[idx], (1, 2))

        X = np.hstack((X1[idx], X2[idx]))
        Y = np.hstack((Y1[idx], Y2[idx]))
        
        # X, Y = shuffle(X, Y, random_state=0)
        np.random.seed(args.seed)
    elif args.datafile.startswith("2dsimple"):
        idim = 2
        odim = 2

        mixcomps = 3
        d = 4
        mu = np.random.uniform(-1, 1, size = (mixcomps, d))
        S = np.array([np.eye(d) * 0.01 for c in range(mixcomps)])
        for s in S:
            s += 0.05 * np.random.uniform(-1, 1, size = s.shape)
            s[np.tril_indices(d, -1)] = s[np.triu_indices(d, 1)]
        pi = np.ones((mixcomps, )) * 1.0/mixcomps

        lr = LearningRules(ndim_out = d, dim = d)
        lr.learnFORCEmdn_setup(mixcomps = mixcomps)
        X = np.zeros((args.numsteps, d))
        for i in range(args.numsteps):
            X[[i]] = lr.mixtureMV(mu, S, pi)

        Y = X[:,odim:]
        X = X[:,:odim]
        print("X.shape", X.shape)
        print("Y.shape", Y.shape)
        
    elif args.datafile.startswith("2devensimpler"):
        idim = 1
        odim = 2

        numcomp_steps = args.numsteps / 3
        setattr(args, 'numsteps', numcomp_steps * 3)
        
        # 3 input clusters
        x1 = np.linspace(-1.0, -0.5, numcomp_steps) # + 0.5
        x2 = np.linspace(-0.25, 0.25, numcomp_steps) 
        x3 = np.linspace(0.5, 1.0, numcomp_steps) # - 0.5

        y1 = generate_2devensimpler_component(x1)
        y2 = generate_2devensimpler_component(x2)
        y3 = generate_2devensimpler_component(x3)
        
        pl.subplot(121)
        pl.plot(x1)
        pl.plot(x2)
        pl.plot(x3)
        pl.subplot(122)
        pl.plot(y1[:,0], y1[:,1], "ko", alpha = 0.1)
        pl.plot(y2[:,0], y2[:,1], "ro", alpha = 0.1)
        pl.plot(y3[:,0], y3[:,1], "go", alpha = 0.1)
        # pl.plot(y1_2)
        pl.show()

        X = np.hstack((x1, x2, x3)).reshape((args.numsteps, 1))
        Y = np.vstack((y1, y2, y3))

        print("X", X.shape, "Y", Y.shape)

        idx = list(range(args.numsteps))
        np.random.shuffle(idx)

        X = X[idx]
        Y = Y[idx]
        
        # mixcomps = 3
        # d = 4
        # mu = np.random.uniform(-1, 1, size = (mixcomps, d))
        # S = np.array([np.eye(d) * 0.01 for c in range(mixcomps)])
        # for s in S:
        #     s += 0.05 * np.random.uniform(-1, 1, size = s.shape)
        #     s[np.tril_indices(d, -1)] = s[np.triu_indices(d, 1)]
        # pi = np.ones((mixcomps, )) * 1.0/mixcomps

        # lr = LearningRules(ndim_out = d, dim = d)
        # lr.learnFORCEmdn_setup(mixcomps = mixcomps)
        # X = np.zeros((args.numsteps, d))
        # for i in range(args.numsteps):
        #     X[[i]] = lr.mixtureMV(mu, S, pi)

        # Y = X[:,odim:]
        # X = X[:,:odim]
        # print("X.shape", X.shape)
        # print("Y.shape", Y.shape)

    else:
        idim = 1
        odim = 1

    X_mu  = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    Y_mu  = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    X -= X_mu
    X /= X_std
    Y -= Y_mu
    Y /= Y_std

    if args.modelclass == "GMM":
        dim = idim + odim
        
    # diagnostics
    print("models_actinf.py: X.shape = %s, idim = %d, Y.shape = %s, odim = %d" % (X.shape, idim, Y.shape, odim))
    # sys.exit()
    # pl.subplot(211)
    # pl.plot(X)
    # pl.subplot(212)
    # pl.plot(Y)
    # pl.show()

    mdlcls = get_class_from_name(args.modelclass)
    mdlcnf = mdlcls.defaults
    mdlcnf['idim'] = idim
    mdlcnf['odim'] = odim
    if args.modelclass == "HebbSOM":
        if args.fitmode == 'incremental':
            args.numepisodes = 1
        # print("HebbianSOM idim", idim, "odim", odim)
        # mdl = mdlcls(idim = idim, odim = odim, numepisodes = args.numepisodes, visualize = True, mapsize_e = 10, mapsize_p = 10)
        mdlcnf['mapsize_e'] = 30
        mdlcnf['mapsize_p'] = 30
        mdlcnf['visualize'] = False # True
        mdlcnf['som_lr'] = 1e-1
        mdlcnf['som_nhs'] = 1e-1
        mdl = mdlcls(conf = mdlcnf)
    elif args.modelclass == "resRLS":
        mdlcnf['alpha'] = 1000.0 # 1.0 # 100.0
        mdlcnf['mixcomps'] = 30 # 3 # 6 # 12
        mdlcnf['sigma_mu'] = 1e-1
        mdlcnf['sigma_sig'] = 1e-1
        mdlcnf['sigma_pi'] = 1e-1 # 1.0/mdlcnf['mixcomps']
        
    mdl = mdlcls(conf = mdlcnf)

    print("Testing model class %s, %s" % (mdlcls, mdl))

    print("Fitting model %s to data with shapes X = %s, Y = %s" % (args.fitmode, X.shape, Y.shape))
    if args.fitmode == 'incremental':
        for i in range(args.numsteps):
            mdl.fit(X[[i]], Y[[i]])
            if i % 1000 == 0:
                if args.modelclass == 'resRLS':
                    print("step = %d, loss = %s, |w| = %s" % (i, np.linalg.norm(mdl.lr.e), np.linalg.norm(mdl.model.wo)))
                else:
                    print("step = %d" % (i, ))
    else:
        # batch fit
        mdl.fit(X, Y)
    
    print("Plotting model %s, %s" % (mdlcls, mdl))
    if args.modelclass == "HebbSOM":

        if X.shape[0] > args.numsteps:
            X = X[:args.numsteps,...]
            Y = Y[:args.numsteps,...]
        
        e_nodes, p_nodes = hebbsom_get_map_nodes(mdl, idim, odim)
        e_nodes_cov = np.tile(np.eye(idim) * 0.05, e_nodes.shape[0]).T.reshape((e_nodes.shape[0], idim, idim))
        p_nodes_cov = np.tile(np.eye(odim) * 0.05, p_nodes.shape[0]).T.reshape((p_nodes.shape[0], odim, odim))

        # print("nodes", e_nodes, p_nodes)
        # print("covs",  e_nodes_cov, p_nodes_cov)
        # print("covs",  e_nodes_cov.shape, p_nodes_cov.shape)

        print("1 plot_nodes_over_data_1d_components")
        fig = plot_nodes_over_data_1d_components_fig(title = args.modelclass, numplots = X.shape[1] + Y.shape[1])
        plot_nodes_over_data_1d_components(fig, X, Y, mdl, e_nodes, p_nodes, e_nodes_cov, p_nodes_cov, saveplot = saveplot)
        
        print("2 plot_nodes_over_data_scattermatrix")
        plot_nodes_over_data_scattermatrix(X, Y, mdl, e_nodes, p_nodes, e_nodes_cov, p_nodes_cov, saveplot = saveplot)

        print("3 hebbsom_predict_full")
        predictions, distances, activities = hebbsom_predict_full(X, Y, mdl)
    
        # print("4 plot_predictions_over_data")
        # plot_predictions_over_data(X, Y, mdl, saveplot = saveplot)
        
        # print("5 plot_predictions_over_data_ts")
        # plot_predictions_over_data_ts(X, Y, mdl, saveplot = saveplot)
        
        print("6 plot_nodes_over_data_scattermatrix_hexbin")
        plot_nodes_over_data_scattermatrix_hexbin(X, Y, mdl, predictions, distances, activities, saveplot = saveplot)
                
        # print("7 plot_hebbsom_links_distances_activations")
        # plot_hebbsom_links_distances_activations(X, Y, mdl, predictions, distances, activities, saveplot = saveplot)
                            
        # nodes_e = filter_e.map.neurons[:,:,i]
        # nodes_p = filter_p.map.neurons[:,:,i]
        # pl.plot(nodes, filter_e.map.neurons[:,:,1], "ko", alpha=0.5, ms=10)
        # pl.show()
    
    elif args.modelclass == "GMM":
        nodes = np.array(mdl.cen_lst)
        covs  = np.array(mdl.cov_lst)

        # print("nodes,covs shape", nodes.shape, covs.shape)
        
        e_nodes = nodes[:,:idim]
        p_nodes = nodes[:,idim:]
        e_nodes_cov = covs[:,:idim,:idim]
        p_nodes_cov = covs[:,idim:,idim:]

        # print("nodes", e_nodes, p_nodes)
        # print("covs",  e_nodes_cov.shape, p_nodes_cov.shape)
        
        plot_nodes_over_data_1d_components(X, Y, mdl, e_nodes, p_nodes, e_nodes_cov, p_nodes_cov, saveplot = saveplot)
        
        plot_nodes_over_data_scattermatrix(X, Y, mdl, e_nodes, p_nodes, e_nodes_cov, p_nodes_cov, saveplot = saveplot)
        
        plot_predictions_over_data_ts(X, Y, mdl, saveplot = saveplot)
        
        plot_predictions_over_data(X, Y, mdl, saveplot = saveplot)

    elif args.modelclass == "resRLS":
        # reservoir, mixture density
        plot_mdn_mues_over_data_scan(X, Y, mdl, saveplot = saveplot)
        
        if odim == 1:
            plot_predictions_over_data(X, Y, mdl, saveplot = saveplot)
        else:
            plot_predictions_over_data_2D(X, Y, mdl, saveplot = saveplot)
        
    else:
        # elif args.modelclass in ["KNN", "SOESGP", "STORKGP"]:
        # print("hello")
        print ("models_actinf.test_model: X", X.shape)
        print ("models_actinf.test_model: Y", Y.shape)
        plot_predictions_over_data_ts(X, Y, mdl, saveplot = saveplot)
        plot_predictions_over_data(X, Y, mdl, saveplot = saveplot)
        
        
    pl.draw()
    pl.pause(1e-9)
    
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile",     type=str, help="datafile containing t x (dim_extero + dim_proprio) matrix ", default="data/simplearm_n1000/EP_1000.npy")
    parser.add_argument("-f", "--fitmode",      type=str, help="Type of fit: batch or incremental [batch]", default='batch')
    parser.add_argument("-m", "--modelclass",   type=str, help="Which model class [all] to test from " + ", ".join(model_classes), default="all")
    parser.add_argument("-n", "--numsteps",     type=int, help="Number of datapoints [1000]", default=1000)
    parser.add_argument("-ne", "--numepisodes", type=int, help="Number of episodes [10]", default=10)
    parser.add_argument("-s",  "--seed",        type=int, help="seed for RNG [0]",        default=0)
    args = parser.parse_args()

    if args.modelclass == "all":
        pl.ion()
        for mdlcls in ["KNN", "SOESGP", "STORKGP", "GMM", "HebbSOM"]:
            args.modelclass = mdlcls
            test_model(args)
    else:
        test_model(args)

    pl.ioff()
    pl.show()
