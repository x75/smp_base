"""smp_base/reservoirs.py

A leaky integrator rate-coded reservoir class. A reservoir is a randomly connected
single hidden layer recurrent neural network. The spectral_radius 'g' design parameter
provides a global gain on the recurrence matrix. In the limit of g -> 0 the layer
becomes a feedforward layer and the model becomes an Extreme Learning Machine (ELM)
rather than an Echo State Network (ESN) or Liquid State Machine (LSM), respectively.

Authors:
 - Oswald Berthold 2012 - 2017
 - Aleke Nolte (learnRLS)

Examples:

This is somewhere close to the plot on https://github.com/ReScience/call-for-replication/issues/2

.. code-block:: shell

   python reservoirs.py -t MSO_s4 -m ol_eh -rs 1000 -l 100000 --plot_interval 20000

TODO
 - x fit/predict interface, batch regression fit: we handle that on the level of the integrating model
 - x intrinsic plasticity: input layer, reservoir layer: use separate standardization layers for inputs
 - x correlated exploration noise: exploration modules: gaussian, colored gaussian, pareto, homeokinesis, ...
"""


import sys, time, argparse

import numpy as np
import numpy.linalg as LA
import scipy as sp
import scipy.sparse as spa
import matplotlib.pyplot as plt
from matplotlib import gridspec


try:
    try:
        from config import RLSPY
    except:
        pass
    else:
        sys.path.append(RLSPY)
    import rlspy
except ImportError:
    print("ImportError for rlspy")
    rlspy = None

# from learners import GHA

############################################################
# utility functions
def create_matrix_sparse_random(rows = 2, cols = 2, density = 0.1, dist = "normal"):
    """reservoirs.create_matrix_sparse_random

    Create a sparse (density p) random (distribution dist) matrix for
    use in the recurrent network.
    """
    m = spa.rand(rows, cols, density)
    m = m.todense()
    validx = m != 0
    valtmp = m[validx]
    # print "validx", validx.shape
    # print "m", m[validx]#.shape
    if dist in ['normal', 'sparse_normal']:
        valtmp_rnd = np.random.normal(0,   1, size=(valtmp.shape[1], ))
    elif dist in ['uniform', 'sparse_uniform']:
        valtmp_rnd = np.random.uniform(-1, 1, size=(valtmp.shape[1], ))
    # why array?
    m[validx] = valtmp_rnd
    return np.array(m)

def create_matrix_reservoir(N, p):
    """reservoirs.create_matrix_reservoir

    Create an NxN reservoir recurrence matrix with density p. Wrapper
    for create_matrix_sparse_random.
    """
    # M = spa.rand(N, N, p)
    # M = M.todense()
    # tmp_idx = M != 0
    # tmp = M[tmp_idx]
    # tmp_r = np.random.normal(0, 1, size=(tmp.shape[1],))
    # M[tmp_idx] = tmp_r
    # # return dense representation
    # return np.array(M).copy()
    # # return spa.bsr_matrix(M)
    
    return create_matrix_sparse_random(N, N, p, dist = "normal")

def normalize_spectral_radius(M, g):
    """reservoirs.normalize_spectral_radius

    Normalize the spectral radius of a given matrix M to scale to g
    """
    
    # compute eigenvalues
    [w,v] = LA.eig(M)
    # get maximum absolute eigenvalue
    lae = np.max(np.abs(w))
    # normalize matrix by max ev
    M /= lae
    # scale normalized matrix to desired spectral radius
    M *= g
    # check for scaling
    [w,v] = LA.eig(M)
    lae = np.max(np.abs(w))
    # print "normalize_spectral_radius: lae post/desired = %f / %f" % (lae, g)
    assert np.abs(g - lae) < 0.1

################################################################################
# input matrix creation
def res_input_matrix_random_sparse(idim = 1, odim = 1, density=0.1, dist = 'normal'):
    """reservoirs.res_input_matrix_random_sparse

    Create a sparse reservoir input matrix. Wrapper for
    create_matrix_sparse_random.

    Arguments:
    idim: input dimension
    odim: hidden dimension
    density: density
    dist: distribution

    Returns:
    wi: input matrix
    """
    # p_wi = density
    # wi_ = spa.rand(odim, idim, p_wi)
    # # print "sparse wi", wi_
    # wi = wi_.todense()
    # tmp_idx = wi != 0
    # tmp = wi[tmp_idx]
    # # tmp_r = np.random.normal(0, 1, size=(tmp.shape[1],))
    # tmp_r = np.random.uniform(-1, 1, size=(tmp.shape[1],))
    # wi[tmp_idx] = tmp_r
    # # return dense repr
    # return np.asarray(wi)
    return create_matrix_sparse_random(odim, idim, density, dist = dist)

def res_input_matrix_disjunct_proj(idim = 1, odim = 1):
    """reservoirs.res_input_matrix_disjunct_proj

    Create an input matrix that projects inputs onto disjunct regions
    of the hidden space.
    """
    
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

def res_input_matrix_random(idim = 1, odim = 1, dist = 'normal'):
    if dist == 'uniform':
        wi = np.random.uniform(-1.0, 1.0, (odim, idim))
    elif dist == 'normal':
        wi = np.random.normal(0.0, 1.0, (odim, idim))
    return wi

################################################################################
# Standalone class for learning rules
# - Recursive Least Squares (RLS, depends on rlspy.py): the vanilla online supervised
#   reservoir training method
# - First-order reduced and controlled error or FORCE learning (Sussillo & Abbott, 2012)
# - FORCEmdn: Mixture density output layer using FORCE rule (Berthold, 2017)
# - Exploratory Hebbian learning (Legenstein & others, 2010)
class LearningRules(object):
    """LearningRules class

    This class implements different learning rules used for training
    the reservoir in online (aka incremental, single-time step,
    closed-loop) mode.
    """
    def __init__(self, ndim_out = 1, dim = 1, alpha = None):
        self.ndim_out = ndim_out
        self.dim = dim
        self.loss = 0
        self.loss_ = np.zeros((1, 1000))
        self.e = np.zeros((self.ndim_out, 1))
        self.perf = self.e # pointer
        self.e_lp = np.zeros_like(self.e)

        # keep counting
        self.cnt = 0

    ############################################################
    ## learning rule: FORCE
    # - David Sussillo, L.F. Abbott, Generating Coherent Patterns of
    # Activity from Chaotic Neural Networks, Neuron, Volume 63, Issue
    # 4, 27 August 2009, Pages 544-557, ISSN 0896-6273,
    # http://dx.doi.org/10.1016/j.neuron.2009.07.018. (http://www.sciencedirect.com/science/article/pii/S0896627309005479)
    # Keywords: SYSNEURO
    def learnFORCEsetup(self, alpha = None, dim = 1):
        if alpha is None:
            self.alpha = 1.0
        else:
            self.alpha = alpha
            
        # FORCE stuff
        self.P = (1.0/self.alpha) * np.eye(dim)

    def learnFORCE_update_P(self, P, r):
        """LearningRules.learnFORCE_update_P

        Perform the covariance update for the FORCE learning rule
        """
        if hasattr(self, 'P'):
           P = self.P
            
        k = np.dot(P, r)
        rPr = np.dot(r.T, k)
        c = 1.0/(1.0 + rPr)
        P = P - np.dot(k, (k.T*c))
        # if hasattr(self, 'P'):
        #     self.P = self.P
        return (P, k, c)

    def learnFORCE(self, target, P, k, c, r, z, channel, error = None):
        """LearningRules.learnFORCE

        The FORCE learning rule for reservoir online supervised learning
        """
        # compute error
        if error is None:
            self.e = z - target
        else:
            self.e = error
        # compute weight update from error times k
        dw = -self.e.T * k * c
        return dw

    def learnDeltamdn(self, target, P, k, c, r, z, channel, x):
        """LearningRules.learnDeltamdn

        Quick hack delta rule for testing MDN learning
        """
        # eta = 5e-5
        eta = 1e-4
        # eta = 5e-4
        self.e = self.mdn_loss(x, r, z, target)
        dw = eta * np.dot(-self.e, r.T).T
        self.cnt += 1
        return dw

    # learning rule: FORCEmdn
    def learnFORCEmdn_setup(self, mixcomps = 3):
        """LearningRules.learnFORCEmdn_setup

        Setup MDN variables before applying FORCEmdn learning rule
        """
        # print "ndim_out", self.ndim_out
        self.loss = 10.0
        self.e = np.zeros((self.ndim_out, 1))
        self.mixcomps = mixcomps

    # use FORCE update with mdn based gradients
    def learnFORCEmdn(self, target, P, k, c, r, z, channel, x):
        """LearningRules.learnFORCEmdn

        FORCE learning rule with MDN loss for reservoir online supervised learning
        """
        # compute error
        if target.shape[0] < 2:
            self.e = self.mdn_loss(x, r, z, target)
        else:
            self.e = self.mdn_lossMV(x, r, z, target, d = target.shape[0])
        # compute weight update from error
        dw = -self.e.T * k * c

        self.cnt += 1
        return dw

    def learnRLSsetup(self, x0 = None, P0 = None, modelsize = 1, noise = 1e-2):
        """LearningRules.learnRLSsetup

        Setup RLS variables before applying the RLS learning rule
        """

        #self.rls_E = rlspy.data_matrix.Estimator(np.zeros(shape=(self.N, 1)) ,(1.0/self.alpha)*np.eye(self.N))
        #self.rls_E = rlspy.data_matrix.Estimator(np.random.uniform(0, 0.0001, size=(self.N, 1)) , np.eye(self.N))

        self.N = modelsize
        self.noise = noise
        
        if  P0 is None:
          print ("reservoirs.LearningRules.learnRLSsetup: random initialization for RLS setup ")
          # self.rls_E = rlspy.data_matrix.Estimator(np.random.uniform(0, 0.1, size=(self.N, 1)) , np.eye(self.N))
          self.rls_estimator = rlspy.data_matrix.Estimator(np.random.uniform(0, 0.01, size=(self.N, 1)) , np.eye(self.N))
          # self.wo = np.random.uniform(-1e-4,1e-4, size=(self.N, self.output_num))
          # self.wo = np.zeros((self.N, self.output_num))
        else:
          print ('reservoirs.LearningRules.learnRLSsetup: taking arguments as initialization for RLS setup')
          # self.wo = wo_init
          self.rls_estimator = rlspy.data_matrix.Estimator(x0, P0)

        #self.wo = np.random.uniform(-1e-3,1e-3, size=(self.N, self.output_num))
        #self.wo = np.random.uniform(0,1, size=(self.N, self.output_num))

    def learnRLS(self, target, r, noise = None, z = None, x = None):
        """LearningRules.learnRLS

        The RLS error-based supervised learning rule
        """
        if noise is not None:
            self.noise = noise

        if x is not None and z is not None:
            e = self.mdn_loss(x, r, z, target)
            target = z - e
            
        # print "%s.learnRLS, target.shape = %s" % (self.__class__.__name__, target.shape)
        # self.rls_estimator.update(self.r.T, target.T, self.theta_state)
        self.rls_estimator.single_update(r.T, target.T, self.noise)
        # self.wo = self.rls_E.x
        self.cnt += 1
        return self.rls_estimator.dx
    
    # mixture density stuff
    def mixtureMV(self, mu, sig, ps):
        """LearningRules.mixtureMV

        Sample from a multivariate gaussian mixture with \mu = means, \Sigma = cov matrix
        """
        # print "%s.mixtureMV: Implement me" % (self.__class__.__name__)

        # print "mixtureMV", "mu", mu.shape, "sig", sig.shape, "ps", ps.shape
        
        mixcomps = self.mixcomps
        d = self.dim
        
        assert mixcomps == ps.shape[0]
        assert not np.isnan(mu).any()
        assert not np.isnan(sig).any()
        assert not np.isnan(ps).any()
        
        # check flat inputs
        if mu.shape[1] == 1:
            mu = mu.reshape((self.mixcomps, self.dim))
            # mu_ = mu.reshape((mixcomps, d))
        if sig.shape[1] == 1:
            # S_diag = sig[:(mixcomps * self.dim)]
            # S_triu = sig[(mixcomps * self.dim):]
            S_diag = sig[:(mixcomps * d)].reshape((mixcomps, d))
            S_triu = sig[(mixcomps * d):].reshape((mixcomps, -1))
            S = []
            for i in range(mixcomps):
                S_ = np.diag(S_diag[i])
                S_[np.triu_indices(d,  1)] = S_triu[i]
                S_[np.tril_indices(d, -1)] = S_triu[i] # check if thats correct
                S.append(S_)
            sig = np.array(S)
            
        # print "mixtureMV", "mu", mu.shape, "sig", sig.shape, "ps", ps.shape
                
        # d = mu.shape[0]
        compidx = self.mixture_sample_prior(ps)

        mu_ = mu[compidx]
        S_  = sig[compidx]

        # print "compidx", compidx, "mu_", mu_, "S_", S_
        
        y = np.random.multivariate_normal(mu_, S_)
        return y

    # mixture, mdn_loss, and softmax are taken from karpathy's MixtureDensityNets.py
    def mixture(self, mu, sig, ps):
        """LearningRules.mixture

        Sample from the univariate gaussian mixture
        """
        # compidx = np.where(np.random.multinomial(1, ps) == 1.0)[0][0]
        compidx = self.mixture_sample_prior(ps)
        # print "compidx", compidx, mu[compidx], "sig", sig[compidx], "p", ps[compidx] # .shape
        # y = np.random.normal(mu[compidx], np.abs(sig[compidx]) + np.random.uniform(0, 1e-3, size=sig[[compidx]].shape))
        y = np.random.normal(mu[compidx], sig[compidx], size = mu[[compidx]].shape)
        return y

    def mixture_sample_prior(self, ps):
        # print "ps", ps.shape
        if len(ps.shape) > 1:
            ps = ps[:,0]
        multinom_sample = np.random.multinomial(1, ps)
        # print "multinom_sample", multinom_sample
        multinom_sample_idx = np.where(multinom_sample == 1.0)
        # print "multinom_sample_idx", ps, multinom_sample_idx
        compidx = multinom_sample_idx[0][0]
        compidx = np.argmax(ps)
        # print "compidx", compidx
        return compidx
    
    def mdn_loss(self, x, r, z, y, loss_only = False):
        """LearningRules.mdn_loss

        Compute the mixture density (MDN) loss
        """
        mixcomps = self.mixcomps
        # predict mean
        mu = z[:mixcomps,[0]]
        # predict log variance
        logsig = z[mixcomps:(2*mixcomps),[0]]
        # unlog it
        sig = np.exp(logsig)
        # predict mixture priors
        piu = z[(2*mixcomps):,[0]]
        # softmax them
        pi = self.softmax(piu)
        # compute the loss: mean negative data log likelihood
        n,k = mu.shape # number of mixture components
        n = float(n)
        # component likelihood
        ps = np.exp(-((y - mu)**2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))
        # mixture likelihood
        pin = ps * pi
        # print "pin", pin.T
        # negloglikelihood, compare with batch estimate
        lp = -np.log(np.sum(pin, axis=0, keepdims=True))
        loss = np.sum(lp) / n
        self.loss = loss
        self.loss_ = np.roll(self.loss_, shift = -1, axis = 1)
        self.loss_[[-1]] = loss

        # print "n", n
        # # compute component errors
        # gammas are pi_i's in bishop94
        # print "pin", pin.shape
        gammas = pin / np.sum(pin, axis=0, keepdims = True)
        # print "gammas", gammas.T, mu - y
        dmu = gammas * ((mu - y)/sig**2) / n
        dlogsig = gammas * (1.0 - (y-mu)**2/(sig**2)) / n
        dpiu = (pi - gammas) / n
        # print "|dmu| = %f" % (np.linalg.norm(dmu))
        
        if self.cnt % 100 == 0:
            print("count[%d]" % self.cnt)
            # print "    loss = %f / %f / %d" % (self.loss/self.cnt, self.loss, self.cnt)
            print("    loss = %f / %d" % (np.mean(self.loss_), self.cnt))
            print("    dmu", np.linalg.norm(np.array(dmu).flatten()))
            print("    dlogisig", np.linalg.norm(dlogsig))
            print("    dpiu", np.linalg.norm(dpiu.flatten()))

        return np.vstack((dmu, dlogsig, dpiu))

    def mdn_lossMV(self, x, r, z, y, d = 2):
        """LearningRules.mdn_lossMV

        Compute the multivariate mixture density (MDN) loss
        """
        mixcomps = self.mixcomps
        num_mu = mixcomps * d
        # num_sig = mixcomps * d ** 2
        num_sig = ((d ** 2 - d)/2 + d) * mixcomps
        num_pi = mixcomps
        
        # predict mean
        mu = z[:num_mu]
        # predict log variance
        logsig = z[num_mu:num_mu + num_sig]
        # unlog it
        sig = np.exp(logsig)
        # predict mixture priors
        piu = z[num_mu + num_sig:]
        # softmax them
        pi = self.softmax(piu)

        # print "mdn_lossMV", "mu", mu.shape, "sig", sig.shape, "pi", pi.shape
        
        # compute the loss: mean negative data log likelihood
        # k, n = mu.shape # number of mixture components
        # n = float(n)

        # print "k", k, "n", n
        n = float(mixcomps)
        
        # component likelihood
        # ps = np.exp( - ( (y - mu) ** 2 ) / ( 2 * sig ** 2 ) ) / ( sig * np.sqrt( 2*np.pi))

        """
                           1                  /    1                                    \
        p(X) = -------------------------- exp | - --- ( X - MU )^T \Sigma^-1 ( X - MU ) |
               (2pi)^(d/2) |\Sigma|^(1/2)     \    2                                    /
        """

        # norm1 = np.tile(x, (mixcomps, 1)) - mu

        # compute mixture component likelihood (posterior)
        p_s = []

        mu_ = mu.reshape((mixcomps, d))
        S_diag = sig[:(mixcomps * d)].reshape((mixcomps, d))
        S_triu = sig[(mixcomps * d):].reshape((mixcomps, -1))
        # print "S_diag", S_diag, "S_triu", S_triu
        
        for i in range(mixcomps):
            S_ = np.diag(S_diag[i])
            S_[np.triu_indices(d,  1)] = S_triu[i]
            S_[np.tril_indices(d, -1)] = S_triu[i]
            # S_ = sig.reshape((mixcomps, d, d)) ** 2
            S_ *= S_
            # print "S_", S_, "sig", sig.shape, "x", x.shape, "mu_", mu_[i].shape

            # print "mu", mu_[[i]].shape
            # print "y",y.shape
            
            # norm1 = x - mu_[[i]].T
            norm1 = y - mu_[[i]].T
            S__ = np.linalg.pinv(S_)
            S___ = sp.linalg.pinv(S_)
            # print "    S__", S__
            # print "    S___", S___
            S__ = np.clip(S__, -1, 1)
            p_ = np.exp(-np.dot(norm1.T, np.dot(S__, norm1))/2.0)
            # print "p_", p_
            p_s.append(p_.flatten())
            # print "S * norm", p_s[-1]

        # print "mdn_lossMV", "x", x.shape, "mu", mu_.shape, "norm1", norm1.shape, "S_", S_.shape
        
        # ps = np.exp(norm1.T np.dot(/ 2.0)
        ps = np.array(p_s)

        # print "ps", ps
        
        # mixture components likelihood
        pin = ps * pi
        # print "pin", pin.T
        
        # negloglikelihood, compare with batch estimate
        lp = -np.log(np.sum(pin, axis=0, keepdims=True))
        # print 'lp', lp
        loss = np.sum(lp) / n
        # self.loss += loss
        self.loss = 0.9 * self.loss + 0.1 * -np.log(np.max(pin)) # loss
        # assert loss > 0.0, "loss must be greater zero?"
        # print "loss", loss

        # compute component errors
        # gammas are pi_i's in bishop94
        gammas = pin / np.sum(pin, axis=0, keepdims = True)

        # print "gammas", pin, gammas
        
        dmu = []
        dlogsig = []
        d1 = []
        d2 = []
        dpiu = []
        for i in range(mixcomps):
            S_ = np.diag(S_diag[i])
            # print "S_ pre  triu", S_[np.triu_indices(d,  1)].shape, S_triu[i].shape
            S_[np.triu_indices(d,  1)] = S_triu[i]
            # print "S_ post triu", S_[np.triu_indices(d,  1)]
            S_[np.tril_indices(d, -1)] = S_[np.triu_indices(d,  1)] # S_triu[i]


            S_2 = S_ ** 2
            S__ = np.linalg.pinv(S_2)

            norm1 = mu_[[i]].T - y

            # print "norm1", norm1, "S__", S__
            # dmu_ = gammas[i] * (np.dot(norm1.T, S__)) / n

            # dmu = prior times (error normalized by vector component variance)
            dmu_ = gammas[i] * (norm1 / np.diag(S_).reshape((-1, 1))**2) / n
            # print "dmu_", dmu_
            dmu.append(dmu_)
            # dlogsig_ = gammas[i] * (1.0 - np.dot(np.dot(S__, norm1), norm1.T)) / n
            # dlogsig_ = gammas[i] * (1.0 - (np.dot(norm1, norm1.T) * S__)) / n
            # print "    norm", norm1
            # print "    S_ = %s\nS_2 = %s\nS_-1 = %s" % ( S_, S_2, S__)
            # print "S_2 < 1e-6",
            if np.any(S_2 < 1e-6):
                S_2 += np.random.uniform(-1e-3, 1e-3, size = S_2.shape)
            mu_y_sig = np.dot(norm1, norm1.T) / S_2
            # print "    mu_y_sig", mu_y_sig
            dlogsig_ = gammas[i] * (1.0 - mu_y_sig) / n
            dlogsig_ = np.clip(dlogsig_, -1.0, 1.0)
            # strip symmetry
            # d1.append(dlogsig_.flatten()[:d]) # variances
            d1.append(np.diag(dlogsig_)) # variances
            d2.append(dlogsig_[np.triu_indices(d,  1)].flatten()) # covariances
            # print "    d1", d1, "d2", d2
            # dlogsig_ = dlogsig_.flatten()
            # print "dlogsig_", dlogsig_.shape
            # dlogsig.append(dlogsig_)
        d_diag = np.hstack(d1)
        d_triu = np.hstack(d2)
        # print "d_diag", d_diag
        # print "d_triu", d_triu
        dlogsig = np.hstack((d_diag, d_triu))
            
        dpiu = (pi - gammas) # / n
        # dpiu.append(dpiu_)
        # print "|dmu| = %f" % (np.linalg.norm(dmu))

        if self.cnt % 100 == 0:
            print("count[%d]" % self.cnt)
            # print "    loss = %f / %f / %d" % (self.loss/self.cnt, self.loss, self.cnt)
            print("    loss = %f / %d" % (self.loss, self.cnt))
            print("    dmu", np.linalg.norm(np.array(dmu).flatten()))
            print("    dlogisig", np.linalg.norm(dlogsig))
            print("    dpiu", np.linalg.norm(dpiu.flatten()))
            
        return np.hstack((np.array(dmu).flatten(), dlogsig, dpiu.flatten())).T
    
    def softmax(self, x):
        """LearningRules.softmax

        Softmax function for MDN sampling
        """
        # softmaxes the columns of x
        #z = x - np.max(x, axis=0, keepdims=True) # for safety
        e = np.exp(x)
        en = e / np.sum(e, axis=0, keepdims=True)
        return en

    def learnEH(
            self, target, r, pred, pred_lp, perf = None,
            perf_lp = 0.0, eta = 1e-4, mdltr_type = 'bin_elem',
            mdltr_thr = 0.0):
        """LearningRules.learnEH

        Exploratory Hebbian learning rule. This function computes the
        reward from the exploration result and modulates the Hebbian
        update with that reward, which can be binary or
        continuous. The exploratory part is happening in execute() by
        adding gaussian noise centered on current prediction.

        Arguments
         - target: target
         - r: network activation state
         - pred: prediction
         - pred_lp: prediction moving avg
         - perf: performance ~ error
         - perf_lp: performance moving avg
        """
        # # debugging
        # print "%s.learnEH args target = %s, r = %s, pred = %s, pred_lp = %s, perf = %s, perf_lp = %s, eta = %s" % (self.__class__.__name__, target.shape, r.shape, pred.shape, pred_lp.shape, perf.shape, perf_lp.shape, eta)
        # print "learnEH", mdltr_type, mdltr_thr

        # if no performance perf is supplied as an argument, the default neg squared error is used
        # FIXME: clean up because it implies the same to be used for perf_lp to be computed somewhere
        #        outside this function
        if perf is None:
            assert target is not None, "Either perf = %s or target = %s need to be not %s" % (perf, target, None)
            self.perf = -np.square(pred - target)
        else:
            self.perf = perf

        # print "reservoirs.learnEH perf", self.perf.shape
        # print "reservoirs.learnEH perf_lp", perf_lp.shape

        if mdltr_type == 'bin_elem':
            # binary clipped modulator element-wise
            mdltr = (np.clip(self.perf - perf_lp, 0, 1) > 0) * 1.0
            # print "mdltr", mdltr
        elif mdltr_type == 'bin_joint_any':
            # binary modulator with error threshold, update all output when at least one element improved
            mdltr = np.any((self.perf - perf_lp) > 0) * 1.0
        elif mdltr_type == 'bin_joint_all':
            # binary modulator with error threshold, update all output when at all elements improved
            mdltr = np.all((self.perf - perf_lp) > 0) * 1.0
        elif mdltr_type == 'cont_elem':
            # continuous modulator element-wise
            mdltr = (self.perf - perf_lp)
            mdltr /= np.sum(np.abs(mdltr))
            # print "reservoirs.learnEH cont_elem", self.perf.shape, perf_lp.shape, mdltr.shape
            
        # dont learn when performance is good enough
        if mdltr_thr > 0.0:
            # print "mdltr_thr kicking"
            # mdltr *= perf_lp <= -0.001 # -0.025
            # mdltr *= perf_lp <= -0.025
            mdltr *= perf_lp <= -mdltr_thr

        # print "reservoirs.learnEH mdltr = %s, pred = %s, pred_lp = %s" % (mdltr.shape, pred.shape, pred_lp.shape)

        # compute dw
        dw = eta * np.dot(r, np.transpose((pred - pred_lp) * mdltr))

        # return, end learnEH
        return dw
    
################################################################################
## Leaky integrator reservoir class
#
# Params: ...
class Reservoir(object):
    """Reservoir

    Leaky integrator single reservoir layer
    """
    def __init__(
            self, N = 100, p = 0.1, g = 1.2, alpha = 1.0, tau = 0.1,
            input_num = 1, output_num = 1, input_scale = 0.05,
            feedback_scale = 0.01, bias_scale = 0.,
            eta_init = 1e-5, theta = 1e-1, theta_state = 1e-2,
            nonlin_func = np.tanh,
            sparse = True,
            ip = False,
            coeff_a = 0.2,
            mtau = False,
            input_coupling = 'normal',
        ):
        """
        Reservoir.__init__

        Initialize a reservoir layer with

        :Arguments:
          N
            Model size [100]
          p
            Weight matrix density [0.1]
          g
            Spectral radius [1.2]

        TODO:
        - unify: wi_amp, input_scaling, res_input_scaling, ...
        """

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
            # self.tau = np.exp(np.random.uniform(-8, -0.5, (self.N, 1)))
            self.tau = np.exp(np.random.uniform(-5, -0.5, (self.N, 1)))
            # self.tau = np.exp(np.random.uniform(-5., -0.2, (self.N, 1)))
            print("self.tau", self.tau.T)
        else:
            self.tau = tau
        self.multiactivation = False # True
        self.alpha = alpha
        # scale gain to spectral radius lambda
        self.scale = 1.0/np.sqrt(self.p*self.N)
        # reservoir connection matrix
        # self.M = create_matrix_reservoir(self.N, self.p)
        self.M = create_matrix_sparse_random(self.N, self.N, self.p)

        normalize_spectral_radius(self.M, self.g)
        self.M = spa.csr_matrix(self.M)

        # inputs and input weight matrix
        self.input_num = input_num
        self.input_coupling = input_coupling
        self.wi_amp = input_scale
        self.init_wi()
        print("reservoirs.init wi = %s" % (self.wi.shape, ))

        # readout feedback term
        self.output_num = output_num
        self.wf_amp = feedback_scale
        # self.wf = np.random.uniform(-self.wf_amp, self.wf_amp, (self.N, self.output_num))
        # self.wf = np.random.normal(0, self.wf_amp, (self.N, self.output_num))
        # enable zero amplitude
        self.wf = np.random.normal(0, 1.0, (self.N, self.output_num)) * self.wf_amp
        # outputs and readout weight matrix
        self.wo = np.zeros((self.N, self.output_num));

        # initialize states to zeros
        self.u = np.zeros(shape=(self.input_num, 1))
        self.x = np.zeros(shape=(self.N, 1))
        self.r = np.zeros(shape=(self.N, 1))
        self.r_idx = list(range(self.N))
        np.random.shuffle(self.r_idx)
        # np.array([]).T
        self.r_nl = np.zeros(shape=(self.N, 1))
        self.r_rbf = np.zeros(shape=(self.N, 1))
        self.r_rbf_base = np.random.uniform(-1, 1, size = (self.N, 1))
        self.r_rbf_scale = np.exp(np.random.uniform(-1, 1, size = (self.N, 1)))
        self.z = np.zeros(shape=(self.output_num, 1))
        self.zn = np.zeros((self.output_num, 1))
        # initialize states randomly
        # self.init_states_random()
        # bias
        self.bias = np.random.uniform(-2., 2., size=self.x.shape)
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

    def rbf(self, x):
        return np.exp(-(x / 1.0)**2)

        
    # setters
    def set_theta(self, theta):
        """Set exploration noise amplitude (theta) and theta_amps"""
        self.theta = theta
        self.theta_amps = np.ones(shape=(self.output_num, 1)) * self.theta
        # print "res: theta_amps", self.theta_amps

    ############################################################
    # save network
    def save(self, filename=""):
        from pickle import Pickler
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
        from pickle import Unpickler
        print("reservoirs.py: loading %s" % filename)
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
    # initialize input weight matrix
    def init_wi(self):
        if hasattr(self, 'input_coupling'):
            if self.input_coupling in ['uniform', 'normal']:
                self.wi = res_input_matrix_random(
                    idim = self.input_num,
                    odim = self.N,
                    dist = self.input_coupling) * self.wi_amp
                
            elif self.input_coupling in ['sparse_uniform', 'sparse_normal']:
                self.wi = res_input_matrix_random_sparse(
                    idim = self.input_num,
                    odim = self.N,
                    density = 0.2, # default
                    dist = self.input_coupling) * self.wi_amp
            elif self.input_coupling in ['disjunct']:
                self.wi = res_input_matrix_disjunct_proj(
                    idim = self.input_num,
                    odim = self.N) * self.wi_amp
            elif self.input_coupling in ['identity']:
                self.init_wi_identity()
            elif self.input_coupling in ['ones']:
                self.init_wi_ones()
            else:
                print("Reservoir.init_wi: unknown input_coupling = '%s'" % (self.input_coupling, ))
                sys.exit(-1)
        else:
            self.wi = res_input_matrix_random(idim = self.input_num, odim = self.N, dist = 'normal') * self.wi_amp
            
    ############################################################
    # initialize input weights to I
    def init_wi_identity(self):
        # self.wo = np.eye((self.N, self.output_num))
        self.wi = np.eye((self.input_num)) # hm?

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

        if self.multiactivation:
            self.r_nl = self.nonlin_func(self.x) # + zeta_state        
            self.r_rbf = self.rbf((self.x - self.r_rbf_base) * self.r_rbf_scale)
            self.r_rbf = self.rbf((self.x - self.r_rbf_base) * self.r_rbf_scale)
            self.r = np.vstack((
                self.r_nl[self.r_idx[:self.N/2]],
                self.r_rbf[self.r_idx[self.N/2:]],
            ))
        else:
            self.r = self.nonlin_func(self.x) + zeta_state
        
        self.r += zeta_state
        
        # print "self.x", self.x.shape
        # print "self.r", self.r, self.r.shape
        # print "self.r_rbf", self.r_rbf, self.r_rbf.shape

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
        # return clean output, no noise
        return self.z

    ############################################################
    # learn: FORCE, EH, recursive regression (RLS)?
    def learnFORCE(self, target):
        """Reservoir.learnFORCE

        FORCE learning rule

        Reservoir class member
        """
        # update statistics
        k = np.dot(self.P, self.r)
        rPr = np.dot(self.r.T, k)
        c = 1.0/(1.0 + rPr)
        self.P = self.P - np.dot(k, (k.T*c))

        # compute error
        e = self.z - target
        # compute dw
        dw = -e.T * k * c
        # apply update
        self.wo += dw

    def learnRLSsetup(self,wo_init, P0_init):
        """Reservoir.learnRLSsetup

        RLS learning rule setup hook

        Reservoir class member
        """
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
        """Reservoir.learnRLS

        RLS learning rule

        Reservoir class member
        """
        # print "%s.learnRLS, target.shape = %s" % (self.__class__.__name__, target.shape)
        self.rls_E.update(self.r.T, target.T, self.theta_state)
        self.cnt += 1
        self.wo = self.rls_E.x

    def learnEH(self, target):
        """Reservoir.learnEH

        Exploratory Hebbian learning rule. This function computes the
        reward from the exploration result and modulates the Hebbian
        update with that reward, which can be binary or
        continuous. The exploratory part is happening in execute() by
        adding gaussian noise centered on current prediction.

        Reservoir class member
        """
        eta = self.eta_init # 0.0003

        self.perf = -np.square(self.zn - target)
        
        # binary modulator
        mdltr = (np.clip(self.perf - self.perf_lp, 0, 1) > 0) * 1.0
        # continuous modulator
        # vmdltr = (self.perf - self.perf_lp)
        # vmdltr /= np.sum(np.abs(vmdltr))
        # OR  modulator
        # mdltr = np.ones_like(self.zn) * np.clip(np.sum(mdltr), 0, 1)
        # AND modulator
        mdltr = np.ones_like(self.zn) * (np.sum(mdltr) >= 1) # FIXME: >=, ?
        # print "Reservoir.learnEH perf = %s, modulator = %s" % (self.perf, mdltr)

        # compute dw
        dw = eta * np.dot(self.r, np.transpose((self.zn - self.zn_lp) * mdltr))
        # update weights
        self.wo += dw
        # update performance prediction
        self.perf_lp = ((1 - self.coeff_a) * self.perf_lp) + (self.coeff_a * self.perf)

    def perfVar(self, x):
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
            self.wo[:,i] += dW[:,0]
            # self.wo[:,i] += np.reshape(dW, (self.N, ))

        self.perf_lp = ((1 - self.coeff_a) * self.perf_lp) + (self.coeff_a * self.perf)

    def learnPI(self, x):
        pass

    # def learnPCA_init(self, eta=1e-4):
    #     # self.ro_dim = 2
    #     eta_gha = eta
    #     self.gha = GHA(eta = eta_gha, ndims = self.N, pdims = self.output_num)
    #     self.gha.w *= 0.1
    #     self.wo = self.gha.w

    # def learnPCA(self):
    #     # resin
    #     # self.res.execute(resin)
    #     y_gha = self.gha.update(self.r)
    #     self.wo = self.gha.w
    #     self.z = y_gha
    #     # self.zn = self.z + np.random.normal(0, self.theta, size=(self.z.shape))
    #     self.zn = self.z + (np.random.normal(0, 1.0, size=(self.z.shape)) * self.theta)
    #     return self.z

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
            print(("sparse False (g = %f)" % self.g))
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
            print("M", self.M.shape)

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
        print("lae pre", lae)
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

        print("ip", self.ip)
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

# usage examples
def get_frequencies(numcomp = 1):
    f_step = 0.001
    f_start = 0.001
    f_numcomp = numcomp
    f_end   = (f_numcomp + 1) * f_step
    freqs = np.linspace(f_start, f_end, f_numcomp) #.reshape((1, -1))
    # freqs_full = np.tile(freqs, (1, outdim)).T
    return freqs

def get_data(elen, outdim, mode="MSO_s1"):
    """Get target data from a datasource"""

    # time / index
    t = np.linspace(0, int(elen), elen)

    d_sig_freqs = np.array([1.]) * 1.0
    f_step = 0.001
    f_start = 0.001
    if mode == "MSO_s1":
        # simple
        # d_sig_freqs = np.array([0.1, 0.2]) * 0.01
        f_numcomp = 1
        f_end   = (f_numcomp + 1) * f_step
        # d_sig_freqs = np.tile(np.linspace(f_start, f_end, f_numcomp), (outdim, f_numcomp))
    elif mode == "MSO_s2":
        # simple
        # d_sig_freqs = np.array([0.1, 0.2]) * 0.01
        # d_sig_freqs = np.array([0.13, 0.26]) * 0.1
        f_numcomp = 2
        f_end   = (f_numcomp + 1) * f_step
    elif mode == "MSO_s3":
        # simple
        # d_sig_freqs = np.array([0.1, 0.2,  0.3]) * 0.02
        # d_sig_freqs = np.array([0.13, 0.26]) * 0.1
        f_numcomp = 3
        f_end   = (f_numcomp + 1) * f_step
    elif mode == "MSO_s4":
        # simple
        # f(t)=(1.3/1.5)*sin(2*pi*t)+(1.3/3)*sin(4*pi*t)+(1.3/9)*sin(6*pi*t)+(1.3/3)*sin(8*pi*t)
        d_sig_freqs = np.array([0.1, 0.2,  0.3]) * 0.02
        # d_sig_freqs = np.array([0.13, 0.26]) * 0.1
        # 400 seconds learning
        t = t/1000.0
        # print "t", t
        data = (1.3/1.5) * np.sin(2*np.pi*t) + (1.3/3) * np.sin(4*np.pi*t) + (1.3/9) * np.sin(6*np.pi*t) + (1.3/3) * np.sin(8*np.pi*t)
        data = data.reshape((1, -1)) * 0.7
        # print "data.shape", data.shape
        return data
    elif mode == "MSO_c1":
        # d_sig_freqs = np.array([0.0085, 0.0174]) * 0.1
        f_numcomp = 2
        d_sig_freqs = np.array([0.0085, 0.011]) * 0.2
        # d_sig_freqs = np.array([0.13, 0.27, 0.53, 1.077]) * 0.3
        # d_sig_freqs = [1.002, 2.004, 3.006, 4.008]
        # d_sig_freqs = np.array([0.1, 0.2, 0.3, 0.4]) * 1.
        # d_sig_freqs = np.array([0.01, 1.01, 1.02, 1.03]) * 0.1
    elif mode == "MSO_c2":
        f_numcomp = 4
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
            print(f(tt))
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
        rate, data = wavfile.read(args.file)
        # rate, data = wavfile.read("drinksonus_mono_short.wav")
        offset = np.random.randint(0, data.shape[0] - elen)
        print("get_data: wav:", data.dtype, offset)
        data = data.astype(np.float)
        # data = data[offset:offset+elen].reshape((outdim, -1))
        data = data[offset:offset+elen].reshape((-1, outdim)).T
        data -= np.mean(data)
        data /= np.max(np.abs(data))
        # print data.shape
        # sys.exit()
        return data
    elif mode == "reg3t1":
        f_numcomp = 1
        f_end   = (f_numcomp + 1) * f_step
        # ds = np.zeros((outdim))
    elif mode == "reg_multimodal_1":
        # build a sinewave with superimposed 1/4 duty cycle pulse
        f_numcomp = 2
        # freqs = np.array([0.001])
        freqs = np.array([0.01])
        phases = np.random.uniform(0, 1, size=(f_numcomp, ))
        amps = np.random.uniform(0, 1, size=(f_numcomp, ))
        # amp normalization to sum 1
        amps /= np.sum(amps)
        sincomp = np.sin(2 * np.pi * (freqs * t.T + 0.0)) # * amps
        # freq_steps = int((2*np.pi)/freqs)
        freq_steps = int(1/freqs)
        print("freq_steps", freq_steps)
        pulse = (np.arange(t.shape[0]) % (2*freq_steps)) > (0.7 * freq_steps) # np.zeros_like(t)
        pulse = pulse * 1.0 - 0.0
        from scipy import signal
        b, a  = signal.butter(4, 0.5)
        pulse = signal.filtfilt(b, a, pulse)
        print("pulse", pulse)
        # pulse *= 1.0
        ds_real = (sincomp + pulse) * (1.0/f_numcomp)
        print("ds_real.shape", ds_real)
        return ds_real.reshape((outdim, -1))

    # generate waveform from frequencies array
    # d_sig_freqs_base = np.linspace(f_start, f_end, f_numcomp).reshape((1, -1))
    if mode == "reg3t1":
        # freqs = np.array([0.1, 0.21, 0.307, 0.417]) * 1e-2
        freqs = np.array([0.1]) * 1e-3
    elif mode in ["MSO_c1", "MSO_c2"]:
        freqs = d_sig_freqs
    else:
        freqs = get_frequencies(numcomp = f_numcomp)

    # print d_sig_freqs_base.shape
    # d_sig_freqs = np.tile(d_sig_freqs_base, (1, outdim)).T
    # print "d_sig_freqs shape = %s, data = %s" % (d_sig_freqs.shape, d_sig_freqs)
    # print (d_sig_freqs * t).shape
    # d_sig_a = np.zeros(shape=(elen, len(d_sig_freqs)))
    # for j in range(len(d_sig_freqs)):
    #     d_sig_a[:,j] = np.sin(2*np.pi*d_sig_freqs[j]*t)

    d_sig = np.zeros((outdim, elen))
    t = np.tile(t, (f_numcomp, 1))
    # print "d_sig.shape", d_sig.shape
    # print "freqs.shape", freqs.shape, t.shape
    for i in range(outdim):
        phases = np.random.uniform(0, 1, size=(f_numcomp, ))
        # freq variation for output components
        freqs += (np.random.randint(1, 3, size=(f_numcomp, )) * 0.001 * np.clip(i, 0, 1))
        # amp variation for output components
        amps = np.random.uniform(0, 1, size=(f_numcomp, ))
        # amp normalization to sum 1
        amps /= np.sum(amps)
        # compute sine waveforms from freqs, phase t and phase offset phases times amp
        sincomps = np.sin(2 * np.pi * (freqs * t.T + phases)) * amps
        # print "sincomps.shape", sincomps.shape
        d = np.sum(sincomps, axis = 1)
        # print "d.shape", d.shape
        d_sig[i] = d
    # print "d_sig.shape", d_sig.shape

    # plt.ioff()
    # plt.subplot(211)
    # plt.plot(d_sig_a)
    # plt.subplot(212)
    # plt.plot(d_sig_b.T)
    # plt.show()

    # d_sig = np.sum(d_sig_a, axis=1)/len(d_sig_freqs)
    # d_sig = d_sig.reshape(1, (len(d_sig)))
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
    plt.subplot(241)
    # plt.plot(u_, "k-", lw=0.3)
    plt.plot(u_ + [i * 10 for i in range(idim)], "k-", lw=0.3)
    plt.ylabel("Neuron input $x$")
    plt.subplot(242)
    plt.hist(u_[sl1], bins=50, normed=True, orientation="horizontal")
    plt.subplot(243)
    plt.hist(u_[sl2], bins=50, normed=True, orientation="horizontal")
    plt.subplot(244)
    plt.hist(u_[sl3], bins=50, normed=True, orientation="horizontal")
    plt.subplot(245)
    # plt.plot(u_ip_, "k-", lw=0.3)
    plt.plot(u_ip_, "k-", lw=0.3)
    plt.ylabel("Neuron output $y$")
    plt.subplot(246)
    plt.hist(u_ip_[sl1], bins=50, normed=True, orientation="horizontal")
    plt.subplot(247)
    plt.hist(u_ip_[sl2], bins=50, normed=True, orientation="horizontal")
    plt.subplot(248)
    plt.hist(u_ip_[sl3], bins=50, normed=True, orientation="horizontal")

    if saveplot:
        plt.gcf().set_size_inches((18,10))
        plt.gcf().savefig("reservoir_test_ip.pdf", dpi=300, bbox_inches="tight")
    plt.show()

class ReservoirPlot(object):
    def __init__(self, args):
        self.gs = gridspec.GridSpec(5, 1)
        self.fig = plt.figure()
        self.fig.suptitle("reservoirs.py mode = %s, target = %s, length = %d" % (args.mode, args.target, args.length))
        self.axs = []
        for plotrow in range(5):
            self.axs.append(self.fig.add_subplot(self.gs[plotrow]))
        self.fig.show()

        self.ressize = args.ressize

        # plotting params for res hidden activation random projection
        self.selsize = 20
        # self.rindex = np.random.randint(self.ressize, size=self.selsize)
        self.rindex = np.random.choice(list(range(self.ressize)), self.selsize, replace=False)

    def plot_data(self, args, data, incr = 1000, lr = None, testing = 100):
        (ds_real, out_t, out_t_mdn_sample, r_t, perf_t, loss_t, wo_t_norm, dw_t_norm) = data
        episode_len = args.length

        assert lr is not None

        for axidx in range(5):
            self.axs[axidx].clear()

        # mixcomp colors mu black, sigma blue, pi red
        cols = ["k", "b", "r"]

        # self.axs[0].clear()
        # backlog = 200
        # print "incr, ds_real.shape", incr, ds_real.shape
        if incr < (ds_real.shape[1] - 5):
            backlog = args.plot_interval
            self.axs[0].set_title("Target (%d-window)" % (backlog))
            self.axs[1].set_title("Target and output (%d-window)" % (backlog))
            pdata     = ds_real[:,(incr-backlog):incr].T
            pdata_mdn = out_t_mdn_sample[:,(incr-backlog):incr].T
            pdata_out = out_t[:,(incr-backlog):incr].T
        else:
            self.axs[0].set_title("Target")
            self.axs[1].set_title("Target and output")
            pdata     = ds_real.T
            pdata_mdn = out_t_mdn_sample.T
            pdata_out = out_t.T
            self.axs[1].axvspan(testing, episode_len, alpha=0.1)

        # axs[0].plot(pdata, "k-", lw=2.0, label="%d-dim tgt" % ds_real.shape[0])
        self.axs[0].plot(pdata, "g-", lw=2.0, label="tgt", alpha=0.5)
        self.axs[0].plot([0, pdata.shape[0]], [0, 0], "k-", lw=0.2)
        self.axs[0].legend(ncol=2, fontsize=8)

        self.axs[1].plot(pdata, "g-", lw=2.0, label="tgt", alpha=0.3)
        self.axs[1].plot([0, pdata.shape[0]], [0, 0], "k-", lw=0.2)

        if args.mode.endswith("mdn"):
            # the sample
            self.axs[1].plot(pdata_mdn, "c-", lw=0.5, label="out_", alpha=0.5)
            for k in range(3):
                pdata = pdata_out[:,(k*args.mixcomps):((k+1)*args.mixcomps)] + (k*2)
                pdata = np.clip(pdata, -10, 10)
                # pdata =     out_t[(k*mixcomps):((k+1)*mixcomps),:] + (k*2)
                # print "pdata[%d].shape = %s, %s" % (k, pdata.shape, out_t.shape)
                if k == 1:
                    pdata = np.exp(pdata)
                if k == 2:
                    # FIXME: fix this, softmax seems to be zero?
                    pdata = pdata # lr.softmax(pdata)
                self.axs[1].plot(pdata, "%s-" % (cols[k]), lw=0.25, label="out%d"%k, alpha=0.5)
        else:
            self.axs[1].plot(pdata_out, lw=0.25, label="out")

        self.axs[1].legend(ncol=2, fontsize=8)

        self.axs[2].set_title("reservoir traces")
        self.axs[2].plot(r_t.T[:,self.rindex], label="r", lw = 0.25, alpha = 0.5)

        self.axs[3].set_title("weight norm |W|")
        if args.mode.endswith("mdn"):
            for k in range(3):
                self.axs[3].plot(wo_t_norm[(k*args.mixcomps):((k+1)*args.mixcomps),:].T, "%s-" % cols[k], label="|W|", lw = 0.25, alpha = 0.5)
                self.axs[3].plot(dw_t_norm[(k*args.mixcomps):((k+1)*args.mixcomps),:].T - 1.0, "%s," % cols[k], label="|dW|", lw = 0.25, alpha = 0.5)
        else:
            self.axs[3].plot(wo_t_norm.T, label="|W|", lw = 0.25, alpha = 0.5)
            self.axs[3].plot(dw_t_norm.T, label="|dW|", lw = 0.25, alpha = 0.5)
        self.axs[3].legend(ncol=2, fontsize=8)

        self.axs[4].set_title("perf (-loss)")
        # if args.mode.endswith("mdn"):
        if args.mode.endswith("mdn"):
            self.axs[4].plot(loss_t.T, "g-", lw=0.5, label="loss", alpha = 0.5)
            for k in range(3):
                self.axs[4].plot(
                    np.clip(
                        perf_t[(k*args.mixcomps):((k+1)*args.mixcomps),:].T,
                        -1, 1),
                    "%s-" % (cols[k]), label="perf", alpha=0.5, lw = 0.25)
        else:
            self.axs[4].plot(perf_t.T, label="perf", alpha=0.5, lw = 0.25)
        self.axs[4].legend(ncol=2, fontsize=8)

        plt.draw()
        plt.pause(1e-9)

def test_mixtureMV(args):
    """test_mixtureMV

    Test mixtureMV() by sampling from dummy statistics mu, S, p
    """
    
    mixcomps = 3
    d = 2
    mu = np.random.uniform(-1, 1, size = (mixcomps, d))
    S = np.array([np.eye(d) * 0.01 for c in range(mixcomps)])
    for s in S:
        s += 0.05 * np.random.uniform(-1, 1, size = s.shape)
        s[np.tril_indices(d, -1)] = s[np.triu_indices(d, 1)]
    pi = np.ones((mixcomps, )) * 1.0/mixcomps

    lr = LearningRules(ndim_out = d, dim = d)
    lr.learnFORCEmdn_setup(mixcomps = mixcomps)
    
    fig = plt.figure()
    gs = gridspec.GridSpec(2, mixcomps)
    for g in gs:
        fig.add_subplot(g)

    ax = fig.axes[0]
    ax.plot(mu[:,0], mu[:,1], "ko", alpha = 0.5)
    # ax.set_xlim((-1.1, 1.1))
    # ax.set_ylim((-1.1, 1.1))


    ax = fig.axes[0] # (2 * mixcomps) + c]
    for i in range(1000):
        y = lr.mixtureMV(mu, S, pi)
        print("y", y)
        ax.plot([y[0]], [y[1]], "ro", alpha = 0.2, markersize = 3.0)

    ax.set_aspect(1)
        
    for c in range(mixcomps):
        ax = fig.axes[mixcomps + c]
        ax.imshow(S[c], cmap = plt.get_cmap('PiYG'))
        
    fig.show()
    plt.show()
    
    print("mu", mu.shape, "S", S.shape)
        
class ReservoirTest(object):

    modes = {"ol_rls": 0, "ol_force": 1, "ol_eh": 2, "ip": 3, "fwd": 4, "ol_pi": 5, "ol_force_mdn": 6, 'mixtureMV': 7}
    targets = {
        "MSO_s1": 0, "MSO_s2": 1, "MSO_s3": 2,
        "MSO_c1": 3, "MSO_c2": 4,
        "MG": 5,
        "wav": 6,
        "reg3t1": 7, "reg_multimodal_1": 8}

def save_wavfile(out_t, timestr):
    try:
        from scipy.io import wavfile
        # wav_out = (out_t.T * 32767).astype(np.int16)
        out_t /= np.max(np.abs(out_t))
        wav_out = (out_t.T * 32767).astype(np.int16)
        wavfile.write("../data/res_out_%s.wav" % (timestr), 44100, wav_out)
    except ImportError:
        print("ImportError for scipy.io.wavfile")

def main(args):
    print("mode", args.mode)
    if args.mode == 'ip':
        test_ip(args)
        sys.exit(0)

    elif args.mode == 'mixtureMV':
        test_mixtureMV(args)
        sys.exit(0)

    # seed the run
    np.random.seed(args.seed)
    episode_len = args.length
    washout_ratio = 0.1
    washout = min(washout_ratio * episode_len, 1000)
    learning_ratio = args.learning_ratio
    testing = learning_ratio * episode_len
    insize = args.ndim_in
    outsize = args.ndim_out
    mixcomps = args.mixcomps
    out_t_mdn_sample = np.zeros(shape=(outsize, episode_len))
    if args.mode == "ol_force_mdn":
        outsize_ = outsize * mixcomps * 3
    else:
        outsize_ = outsize
    # for feedback
    percent_factor = 1./(episode_len/100.)
    alpha = 1.0
    input_scale = 1.2 # 2.0
    feedback_scale = args.scale_feedback
    w_bias = 0.5
    g = 1.5
    # tau = 0.1
    # tau = 0.075
    tau = 0.05
    # tau = 0.025
    # tau = 0.01
    # tau = 0.005
    # tau = 0.0025
    # tau = 0.001
    if args.mode.endswith("mdn"):
        # alpha = 100.0
        # input_scale = 1.0
        # g = 0.01
        # tau = 1.0

        # alpha = 200.0
        # alpha = 100.0
        # alpha = 50.0
        alpha = 10.0
        # alpha = 4.0
        # alpha = 1.0
        # alpha = 0.2
        input_scale = 1.0
        w_bias = 0.0
        g = 1.5
        # tau = 0.2
        # tau = 0.1
        tau = 0.05
        # tau = 0.025
        # tau = 0.01
    # eta_init_ = 1e-4
    eta_init_ = 5e-4
    # eta_init_ = 1e-3
    # eta_init_ = 2e-3
    # eta_init_ = 4e-3

    # get training data
    ds_real = get_data(episode_len+1, outsize, args.target)
    print("ds_real.shape", ds_real.shape)

    # compute effective tapping for these timeseries problems
    # non AR regression setup
    # mdn
    # entropy
    # loop over different hyperparams (ressize, tau, eta, ...)
    for i in [args.ressize]:
        out_t = np.zeros(shape=(outsize_, episode_len))
        out_t_n = np.zeros(shape=(outsize_, episode_len))
        r_t = np.zeros(shape=(i, episode_len))
        perf_t = np.zeros(shape=(outsize_, episode_len))
        loss_t = np.zeros(shape=(1, episode_len))
        wo_t = np.zeros(shape=(i, outsize_, episode_len))
        wo_t_norm = np.zeros((outsize_, episode_len))
        dw_t_norm = np.zeros((outsize_, episode_len))
        # test save and restore
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        filename = "reservoir-%s.bin" % timestamp
        # res = Reservoir(N=10)
        # for larger reservoirs: decrease eta, increase input scale
        # res = Reservoir(N = i, input_num = insize, output_num = outsize, g = 1.5, tau = 0.01,
        #                 feedback_scale=0.1, input_scale=2.0, bias_scale=0.2, eta_init=1e-3,
        #                 theta = 1e-1, theta_state = 1e-2, coeff_a = 0.2, mtau=args.multitau)
        # tau was 0.01
        res = Reservoir(
            N = i, input_num = insize, output_num = outsize_, g = g, tau = tau,
            alpha = alpha, feedback_scale = feedback_scale, input_scale = input_scale,
            bias_scale = w_bias, eta_init = eta_init_, theta = 2.5e-1, theta_state = 1e-3,
            coeff_a = 0.2, mtau=args.multitau
        )

        # res.save(filename)
        # print ("first", res.M)

        # res = Reservoir2(N=i, input_num=insize, output_num=outsize, g = 1.5, tau = 0.1,
        #                 feedback_scale=0.0, input_scale=1.0, bias_scale=0.0, eta_init=1e-3,
        #                 sparse=True, coeff_a = 0.2,
        #                 ip=False, theta = 1e-5, theta_state = 1e-2)

        # print ("second", res.M)

        # res.load(filename)
        # print "third", res.M

        # print "res size", res.get_reservoir_size()
        # print "res matrix", res.get_reservoir_matrix()
        # print "res input matrix", res.get_input_matrix()
        # print "res feedback matrix", res.get_feedback_matrix()
        # print "res readout matrix", res.get_readout_matrix()

        # output weight init
        if args.mode.endswith("mdn"):
            # res.init_wo_random(0, 1e-1)
            # sigmas = [1e-6] * mixcomps + [1e-6] * mixcomps + [1e-6] * mixcomps
            # good with deltamdn mixcomps 6 and teacherforcing and eta ~ 1e-4
            sigmas = [1e-4] * mixcomps + [1e-4] * mixcomps + [1e-4] * mixcomps
            # sigmas = [1e-3] * mixcomps + [1e-3] * mixcomps + [1e-3] * mixcomps
            # good with rlsmdn mixcomps 3 and teacherforcing and alpha = 10
            # sigmas = [1e-2] * mixcomps + [1e-2] * mixcomps + [1e-2] * mixcomps
            # sigmas = [1e-1] * mixcomps + [1e-1] * mixcomps + [1e-1] * mixcomps
            # print "sigmas xxx", sigmas
            res.init_wo_random(np.zeros((1, outsize_)), np.array(sigmas))

        # learning rule module
        lr = LearningRules(ndim_out = outsize_)

        # do some setup
        if ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_rls"]:
            if rlspy is None:
                print("Dont have rlspy, exiting")
                sys.exit()
            # initialize rlspy
            # res.learnRLSsetup(None, None)
            lr.learnRLSsetup(x0 = res.wo, P0 = np.eye(res.N))
        if args.mode == "ol_force_mdn":
            # initialize FORCEmdn
            lr.learnFORCEmdn_setup(mixcomps = mixcomps)
            
            lr.learnRLSsetup(x0 = res.wo, P0 = np.eye(res.N))
            
        elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_pi"]:
            print("ol_pi in progress, exiting")
            sys.exit(1)

        # interactive plotting
        plt.ion()
        # init reservoir plotting
        rp = ReservoirPlot(args) # bad naming, i is instance of res model size

        # loop over timesteps
        for j in range(episode_len):

            # teacher forcing
            if args.teacher_testing or (args.teacher_forcing and j < testing):
                inputs = ds_real[:,[j-1]] # .reshape((insize, 1))
            elif args.feedback_only:
                inputs = np.zeros((args.ndim_in, 1))
            else:
                if args.mode.endswith("mdn"):
                    inputs = out_t_mdn_sample[:,[j-1]]
                else:
                    inputs = out_t[:,[j-1]]
            # target
            target = ds_real[:,[j]]

            # save network state
            res_x_ = res.x.copy()
            res_r_ = res.r.copy()

            # update network, log activations
            res.execute(inputs)
            out_t[:,[j]]   = res.z
            out_t_n[:,[j]] = res.zn
            r_t[:,[j]] = res.r

            # learning window
            if j < testing and j > washout:
                if ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_rls"]:
                    # res.learnRLS(target)
                    dw = lr.learnRLS(target = target, r = res.r)
                    res.wo += dw
                    res.perf = lr.rls_estimator.y.T
                    for k in range(outsize_):
                        dw_t_norm[k,j] = LA.norm(dw[:,k])
                        
                elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_force"]:
                    # reservoir class built-in learning rule
                    # res.learnFORCE(target)

                    # modular learning rule (ugly call)
                    (res.P, k, c) = lr.learnFORCE_update_P(res.P, res.r)
                    dw = lr.learnFORCE(target, res.P, k, c, res.r, res.z, 0)
                    res.wo += dw
                    for k in range(outsize_):
                        dw_t_norm[k,j] = LA.norm(dw[:,k])
                    res.perf = lr.e # mdn_loss_val

                elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_force_mdn"]:
                    # reservoir class built-in learning rule
                    # res.learnFORCE(target)


                    # modular learning rule (ugly call)
                    # (res.P, k, c) = lr.learnFORCE_update_P(res.P, res.r)
                    # dw = lr.learnFORCEmdn(target, res.P, k, c, res.r, res.z, 0, inputs)
                    # noise for reg_multimodal_tp1 5e-1
                    # dw = lr.learnRLS(target = target, r = res.r, noise = args.eta, z = res.z, x = inputs)
                    dw = lr.learnDeltamdn(
                        target = target, P = res.P, k = None, c = None,
                        r = res.r, z = res.z, channel = 0, x = inputs)
                    # res.wo += (1e-1 * dw)
                    # print "dw.shape", dw
                    # when using delta rule
                    leta = (1/np.log((j*0.1)+2)) * 1e-4
                    leta = 1.0
                    res.wo = res.wo + (leta * dw)
                    for k in range(outsize_):
                        # wo_t_norm[k,j] = LA.norm(wo_t[:,k,j])
                        dw_t_norm[k,j] = LA.norm(dw[:,k])

                    loss_t[0,j] = lr.loss
                    # print "mdn_loss = %s" % mdn_loss_val
                    res.perf = lr.e # mdn_loss_val

                elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_eh"]:
                    res.eta_init = eta_init_ / (1 + ((j - washout)/20000.0))
                    # print "res.eta_init", res.eta_init
                    # res.learnEH(target)
                    
                    dw = lr.learnEH(target = target, r = res.r, pred = res.zn, pred_lp = res.zn_lp, perf = res.perf, perf_lp = res.perf_lp, eta = res.eta_init)
                    res.wo += dw
                    res.perf = lr.perf
                    res.perf_lp = ((1 - res.coeff_a) * res.perf_lp) + (res.coeff_a * res.perf)

                elif ReservoirTest.modes[args.mode] == ReservoirTest.modes["ol_pi"]:
                    if j > 100:
                        res.perfVar(out_t)
                        res.learnEHPerf()

                if True: # if recompute_output
                    # recompute activations, only when learning
                    # restore states
                    res.x = res_x_
                    res.r = res_r_
                    # update activations with corrected ouput
                    res.execute(inputs)
                    out_t[:,[j]] =   res.z
                    out_t_n[:,[j]] = res.zn
                    r_t[:,[j]] = res.r

            if args.mode.endswith("mdn"):
                out_t_mdn_sample[:,[j]] = lr.mixture(res.z[:mixcomps,0], np.exp(res.z[mixcomps:(2*mixcomps),0]), lr.softmax(res.z[(2*mixcomps):,0]))

            perf_t[:,[j]] = res.perf
            wo_t[:,:,j] = res.wo
            for k in range(outsize_):
                wo_t_norm[k,j] = LA.norm(wo_t[:,k,j])

            # print "state", res.x
            if j > washout and (j+1) % 1000 == 0:
                # # old style
                # print "eta", res.eta_init
                # print("%d of %d" % (j,  episode_len))
                # new style
                progress = j * percent_factor
                # print '\r[{0}] {1}%'.format('#'*(progress/10), progress)
                sys.stdout.write( '\r[{0}] {1}%'.format('#'*int(progress), progress))
                sys.stdout.flush()

            if j > washout and j < (args.length - 1) and (j+1) % args.plot_interval == 0:
                rpdata = (ds_real, out_t, out_t_mdn_sample, r_t, perf_t, loss_t, wo_t_norm, dw_t_norm)
                rp.plot_data(args = args, data = rpdata, incr = j, lr = lr, testing = testing)

    # FIXME: if testing
    # compute performance for testing: MSE, freq match without phase

    # final plot
    plt.ioff()

    rpdata = (ds_real, out_t, out_t_mdn_sample, r_t, perf_t, loss_t, wo_t_norm, dw_t_norm)
    rp.plot_data(args = args, data = rpdata, incr = j, lr = lr, testing = testing)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    # rp.fig.savefig("../data/res_plot_%s.pdf" % (timestr), dpi=300, bbox_inches="tight")
    plt.show()

    if args.mode.endswith("mdn"):
        save_wavfile(out_t_mdn_sample, timestr)
    else:
        save_wavfile(out_t, timestr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reservoir library, call main for testing: python reservoirs.py -t MSO_s1 -m ol_eh -rs 500 -l 100000")
    parser.add_argument("-l", "--length", help="Episode length [30000]", default=30000, type=int)
    parser.add_argument("-lr", "--learning_ratio", help="Ratio of learning to episode len [0.8]", default=0.8, type=float)
    parser.add_argument("-e", "--eta", help="Learning rate eta [1e-3]", default=1e-3, type=float)
    parser.add_argument("-m", "--mode", help="Mode [ol_rls], one of " + str(list(ReservoirTest.modes.keys())), default = "ol_rls")
    parser.add_argument("-mt", "--multitau", dest="multitau", action="store_true",
                        help="Use multiple random time constants in reservoir, doesn't seem to work so well with EH [False]")
    parser.add_argument("-mc", "--mixcomps", help="Number of mixture components for mixture network [3]", type=int, default=3)
    parser.add_argument("-ndo", "--ndim_out", help="Number of output dimensions [1]", default=1, type=int)
    parser.add_argument("-ndi", "--ndim_in",  help="Number of input dimensions [1]",  default=1, type=int)
    parser.add_argument("-pi", "--plot_interval",  help="Time step interval at which to update plot [1000]",  default=1000, type=int)
    parser.add_argument("-rs", "--ressize", help="Reservoir (hidden layer) size [300]", default=300, type=int)
    parser.add_argument("-s", "--seed", help="RNG seed [101]", default=101, type=int)
    parser.add_argument("-sf", "--scale_feedback", help="Global feedback strength (auto-regressive) [0.0]", default=0.0, type=float)
    parser.add_argument("-t", "--target", help="Target [MSO_s1], one of " + str(list(ReservoirTest.targets.keys())), default = "MSO_s1")
    parser.add_argument("-f", "--file", help="File to use as data input", dest="file", default="data/notype_mono_short.wav")
    parser.add_argument("-tf", "--teacher_forcing", dest="teacher_forcing", action="store_true",
                        help="Use teacher forcing during training [False]")
    parser.add_argument("-tt", "--teacher_testing", dest="teacher_testing", action="store_true",
                        help="Use teacher forcing during testing [False]")
    parser.add_argument("-fo", "--feedback_only", dest="feedback_only", action="store_true",
                        help="Use only the global output feedback [False]")

    args = parser.parse_args()

    main(args)
