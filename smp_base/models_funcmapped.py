"""extracted from smp_graphs.block_model

try to separate / remove one level of hierarchy in this stack:

    `modelblock -> model -> [init_, step_]_modelfunc -> [sklearn, smpmodel, reservoir, ...]`

to have 

    `modelblock -> smpmodel -> [sklearn, reservoir, ...]`

TODO
- separate ref and mref tasks [wip]
- convert these to smpmodels and put them into smp_base
- expansions: poly, res, ...
- unsup: incsfa, ...
"""

from os import path as ospath
from functools import partial
import joblib

import numpy as np

# mdp
from mdp.nodes import PolynomialExpansionNode

# scikit-learn is sklearn
from sklearn import linear_model, kernel_ridge

# smp_base
from smp_base.common import get_module_logger
from smp_base.measures import meas as measf
from smp_base.measures_probes import meas_linear_regression_probe
from smp_base.models import iir_fo
from smp_base.models_actinf  import smpKNN, smpGMM, smpIGMM, smpHebbianSOM
from smp_base.models_learners import smpSHL, learnerReward, Eligibility
from smp_base.models_reservoirs import Reservoir, LearningRules
from smp_base.models_reservoirs import res_input_matrix_from_conf, create_matrix_sparse_from_conf
from smp_base.models_reservoirs import res_input_matrix_random_sparse, res_input_matrix_disjunct_proj
from smp_base.models_selforg import HK

# .. TODO:: use smpi dynamic importer
try:
    from smp_base.models_actinf import smpOTLModel, smpSOESGP, smpSTORKGP
    HAVE_SOESGP = True
except ImportError as e:
    print("couldn't import online GP models", e)
    HAVE_SOESGP = False

# .. TODO:: merge with embedding code and move to smp_base
from smp_base.codeops import code_compile_and_run, get_input

from smp_graphs.tapping import tap_tupoff, tap, tap_flat, tap_unflat
from smp_graphs.tapping import tap_imol, tap_stack

# from smp_graphs.tapping import tap_imol_fwd, tap_imol_fwd_time, tap_imol_fwd_modality
# from smp_graphs.tapping import tap_imol_inv_old, tap_imol_inv_time, tap_imol_inv_modality

# from logging import DEBUG as LOGLEVEL
# logger = get_module_logger(modulename = 'funcs_models', loglevel = LOGLEVEL - 0)

# # joblib caching
# from joblib import Memory
# location = './cachedir'
# memory = Memory(location, verbose=0)

# def array_fix(a = None, col = True):
#     """smp_graphs.common.array_fix

#     Fix arrays once and forever.
#     - if scalar or list convert to array
#     - if one-dimensional add single second dimension / axis (atleast_2d)
#     - if no col-type shape, transpose
#     """
#     assert a is not None, "array_fix needs argument a"
#     if type(a) is list:
#         a = np.array(a)
#     if len(a.shape) == 1:
#         a = np.atleast_2d(a)

#     if a.shape[0] > a.shape[1]:
#         if col:
#             return a
#         else:
#             return a.T
#     else:
#         if col:
#             return a.T
#         else:
#             return a

# # test for learning modulation
# def isindevelphase(cnt, model, phase='n_train'):
#     # logger.debug('isindevelphase model = %s, phase = %s', dir(model), phase)
#     if hasattr(model, phase):
#         r0 = getattr(model, phase)[0]
#         r1 = getattr(model, phase)[1]
#     else:
#         r0 = model[phase][0]
#         r1 = model[phase][1]
#     return r0 <= cnt < r1

# def istraining(cnt, model):
#     return isindevelphase(cnt, model, 'n_train')
        
# def iswashouting(cnt, model):
#     # return model['n_washout'][0] <= cnt < model['n_washout'][1]
#     return isindevelphase(cnt, model, 'n_washout')
        
# def istesting(cnt, model):
#     # return model['n_test'][0] <= cnt < model['n_test'][1]
#     return isindevelphase(cnt, model, 'n_test')
        
# # def init_identity(ref):
# #     return None
                        
# # def step_identity(ref, ins = {}):
# #     return None

# def init_musig(ref, mref, conf, mconf):
#     """model.expansion.mu-sigma

#     Expand the input into the components: mu, sigma

#     .. note::

#     Single time step incremental iir estimator.
#     """
#     params = conf['params']
#     # params = mconf
#     mref.a1 = mconf['a1']
#     for ink, inv in list(params['inputs'].items()):
#         # print "init_musig", inv
#         for outk in ["mu", "sig"]:
#             # outk_full = "%s/%s_%s" % (mref.modelkey, ink, outk)
#             outk_full = "%s_%s" % (ink, outk)
#             params['outputs'][outk_full] = {'shape': inv['shape']}
#             setattr(mref, outk_full, np.zeros(inv['shape']))
#     logger.debug('musig.outputs = %s' % (params['outputs']))
#     # return None

# def step_musig(ref, mref, *args, **kwargs):
#     # for ink, inv in ref.inputs.items():
#     for ink, inv in list(mref.mconf['inputs'].items()):
#         for outk_ in ["mu", "sig"]:
#             outk = "%s_%s" % (ink, outk_)
#             outv_ = getattr(mref, outk)

#             if outk.endswith("mu"):
#                 setattr(mref, outk, mref.a1 * outv_ + (1 - mref.a1) * inv['val'])
#             elif outk.endswith("sig"):
#                 setattr(mref, outk, mref.a1 * outv_ + (1 - mref.a1) * np.sqrt(np.square(inv['val'] - getattr(mref, ink + "_mu"))))
                
#             # logger.debug('musig.output %s = %s' % (outk, getattr(mref, outk)))

# def init_msr(ref, mref, conf, mconf):
#     """model.expansion.mu-sigma-residual

#     Expand the input into the components: mu, sigma, residual
#     """
#     params = conf['params']
#     # params = mconf
#     # expansion spec: mu, sigma, residual
#     axis = -1
#     mref.expansion = dict(list(zip(['m', 's', 'r'], [
#         lambda x: np.mean(x, axis=axis, keepdims=True) + np.zeros_like(x),
#         lambda x: np.std(x, axis=axis, keepdims=True) + np.zeros_like(x),
#         lambda x: x - np.mean(x, axis=axis, keepdims=True),
#     ])))
    
#     # for each input
#     for ink, inv in list(params['inputs'].items()):
#         # compute msr
#         for outk in mref.expansion:
#             # outk_full = "%s/%s_%s" % (mref.modelkey, ink, outk)
#             outk_full = "%s_%s" % (ink, outk)
#             params['outputs'][outk_full] = {'shape': inv['shape']}
#             setattr(mref, outk_full, np.zeros(inv['shape']))
            
#     logger.debug('msr.outputs = %s' % (params['outputs']))
#     # return None

# def step_msr(ref, mref, *args, **kwargs):
#     # for ink, inv in ref.inputs.items():
#     for ink, inv in list(mref.mconf['inputs'].items()):
#         for outk_ in mref.expansion:
#             outk = "%s_%s" % (ink, outk_)
#             setattr(mref, outk, mref.expansion[outk_](inv['val']))
#             # logger.debug('musig.output %s = %s' % (outk, getattr(mref, outk)))

# # model func: reservoir expansion
# def init_res(ref, mref, conf, mconf):
#     """model.expansion.res

#     Use an open-loop reservoir aka single hidden layer network (shln)
#     expansion. In the feed forward case this yields a kernel expansion
#     like polynomial, sin/cos, rbf, or exponential. In the recurrent
#     case this yield a kernel expansion with local memory with some
#     interesting corner cases like a linear delay bank, tdnn,
#     embedding, tapping, conv kernel.
#     """
#     params = conf['params']
#     mref.oversampling = mconf['oversampling']
    
#     #  create input matrices
#     if 'restype' not in mconf:
#         wi = res_input_matrix_random_sparse(mconf['input_num'], mconf['N'], density = 0.2) * mconf['input_scale']
#         wres = None
#         if 'g' not in mconf: mconf['g'] = 0.99
#         if 'tau' not in mconf: mconf['tau'] = 0.1
#         if 'nonlin_func' not in mconf: mconf['nonlin_func'] = np.tanh
#         if 'theta_state' not in mconf: mconf['theta_state'] = 1e-2
#     else:
#         wi = res_input_matrix_from_conf(mconf)
#         wres = create_matrix_sparse_from_conf(mconf)
#         logger.debug('wi = %s', wi)
#         logger.debug('wres = %s', wres)
#         # tdnn config
#         if mconf['restype'] in ['tdnn', 'lpfb']:
#             mconf['g'] = 1.0
#             mconf['tau'] = 1.0
#             mconf['nonlin_func'] = lambda x: x
#             mconf['theta_state'] = 0.0
        
#     # create reservoir
#     mref.res = Reservoir(
#         N = mconf['N'],
#         input_num = mconf['input_num'],
#         output_num = mconf['output_num'],
#         input_scale = mconf['input_scale'], # 0.03,
#         bias_scale = mconf['bias_scale'], # 0.0,
#         feedback_scale = 0.0,
#         g = mconf['g'],
#         tau = mconf['tau'],
#         nonlin_func = mconf['nonlin_func'],
#     )
#     logger.debug('mref.res.M = %s', mref.res.M.shape)
#     # set input weights
#     mref.res.wi = wi
#     # set reservoir weights
#     if wres is not None:
#         mref.res.M = wres

#     logger.debug('mref.res.M = %s', mref.res.M.shape)
#     params['outputs']['x_res'] = {'shape': (mconf['N'], 1)}

# def step_res(ref, mref, *args, **kwargs):
#     """model.expansion.res.step

#     Step the shln expansion.
#     """
#     # print ref.inputs['x']['val'].shape
#     for i in range(mref.oversampling):
#         mref.res.execute(ref.inputs['x']['val'])
#     # ref._debug('step_res  mref.res.r = %s' % (mref.res.r))
#     setattr(mref, 'x_res', mref.res.r)

# # TODO: model func: attractor neural network expansion (ANN-x)
# #       the idea is:
# #       a) a forward autonomous reservoir which computes out its eigendynamics given constant input (re shriki)
# #       b) 'conventional' ANN with random or structured design which is excited and computed to internal
# #          convergence for given constant input (repose-cell network, bayes filter, ...)
# # init_annx(ref, conf, mconf):
# #     pass

# # step_annx(ref):
# #     pass
    
# # model func: polynomial expansion using mdp
# def init_polyexp(ref, mref, conf, mconf):
#     params = conf['params']
#     if 'degree' not in mconf:
#         mconf['degree'] = 3
#     mref.polyexpnode = PolynomialExpansionNode(mconf['degree'])
#     mref.polyexpnode.execute(np.zeros((1, params['inputs']['x']['shape'][0])))
#     # logger.debug('init_polyexp mref.polyexpnode = %s' % (dir(mref.polyexpnode), ))
#     # params['outputs']['polyexp'] = {'shape': params['inputs']['x']['shape']}
#     params['outputs']['y'] = {'shape': (mref.polyexpnode.get_output_dim(), 1)} # ??? magic number computed from expansion size with degree 3 FIXME
#     logger.debug('init_polyexp outputs = %s' % (params['outputs'], ))

# def step_polyexp(ref, mref, *args, **kwargs):
#     setattr(mref, 'y', mref.polyexpnode.execute(ref.inputs['x']['val'].T).T)

# # model func: function generator model
# def init_function_generator(ref, mref, conf, mconf):
#     params = conf['params']
#     # function
#     func_ = mconf['func']
#     if type(func_) is str:
#         # eval string
#         ref.func = eval(func_)
#     else:
#         ref.func = func_
#     ref.funcargs = {}

#     # outputs
#     for outk, outv in list(params['outputs'].items()):
#         lo = -np.ones(( outv['shape'] ))
#         hi = np.ones(( outv['shape'] ))
#         setattr(ref, outk, np.zeros(outv['shape']))
#         logger.debug('init_function_generator %s = %s' % (outk, getattr(ref, outk)))

# def step_function_generator(ref, mref, *args, **kwargs):
#     # logger.debug('step_function_generator[%d]' % (ref.cnt))
#     if hasattr(ref, 'rate'):
#         if (ref.cnt % ref.rate) not in ref.blockphase: return

#     # # copy inputs to function arguments
#     # for ink in ref.inputs:
#     #     ref.funcargs[ink] = ref.get_input(ink)

#     # evaluate function
#     funcval = ref.func(ref.inputs)
    
#     for outk, outv in list(ref.outputs.items()):
#         setattr(ref, outk, funcval)
#         # logger.debug('step_function_generator outk %s = %s' % (outk, getattr(ref, outk)))
        
#         # if ref.cnt % (ref.rate * 1) == 0:
#         #     # print ref.__class__.__name__, ref.id, "lo, hi, out shapes", lo.shape, hi.shape, outv['shape']
#         #     setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
#         # else:
#         #     setattr(ref, outk, np.random.uniform(-1e-3, 1e-3, size = outv['shape']))
        
#         # setattr(ref, outk, np.random.choice([-1.0, 1.0], size = outv['shape']))
        
#         # np.random.uniform(lo, hi, size = outv['shape']))
#         # print "%s-%s[%d]model.step_function_generator %s = %s" % (
#         #     ref.cname, ref.id, ref.cnt, outk, getattr(ref, outk))
#         # print "block_models.py: function_generator_step %s = %s" % (outk, getattr(ref, outk))

    
# def init_random_lookup(ref, mref, conf, mconf):
#     """random_lookup: init and setup

#     Random lookup is a model based on a transfer function.

#     .. note:: 

#         Only tested on 1-d data. Correspondence / hysteresis style transfer not implemented.

#     Arguments:
#      - ref(Block2): ModelBlock2 reference
#      - conf(dict): ModelBlock2's configuration dict
#      - mconf(dict): this model's configuration dict with items
#       - l_a(float, [0, 1]): linear comp amplitude 
#       - d_a(float, [0, 1]): gaussian comp amplitude 
#       - d_s(float, [0, 1]): gaussian comp sigma
#       - s_a(float, [0, 1]): noise comp amplitude
#       - s_f(float, [0, 1]): noise comp color (beta in 1/f noise)
#       - e(float, [0, 1]): external entropy / independent noise amplitude
#     """
#     from scipy.stats import norm
#     # setup
#     params = conf['params']
#     if 'numelem' not in mconf:
#         mconf['numelem'] = 1001
#     inshape = params['inputs']['x']['shape']
#     # logger.debug("init_random_lookup: inshape = %s", inshape)
    
#     # linear ramp
#     mref.h_lin = np.linspace(-1.0, 1.0, mconf['numelem']).reshape(1, mconf['numelem'])

#     # three factors controlling information distance
#     #  1. contraction / expansion: transfer function y = h(x)
#     #  2. smoothness: noisy transfer function y = h(x + noise(f, amp))
#     #  3. external entropy: y = h(x) + E
    
#     # information distance parameters
#     mref.l_a = mconf['l_a'] # coef linear
#     mref.d_a = mconf['d_a'] # coef  gaussian
#     mref.d_s = mconf['d_s'] # sigma gaussian
#     mref.s_a = mconf['s_a'] # coef noise
#     mref.s_f = mconf['s_f'] # beta noise
#     mref.e = mconf['e']

#     # contraction / expansion: transforming uniform to gaussian, d to var
#     # var = 0.25**2
#     var = mref.d_s**2
#     mref.h_gauss = np.exp(-0.5 * (0.0 - mref.h_lin)**2/var)
#     # invert the density for lookup
#     # mref.h_gauss_inv = mref.h_gauss
#     mref.h_gauss_inv = np.max(mref.h_gauss) - mref.h_gauss
#     mref.h_gauss_inv_int = np.cumsum(mref.h_gauss_inv).reshape(mref.h_lin.shape)
#     # mref.h_gauss_inv_int = np.cumsum(mref.h_gauss_inv).reshape(mref.h_lin.shape)
#     # mref.h_gauss_inv_int = np.exp(-0.5 * (0.0 - mref.h_lin)**2/var)
#     # print "mref.h_gauss_inv_int", mref.h_gauss_inv_int.shape, mref.h_gauss_inv_int
#     # mref.h_lin_gauss_inv = mref.h_lin + 1.0)/2.0
#     mref.h_lin_gauss_inv = np.linspace(norm.cdf(-1.0, loc = 0, scale = mref.d_s), norm.cdf(1.0, loc = 0, scale = mref.d_s), mconf['numelem'])
#     mref.h_gauss_inv_int = np.clip(norm.ppf(mref.h_lin_gauss_inv, loc = 0.0, scale = mref.d_s), -1 * 1, 1 * 1)

#     # print("mref.h_gauss_inv_int = %s" % (mref.h_gauss_inv_int,))
#     # logger.debug("mref.h_gauss_inv_int = %s" % (mref.h_gauss_inv_int.shape,))
#     # logger.debug("mref.h_gauss_inv_int = %s" % (mref.h_gauss_inv_int,))

#     mref.h_gauss_inv_int -= np.mean(mref.h_gauss_inv_int)
#     mref.h_gauss_inv_int /= np.max(np.abs(mref.h_gauss_inv_int))
#     # mref.h_gauss_inv_int *= min(1.0, mref.d_s)

#     # additive noise on base h
#     # mref.h_noise = np.random.uniform(-1, 1, (mconf['numelem'], )) # .reshape(1, mconf['numelem'])
#     # mref.h_noise = np.exp(-0.5 * (0.0 - mref.h_noise)**2)
#     from smp_base.gennoise import Noise
#     noise = Noise.oneoverfnoise(N = mconf['numelem'], beta = mref.s_f, normalize = True)
#     # is complex?
#     if hasattr(noise[1], 'real'):
#         mref.h_noise = noise[1].real.reshape(mref.h_lin.shape)
#     else:
#         mref.h_noise = noise[1].reshape(mref.h_lin.shape)
#     # logger.debug("    model init_random_lookup mref.h_noise = %s/%s, %s" % (mref.h_noise.real, mref.h_noise.imag, mref.h_noise.shape))
    
#     # noise: color (1/f)
#     # mref.
#     # mref.h = (1 - mref.s_a) * mref.h_gauss_inv_int + mref.s_a * mref.h_noise
#     # mref.h *= 0.5
#     # mref.h = mref.h_gauss_inv_int
#     # components are all normalized to [-1, 1]
#     mref.h = mref.l_a * mref.h_lin + mref.d_a * mref.h_gauss_inv_int + mref.s_a * mref.h_noise
#     if mref.e > 0.0:
#         mref.h += np.random.normal(0, 1.0, mref.h.shape) * mref.e
    
#     # mref.h /= np.max(np.abs(mref.h))
#     mref.x = np.zeros((inshape))
#     mref.y = np.zeros_like(mref.x)
#     # logger.debug("    model init_random_lookup mref.x = %s, mref.y = %s, mref.h = %s, mref.h_lin = %s, mref.h_noise = %s" % (
#     #     mref.x.shape, mref.y.shape, mref.h.shape, mref.h_lin.shape, mref.h_noise.dtype))
#     # do_random_lookup(ref)
    
# def step_random_lookup(ref, mref, *args, **kwargs):
#     # logger.debug('    step_random_lookup pre do_: args = %s, kwargs = %s', args, kwargs)
#     if len(args) > 0:
#         mref.x = args[0]
#         # logger.debug('    step_random_lookup pre do_: mref.x = %s from args', mref.x)
#         return do_random_lookup(ref, mref)
#     else:
#         mref.x = np.clip(ref.inputs['x']['val'], -1, 1)
#         # logger.debug('    step_random_lookup pre do_: mref.x = %s', mref.x)
#         do_random_lookup(ref, mref)        

# # model func: lookup table expansion with parametric map randomness
# def do_random_lookup(ref, mref):
#     """random_lookup: perform the lookup

#     Compute bin index $i$ of input $x$ in linear range and output $y$
#     as the transfer function's value at $i$.
#     """
#     # logger.debug("%sdo_random_lookup[%s] mref.x = %s, mref.h_lin = %s" % ('    ', ref.cnt, mref.x.shape, mref.h_lin.shape)) # , mref.h.shape
#     mref.x_idx = np.searchsorted(mref.h_lin[0], mref.x, side = 'right')[0]
#     # logger.debug("%sdo_random_lookup[%s] mref.x_idx = %s", '    ', ref.cnt, mref.x_idx) # , mref.h.shape
#     mref.y = mref.h[...,mref.x_idx - 1]
#     # logger.debug("%sdo_random_lookup[%s] mref.y = %s" % ('    ', ref.cnt, mref.y))
#     mref.y = mref.y * (1 - mref.e) + np.random.normal(0, 1.0, mref.y.shape) * mref.e
#     # logger.debug("%sdo_random_lookup[%s] mref.y = %s" % ('    ', ref.cnt, mref.y))
#     return mref.y

# # model func: random_uniform model
# def init_random_uniform(ref, mref, conf, mconf):
#     params = conf['params']
#     for outk, outv in list(params['outputs'].items()):
#         lo = -np.ones(( outv['shape'] ))
#         hi = np.ones(( outv['shape'] ))
#         setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
#         # setattr(ref, outk, np.ones(outv['shape']))
#         # print "block_models.py: random_uniform_init %s = %s" % (outk, getattr(ref, outk))

# def step_random_uniform(ref, mref, *args, **kwargs):
#     if hasattr(ref, 'rate'):
#         if (ref.cnt % ref.rate) not in ref.blockphase: return
            
#     lo = ref.inputs['lo']['val'] # .T
#     hi = ref.inputs['hi']['val'] # .T
#     for outk, outv in list(ref.outputs.items()):
#         if ref.cnt % (ref.rate * 1) == 0:
#             # logger.debug('lo = %s, hi = %s, out = %s', lo.shape, hi.shape, outv['shape'])
#             assert lo.shape == hi.shape, "lo/hi shapes need to agree %s/%s" % (lo.shape, hi.shape)
#             assert lo.shape == outv['shape'], "lo/hi and out shapes need to agree %s/%s" % (lo.shape, outv['shape'])
#             setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
#         else:
#             setattr(ref, outk, np.random.uniform(-1e-3, 1e-3, size = outv['shape']))
        
#         # setattr(ref, outk, np.random.choice([-1.0, 1.0], size = outv['shape']))
        
#         # np.random.uniform(lo, hi, size = outv['shape']))
#         # print "%s-%s[%d]model.step_random_uniform %s = %s" % (
#         #     ref.cname, ref.id, ref.cnt, outk, getattr(ref, outk))
        
#         # logger.debug('step_random_uniform %s = %s', outk, getattr(ref, outk))

# # model func: random_uniform_pi_2 model
# def init_random_uniform_pi_2(ref, conf, mconf):
#     params = conf['params']
#     for outk, outv in list(params['outputs'].items()):
#         lo = -np.ones(( outv['shape'] ))
#         hi = np.ones(( outv['shape'] ))
#         setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
#         # setattr(ref, outk, np.ones(outv['shape']))
#         # print "block_models.py: random_uniform_pi_2_init %s = %s" % (outk, getattr(ref, outk))
#     ref.prerr_ = np.ones((ref.prerr.shape[0], 40)) * 1.0

# def step_random_uniform_pi_2(ref):
#     if hasattr(ref, 'rate'):
#         if (ref.cnt % ref.rate) not in ref.blockphase: return
            
#     lo = ref.inputs['lo']['val'] # .T
#     hi = ref.inputs['hi']['val'] # .T
#     meas_l0 = ref.inputs['meas_l0']['val'][...,[-1]]
#     for outk, outv in list(ref.outputs.items()):
#         if ref.cnt % (ref.rate * 1) == 0:
#             # pred = np.random.normal(0, 0.05, size = outv['shape'])
#             # pred[1,0] = pred[0,0]
#             # pred = np.random.uniform(lo, hi, size = outv['shape'])
#             # pred[1,0] = pred[0,0] - 0.5
#             # print meas_l0.shape
#             prerr = ref.pre - meas_l0
#             np.roll(ref.prerr_, -1, axis = 1)
#             ref.prerr_[...,[-1]] = prerr.copy()
#             pred = ref.pre
#             print("uniform_pi_2 small error", prerr, np.mean(np.abs(ref.prerr_)))
#             if np.mean(np.abs(ref.prerr_)) < 0.1:
#                 print("uniform_pi_2 small error sampling")
#                 pred = np.random.normal(meas_l0, scale = np.mean(np.abs(ref.prerr_))) # , size = outv['shape']) # * 1e-3
#             else:
#                 # pred = np.random.normal(meas_l0, scale = 0.001) # , size = outv['shape']) # * 1e-3
#                 if ref.cnt % (ref.rate * 200) == 0:
#                     pred = np.random.normal(meas_l0, scale = 0.001) # , size = outv['shape']) # * 1e-3
                
            
#             # pred = np.zeros(outv['shape'])
#             setattr(ref, outk, pred)
#             print("step_random_uniform_pi_2 ref.outk", getattr(ref, outk))
#         else:
#             setattr(ref, outk, np.random.uniform(-1e-3, 1e-3, size = outv['shape']))
        
#         # setattr(ref, outk, np.random.choice([-1.0, 1.0], size = outv['shape']))
        
#         # np.random.uniform(lo, hi, size = outv['shape']))
#         # print "%s-%s[%d]model.step_random_uniform_pi_2 %s = %s" % (
#         #     ref.cname, ref.id, ref.cnt, outk, getattr(ref, outk))
#         # print "block_models.py: random_uniform_pi_2_step %s = %s" % (outk, getattr(ref, outk))

# # model func: agent budget: energy,
# def init_budget(ref, mref, conf, mconf):
#     params = conf['params']
#     mref.credit = np.ones((1, 1)) * params['credit']
#     mref.credit_ = mref.credit.copy()
#     mref.resets = np.ones((1, 1))

# def step_budget(ref, mref, *args, **kwargs):
#     if hasattr(ref, 'rate'):
#         if (ref.cnt % ref.rate) not in ref.blockphase: return
    
#     mdltr = ref.inputs['s0']['val'] # .T
#     mdltr_ref = ref.inputs['s0_ref']['val']
#     # refk = ref.outputs.keys()[0]
#     # print "refk", refk, "mdltr_ref", mdltr_ref
#     d_raw = mdltr - mdltr_ref
#     # print "refk", refk, "mdltr_ref", mdltr_ref, "d_raw", d_raw
#     mdltr_ = np.sqrt(np.sum(np.square(d_raw)))
    
#     if ref.cnt % (ref.rate * 1) == 0 and mdltr_ < ref.goalsize:
#         # print ref.__class__.__name__, ref.id, "lo, hi, out shapes", lo.shape, hi.shape, outv['shape']
#         # setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
#         mref.credit = mref.credit_.copy()
#         mref.resets = mdltr.copy()
#     else:
#         mref.credit -= 1
#         mref.resets = np.zeros_like(mdltr)
#         # if ref.credit
            
    
# # model func: random_uniform_modulated model
# def init_random_uniform_modulated(ref, mref, conf, mconf):
#     params = conf['params']
#     # for outk, outv in params['outputs'].items():
    
#     outk = 'pre'
#     outv = params['outputs'][outk]
#     lo = -np.ones(( outv['shape'] )) * 1e-3
#     hi = np.ones(( outv['shape'] )) * 1e-3
#     setattr(mref, outk, np.random.uniform(lo, hi, size = outv['shape']))
    
#     outk_1 = 'd_pre'
#     outv = params['outputs'][outk_1]
#     setattr(mref, outk_1, getattr(mref, outk))
 
#     # ref.credit = np.ones((1, 1)) * params['credit']
#     # ref.credit_ = ref.credit.copy()
#     mref.goalsize = params['goalsize']
#     # print "ref.credit", ref.credit
#     # setattr(ref, outk, np.ones(outv['shape']))
#     # print "block_models.py: random_uniform_modulated_init %s = %s" % (outk, getattr(ref, outk))

# def step_random_uniform_modulated(ref, mref, *args, **kwargs):
#     if hasattr(ref, 'rate'):
#         if (ref.cnt % ref.rate) not in ref.blockphase: return

#     # if ref.credit <= 0:
#     #     return
            
#     lo = ref.inputs['lo']['val'] # .T
#     hi = ref.inputs['hi']['val'] # .T
#     mdltr = ref.inputs['mdltr']['val'] # .T
#     refk = list(ref.outputs.keys())[0]
#     mdltr_ref = getattr(mref, 'pre')
#     # print "refk", refk, "mdltr_ref", mdltr_ref
#     d_raw = mdltr - mdltr_ref
#     # print "refk", refk, "mdltr_ref", mdltr_ref, "d_raw", d_raw
#     mdltr_ = np.sqrt(np.sum(np.square(d_raw)))
#     # print "mdltr", mdltr_
#     # for outk, outv in ref.outputs.items():
#     outk = 'pre'
#     outv = ref.outputs[outk]
#     if ref.cnt % (ref.rate * 1) == 0 and mdltr_ < mref.goalsize:
#         # print ref.__class__.__name__, ref.id, "lo, hi, out shapes", lo.shape, hi.shape, outv['shape']
#         setattr(mref, outk, np.random.uniform(lo, hi, size = outv['shape']))
#         # ref.credit = ref.credit_.copy()
#         setattr(mref, 'd_pre', getattr(mref, outk))
#     else:
#         setattr(mref, 'd_pre', np.zeros_like(getattr(mref, outk)))
        
#     # else:
#     #     ref.credit -= 1
#     #     # if ref.credit

# # model func: alternating_sign model
# def init_alternating_sign(ref, mref, conf, mconf):
#     params = conf['params']
#     for outk, outv in list(params['outputs'].items()):
#         lo = -np.ones(( outv['shape'] ))
#         hi = np.ones(( outv['shape'] ))
#         # setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
#         setattr(ref, outk, np.ones(outv['shape']))
#         # print "block_models.py: alternating_sign_init %s = %s" % (outk, getattr(ref, outk))

# def step_alternating_sign(ref, mref, *args, **kwargs):
#     if hasattr(ref, 'rate'):
#         if (ref.cnt % ref.rate) not in ref.blockphase: return
            
#     lo = ref.inputs['lo']['val'] # .T
#     hi = ref.inputs['hi']['val'] # .T
#     for outk, outv in list(ref.outputs.items()):
#         # setattr(ref, outk, np.random.uniform(lo, hi, size = outv['shape']))
        
#         # setattr(ref, outk, np.random.choice([-1.0, 1.0], size = outv['shape']))
        
#         if np.sum(np.abs(getattr(ref, outk))) == 0.0:
#             setattr(ref, outk, np.ones(outv['shape']))
#         setattr(ref, outk, getattr(ref, outk) * -1.0)
        
#         # np.random.uniform(lo, hi, size = outv['shape']))
#         # print "%s-%s[%d]model.step_alternating_sign %s = %s" % (
#         #     ref.cname, ref.id, ref.cnt, outk, getattr(ref, outk))
#         # print "block_models.py: alternating_sign_step %s = %s" % (outk, getattr(ref, outk))

# smpmodel_defaults = {'algo': 'knn', 'idim': 1, 'odim': 1}
        
# # used by: actinf, imol, homeokinesis, e2p, eh (FIXME: rename reward)
# def init_smpModel(ref, mref, conf, mconf):
#     """init_smpModel

#     Initialize an smp model for use in an agent self-exploration and
#     learning model.

#     Model interface:

#     Init with one dictionary parameter holding the configuration
#     composed from smpModel.defaults and model configuration in the
#     BlockModel params.

#     init(conf = mconf)

#     Fit the model to some supervised training data input X, target Y
#     or unsupervised training data input X, target = None.

#     fit(X, Y)

#     Compute a prediction of the model given an input X.

#     Y_ = predict(X)

#     TODO
#     - dim1: number of variables
#     - dim2: number of representatives of single variable (e.g. mean coding, mixture coding, ...)
#     """
#     # check conf, set defaults
#     for required in ['algo', 'idim', 'odim']:
#         if required not in mconf:
#             mconf[required] = smpmodel_defaults[required]
            
#     # shortcut handles
#     algo = mconf['algo']
#     idim = mconf['idim']
#     odim = mconf['odim']

#     if not HAVE_SOESGP:
#         algo = "knn"
#         logger.warning('soesgp/storkgp not available, defaulting to knn')
            
#     if algo == "knn":
#         # mdl = KNeighborsRegressor(n_neighbors=5)
#         # mdl = smpKNN(idim, odim)
#         mdl = smpKNN(conf = mconf)
#     elif algo == "gmm":
#         mdl = smpGMM(conf = mconf)
#     elif algo == "igmm":
#         mdl = smpIGMM(conf = mconf)
#     elif algo == "hebbsom":
#         # mconf.update({'numepisodes': 1, 'mapsize_e': 140, 'mapsize_p': 60, 'som_lr': 1e-1, 'visualize': False})
#         mconf.update({'numepisodes': 1, 'mapsize_e': 40, 'mapsize_p': 100, 'som_lr': 1e-0, 'som_nhs': 0.05, 'visualize': False})
#         print("mconf", mconf)
#         mdl = smpHebbianSOM(conf = mconf)
#         # mdl = smpHebbianSOM(idim, odim, numepisodes = 1, mapsize_e = 1000, mapsize_p = 100, som_lr = 1e-1)
#     elif algo == "soesgp":
#         print("soesgp conf", mconf)
#         mdl = smpSOESGP(conf = mconf)
#     elif algo == "storkgp":
#         mdl = smpSTORKGP(conf = mconf)
#     elif algo in ['resrls', 'res_eh']:
#         if algo == 'resrls':
#             # mconf['lrname'] = 'RLS'
#             pass
#         # only copy unset fields from the source
#         # mconf.update(smpSHL.defaults)
#         # mconf.update({'numepisodes': 1, 'mapsize_e': 140, 'mapsize_p': 60, 'som_lr': 1e-1, 'visualize': False})
#         # mconf.update({'idim': idim, 'odim': odim})
#         mdl = smpSHL(conf = mconf)
#     elif algo == 'copy':
#         targetid = mconf['copyid']
#         # # debugging
#         # print "topgraph", ref.top.nxgraph.nodes()
#         # print "topgraph.node[0]['block_'].nxgraph", ref.top.nxgraph.node[0]['block_'].nxgraph.nodes()
#         # for n in ref.top.nxgraph.node[0]['block_'].nxgraph.nodes():
#         #     print "    node.id = %s, graph = %s" % (
#         #         ref.top.nxgraph.node[0]['block_'].nxgraph.node[n]['block_'].id,
#         #         ref.top.nxgraph.node[0]['block_'].nxgraph.node[n]['block_'].nxgraph.nodes(), )
            
#         targetnode = nxgraph_node_by_id_recursive(ref.top.nxgraph, targetid)
#         print("targetid", targetid, "targetnode", targetnode)
#         if len(targetnode) > 0:
#             # print "    targetnode id = %d, node = %s" % (
#             #     targetnode[0][0],
#             #     targetnode[0][1].node[targetnode[0][0]])
#             # copy node
#             clone = {}
#             tnode = targetnode[0][1].node[targetnode[0][0]]
#         mdl = tnode['block_'].mdl
#     elif algo == 'homeokinesis':
#         mdl = HK(conf = mconf)
#     else:
#         print("unknown model algorithm %s, exiting" % (algo, ))
#         # import sys
#         # sys.exit(1)
#         mdl = None

#     assert mdl is not None, "Model (algo = %s) shouldn't be None, check your config" % (algo,)
        
#     return mdl

# ################################################################################
# # tapping, uh ah
# def tapping_SM(ref, mode = 'm1'):
#     """block_models.tapping_SM

#     Tap the incoming sensorimotor data stream as specified by each input's lag configuration
    
#     # FIXME: rewrite in general form (ref, invariable) -> (tapped invariable)
    
#     # maxlag == windowsize
#     # individual lags as array
#     """
    
#     # current goal[t] prediction descending from layer above
#     if 'blk_mode' in ref.inputs and ref.inputs['blk_mode']['val'][0,0] == 2.0:
#         # that's a wild HACK for switching the top down goal input of the current predictor
#         ref.pre_l1_inkey = 'e2p_l1'
#     else:
#         ref.pre_l1_inkey = 'pre_l1'
        
#     ############################################################
#     # instantaneous inputs: the full input buffer as specified by the minlag-maxlag range
#     pre_l1   = ref.inputs[ref.pre_l1_inkey]['val']
#     # measurement[t] at current layer input
#     meas_l0 = ref.inputs['meas_l0']['val']
#     # prediction[t-1] at current layer input
#     pre_l0   = ref.inputs['pre_l0']['val']   
#     # prediction error[t-1] at current layer input
#     prerr_l0 = ref.inputs['prerr_l0']['val']

#     ############################################################
#     # tapped inputs: a buffer containing only selected (receptive field, kernel, ...) dimensions and times
#     # get lag spec: None (lag = 1), int d (lag = d), array a (lag = a)
#     pre_l1_tap_spec = ref.inputs[ref.pre_l1_inkey]['lag']
#     # print "pre_l1_tap_spec", pre_l1_tap_spec
#     pre_l1_tap_full = ref.inputs[ref.pre_l1_inkey]['val'][...,pre_l1_tap_spec]
#     # print "pre_l1_tap_full", pre_l1_tap_full.shape
#     pre_l1_tap_flat = pre_l1_tap_full.reshape((-1, 1))

#     # target
#     pre_l1_tap_full_target = ref.inputs[ref.pre_l1_inkey]['val'][...,list(range(-ref.laglen_future - 1, -1))]
#     pre_l1_tap_flat_target = pre_l1_tap_full_target.reshape((-1, 1))

#     # meas
#     meas_l0_tap_spec = ref.inputs['meas_l0']['lag']
#     meas_l0_tap_full = ref.inputs['meas_l0']['val'][...,meas_l0_tap_spec]
#     meas_l0_tap_flat = meas_l0_tap_full.reshape((ref.odim, 1))
#     meas_l0_tap_full_input = ref.inputs['meas_l0']['val'][...,list(range(-ref.laglen_past, 0))]
#     meas_l0_tap_flat_input = meas_l0_tap_full_input.reshape((-1, 1))

#     # pre_l0
#     pre_l0_tap_spec = ref.inputs['pre_l0']['lag']
#     pre_l0_tap_full = ref.inputs['pre_l0']['val'][...,pre_l0_tap_spec]
#     pre_l0_tap_flat = pre_l0_tap_full.reshape((-1, 1))

#     prerr_l0_tap_spec = ref.inputs['prerr_l0']['lag']
#     # print "prerr_l0_tap_spec", prerr_l0_tap_spec
#     prerr_l0_tap_full = ref.inputs['prerr_l0']['val'][...,prerr_l0_tap_spec]
#     prerr_l0_tap_flat = prerr_l0_tap_full.reshape((-1, 1))
    
#     # print "meas", meas_l0[...,[-1]], "prel1", pre_l1[...,[pre_l1_tap_spec[-1]]]
    
#     # compute prediction error PE with respect to top level prediction (goal)
#     # momentary PE
#     prerr_l0_  = meas_l0[...,[-1]] - pre_l1[...,[pre_l1_tap_spec[-1]]]
#     # embedding PE
#     prerr_l0__ = meas_l0_tap_flat_input - pre_l1_tap_flat # meas_l0[...,[-1]] - pre_l1[...,[-lag]]
#     prerr_l0___ = meas_l0_tap_flat - pre_l1_tap_flat_target # meas_l0[...,[-1]] - pre_l1[...,[-lag]]
#     # prerr_l0__ = (meas_l0_tap_full - pre_l1_tap_full[...,[-1]]).reshape((-1, 1))
    
#     # FIXME: future > 1, shift target block across the now line completely and predict entire future segment
#     # X__ = np.vstack((pre_l1[...,[-lag]], prerr_l0[...,[-(lag-1)]]))
    
#     # return (pre_l1_tap_flat, pre_l0_tap_flat, meas_l0_tap_flat, prerr_l0_tap_flat, prerr_l0_, X, Y, prerr_l0__)
#     return (pre_l1_tap_flat, pre_l0_tap_flat, meas_l0_tap_flat, prerr_l0_tap_flat, prerr_l0_, prerr_l0__, prerr_l0___)

# def tapping_XY(ref, pre_l1_tap_flat, pre_l0_tap_flat, prerr_l0_tap_flat, prerr_l0__, mode = 'm1'):
#     """block_models.tapping_XY

#     Tap data from the sensorimotor data stream and build a supervised
#     training set of inputs X and targets Y suitable for machine
#     learning algorithms.
#     """
#     # print "tapping pre_l1", pre_l1_tap_flat.shape, prerr_l0_tap_flat.shape, ref.idim
#     # print "tapping reshape", pre_l1_tap.reshape((ref.idim/2, 1)), prerr_l0_tap.reshape((ref.idim/2, 1))

#     # wtf?
#     # orig tap at past+1 because recurrent
#     tmp = ref.inputs['pre_l0']['val'][...,ref.inputs['pre_l0']['lag']]
#     # truncate to length of future tap?
#     tmp_ = tmp[...,ref.inputs['meas_l0']['lag']].reshape((-1, 1))
    
#     if ref.type == 'm1' or ref.type == 'm3':
#         X = np.vstack((pre_l1_tap_flat, prerr_l0_tap_flat))
#         # compute the target for the  forward model from the embedding PE
#         # Y = (pre_l0_tap_flat - (prerr_l0__ * ref.eta)) # .reshape((ref.odim, 1)) # pre_l0[...,[-lag]] - (prerr_l0_ * ref.eta) #
#         Y = (tmp_ - (prerr_l0__ * ref.eta)) # .reshape((ref.odim, 1)) # pre_l0[...,[-lag]] - (prerr_l0_ * ref.eta) #
#     elif ref.type == 'm2':
#         X = np.vstack((prerr_l0_tap_flat, ))
#         Y = -prerr_l0__ * ref.eta # .reshape((ref.odim, 1)) # pre_l0[...,[-lag]] - (prerr_l0_ * ref.eta) #
#     elif ref.type == 'eh':
#         return (None, None)
#     else:
#         return (None, None)
#     # print "X", X.shape
    
#     # ref.mdl.fit(X__.T, ref.y_.T) # ref.X_[-lag]
    
#     return (X, Y)

# def tapping_X(ref, pre_l1_tap_flat, prerr_l0__):
#     """block_models.tapping_X

#     Tap data from the sensorimotor data stream and build an
#     unsupervised training set of inputs X suitable for machine
#     learning algorithms.
#     """
#     # print prerr_l0__.shape
#     if ref.type == 'm1' or ref.type == 'm3':
#         X = np.vstack((pre_l1_tap_flat, prerr_l0__))
#     elif ref.type == 'm2':
#         X = np.vstack((prerr_l0__, ))

#     return X

# def tapping_EH(
#         ref, pre_l1_tap_flat, pre_l0_tap_flat, meas_l0_tap_flat,
#         prerr_l0_tap_flat, prerr_l0_, prerr_l0__):
#     """block_models.tapping_EH

#     Tap data from the sensorimotor data stream and build a reward
#     modulated training set of inputs X and targets Y suitable for RL.
#     """
#     # print "tapping pre_l1", pre_l1_tap_flat.shape, prerr_l0_tap_flat.shape, ref.idim
#     # print "tapping reshape", pre_l1_tap.reshape((ref.idim/2, 1)), prerr_l0_tap.reshape((ref.idim/2, 1))
#     if ref.type == 'eh':
#         X = np.vstack((pre_l0_tap_flat, prerr_l0_tap_flat, meas_l0_tap_flat))
#         # compute the target for the forward model from the embedding PE
#         Y = (pre_l0_tap_flat - (prerr_l0__ * ref.eta))

#     else:
#         return (None, None)
    
#     return (X, Y)

# # def tapping_EH2():
    
# #     # current goal[t] prediction descending from layer above
# #     if ref.inputs.has_key('blk_mode') and ref.inputs['blk_mode']['val'][0,0] == 2.0:
# #         # that's a wild HACK for switching the top down goal input of the current predictor
# #         ref.pre_l1_inkey = 'e2p_l1'
# #     else:
# #         ref.pre_l1_inkey = 'pre_l1'
        
# #     ############################################################
# #     # instantaneous inputs: the full input buffer as specified by the minlag-maxlag range
# #     pre_l1   = ref.inputs[ref.pre_l1_inkey]['val']
# #     # measurement[t] at current layer input
# #     meas_l0 = ref.inputs['meas_l0']['val']
# #     # prediction[t-1] at current layer input
# #     pre_l0   = ref.inputs['pre_l0']['val']   
# #     # prediction error[t-1] at current layer input
# #     prerr_l0 = ref.inputs['prerr_l0']['val']

# #     ############################################################
# #     # tapped inputs: a buffer containing only selected (receptive field, kernel, ...) dimensions and times
# #     # get lag spec: None (lag = 1), int d (lag = d), array a (lag = a)
# #     pre_l1_tap_spec = ref.inputs[ref.pre_l1_inkey]['lag']
# #     # print "pre_l1_tap_spec", pre_l1_tap_spec
# #     pre_l1_tap_full = ref.inputs[ref.pre_l1_inkey]['val'][...,pre_l1_tap_spec]
# #     # print "pre_l1_tap_full", pre_l1_tap_full
# #     pre_l1_tap_flat = pre_l1_tap_full.reshape((ref.odim, 1))

# #     meas_l0_tap_spec = ref.inputs['meas_l0']['lag']
# #     meas_l0_tap_full = ref.inputs['meas_l0']['val'][...,meas_l0_tap_spec]
# #     meas_l0_tap_flat = meas_l0_tap_full.reshape((ref.odim, 1))

# #     pre_l0_tap_spec = ref.inputs['pre_l0']['lag']
# #     pre_l0_tap_full = ref.inputs['pre_l0']['val'][...,pre_l0_tap_spec]
# #     pre_l0_tap_flat = pre_l0_tap_full.reshape((ref.odim, 1))

# #     prerr_l0_tap_spec = ref.inputs['prerr_l0']['lag']
# #     # print "prerr_l0_tap_spec", prerr_l0_tap_spec
# #     prerr_l0_tap_full = ref.inputs['prerr_l0']['val'][...,prerr_l0_tap_spec]
# #     prerr_l0_tap_flat = prerr_l0_tap_full.reshape((ref.odim, 1))
    
# #     # print "meas", meas_l0[...,[-1]], "prel1", pre_l1[...,[pre_l1_tap_spec[-1]]]
    
# #     # compute prediction error PE with respect to top level prediction (goal)
# #     # momentary PE
# #     prerr_l0_  = meas_l0[...,[-1]] - pre_l1[...,[pre_l1_tap_spec[-1]]]
# #     # embedding PE
# #     prerr_l0__ = meas_l0_tap_flat - pre_l1_tap_flat # meas_l0[...,[-1]] - pre_l1[...,[-lag]]
    
# #     # FIXME: future > 1, shift target block across the now line completely and predict entire future segment
# #     # X__ = np.vstack((pre_l1[...,[-lag]], prerr_l0[...,[-(lag-1)]]))
    
# #     # return (pre_l1_tap_flat, pre_l0_tap_flat, meas_l0_tap_flat, prerr_l0_tap_flat, prerr_l0_, X, Y, prerr_l0__)
# #     return (pre_l1_tap_flat, pre_l0_tap_flat, meas_l0_tap_flat, prerr_l0_tap_flat, prerr_l0_, prerr_l0__)

# ################################################################################
# # active inference model
# # model func: actinf_m2
# def init_actinf(ref, mref, conf, mconf):
#     """init_actinf

#     Initialize active inference model
#     """
#     # params = conf['params']
#     # hi = 1
#     # for outk, outv in params['outputs'].items():
#     #     setattr(ref, outk, np.random.uniform(-hi, hi, size = outv['shape']))
#     ref.pre_l1_inkey = 'pre_l1'
#     ref.X_  = np.zeros((mconf['idim'], 1))
#     ref.y_  = np.zeros((mconf['odim'], 1))
#     ref.laglen  = mconf['laglen']
#     ref.lag_past  = mconf['lag_past']
#     ref.lag_future  = mconf['lag_future']
#     ref.lag_off = ref.lag_future[1] - ref.lag_past[1]

#     ref.laglen_past = ref.lag_past[1] - ref.lag_past[0]
#     ref.laglen_future = ref.lag_future[1] - ref.lag_future[0]
#     ref.pre_l1_tm1 = np.zeros((mconf['idim']/2/ref.laglen_past, 1))
#     ref.pre_l1_tm2 = np.zeros((mconf['idim']/2/ref.laglen_past, 1))

#     # reservoir extras
#     mconf.update({'memory': ref.laglen})
    
#     if mconf['type'] == 'actinf_m1':
#         ref.type = 'm1'
#     elif mconf['type'] == 'actinf_m2':
#         ref.type = 'm2'
#     elif mconf['type'] == 'actinf_m3':
#         ref.type = 'm3'

#     if mconf['type'].startswith('actinf'):
#         ref.tapping_SM = partial(tapping_SM, mode = ref.type)
#         ref.tapping_XY = partial(tapping_XY, mode = ref.type)
#         ref.tapping_X = partial(tapping_X)
        
#     # goal statistics
#     ref.dgoal_fit_ = np.linalg.norm(ref.pre_l1_tm1 - ref.pre_l1_tm2)
#     ref.dgoal_ = np.linalg.norm(-ref.pre_l1_tm1)

#     #  initialize the learner
#     ref.mdl = init_smpModel(ref, mref, conf, mconf)
    
# def step_actinf(ref, mref, *args, **kwargs):
#     """step_actinf

#     compute one step of active inference algorithm
#     """
#     # get prediction taps
#     (pre_l1, pre_l0, meas_l0, prerr_l0, prerr_l0_, prerr_l0__, prerr_l0___) = ref.tapping_SM(ref)
#     # get model fit taps
#     (X, Y) = ref.tapping_XY(ref, pre_l1, pre_l0, prerr_l0, prerr_l0___)
    
#     # print "cnt", ref.cnt, "pre_l1.shape", pre_l1.shape, "pre_l0.shape", pre_l0.shape, "meas_l0.shape", meas_l0.shape, "prerr_l0.shape", prerr_l0.shape, "prerr_l0_", prerr_l0_.shape, "X", X.shape, "Y", Y.shape

#     # print "ref.pre.shape", ref.pre.shape, "ref.err.shape", ref.err.shape
    
#     assert pre_l1.shape[-1] == pre_l0.shape[-1] == meas_l0.shape[-1], "step_actinf_m2: input shapes need to agree"

#     # loop over block of inputs if pre_l1.shape[-1] > 0:
#     prerr = prerr_l0_

#     logger.debug(
#         'step_actinf, pre_l1 = %s, pre_l0 = %s, meas_l0 = %s, prerr_l0 = %s',
#         pre_l1, pre_l0, meas_l0, prerr_l0)
#     ################################################################################
#     # fit the model
    
#     # dgoal for fitting lag additional time steps back
#     dgoal_fit = np.linalg.norm(ref.pre_l1_tm1 - ref.pre_l1_tm2)
#     y_ = Y.reshape((ref.odim / ref.laglen_future, -1))[...,[-1]]
#     if dgoal_fit < 5e-1: #  and np.linalg.norm(prerr_l0_) > 5e-2:        
#     # if np.linalg.norm(dgoal_fit) <= np.linalg.norm(ref.dgoal_fit_): #  and np.linalg.norm(prerr_l0_) > 5e-2:
#         # prerr = prerr_l0_.reshape((ref.odim / ref.laglen, -1))[...,[-1]]
#         # FIXME: actually, if ref.mdl.hasmemory
#         if isinstance(ref.mdl, smpOTLModel) or isinstance(ref.mdl, smpSHL):
#             print("Fitting without update")
#             ref.mdl.fit(X.T, Y.T, update = False)
#         else:
#             ref.mdl.fit(X.T, Y.T)
#     else:
#         # print "not fit[%d], dgoal_fit = %s, dgoal_fit_ = %s" % (ref.cnt, dgoal_fit, ref.dgoal_fit_)
#         pass

#     ref.dgoal_fit_ = 0.9 * ref.dgoal_fit_ + 0.1 * dgoal_fit
            
#     # ref.X_ = np.vstack((pre_l1[...,[-1]], prerr_l0[...,[-1]]))
#     ref.debug_print("step_actinf_m2_single ref.X_.shape = %s", (ref.X_.shape, ))

#     ################################################################################
#     # predict
    
#     # predict next values at current layer input
#     # pre_l1_tap_spec = ref.inputs[ref.pre_l1_inkey]['lag']
#     pre_l1_tap_full = ref.inputs[ref.pre_l1_inkey]['val'][...,-ref.laglen_past:]
#     pre_l1_tap_flat = pre_l1_tap_full.reshape((-1, 1))
    
#     dgoal = np.linalg.norm(ref.inputs[ref.pre_l1_inkey]['val'][...,[-1]] - ref.pre_l1_tm1)
#     if dgoal > 5e-1: # fixed threshold
#     # if np.linalg.norm(dgoal) > np.linalg.norm(ref.dgoal_): #  and np.linalg.norm(prerr_l0_) > 5e-2:
#         # goal changed
#         # m = ref.inputs['meas_l0']['val'][...,[-1]].reshape((ref.odim / ref.laglen, 1))
#         # p = ref.inputs[ref.pre_l1_inkey]['val'][...,[-1]].reshape((ref.odim / ref.laglen, 1))
#         # prerr_l0_ = (m - p) # * 0.1
#         # prerr_l0_ = -p.copy()
#         prerr_l0_ = np.random.uniform(-1e-3, 1e-3, prerr_l0_.shape)
#         # print "goal changed predict[%d], |dgoal| = %f, |PE| = %f" % (ref.cnt, dgoal, np.linalg.norm(prerr_l0_))

#         # prerr_l0__ = meas_l0 - pre_l1_tap_flat
#         tmp = prerr_l0__.reshape((-1, ref.laglen_past))
#         tmp[...,[-1]] = prerr_l0_.copy()
#         prerr_l0__ = tmp.reshape((-1, 1)) # meas_l0_tap_flat - pre_l1_tap_flat # meas_l0[...,[-1]] - pre_l1[...,[-lag]]
#         # pre_l1[...,[-1]]).reshape((ref.odim, 1))
        
#         prerr = prerr_l0_.reshape((ref.odim / ref.laglen_future, -1))[...,[-1]]
        
#     ref.dgoal_ = 0.9 * ref.dgoal_ + 0.1 * dgoal

#     # print "prerr_l0__", prerr_l0__.shape
#     # print "pre_l1_tap_flat", pre_l1_tap_flat.shape
#     ref.X_ = tapping_X(ref, pre_l1_tap_flat, prerr_l0__)
    
#     # print "step_actinf X[%d] = %s" % (ref.cnt, ref.X_.shape)
#     pre_l0_ = ref.mdl.predict(ref.X_.T)
#     # print "cnt = %s, pre_l0_" % (ref.cnt,), pre_l0_, "prerr_l0_", prerr_l0_.shape
    
#     # compute the final single time-step output from the multi-step prediction
#     # FIXME: put that mapping into config?
#     # fetch the logically latest prediction
#     pre = pre_l0_.reshape((ref.odim / ref.laglen_future, -1))[...,[-1]]
#     # # fetch the logically earliest prediction, might already refer to a past state
#     # pre = pre_l0_.reshape((ref.odim / ref.laglen_future, -1))[...,[-ref.laglen_future]]
#     # # fetch the minimally delayed prediction from multi step prediction
#     # pre = pre_l0_.reshape((ref.odim / ref.laglen_future, -1))[...,[max(-ref.laglen_future, ref.lag_past[1])]]
    
#     # pre = np.mean(pre_l0_.reshape((ref.odim / ref.laglen_future, -1))[...,-3:], axis = 1).reshape((-1, 1))
#     # prerr = prerr_l0_.reshape((ref.odim / ref.laglen, -1))[...,[-1]]
                
#     pre_ = getattr(ref, 'pre')
#     if ref.type == 'm1' or ref.type == 'm3':
#         pre_[...,[-1]] = pre
#     elif ref.type == 'm2':
#         pre_[...,[-1]] = np.clip(pre_[...,[-1]] + pre, -1, 1)
#     err_ = getattr(ref, 'err')
#     err_[...,[-1]] = prerr
#     tgt_ = getattr(ref, 'tgt')
#     tgt_[...,[-1]] = y_

#     # publish model's internal state
#     setattr(mref, 'pre', pre_)
#     setattr(mref, 'err', err_)
#     setattr(mref, 'tgt', tgt_)

#     # remember stuff
#     ref.pre_l1_tm2 = ref.pre_l1_tm1.copy()
#     ref.pre_l1_tm1 = ref.inputs[ref.pre_l1_inkey]['val'][...,[-1]].copy() # pre_l1[...,[-1]].copy()

# ################################################################################
# # step_actinf_2
# def step_actinf_2(ref, mref, *args, **kwargs):
#     """step_actinf_2

#     Compute on step of the actinf model, version 2, lean tapping code

#     # FIXME: single time slice taps only: add flattened version, reshape business
#     # FIXME: what's actinf specific, what's general?
#     """
#     # logger.debug(
#     #     'step_actinf_2 lag_past = %s, lag_future = %s, lag_off = %s',
#     #     ref.lag_past, ref.lag_future, ref.lag_off)
    
#     # prerr_t  = pre_l1_{lagf[0] - lag_off, lagf[1] - lag_off} - meas_l0_{lagf[0], lagf[1]}    
#     def tapping_prerr_fit(ref):
#         # prerr_fit = tap(ref, 'pre_l1', tap_tupoff(ref.lag_future, -ref.lag_off)) - tap(ref, 'meas_l0', ref.lag_future)
#         goal_ = tap(ref, 'pre_l1', ref.lag_future, -ref.lag_off - 1, 'inputs')
#         meas_ = tap(ref, 'meas_l0', ref.lag_future, 0, 'inputs')
#         # logger.debug('tapping_prerr_fit goal = %s, meas = %s', goal_, meas_)
#         prerr_fit = goal_ - meas_
#         return (prerr_fit, )
    
#     # pre_l0_t = pre_l0_{lagf[0] - lag_off + 1, lagf[1] - lag_off + 1}
#     def tapping_pre_l0_fit(ref):
#         # pre_l0_fit = tap(ref, 'pre_l0', tap_tupoff(ref.lag_future, -ref.lag_off + 1))
#         pre_l0_fit = tap(ref, 'pre_l0', ref.lag_future, -ref.lag_off + 1)
#         return (pre_l0_fit,)
        
#     # X_t-lag  = [pre_l1_{lagp[0], lagp[1]}, prerr_l0_{lagp[0]+1, lagp[1]+1}]
#     # Y_t      = pre_l0_t - (prerr_t * eta) # * d_prerr/d_params
#     def tapping_XY_fit(ref):
#         if ref.cnt < 2: logger.debug('tap_actinf %s, %s', 'pred', 'X_fit')
#         X_fit_pre_l1 = tap(ref, 'pre_l1', ref.lag_past)
#         X_fit_prerr_l0 = tap(ref, 'prerr_l0', ref.lag_past)
#         X_fit_flat = np.vstack((
#             tap_flat(X_fit_pre_l1),
#             tap_flat(X_fit_prerr_l0),
#         ))
        
#         if ref.cnt < 2: logger.debug('tap_actinf %s, %s', 'pred', 'Y_fit')
#         Y_fit_prerr_l0, = tapping_prerr_fit(ref)
#         Y_fit_prerr_l0_flat = tap_flat(Y_fit_prerr_l0)
#         Y_fit_pre_l0,   = tapping_pre_l0_fit(ref)
#         Y_fit   = Y_fit_pre_l0 + (Y_fit_prerr_l0 * ref.eta)
#         Y_fit_flat = tap_flat(Y_fit)
#         return (X_fit_flat, Y_fit_flat, Y_fit_prerr_l0_flat)

#     # get data and fit the model
#     X_fit_flat, Y_fit_flat, prerr_fit_flat = tapping_XY_fit(ref)
#     # logger.debug('X_fit_flat = %s, Y_fit_flat = %s, prerr_fit_flat = %s', X_fit_flat.T, Y_fit_flat.T, prerr_fit_flat.T)
#     if istraining(ref.cnt, ref.mdl):
#         ref.mdl.fit(X_fit_flat.T, Y_fit_flat.T)

#     # prerr_t  = pre_l1_{lagf[0] - lag_off, lagf[1] - lag_off} - meas_l0_{lagf[0], lagf[1]}
#     def tapping_prerr_predict(ref):
#         prerr_predict = tap(ref, 'pre_l1', ref.lag_past, -1) - tap(ref, 'meas_l0', tap_tupoff(ref.lag_past, ref.lag_off))
#         return (prerr_predict, )
    
#     def tapping_pre_l1_predict(ref):
#         pre_l1_predict = tap(ref, 'pre_l1', (ref.lag_future[1] - ref.laglen_past, ref.lag_future[1]))
#         return (pre_l1_predict,)
    
#     # X_t = [pre_l1_{lagf[1] - lagp_len, lagf[1]}, prerr_t]
#     def tapping_X_predict(ref):
#         if ref.cnt < 2: logger.debug('tap_actinf %s, %s', 'pred', 'X_pre')
#         prerr_predict, = tapping_prerr_predict(ref)
#         prerr_predict_flat = tap_flat(prerr_predict)
#         pre_l1_predict, = tapping_pre_l1_predict(ref)
#         pre_l1_predict_flat = tap_flat(pre_l1_predict)
#         X_predict = np.vstack((
#             pre_l1_predict_flat,
#             prerr_predict_flat,
#         ))
#         return (X_predict, )

#     # get data and predict
#     X_predict, = tapping_X_predict(ref)
#     # logger.debug('X_predict = %s', X_predict.T)
#     pre_l0 = ref.mdl.predict(X_predict.T)
#     # logger.debug('pre_l0 = %s', pre_l0)
    
#     # block outputs prepare
#     pre_ = tap_unflat(pre_l0, ref.laglen_future).copy()
#     pre = pre_[...,[-1]] # earliest relevant prediction
#     err_ = tap_unflat(prerr_fit_flat.T, ref.laglen_future).copy() # prerr_fit.copy()
#     err = err_[...,[-1]]
#     tgt_ = tap_unflat(Y_fit_flat.T, ref.laglen_future).copy()
#     tgt = tgt_[...,[-1]]
#     X_fit = X_fit_flat.copy()
#     Y_pre = pre_l0.T.copy()

#     # # debug
#     # logger.debug('pre = %s, err = %s, tgt = %s, X_fit = %s, Y_pre = %s', pre.shape, err.shape, tgt.shape, X_fit.shape, Y_pre.shape)
#     # logger.debug('pre = %s, err = %s, tgt = %s, X_fit = %s, Y_pre = %s', pre.T, err.T, tgt.T, X_fit.T, Y_pre.T)
    
#     # block outputs set
#     setattr(ref, 'pre', pre)
#     setattr(ref, 'err', err)
#     setattr(ref, 'tgt', tgt)
#     setattr(ref, 'X_fit', X_fit)
#     setattr(ref, 'Y_pre', Y_pre)

    
# # def step_actinf_prediction_errors_extended(ref, mref, *args, **kwargs):
# #     # if np.sum(np.abs(ref.goal_prop - ref.goal_prop_tm1)) > 1e-2:
# #     #     ref.E_prop_pred_fast = np.random.uniform(-1e-5, 1e-5, ref.E_prop_pred_fast.shape)
# #     #     ref.E_prop_pred_slow = np.random.uniform(-1e-5, 1e-5, ref.E_prop_pred_slow.shape)
# #     #     # recompute error
# #     #     # ref.E_prop_pred = ref.M_prop_pred - ref.goal_prop
# #     #     # ref.E_prop_pred[:] = np.random.uniform(-1e-5, 1e-5, ref.E_prop_pred.shape)
# #     #     #else:            
                
# #     E_prop_pred_tm1 = ref.E_prop_pred.copy()

# #     # prediction error's
# #     ref.E_prop_pred_state = ref.S_prop_pred - ref.M_prop_pred
# #     ref.E_prop_pred_goal  = ref.M_prop_pred - ref.goal_prop
# #     ref.E_prop_pred = ref.E_prop_pred_goal
        
# #     ref.E_prop_pred__fast = ref.E_prop_pred_fast.copy()
# #     ref.E_prop_pred_fast  = ref.coef_smooth_fast * ref.E_prop_pred_fast + (1 - ref.coef_smooth_fast) * ref.E_prop_pred

# #     ref.E_prop_pred__slow = ref.E_prop_pred_slow.copy()
# #     ref.E_prop_pred_slow  = ref.coef_smooth_slow * ref.E_prop_pred_slow + (1 - ref.coef_smooth_slow) * ref.E_prop_pred
                
# #     ref.dE_prop_pred_fast = ref.E_prop_pred_fast - ref.E_prop_pred__fast
# #     ref.d_E_prop_pred_ = ref.coef_smooth_slow * ref.d_E_prop_pred_ + (1 - ref.coef_smooth_slow) * ref.dE_prop_pred_fast

# # def step_actinf_sample_error_gradient(ref, mref, *args, **kwargs):
# #     # sample error gradient
# #     numsamples = 20
# #     # was @ 50
# #     lm = linear_model.Ridge(alpha = 0.0)
            
# #     S_ = []
# #     M_ = []
# #     for i in range(numsamples):
# #         # S_.append(np.random.normal(self.S_prop_pred, 0.01 * self.environment.conf.m_maxs, self.S_prop_pred.shape))
# #         # larger sampling range
# #         S_.append(np.random.normal(self.S_prop_pred, 0.3 * self.environment.conf.m_maxs, self.S_prop_pred.shape))
# #         # print "S_[-1]", S_[-1]
# #         M_.append(self.environment.compute_motor_command(S_[-1]))
# #         S_ext_ = self.environment.compute_sensori_effect(M_[-1]).reshape((1, self.dim_ext))
# #     S_ = np.array(S_).reshape((numsamples, self.S_prop_pred.shape[1]))
# #     M_ = np.array(M_).reshape((numsamples, self.S_prop_pred.shape[1]))
# #     print "S_", S_.shape, "M_", M_.shape
# #     # print "S_", S_, "M_", M_

# #     lm.fit(S_, M_)
# #     self.grad = np.diag(lm.coef_)
# #     print "grad", np.sign(self.grad), self.grad
            
# #     # pl.plot(S_, M_, "ko", alpha=0.4)
# #     # pl.show()
    
# ################################################################################
# # selforg / playful: hs, hk, pimax/tipi?
# def init_homoekinesis(ref, mref, conf, mconf):
#     # params = conf['params']
#     # hi = 1
#     # for outk, outv in params['outputs'].items():
#     #     setattr(ref, outk, np.random.uniform(-hi, hi, size = outv['shape']))
#     ref.mdl = init_smpModel(ref, mref, conf, mconf)
#     ref.X_  = np.zeros((mconf['idim'], 1))
#     ref.y_  = np.zeros((mconf['odim'], 1))
#     ref.pre_l1_tm1 = 0
#     # # eta = 0.3
#     # eta = ref.eta
#     # lag = ref.lag
#     # # print "Lag = %d" % (lag,)

# def step_homeokinesis(ref, mref, *args, **kwargs):
#     # get lag
#     # lag = ref.inputs['']['val'][...,lag]
#     # lag = 0
#     # current goal[t] prediction descending from layer above
#     pre_l1   = ref.inputs['pre_l1']['val']
#     # measurement[t] at current layer input
#     meas_l0 = ref.inputs['meas_l0']['val']
#     # prediction[t-1] at current layer input
#     pre_l0   = ref.inputs['pre_l0']['val']   
#     # prediction error[t-1] at current layer input
#     prerr_l0 = ref.inputs['prerr_l0']['val']

#     m_mins   = ref.mdl.m_mins # 0.1
#     m_mins_2 = m_mins * 2 # 0.2
#     one_over_m_mins = 1.0/m_mins
#     m_ranges = ref.mdl.m_maxs - ref.mdl.m_mins
    
#     # predict next values at current layer input
#     # pre_l0_ = ref.mdl.predict(ref.X_.T)
#     # pre_l0_ = ref.mdl.step((meas_l0 - m_mins_2) * one_over_m_mins) # bha m_mins/m_maxs
#     pre_l0_ = ref.mdl.step(meas_l0)
#     err_ = ref.mdl.xsi 
#     tgt_ = ref.mdl.v
   
#     # print "meas_l0", meas_l0.shape
#     # print "pre_l0_", pre_l0_.shape

#     setattr(ref, 'pre', ((pre_l0_[:,[-1]] + 1) * 0.5) * m_ranges + m_mins) # m_mins + m_mins_2) # bha m_mins/m_maxs
#     # setattr(ref, 'pre', pre_l0_[:,[-1]])#  - pre_l1[:,[-1]])
#     setattr(ref, 'err', err_)
#     setattr(ref, 'tgt', tgt_)
#     # return (pre_l0_.T.copy(), )
#     # return {
#     #     's_proprio': pre_l0.copy(),
#     #     's_extero': pre_l0.copy()}

# ################################################################################
# # sklearn based model
# def init_sklearn(ref, mref, conf, mconf):
#     # insert defaults
#     assert 'skmodel' in mconf
#     assert 'skmodel_params' in mconf
#     # sklearn models are saveable with pickle
#     mref.saveable = True
#     # check mconf
#     skmodel = mconf['skmodel']
#     skmodel_params = mconf['skmodel_params']
#     skmodel_comps = skmodel.split('.')
#     code = 'from sklearn.{0} import {1}\nmdl = {1}(**skmodel_params)'.format(skmodel_comps[0], skmodel_comps[1])
#     gv = {'skmodel_params': skmodel_params}
#     r = code_compile_and_run(code, gv)
#     # logger.debug("result from compile_and_run code = %s" % (code, ))
#     # logger.debug("    r = %s" % (r, ))
#     mref.mdl = r['mdl']
#     mref.mdl_init = False
#     mref.y = 0
#     # proper models have self.h transfer func
#     mref.h = np.zeros((
#         conf['params']['outputs']['y']['shape'][0],
#         # ref.defaults['model_numelem']
#         # mref.mdl.odim,   # dim
#         mconf['numelem'] # number observations
#     ))
#     # logger.debug('ref.h = %s', ref.h.shape)
#     # self.h_sample = 
#     # print "mref.mdl", mref.mdl
    
#     # set trigger callback
#     trigger_funcs = {'h': partial(trig_sklearn_h, mref = mref)}
#     for outk, outv in list(conf['params']['outputs'].items()):
#         if 'trigger' in outv and 'trigger_func' in outv:
#             outv['trigger_func'] = trigger_funcs[outv['trigger_func']] # code_compile_and_run('trig_sklearn_{0}'.format(outv['trigger_func']), gv)
#             logger.debug('converted trigger_func to %s' % (outv['trigger_func'], ))

# def step_sklearn(ref, mref, *args, **kwargs):
#     # pass
#     x_in = ref.inputs['x_in']['val'].T
#     x_tg = ref.inputs['x_tg']['val'].T
    
#     if not mref.mdl_init:
#         mref.mdl.fit(x_in, x_tg)
#     x_tg_ = mref.mdl.predict(x_in)
#     # print "x_tg_", x_tg_
#     mref.y = x_tg_.T
#     # logger.debug('x_in = %s', x_in.shape)
#     # mref.h_sample = np.atleast_2d(np.hstack([np.linspace(np.min(x_in_), np.max(x_in_), ref.defaults['model_numelem']) for x_in_ in x_in.T]))
#     # logger.debug('ref.h_sample = %s', ref.h_sample.shape)
#     # FIXME: meshgrid or random samples if dim > 4
#     # mref.h = mref.mdl.predict(mref.h_sample.T).T
#     # logger.debug('ref.h = %s', ref.h.shape)

# def trig_sklearn_h(ref, mref, *args, **kwargs):
#     # need idim
#     # idim = mref.mdl.idim
#     # numelem = mref.mdl.numelem
#     idim = ref.inputs['x_in']['shape'][0]
#     numelem = ref.outputs['h']['shape'][1]
#     mref.h_sample = np.atleast_2d(np.hstack([np.linspace(-1.1, 1.1, numelem) for _ in range(idim)]))
#     # logger.debug('mref.h_sample = %s', mref.h_sample.shape)
#     # FIXME: meshgrid or random samples if dim > 4
#     mref.h = mref.mdl.predict(mref.h_sample.T).T
#     # hack because we are called by step wrapper _after_ ModelBlock2 has copied mref to ref outputs
#     ref.h = mref.h
#     # logger.debug('mref.h = %s', mref.h)
#     # logger.debug(' ref.h = %s',  ref.h)
    
# def load_sklearn(ref, mref):
#     modelfileext = 'pkl'
#     modelfilenamefull = '{0}.{1}'.format(mref.modelfilename, modelfileext)
#     if ospath.exists(modelfilenamefull):
#         logger.debug("Trying to load model %s/%s from existing file %s" % (ref.id, mref.modelstr, modelfilenamefull))
#         try:
#             mref.mdl = joblib.load(modelfilenamefull)
#             mref.mdl_init = True
#         except Exception as e:
#             ref._error('load_sklearn failed with %s' % (e, ))
    
# def save_sklearn(ref, mref):
#     modelfileext = 'pkl'
#     modelfilenamefull = '{0}.{1}'.format(mref.modelfilename, modelfileext)
#     # logger.debug("Dumping model %s/%s to file %s" % (ref.id, mref.modelstr, modelfilenamefull))
#     joblib.dump(mref.mdl, modelfilenamefull)
    
# ################################################################################
# # extero-to-proprio map learning (e2p)
# def init_e2p(ref, conf, mconf):
#     ref.mdl = init_smpModel(ref, mref, conf, mconf)
#     ref.X_  = np.zeros((mconf['idim'], 1))
#     ref.y_  = np.zeros((mconf['odim'], 1))
#     ref.pre = np.zeros_like(ref.y_)
#     ref.pre_l1_tm1 = 0

# def step_e2p(ref):
#     # current goal[t] prediction descending from layer above
#     proprio   = ref.inputs['meas_l0_proprio']['val'][...,[-1]]
#     # measurement[t] at current layer input
#     extero    = ref.inputs['meas_l0_extero']['val'][...,[-1]]

#     # print "proprio", proprio.shape
#     # print "extero", extero.shape
    
#     ref.mdl.fit(extero.T, proprio.T)

#     # if ref.inputs['blk_mode']['val'] == 2.0:
#     # if True:
#     if 'blk_mode' in ref.inputs and ref.inputs['blk_mode']['val'][0,0] == 2.0:
#         if ref.cnt % 400 == 0:
#             # uniform prior
#             # extero_ = np.random.uniform(-1e-1, 1e-1, extero.shape)
#             # model prior?
#             extero_ = ref.mdl.sample_prior()
#             # print "extero_", extero_.shape
#             # print "sample", sample.shape
#             sample = np.clip(ref.mdl.predict(extero_.T), -3, 3)
#         elif ref.cnt % 400 in [100, 200, 300]:
#             # resting state
#             extero_ = np.random.uniform(-1e-3, 1e-3, extero.shape)

#         if ref.cnt % 200 in [0, 100]:
#             sample = np.clip(ref.mdl.predict(extero_.T), -3, 3)
#             setattr(ref, 'pre', sample.T)
#             setattr(ref, 'pre_ext', extero_)

# ################################################################################
# # dm: internal model online learning (imol)
# def init_imol(ref, mref, conf, mconf):
#     """initialize imol model pool

#     Implements a forward / inverse model pair with online learning
#     algorithms.
#     TODO: generalize for n models
#     """
#     # params shortcut
#     params = conf['params']

#     # # debug
#     # logger.debug('mconf = %s', mconf.keys())
#     # for mk in mconf.keys():
#     #     logger.debug('mconf[%s] = %s', mk, mconf[mk])

#     # submodel names in the pool (pair, triple, n-tuple)
#     ref.submodels = ['fwd', 'inv']
#     # submodel step func components specific for each submodel
#     func_steps = {

#         # 'fwd': [step_imol_update_prerr_l0_2, step_imol_fwd, step_imol_common],
#         # 'inv': [
#         #     step_imol_update_prerr_l0, step_imol_inv,
#         #     step_imol_common, step_imol_update_pre_l0],

#         'fwd': [
#             step_imol_update_prerr_l0_2,
#             tap_imol_x,
#             step_imol_common
#         ],
#         'inv': [
#             step_imol_update_prerr_l0,
#             tap_imol_x,
#             step_imol_common,
#             step_imol_update_pre_l0
#         ],
#     }
#     # submodel pool
#     ref.mdl = {}

#     # ref.store = {}
    
#     # initialize each submodel
#     for mk in ref.submodels:
#         # debug
#         logger.debug('mconf[%s] = %s', mk, mconf[mk])
#         # init submodel
#         init_imol_submodel(ref, mref, conf, mconf, mk)
#         # step func pointer
#         ref.mdl[mk]['func_step'] = func_steps[mk]
#         # set random additional ad-hoc params
#         ref.mdl[mk]['coef_err_avg'] = 0.9
        
#         # learning modulation
#         if 'fit_offset' not in ref.mdl[mk]:
#             ref.mdl[mk]['fit_offset'] = 200
            
# def init_imol_submodel(ref, mref, conf, mconf, mk):
#     """initialize imol submodel

#     1. copy mconf into ref.mdl at location modelkey
#     2. init_smpModel for configuration mconf
#     """
#     # params shortcut
#     params = conf['params']
    
#     # copy configuration
#     ref.mdl[mk] = mconf[mk]
    
#     # instantiate the model, returns smpModel
#     ref.mdl[mk]['inst_'] = init_smpModel(ref, mref, conf = conf, mconf = mconf[mk])
    
#     # update model mconf with inferred configuration
#     # 1. timing: explicit prior lag information
#     ref.mdl[mk]['laglen_past'] = ref.mdl[mk]['lag_past'][1] - ref.mdl[mk]['lag_past'][0]
#     ref.mdl[mk]['laglen_future'] = ref.mdl[mk]['lag_future'][1] - ref.mdl[mk]['lag_future'][0]
#     ref.mdl[mk]['lag_gap_f2p'] = ref.mdl[mk]['lag_future'][0] - ref.mdl[mk]['lag_past'][1]
#     ref.mdl[mk]['lag_off_f2p'] = ref.mdl[mk]['lag_future'][1] - ref.mdl[mk]['lag_past'][1]
#     ref.mdl[mk]['lag_off'] = ref.mdl[mk]['lag_off_f2p']
    
#     # 2. introspection: errors, statistics, modulation of learning params
#     ref.mdl[mk]['prerr_avg'] = 1e-3
#     ref.mdl[mk]['prerr_rms_avg'] = 1e-3
#     # exploration: cnt based suppression of prediction
#     ref.mdl[mk]['thr_predict'] = 1

#     # 3. outputs
#     # random projection of hidden state for output and debugging
#     if 'modelsize' in ref.mdl[mk]:
#         # 1. projection size
#         ref.mdl[mk]['selsize'] = params['outputs']['hidden']['shape'][0]
#         # 2. one-time random projection indices
#         ref.mdl[mk]['hidden_output_index'] = np.random.choice(
#             list(range(ref.mdl[mk]['modelsize'])),
#             ref.mdl[mk]['selsize'],
#             replace=False
#         )

#     # recurrent models
#     ref.mdl[mk]['recurrent'] = False
#     if mref.isrecurrent(ref.mdl[mk]['inst_']):
#         ref.mdl[mk]['recurrent'] = True

# def step_imol_update_pre_l0(ref, mref, mk, *args, **kwargs):
#     # mk was inv
#     tmp_ = np.roll(ref.get_input('pre_l0'), shift=-1, axis=-1).copy()
#     tmp_[...,[-1]] = ref.mdl[mk]['pre_l0']
#     setattr(ref, 'pre_l0_%s' % mk, tmp_)

# def step_imol_update_prerr_l0(ref, mref, mk, *args, **kwargs):
#     # mk was inv
#     # compute current prediction error
#     err_ = ref.get_input('pre_l1')[...,[-ref.mdl[mk]['lag_off_f2p'] - 1]] - ref.get_input('meas_l0')[...,[-1]]
#     tmp_ = np.roll(ref.get_input('prerr_l0'), shift=-1, axis=-1).copy()
#     tmp_[...,[-1]] = err_
#     setattr(ref, 'prerr_l0_%s' % mk, tmp_)

#     # fit error meas_now - meas_then?
#     if not hasattr(ref, 'prerr_l0_%s' % 'invfit'):
#         setattr(ref, 'prerr_l0_%s' % 'invfit', np.zeros_like(tmp_))
#     tmp_2 = np.roll(getattr(ref, 'prerr_l0_%s' % 'invfit'), shift=-1, axis=-1)
#     err_2 = ref.get_input('meas_l0')[...,[-1]] - ref.get_input('meas_l0')[...,[ref.mdl[mk]['lag_off_f2p']]]
#     tmp_2[...,[-1]] = err_2
#     setattr(ref, 'prerr_l0_%s' % 'invfit', tmp_2)
    
# def step_imol_update_prerr_l0_2(ref, mref, mk, *args, **kwargs):
#     # update current local predicition error
#     err_ = ref.get_input('pre_l0_fwd')[...,[-ref.mdl['fwd']['lag_off_f2p']]] - ref.get_input('meas_l0')[...,[-1]]
#     # ref.mdl['fwd']['prerr_l0_pre'] = ref.inputs['pre_l0_fwd']['val'][...,[-ref.mdl['fwd']['lag_off_f2p']]] - tap_pre_fwd_X['meas_l0'][...,[-1]]
#     # get past prediction errors
#     tmp_ = np.roll(ref.get_input('prerr_l0_fwd'), shift=-1, axis=-1)
#     tmp_[...,[-1]] = err_ # ref.mdl['fwd']['prerr_l0_pre']
#     ref.mdl['fwd']['prerr_l0'] = tmp_
#     setattr(ref, 'prerr_l0_%s' % mk, tmp_)
    
# def step_imol(ref, mref, *args, **kwargs):
#     """developmental model imol step func

#     - loop over fwd and inv models and for each do
#      - tap data
#      - fit old forward with current feedback
#      - predict new forward after update
#      - fit old inverse with current feedback
#      - predict new inverse with current state
#      - return new (inverse) prediction (command)

#     - compute block output from individual model predictions

#     variations
#     1. different low-level algorithms
#     2. different exploration/training regimes: both fwd and inv can
#       influence the sampling, both can sample confidences (certain
#       vs. uncertain areas), e.g. sample motor conf until flat and
#       converged, sample prediction under fixed motor, iterate
#     3. different tapping size for past and future (RL lambda / gamma)
#     """
#     # common header
    
#     # inverse first
#     # forward second (trigger reiteration on-demand)
#     for mk in ['inv', 'fwd']:
#         # # debug
#         # # if ref.cnt % 100 == 0:
#         # logger.debug('step_imol[%d] mdl key = %s', ref.cnt, mk)
#         # call step func for submodel
#         if type(ref.mdl[mk]['func_step']) is list:
#             for i, f_step_i in enumerate(ref.mdl[mk]['func_step']):
#                 f_step_i(ref, mref, mk, *args, **kwargs)
#         else:
#             ref.mdl[mk]['func_step'](ref, mref, *args, **kwargs)
#             step_imol_common(ref, mref, mk, *args, **kwargs)
            
#         # if fwd disagrees with inv, reiterate?

#     # logger.debug("step_imol fwd_pre_l0 = %s, %s", ref.mdl['fwd']['pre_l0'], ref.mdl['fwd']['pre_l0_var'])
    
#     # set imol block outputs after fit / predicting all models
#     # inv 
#     setattr(ref, 'pre_inv', ref.mdl['inv']['pre_l0'].copy())
#     setattr(ref, 'prerr_inv', ref.mdl['inv']['prerr_l0_'].copy())
#     setattr(ref, 'prerr_avg_inv', ref.mdl['inv']['prerr_avg'].copy())
#     setattr(ref, 'prerr_rms_avg_inv', ref.mdl['inv']['prerr_rms_avg'].copy())
#     setattr(ref, 'wo_norm_inv', ref.mdl['inv']['inst_'].get_params(param='w_norm'))
    
#     # fwd sidechannel
#     setattr(ref, 'pre_fwd', ref.mdl['fwd']['pre_l0'].copy())
#     setattr(ref, 'prerr_fwd', ref.mdl['fwd']['prerr_l0_'].copy())
#     setattr(ref, 'prerr_avg_fwd', ref.mdl['fwd']['prerr_avg'].copy())
#     setattr(ref, 'prerr_rms_avg_fwd', ref.mdl['fwd']['prerr_rms_avg'].copy())
#     setattr(ref, 'wo_norm_fwd', ref.mdl['fwd']['inst_'].get_params(param='w_norm'))

#     # legacy
#     setattr(ref, 'pre', getattr(ref, 'pre_inv'))
#     setattr(ref, 'err', getattr(ref, 'prerr_inv'))
#     setattr(ref, 'prerr_avg', getattr(ref, 'prerr_avg_inv'))
#     setattr(ref, 'prerr_rms_avg', getattr(ref, 'prerr_rms_avg_inv'))
#     setattr(ref, 'wo_norm', getattr(ref, 'wo_norm_inv'))
#     setattr(ref, 'tgt', ref.mdl['inv']['tgt'].copy())
#     setattr(ref, 'X', ref.mdl['inv']['X_fit'].copy())
#     setattr(ref, 'Y', ref.mdl['inv']['Y_fit'].copy())
    
# def step_imol_common(ref, mref, mk, *args, **kwargs):
#     """step imol common

#     shared computation for forward and inverse model
#     """
    
#     # momentary local prediction error
#     ref.mdl[mk]['prerr_l0_'] = getattr(ref, 'prerr_l0_%s' % mk)[...,[-1]]
    
#     # average local prediction error
#     ref.mdl[mk]['prerr_avg'] = ref.mdl[mk]['coef_err_avg'] * ref.mdl[mk]['prerr_avg'] + (1-ref.mdl[mk]['coef_err_avg']) * ref.mdl[mk]['prerr_l0_']
#     ref.mdl[mk]['prerr_rms_avg'] = np.sqrt(np.mean(np.square(ref.mdl[mk]['prerr_avg']), keepdims=True))

#     # perform a fit / predict cycle
#     ref.mdl[mk]['pre_l0'], ref.mdl[mk]['pre_l0_var'] = fit_predict_imol(ref, mref, mk=mk, *args, **kwargs)
#     # add noise to prediction (exploration?)
#     ref.mdl[mk]['pre_l0'] += ref.mdl[mk]['pre_l0_var']
#     # bound outputs
#     ref.mdl[mk]['pre_l0'] = np.clip(ref.mdl[mk]['pre_l0'], -1.1, 1.1)

#     # debugging
#     if ref.cnt % 100 == 0:
#         logger.debug(
#             '%s-%s.step_imol[%d] %s prerr_rms_avg = %s',
#             ref.__class__.__name__,
#             ref.id,
#             ref.cnt,
#             mk,
#             ref.mdl[mk]['prerr_rms_avg'])

#     # check n-step prediction
#     pre_l0_, tgt_ = step_imol_nstep(ref, mref, mk, *args, **kwargs)
    
# def step_imol_nstep(ref, mref, mk, *args, **kwargs):
#     """special treatment for n-step prediction
#     """
#     if ref.mdl[mk]['laglen_future'] > 1: # n-step prediction
#         # assume ref.mdl[mk]['pre_l0'] is row vector with shape:
#         # - (x1_t-n, x2_t-n, ..., x1_t-1, x2_t-1, ..., x1_t, x2_t, ...)
#         # 
#         # restore ref.inputs format rows = dims, cols = time, most recent at -1
#         pre_l0_t = ref.mdl[mk]['pre_l0'].reshape((ref.mdl[mk]['laglen_future'], -1)).T
#         # print "pre_l0_t", pre_l0_t.shape

#         # restore most recent prediction
#         ref.mdl[mk]['pre_l0'] = pre_l0_t[...,[-1]].copy()
        
#         # ref.mdl[mk]['pre_l0'] = pre_l0_t[...,[0]].copy()
#         # ref.mdl[mk]['pre_l0'] = np.sum(pre_l0_t * (1.0/np.arange(1, ref.laglen_future+1)), axis = -1).reshape((-1, 1))
#         # ref.mdl[mk]['pre_l0'] = np.mean(pre_l0_t, axis = -1).reshape((-1, 1))
#         # print "ref.mdl[mk]['pre_l0']", ref.mdl[mk]['pre_l0'].shape

#         # prepare target
#         tgt_t = ref.mdl[mk]['Y_fit'].reshape((ref.mdl[mk]['laglen_future'], -1)).T
#         ref.mdl[mk]['tgt'] = tgt_t[...,[-1]]
#     # 1-step prediction
#     else:
#         ref.mdl[mk]['pre_l0'] = ref.mdl[mk]['pre_l0'].T
#         ref.mdl[mk]['tgt'] = ref.mdl[mk]['Y_fit']

#     return ref.mdl[mk]['pre_l0'], ref.mdl[mk]['tgt']
    
# def step_imol_fwd(ref, mref, mk, *args, **kwargs):
#     """step imol forward model

#      - tap data
#      - fit old forward with current feedback
#      - predict new forward after update
#      - return new forward prediction
#     """
#     # # debug
#     # logger.debug('step_imol_fwd[%d] ref.mdl[\'fwd\'] = %s', ref.cnt, ref.mdl['fwd'].keys())

#     # tap
#     # tap_pre_fwd, tap_fit_fwd = tap_imol_fwd(ref)
#     # X_fit_fwd, Y_fit_fwd, X_pre_fwd = tap_imol_fwd_modality(tap_pre_fwd, tap_fit_fwd)
#     # tap spec:
#     # 1: channels: ['pre_l0_flat', 'meas_l0_flat', 'prerr_l0_flat']
#     # 2a:   times: lag_start:lag_end,    contiguous
#     # 2b:   times: [i_0, i_1, ..., i_k], intermittent / sparse
    
#     # tap_pre_fwd, tap_fit_fwd, X_fit_fwd, Y_fit_fwd, X_pre_fwd = tap_imol_fwd(ref)
#     # ref.mdl['fwd']['X_fit_fwd'] = X_fit_fwd
#     # ref.mdl['fwd']['Y_fit_fwd'] = Y_fit_fwd
#     # ref.mdl['fwd']['X_pre_fwd'] = X_pre_fwd

#     mk = 'fwd'
    
#     tap_fit_fwd_X = tap_imol(
#         ref,
#         ['pre_l0', 'meas_l0', 'prerr_l0'],
#         ['lag_past'] * 3,
#         [1, 0, 1],
#         mk=mk)
#     ref.mdl['fwd']['X_fit'] = tap_fit_fwd_X['X'] # tap_stack(tap_fit_fwd_X, ['meas_l0', 'prerr_l0', 'pre_l0'])

#     tap_fit_fwd_Y = tap_imol(ref, ['meas_l0'], ['lag_future'], [0], mk=mk)
#     ref.mdl['fwd']['Y_fit'] = tap_fit_fwd_Y['X'] # tap_stack(tap_fit_fwd_Y, ['meas_l0'])

#     tap_pre_fwd_X = tap_imol(
#         ref,
#         ['pre_l0_inv', 'meas_l0', 'prerr_l0_fwd'],
#         ['lag_past'] * 3,
#         [ref.mdl['fwd']['lag_off_f2p'] - 1] + [ref.mdl['fwd']['lag_off_f2p']] * 2,
#         mk,
#         ['attr', 'inputs', 'inputs']
#     )
#     ref.mdl['fwd']['X_pre'] = tap_pre_fwd_X['X']

#     # do computations outside of tappings
#     # get motor
#     # tmp_ = np.roll(tap_pre_fwd_X['pre_l0'], shift=-1, axis=-1).copy()
#     # update motor with most recent inverse prediction
#     # tmp_[...,[-1]] = ref.mdl['inv']['pre_l0']
#     # update tap
#     # tap_pre_fwd_X['pre_l0'] = tmp_

#     # # update current local predicition error
#     # err_ = ref.inputs['pre_l0_fwd']['val'][...,[-ref.mdl['fwd']['lag_off_f2p']]] - tap_pre_fwd_X['meas_l0'][...,[-1]]
#     # # ref.mdl['fwd']['prerr_l0_pre'] = ref.inputs['pre_l0_fwd']['val'][...,[-ref.mdl['fwd']['lag_off_f2p']]] - tap_pre_fwd_X['meas_l0'][...,[-1]]
#     # # get past prediction errors
#     # tmp_2 = np.roll(tap_pre_fwd_X['prerr_l0_fwd'], shift=-1, axis=-1)
#     # tmp_2[...,[-1]] = err_ # ref.mdl['fwd']['prerr_l0_pre']
#     # ref.mdl['fwd']['prerr_l0'] = tmp_2
#     # setattr(mref, 'prerr_l0_fwd', tmp_2)

#     # # update tap
#     # tap_pre_fwd_X['prerr_l0_fwd'] = ref.mdl['fwd']['prerr_l0']
    
#     # # recompute the input stack
#     # ref.mdl['fwd']['X_pre'] = tap_stack(tap_pre_fwd_X, ['pre_l0_inv', 'meas_l0', 'prerr_l0_fwd'])
    
#     # # prediction error local
#     # ref.mdl['fwd']['prerr_l0_'] = err_ # tap_pre_fwd_X['prerr_l0_fwd'][...,[-1]]
#     # # ref.mdl['fwd']['prerr_l0_fit'] = tap_fit_fwd_X['prerr_l0'][...,[-1]]

#     # # compactify 2->1 oben
#     # ref.mdl['fwd']['X_fit'] = tap_imol_fwd(ref, ['pre_l0', 'meas_l0', 'prerr_l0'], ['lag_past'] * 3, [1, 0, 1])['X']
#     # ref.mdl['fwd']['Y_fit'] = tap_imol_fwd(ref, ['meas_l0'], ['lag_future'], [0])['X']
#     # ref.mdl['fwd']['X_pre'] = tap_imol_fwd(
#     #     ref, ['pre_l0', 'meas_l0', 'prerr_l0'],
#     #     ['lag_past'] * 3,
#     #     [ref.mdl['fwd']['lag_off_f2p'], ref.mdl['fwd']['lag_off_f2p'], ref.mdl['fwd']['lag_off_f2p'] - 1]
#     # )['X']
    
# def step_imol_inv(ref, mref, mk, *args, **kwargs):
#     """step imol inverse model

#      - tap data
#      - fit old forward with current feedback
#      - predict new forward after update
#      - fit old inverse with current feedback
#      - predict new inverse with current state
#      - return new (inverse) prediction (command)
#     """
#     # # debug
#     # logger.debug('step_imol_inv ref.mdl[\'inv\'] = %s', ref.mdl['inv'].keys())

#     # # tappings for predict and fit as dicts with channels as keys
#     # tap_pre_inv, tap_fit_inv = tap_imol_inv_old(ref)
#     # X_fit_inv, Y_fit_inv, X_pre_inv = tap_imol_inv_modality(tap_pre_inv, tap_fit_inv)
#     # ref.mdl['inv']['X_fit_inv'] = X_fit_inv
#     # ref.mdl['inv']['Y_fit_inv'] = Y_fit_inv
#     # ref.mdl['inv']['X_pre_inv'] = X_pre_inv
#     # # logger.debug('step_imol_inv X_fit_inv = %s', X_fit_inv)

#     mk = 'inv'
    
#     tap_fit_inv = tap_imol(
#         ref, [('pre_l1', 'meas_l0'), 'meas_l0', 'prerr_l0'],
#         ['lag_past'] * 3, [ref.mdl[mk]['lag_off_f2p'], 0, 1], mk=mk)
#     ref.mdl[mk]['X_fit'] = tap_fit_inv['X']

#     tap_fit_inv = tap_imol(
#         ref, ['pre_l0'],
#         ['lag_future'], [-ref.mdl[mk]['lag_off_f2p'] + 1], mk=mk)
#     ref.mdl[mk]['Y_fit'] = tap_fit_inv['X']
    
#     tap_pre_inv = tap_imol(
#         ref, ['pre_l1', 'meas_l0', 'prerr_l0_%s' % mk],
#         ['lag_past'] * 3,
#         [ref.mdl[mk]['lag_off_f2p']] * 3,
#         mk,
#         ['inputs', 'inputs', 'attr']
#         )
#     ref.mdl[mk]['X_pre'] = tap_pre_inv['X']

#     # # compute current prediction error
#     # err_ = ref.inputs['pre_l1']['val'][...,[-ref.mdl['inv']['lag_off_f2p']]] - tap_pre_inv['meas_l0'][...,[-1]]
#     # ref.mdl['inv']['prerr_l0'] = np.roll(tap_pre_inv['prerr_l0'], shift=-1, axis=-1)
#     # ref.mdl['inv']['prerr_l0'][...,[-1]] = err_
#     # tap_pre_inv['prerr_l0'] = ref.mdl['inv']['prerr_l0'] # copy? check view or not
#     # # recompute the input tap stack
#     # ref.mdl['inv']['X_pre'] = tap_stack(tap_pre_inv, ['pre_l1', 'meas_l0', 'prerr_l0'])

#     # # prediction error local
#     # ref.mdl['inv']['prerr_l0_'] = ref.prerr_l0_inv[...,[-1]]
    
#     # return to step_imol for fit_predict_imol
#     return

# def tap_imol_x(ref, mref, mk, *args, **kwargs):
#     """pure tapping for imol model

#     - take tap config
#     - tap block inputs and attr data
#     - return raw ndarray tensor stack
#     """
    
#     for k in ['X_fit', 'Y_fit', 'X_pre']:
#         # debug tapping
#         if ref.cnt < 2:
#             logger.debug('tap_imol_x %s, %s', mk, k)
#         # for tk in ['vars', 'taps', 'offs', 'srcs']:
#         #     tapk = 'tap_%s_%s' % (k, tk)
#         #     logger.debug('tap_imol_x ref.mdl[%s][%s] = %s', mk, tapk, ref.mdl[mk][tapk])
#         tap_fit_inv = tap_imol(
#             ref, ref.mdl[mk]['tap_%s_vars' % k],
#             ref.mdl[mk]['tap_%s_taps' % k],
#             ref.mdl[mk]['tap_%s_offs' % k],
#             mk,
#             ref.mdl[mk]['tap_%s_srcs' % k],
#             # ['attr', 'inputs', 'inputs']
#             # ['inputs', 'inputs', 'attr']
#             )
#         ref.mdl[mk][k] = tap_fit_inv['X']

#     return

# def fit_predict_imol(ref, mref, mk='inv', *args, **kwargs):
#     """fit-predict imol model

#     assume tapped inputs are in ['X_fit', 'Y_fit', 'X_pre'] of
#     ref.mdl[modelkey=mk]
#     """
#     # get tapped data
#     X_fit = ref.mdl[mk]['X_fit']
#     Y_fit = ref.mdl[mk]['Y_fit']
#     X_pre = ref.mdl[mk]['X_pre']

#     # fit predict model acoording to low-level algorithm
#     # otl model
#     if isinstance(ref.mdl[mk]['inst_'], smpOTLModel):
#         # ref.mdl[mk]['inst_'].fit(X = X_fit.T, y = Y_fit.T, update = True) # False)
#         # if True: # np.any(ref.mdl[mk]['prerr_l0_pre'] < ref.mdl[mk]['prerr_avg']):
#         # fit only after two updates
#         if istraining(ref.cnt, ref.mdl[mk]): # washout?
#             ref.mdl[mk]['inst_'].fit(X = X_fit.T, y = Y_fit.T, update = False)

#         # predict for external goal
#         pre_l1_local = ref.mdl[mk]['inst_'].predict(X = X_pre.T, rollback = True)
        
#         # save prediction
#         pre_l0 = ref.mdl[mk]['inst_'].pred.copy().reshape(Y_fit.T.shape)
#         # print "soesgp pre_l0", pre_l0
        
#         # predict with same fit X and update network
#         ref.mdl[mk]['inst_'].predict(X = X_fit.T)

#         # set hidden attribute for output
#         if hasattr(ref.mdl[mk]['inst_'], 'r_'):
#             hidden = ref.mdl[mk]['inst_'].r_[ref.mdl[mk]['hidden_output_index'],[-1]].reshape((ref.mdl[mk]['hidden_output_index'].shape[0], 1))
#             setattr(ref, 'hidden', hidden)
            
#     # single hidden layer model
#     elif isinstance(ref.mdl[mk]['inst_'], smpSHL):
#         # fit only after two updates, 0.3 barrel
#         if istraining(ref.cnt, ref.mdl[mk]): #  and ref.mdl[mk]['prerr_rms_avg'] >= 0.01: #  and np.mean(np.square(ref.mdl[mk]['prerr_l0_pre'])) < 0.1:
#             ref.mdl[mk]['inst_'].fit(X = X_fit.T, Y = Y_fit.T * 1.0, update = False)
#             # print "mdl_inv e", ref.mdl[mk]['inst_'].lr.e
            
#         # predict for external goal
#         # r_local = ref.mdl[mk]['inst_'].model.r.copy()
#         # x_local = ref.mdl[mk]['inst_'].model.x.copy()
#         # print "mdl_inv.model.r", 
#         pre_l1_local = ref.mdl[mk]['inst_'].predict(X = X_pre.T, rollback = True)
#         # print "mdl_inv.model.r", np.mean(np.abs(ref.mdl[mk]['inst_'].model.r - r_local))
#         # print "mdl_inv.model.x", np.mean(np.abs(ref.mdl[mk]['inst_'].model.x - x_local))
#         # print "pre_l1_local", pre_l1_local
            
#         if ref.mdl[mk]['inst_'].lrname == 'FORCEmdn':
#             pre_l0 = ref.mdl[mk]['inst_'].y
#         else:
#             # pre_l0 = ref.mdl[mk]['inst_'].model.zn.T
#             pre_l0 = ref.mdl[mk]['inst_'].model.z.T.copy()
#             # print "step_imol smpSHL pre_l0", pre_l0
            
#         # predict with same X and update network
#         ref.mdl[mk]['inst_'].predict(X = X_fit.T)

#         # prepare hidden states for output
#         if hasattr(ref.mdl[mk]['inst_'].model, 'r'):
#             hidden = ref.mdl[mk]['inst_'].model.r[ref.mdl[mk]['hidden_output_index']]
#             # print "hidden", hidden.shape
#             setattr(ref, 'hidden', hidden)
#             setattr(ref, 'wo_norm', np.array([[np.linalg.norm(ref.mdl[mk]['inst_'].model.wo, 2)]]))
#             # print "wo_norm", ref.mdl[mk]['wo_norm']
            
#     # other models
#     else:
#         # feedforward case
#         # model fit
#         if istraining(ref.cnt, ref.mdl[mk]):
#             ref.mdl[mk]['inst_'].fit(X = X_fit.T, y = Y_fit.T)
#         # model prediction
#         pre_l0 = ref.mdl[mk]['inst_'].predict(X = X_pre.T)
#         # output weights
#         # logger.debug("model params = %s", ref.mdl[mk]['inst_'].get_params())
#         # setattr(ref, 'wo_norm', ref.mdl[mk]['inst_'].get_params())
        
#         # set hidden attribute for output
#         if hasattr(ref.mdl[mk]['inst_'], 'hidden_dist'):
#             hidden = np.hstack((
#                 ref.mdl[mk]['inst_'].hidden_dist,
#                 ref.mdl[mk]['inst_'].hidden_dist_sum_avg,
#                 ref.mdl[mk]['inst_'].hidden_idx_norm)).T
#             # logger.debug('hidden = %s', hidden.shape)
#             setattr(ref, 'hidden', hidden)

#     # output sampling according to low-level algorithm
#     # otl model
#     if isinstance(ref.mdl[mk]['inst_'], smpOTLModel):
#         # pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * (1.0/np.sqrt(ref.mdl[mk]['inst_'].var) * 1.0) * ref.mdl[mk]['prerr_avg'] * 1.0

#         # amp = 1.0
#         # amp = 0.2
#         # amp = 0.1
#         amp = 0.001 / ref.mdl[mk]['inst_'].noise
#         # print "amp", amp
#         # amp = 0.02
#         # amp = 0.01
#         # amp = 0.001
#         # amp = 0.0
        
#         if ref.cnt % 100 == 0:
#             print("soesgp var", ref.mdl[mk]['inst_'].var, ref.cnt) # np.sqrt(np.mean(ref.mdl[mk]['inst_'].var))
        
#         # if np.sqrt(np.mean(ref.mdl[mk]['inst_'].var)) < 0.4:
#         #     pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * 0.1
#         # else:
#         #     pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * 0.1
#         # pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * np.square(ref.mdl[mk]['prerr_avg']) * 0.2 # 
#         if ref.cnt < ref.mdl[mk]['thr_predict']:
#             pre_l0 = np.random.uniform(-1.0, 1.0, size = pre_l0.shape)
#             pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * 0.001
#         else:
#             # pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * amp
#             pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * amp * np.sqrt(ref.mdl[mk]['inst_'].var)
#             # pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * np.sqrt(ref.mdl[mk]['inst_'].var) * ref.mdl[mk]['prerr_rms_avg'] * amp # 0.3
            
#     # single hidden layer model
#     elif isinstance(ref.mdl[mk]['inst_'], smpSHL):

#         # real goal prediction
#         # pre_l0 = pre_l1_local.copy()
#         # fake goal prediction
#         # pre_l0 = ref.mdl[mk]['inst_'].model.z.T.copy()

#         # amp = 1.0
#         amp = 0.1
#         # amp = 0.05
#         # amp = 0.01 / 0.6
#         # amp = 0.01 # lag = 1
#         # amp = 1e-3
#         # amp = 0.0
        
#         if ref.cnt < ref.mdl[mk]['thr_predict']:
#         #     pre_l0 = np.random.uniform(-1.0, 1.0, size = pre_l0.shape)
#         #     pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * 0.001
#         # else:
#             pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * ref.mdl[mk]['prerr_rms_avg'] * 0.0001 # np.sqrt(ref.mdl[mk]['inst_'].var) # * ref.mdl[mk]['inst_'].noise
#         else:
#             pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * np.abs(ref.mdl[mk]['prerr_avg']).T * amp # np.sqrt(ref.mdl[mk]['inst_'].var) # * ref.mdl[mk]['inst_'].noise
#             # pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * amp
            
#         # ref.mdl[mk]['inst_'].theta = ref.mdl[mk]['prerr_rms_avg'] * amp
#         ref.mdl[mk]['inst_'].theta = amp
#         # pre_l0_var = np.ones_like(pre_l0) * ref.mdl[mk]['prerr_avg'] * 0.1
#         if ref.mdl[mk]['inst_'].lrname == 'FORCEmdn':
#             pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * 1e-4
            
#     # hebbSOM model
#     elif isinstance(ref.mdl[mk]['inst_'], smpHebbianSOM):
#         pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * ref.mdl[mk]['prerr_rms_avg'] * 0.0001
        
#     # other models
#     else:
#         pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * ref.mdl[mk]['prerr_rms_avg'] * 0.25 # 1.0
#     # pre_l0_var = np.random.normal(0.0, 1.0, size = pre_l0.shape) * ref.mdl[mk]['prerr_avg'] * 0.25
    
#     return pre_l0, pre_l0_var

# ################################################################################
# # exploratory hebbian direct inverse model learning (eh diml)
# def init_eh(ref, mref, conf, mconf):
#     """init_eh

#     Reward modulated exploratory Hebbian learning initialization

#     TODO
#     - x Base version ported from point_mass_learner_offline.py and learners.py
#     - x Consolidate: step_eh, smpSHL, learnEH and learn*
#     - Tappings, eligibility, dev-model vs. smpSHL vs. LearningRules vs. Explorer
#     - Integrate and merge tapping with earlier Eligibility / learnEHE approach
#     - Use tapping to build a supervised learning version of the algorithm?
#     - Implement and compare CACLA
#     - Tapping past/future cleanup and evaluate -1/0, -n:-1/0, -mask/0, -n/k, -mask/mask, -n/n
#     - Stabilization: error thresholding, weight bounding, decaying eta, IP + mean removal + moment coding
#     """

    
#     logger.debug('ModelBlock2.model.init_eh mconf = {')
#     for k, v in list(mconf.items()):
#             logger.debug('   %s = %s', k, v)
#     # print "mconf.eta", mconf['eta']
#     # print "mconf.eta_init", mconf['eta_init']
    
#     # params variable shortcut
#     params = conf['params']

#     # FIXME: definition of all variables of this model (eh) for logging / publishing
#     # parameter aliasing
#     # algo -> type -> lrname
#     # N -> modelsize
#     # g -> spectral_radius
#     # p -> density
    
#     # model type / algo / lrname
#     # ref.type = mconf['type']
#     # ref.perf_measure = mconf['perf_measure']
#     # ref.minlag = mconf['minlag']
#     for k in ['type', 'perf_measure', 'minlag', 'maxlag', 'lag_future', 'lag_past']:
#         setattr(ref, k, mconf[k])

#     # compute the tapping lengths for past and future
#     ref.laglen_past = ref.lag_past[1] - ref.lag_past[0]
#     ref.laglen_future = ref.lag_future[1] - ref.lag_future[0]
#     # lag offset
#     ref.lag_off = ref.lag_future[1] - ref.lag_past[1]

#     mconf['visualize'] = False
    
#     # reservoir network
#     ref.mdl = init_smpModel(ref, mref, conf, mconf)

#     # FIXME: parameter configuration post-processing
#     # expand input coupling matrix from specification
#     # ref.use_icm = True
#     # ref.input_coupling_mtx = np.zeros((mconf['idim'], mconf['idim']))
#     # for k,v in mconf['input_coupling_mtx_spec'].items():
#     #     ref.input_coupling_mtx[k] = v
#     # print ("input coupling matrix", ref.input_coupling_mtx)
        
#     # # eligibility traces (devmdl)
#     # ref.ewin_off = 0
#     # ref.ewin = mconf['et_winsize']
#     # # print "ewin", ref.ewin
#     # ref.ewin_inv = 1./ref.ewin
#     # funcindex = 0 # rectangular
#     # # funcindex = 3 # double_exponential
#     # ref.etf = Eligibility(ref.ewin, funcindex)
#     # ref.et_corr = np.zeros((1, mconf['et_winsize']))

#     # predictors (unclear)
#     if mconf['use_pre']:
#         ref.pre = PredictorReservoir(
#             mconf['pre_inputs'],
#             mconf['pre_delay'],
#             mconf['len_episode'],
#             mconf['modelsize'])

#     # use weight bounding (smmdl)
#     if mconf['use_wb']:
#         self.bound_weight_fit(mconf['wb_thr'])

#     # self.selsize = ref.outputs['hidden']['shape'][0]
#     ref.selsize = params['outputs']['hidden']['shape'][0]
#     # hidden state output random projection
#     ref.hidden_output_index = np.random.choice(
#         list(range(mconf['modelsize'])), ref.selsize, replace=False)
#     # initialize tapping (devmdl)
#     ref.tapping_SM = partial(tapping_SM, mode = ref.type)
#     ref.tapping_EH = partial(tapping_EH)
#     ref.tapping_X = partial(tapping_X)

# def step_eh(ref, mref, *args, **kwargs):
#     """step_eh

#     Reward modulated exploratory Hebbian learning predict/update step
#     """
#     # new incoming measurements
    
#     # deal with the lag specification for each input (lag, delay, temporal characteristic)
#     # (pre_l1, pre_l0, meas_l0, prerr_l0, prerr_l0_, prerr_l0__, prerr_l0___) = ref.tapping_SM(ref)
#     # (pre_l1, pre_l0, meas_l0, prerr_l0, prerr_l0_, prerr_l0__) = tapping_EH2(ref)
#     # (X, Y) = ref.tapping_XY(ref, pre_l1, pre_l0, prerr_l0, prerr_l0__)
    
#     # (X, Y) = ref.tapping_EH(
#     #     ref, pre_l1, pre_l0, meas_l0,
#     #     prerr_l0, prerr_l0_, prerr_l0__)

#     # tapping_EH_input:  pre_l1, prerr_l0, meas_l0
#     def tapping_EH_input(ref):
#         # pre_l1 = ref.inputs['pre_l1']['val'][...,ref.inputs['pre_l1']['lag']]
#         # prerr_l0 = ref.inputs['prerr_l0']['val'][...,ref.inputs['prerr_l0']['lag']]
#         # meas_l0 = ref.inputs['meas_l0']['val'][...,np.array(ref.inputs['pre_l1']['lag'])+1]
#         pre_l1 = ref.inputs['pre_l1']['val'][...,[-1]] # most recent goal prediction
#         pre_l0 = ref.inputs['pre_l0']['val'][...,[-1]] # most recent goal prediction
#         prerr_l0 = ref.inputs['prerr_l0']['val'][...,[-1]] * 0.0 # our own most recent prediction error
#         meas_l0 = ref.inputs['meas_l0']['val'][...,[-1]] # most recent measurement
#         return (pre_l1, pre_l0, prerr_l0, meas_l0)
#     (pre_l1, pre_l0, prerr_l0, meas_l0) = tapping_EH_input(ref)
     
#     # tapping_EH_target: pre_l1, meas_l0
#     def tapping_EH_target(ref):
#         # pre_l1 = ref.inputs['pre_l1']['val'][...,np.array(ref.inputs['meas_l0']['lag'])-1]
#         # meas_l0 = ref.inputs['meas_l0']['val'][...,ref.inputs['meas_l0']['lag']]
        
#         # # this also works because goals change slowly
#         # pre_l1 = ref.inputs['pre_l1']['val'][...,range(ref.lag_future[0] - 1, ref.lag_future[1] - 1)]
#         # future - lag offset
#         pre_l1 = ref.inputs['pre_l1']['val'][...,list(range(ref.lag_future[0] - ref.lag_off, ref.lag_future[1] - ref.lag_off))]
#         # pre_l1 = ref.inputs['pre_l1']['val'][...,range(ref.lag_past[0]-1, ref.lag_past[1]-1)]
#         meas_l0 = ref.inputs['meas_l0']['val'][...,list(range(ref.lag_future[0], ref.lag_future[1]))]
#         return(pre_l1, meas_l0)

#     (pre_l1_t, meas_l0_t) = tapping_EH_target(ref)

#     def tapping_EH_target_corr(ref):
#         # pre_l1 = ref.inputs['pre_l1']['val'][...,np.array(ref.inputs['meas_l0']['lag'])-1]
#         # meas_l0 = ref.inputs['meas_l0']['val'][...,ref.inputs['meas_l0']['lag']]
#         lag_error = (-100, 0)
#         pre_l1 = ref.inputs['pre_l1']['val'][...,list(range(lag_error[0]-1, lag_error[1]-1))]
#         meas_l0 = ref.inputs['meas_l0']['val'][...,list(range(lag_error[0], lag_error[1]))]
#         # meas_l0
#         return(pre_l1, meas_l0)

#     (pre_l1_t_corr, meas_l0_t_corr) = tapping_EH_target_corr(ref)
     
#     # print "tap input pre_l1 = %s, prerr_l0 = %s, meas_l0 = %s" % (pre_l1.shape, prerr_l0.shape, meas_l0.shape)
#     # print "tap target pre_l1_t = %s, meas_l0_t = %s" % (pre_l1_t.shape, meas_l0_t.shape)

#     ############################################################
#     # shorthands for inputs
#     goal_i = pre_l1.reshape((-1, 1))
#     meas_i = meas_l0.reshape((-1, 1))
#     pre_i = pre_l0.reshape((-1, 1))
#     err_i = goal_i - meas_i
#     perf_i = -ref.perf_measure(err_i)
    
#     # shorthands for target
#     goal_t = pre_l1_t.reshape((-1, 1))
#     meas_t = meas_l0_t.reshape((-1, 1))
    
#     # use model specific error func
#     # err = goal - meas # component-wise error
#     # err = prerr_l0_
#     # err = prerr_l0___
#     # print "prerr_l0___", prerr_l0___.shape
#     err_t = goal_t - meas_t
#     # print "goal_t", goal_t
#     # print "meas_t", meas_t
#     # print "err_t", err_t
#     # err_t1 = np.corrcoef(pre_l1_t_corr, meas_l0_t_corr)
    
#     # # compute correlation
#     # err_t1 = np.array([
#     #     np.array([
#     #         np.correlate(np.roll(pre_l1_t_corr[j,:], shift = i), meas_l0_t_corr[j,:]) for i in range(-200, 0)
#     #     ]) for j in range(2)
#     # ])
    
#     # # print "err_t1 0", np.argmax(np.abs(err_t1[0].T[0]))
#     # # print "err_t1 1", np.argmax(np.abs(err_t1[1].T[0]))
#     # err_t = np.array([
#     #     [
#     #         # np.abs(err_t1[0].T[0][np.argmax(np.abs(err_t1[0].T[0]))])
#     #         np.argmax(np.abs(err_t1[0].T[0]))
#     #     ],
#     #     [
#     #         # np.abs(err_t1[1].T[0][np.argmax(np.abs(err_t1[1].T[0]))])
#     #         np.argmax(np.abs(err_t1[1].T[0]))
#     #     ]
#     # ])
    
#     # err = pre_l1_t[...,[-1]] - meas_l0_t[...,[-1]]
#     # print "err == pre_l0__", err == pre_l0__

#     # prepare model update
    
#     # error / performance: different variations
#     # FIXME: perf: order 0, 1, 2, -1, -2, differential relation between output and measurement, e.g. use int/diff expansions 
#     # FIXME: perf: fine-grained error, binary goal reached, selforg via mi, pi, novelty, ...
#     # FIXME: perf: learn perf from sparse and coarse reward aka Q-learning ;)
#     # x: perf: element-wise, global, partially coupled, ...

#     # set perf to EH specific perf (neg error with perf = 0 optimal performance)
#     ref.mdl.learnEH_prepare(perf = ref.perf_measure(err_t))
#     perf = ref.mdl.perf
#     # perf_i = np.ones_like(goal) * ref.perf_measure(goal - meas)
#     # print "perf", perf.shape, "err_t", err_t.shape
    
#     # compose new network input
#     x = np.vstack((
#         goal_i,
#         perf_i * 1.0, # 0.25,
#         meas_i,
# #        pre_i,
#         ))
#     # print "x", x.shape
#     y = pre_i
#     # update model
#     # if ref.cnt < 200: # washout 
#     if isindevelphase(ref.cnt, ref.mdl, 'n_train'):
#         ref.mdl.eta = ref.mdl.eta2
#     else:
#         ref.mdl.eta = 0.0
        
#     # print "perf", np.mean(np.square(ref.mdl.perf_lp))
#     # print "eta", ref.mdl.eta
#     y_mdl_ = ref.mdl.step(
#         X = x.T,
#         Y = y.T # dummy
#     )
#     # print "y_mdl_", y_mdl_.shape
#     # print "y_mdl_", y_mdl_

#     # # update perf prediction
#     # X_perf = np.vstack((goal_i, meas_i, perf_i)).T # , y_mdl_.T)).T # , pre_i
#     # Y_perf = perf_i.T
#     # # print "X_perf", X_perf.shape
#     # # print "Y_perf", Y_perf.shape
#     # perf_lp_fancy = ref.mdl.perf_model_fancy.step(X = X_perf, Y = Y_perf)
#     # # print "perf pred", ref.mdl.perf_lp, perf_lp_fancy
#     # # perf_lp_m1 = ref.mdl.perf_lp.copy()
#     # ref.mdl.perf_lp = perf_lp_fancy.T.copy()
    
#     # prepare block outputs
#     # print "ref.laglen", ref.laglen
#     pre_ = y_mdl_.reshape((-1, ref.laglen_future))
#     # print "pre_", pre_
#     err_ = ref.mdl.perf.reshape((-1, ref.laglen_future))

#     # print "block_models.step_eh: pre_", pre_
#     # print "block_models.step_eh: err_", err_
#     # setattr(ref, 'pre', np.sum(pre_[:,-3:]) * np.ones_like(ref.pre))
#     setattr(ref, 'pre', pre_[:,[-1]]) # FIXME: output scaling, e.g. bha * 0.5 + 0.2)
#     # setattr(ref, 'pre', pre_[:,[-2]])
#     setattr(ref, 'err', err_[:,[-1]])
#     setattr(ref, 'perflp', ref.mdl.perf_lp)
#     # setattr(ref, 'perflp', perf_lp_m1)
#     hidden = ref.mdl.model.r[ref.hidden_output_index]
#     # print "hidden", hidden.shape
#     setattr(ref, 'hidden', hidden)

#     wo_norm = np.linalg.norm(ref.mdl.model.wo, keepdims=True)
#     setattr(ref, 'wo_norm', wo_norm)
    
#     if ref.cnt % 500 == 0:
#         print("iter[%d]: |W_o| = %f, eta = %f" % (ref.cnt, wo_norm, ref.mdl.eta, ))
    
#     # return to execute prediction on system and wait for new measurement

# def init_smpmodel(ref, mref, conf, mconf):
#     """smpmodel is a simple wrapper for smpModels

#     This implements very simple single batch fit and open-loop prediction scenarios.

#     For temporal embedding use smpmodel2, actinf, imol, eh
#     """
#     mref.mdl = init_smpModel(ref, mref, conf, mconf)
#     mref.h = np.zeros((
#         # conf['params']['outputs']['y']['shape'][0], # dim
#         # ref.defaults['model_numelem']               # number observations
#         mref.mdl.odim,   # dim
#         mconf['numelem'] # number observations
#     ))
    
#     # set trigger callback
#     trigger_funcs = {'h': partial(trig_smpmodel_h, mref = mref)}
#     for outk, outv in list(conf['params']['outputs'].items()):
#         if 'trigger' in outv and 'trigger_func' in outv:
#             outv['trigger_func'] = trigger_funcs[outv['trigger_func']] # code_compile_and_run('trig_smpmodel_{0}'.format(outv['trigger_func']), gv)
#             logger.debug('converted trigger_func to %s' % (outv['trigger_func'], ))

# def step_smpmodel(ref, mref, *args, **kwargs):
#     """smpmodel step

#     smpmodel's computation step
#     """
#     # get fit input handles
#     X = np.atleast_2d(ref.get_input('x_in')).T
#     Y = ref.get_input('x_tg')

#     # logger.debug('X = %s, Y = %s', X.shape, Y.shape)
#     if len(Y.shape) > 2: Y = np.reshape(Y, (-1, ref.blocksize))
    
#     # fit the model
#     mref.mdl.fit(X, Y)

#     # predict
#     Y_ = mref.mdl.predict(X)

#     # get predict input handles and use separate input if available
#     if 'x_in2' in ref.inputs:
#         X2 = ref.get_input('x_in2')
#         # predict
#         Y2_ = mref.mdl.predict(X2)
#         # prepare for block output
#         setattr(mref, 'y2', Y2_)
        
#     # prepare for block output
#     setattr(mref, 'y', Y_)

# def trig_smpmodel_h(ref, mref, *args, **kwargs):
#     # logger.debug('trig_smpmodel_h dir(ref) = %s', dir(ref))
#     # logger.debug('trig_smpmodel_h dir(mref) = %s', dir(mref))
#     # logger.debug('trig_smpmodel_h dir(mref.mdl) = %s', dir(mref.mdl))
#     # ref.h_sample = np.atleast_2d(np.hstack([np.linspace(np.min(x_in_), np.max(x_in_), ref.defaults['model_numelem']) for x_in_ in X.T]))
#     # need idim
#     mref.h_sample = np.atleast_2d(np.hstack([np.linspace(-1.1, 1.1, mref.mdl.numelem) for _ in range(mref.mdl.idim)]))
#     # logger.debug('mref.h_sample = %s', mref.h_sample.shape)
#     # FIXME: meshgrid or random samples if dim > 4
#     mref.h = mref.mdl.predict(mref.h_sample.T).T
#     # hack because we are called by step wrapper _after_ ModelBlock2 has copied mref to ref outputs
#     ref.h = mref.h
#     # logger.debug('mref.h = %s', mref.h)
#     # logger.debug(' ref.h = %s',  ref.h)

# def init_qtap(ref, mref, conf, mconf):
#     """qtap - quantitative tapping

#     Compute binary tapping from continuous valued scan.
#     """
#     mref.thr = 0.5
#     if 'thr' in mconf:
#         mref.thr = mconf['thr']
    
# def step_qtap(ref, mref, *args, **kwargs):
#     """qtap step

#     qtap's computation step
#     """
#     qtap = np.atleast_2d(ref.get_input('qtap')).T
#     qtap_idx_sorted = np.argsort(qtap, axis = 0)[::-1]
#     qtap_sorted_cumsum = np.cumsum(qtap[qtap_idx_sorted])/np.sum(qtap)
#     # ref._debug("qtap = %s" % qtap)
#     # ref._debug("qtap_idx_sorted = %s" %( qtap_idx_sorted))
#     # ref._debug("qtap_sorted_cumsum = %s" %( qtap_sorted_cumsum))
#     thr_idx = np.searchsorted(qtap_sorted_cumsum, mref.thr)
#     tap = qtap_idx_sorted.copy()
#     tap[thr_idx:] = -1.
#     # tap.sort(axis=0)
#     ref._debug("thr = %s" % (mref.thr,))
#     ref._debug("tap = %s" % (tap.T, ))
#     mref.tap_x = tap.T
#     mref.tap_y = tap.T
    
# def init_linear_regression_probe(ref, mref, conf, mconf):
#     """linear_regression_probe - quantitative tapping

#     Compute binary tapping from continuous valued scan.
#     """
#     mref.alpha = 1e-2
#     if 'alpha' in mconf:
#         mref.alpha = mconf['alpha']

#     mref.meas = 'mse'
#     if 'meas' in mconf:
#         mref.meas = mconf['meas']

# def step_linear_regression_probe(ref, mref, *args, **kwargs):
#     """linear_regression_probe step

#     `linear_regression_probe`'s computation step:
    
#     """
#     # get raw tapping input from qtap model with shape (-1, scanlen)
#     tap_raw = np.atleast_2d(ref.get_input('tap')).T
#     ref._debug('tap_raw = %s' % (tap_raw.T, ))
#     # get input x
#     x = np.atleast_2d(ref.get_input('x')).T
#     # get target y
#     y = np.atleast_2d(ref.get_input('y')).T
#     # get base index running through raw tapping length
#     idx_base = np.arange(x.shape[0])
#     # get the selected time shifts with effective contributions
#     tap_clean = tap_raw[...,tap_raw>=0,np.newaxis]
#     # ref._debug("tap_clean", tap_clean)
#     # compute input tapping index from base index and effective taps
#     idx_tap = (idx_base + (tap_clean * -1)).T
#     ref._debug("idx_tap = %s" % ( idx_tap.shape,))
#     # ref._debug("idx_tap", idx_tap)
#     # do zero padding in the beginning to handle negative indices (FIXME: not used?)
#     x_ = np.vstack((np.zeros((tap_raw.shape[0], x.shape[1])), x))
#     # tap the values from the raw input (FIXME: use x_?)
#     x_tapped = x[idx_tap].reshape((x.shape[0], -1))
#     ref._debug("x_tapped = %s" % (x_tapped.shape,))
#     # ref._debug("x_tapped = %s" %(x_tapped))
#     # compute regression probe
#     # - fit linear model to data x/y and l2 regularization parameter $\alpha$
#     # - return prediction, prediction error, post fit parameter norm
#     y_, y_res, w_norm, i_norm = meas_linear_regression_probe(data={'X': x_tapped, 'Y': y}, alpha = mref.alpha, meas = mref.meas)
#     # data bookkeeping
#     mref.y     = y_.T.copy()
#     mref.y_res = np.array([[y_res.copy()]])
#     # prepare effective tapping index for output as binary mask
#     x__ = np.zeros((1, tap_raw.shape[0]))
#     x__[0,tap_clean[:,0]] = 1
#     ref._debug('x__ = %s' % (x__,))
#     ref._debug('tap_clean.T = %s' % (tap_clean.T,))
#     # compute effective tapping index for output converting bool to numeric
#     mref.y_idx = x__ * 1.
#     ref._debug("lrp.y_res = %s" % (mref.y_res,))
#     # set model parameter norm outputs
#     mref.w_norm = np.array([[w_norm]])
#     mref.b_norm = np.array([[i_norm]])
#     # done

# def f_cache(f):
#     f_cache_flag = False
#     if f_cache_flag:
#         return memory.cache(f)
#     else:
#         return f
    
# class model(object):
#     """model class

#     Generic model class used by ModelBlock2 for all low-level models
#     and algorithms. Low-level models are implemented via init_<model>
#     and step_<model> functions.

#     TODO: make them proper model child classes

#     :attr:`model.models`, :data:`model.models`.
#     """
#     defaults = {
#         'type': 'random_uniform',
#         'numelem': 1001,
#         # 'modelsize': 1,
#     }

#     # maybe using classes is appropriate? ;)
#     models = {
#         # open-loop models
#         # 'identity': {'init': init_identity, 'step':  f_cache(step_identity)},
#         # expansions
#         'musig': {'init': init_musig, 'step':  f_cache(step_musig)},
#         'msr': {'init': init_msr, 'step':  f_cache(step_msr)},
#         'res': {'init': init_res, 'step':  f_cache(step_res)},
#         'polyexp': {'init': init_polyexp, 'step':  f_cache(step_polyexp)},
#         'random_lookup': {
#             'init': init_random_lookup, 'step':  f_cache(step_random_lookup)},
#         # budget
#         'budget_linear': {
#             'init': init_budget, 'step':  f_cache(step_budget)},
#         # constants
#         'alternating_sign': {
#             'init': init_alternating_sign, 'step':  f_cache(step_alternating_sign)},
#         # function generators
#         'function_generator': {
#             'init': init_function_generator, 'step':  f_cache(step_function_generator)},
#         # active randomness
#         'random_uniform': {
#             'init': init_random_uniform, 'step':  f_cache(step_random_uniform)},
#         'random_uniform_pi_2': {
#             'init': init_random_uniform_pi_2, 'step':  f_cache(step_random_uniform_pi_2)},
#         'random_uniform_modulated': {
#             'init': init_random_uniform_modulated,
#             'step':  f_cache(step_random_uniform_modulated)},
#         # smp model
#         'smpmodel': {'init': init_smpmodel, 'step':  f_cache(step_smpmodel)},
#         # closed-loop models
#         # active inference
#         'actinf_m1': {'init': init_actinf, 'step':  f_cache(step_actinf_2)},
#         'actinf_m2': {'init': init_actinf, 'step':  f_cache(step_actinf)},
#         'actinf_m3': {'init': init_actinf, 'step':  f_cache(step_actinf)},
#         'e2p':       {'init': init_e2p,    'step':  f_cache(step_e2p)},
#         'sklearn':   {
#             'init': init_sklearn, 'step':  f_cache(step_sklearn),
#             'save': save_sklearn, 'load': load_sklearn},
#         # direct forward/inverse model pair learning
#         'imol': {'init': init_imol, 'step':  f_cache(step_imol)},
#         # reward based learning
#         'eh':        {'init': init_eh,     'step':  f_cache(step_eh)},
#         # self-organization of behaviour: hk, pimax/tipi, infth_pi, infth_ais, ...
#         'homeokinesis': {'init': init_homoekinesis, 'step':  f_cache(step_homeokinesis)},

#         # unspervised
#         'qtap': {'init': init_qtap, 'step':  f_cache(step_qtap)},
#         'linear_regression_probe': {
#             'init': init_linear_regression_probe, 'step':  f_cache(step_linear_regression_probe)},
#     }

#     def __init__(self, ref, conf, mref = None, mconf = {}):
#         """model.init

#         Initialize the core model of a ModelBlock2.

#         Uses configuration dict from block config and implements many
#         model variants.
        
#         Arguments
#          - ref: Block reference for parent block
#          - conf: Block configuration for parent block
#          - mref: Model reference [self] FIXME: actually it is modelkey
#          - mconf: Model configuration
#         """
#         # defaults
#         mconf_ = {}
#         mconf_.update(model.defaults)
#         mconf_.update(mconf)
#         mconf.update(mconf_)
        
#         assert mconf['type'] in list(self.models.keys()), "in %s.init: unknown model type, %s not in %s" % (self.__class__.__name__, mconf['type'], list(self.models.keys()))
#         assert 'numelem' in list(mconf.keys()), "in %s.init: %s not in mconf %s" % (self.__class__.__name__, 'numelem', list(mconf.keys()))
#         # FIXME: ignoring multiple entries taking 'last' one, in dictionary order
#         if mref is None:
#             mref = self.__class__.__name__

#         # model configuration
#         self.mconf = mconf

#         # use mref as modelkey and forget it
#         self.modelkey = mref
#         # this is braindead
#         self.modelstr = mconf['type']
#         # because mref is set to self here
#         self.models[self.modelstr]['init'](ref, self, conf, mconf)

#         # try to load existing model params
#         # self.load()

#     def init_modelfilename(self, ref):
#         if not hasattr(self, 'modelfilename'):
#             self.modelfilename = '{0}/model_{1}_{2}_{3}_{4}'.format(
#                 ref.top.datadir_expr, ref.id, self.modelstr, self.modelkey, ref.md5)
#         # else:
#         #     self.modelfilename = '{0}/model_{1}_{2}_{3}_{4}_{5}'.format(
#         #         ref.top.datadir_expr, ref.id, self.modelstr, self.modelkey, self.mconf['skmodel'], ref.md5)

#     def load(self, ref):
#         if hasattr(self, 'saveable') and self.saveable and 'load' in self.models[self.modelstr]:
#             ref._info("Trying to load model %s from file %s" % (self.modelstr, self.modelfilename))
#             self.models[self.modelstr]['load'](ref, self)
            
#     def save(self, ref):
#         """Dump the model into a file
#         """
#         if hasattr(self, 'saveable') and self.saveable and not self.mdl_init and 'save' in self.models[self.modelstr]:
#             ref._info("funcs_models: saving model %s into file %s" % (self.modelstr, self.modelfilename))
#             self.models[self.modelstr]['save'](ref, self)
        
#     def isrecurrent(self, inst):
#         """Check if the model instance 'inst' is recurrent
#         """
#         models_recurrent = [smpOTLModel, smpSHL]
#         return np.any([isinstance(inst, m) for m in models_recurrent])
    
#     def predict(self, ref):
#         """predict func for standard modelblock use

#         function arguments implicit in block input foo
#         """
#         self.models[self.modelstr]['step'](ref, self)
        
#     def predict2(self, ref, mref, x):
#         """predict func for input from argument

#         function arguments explicit in predict arg 'x'

#         x not None triggers return not None
#         """
#         return self.models[self.modelstr]['step'](ref, mref, x)
