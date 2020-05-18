"""tapping.py

Tapping is a graphical model mapping sensorimotor data indices (time
t, modality s) to a single machine learning input data point at
time t, see [1]

Full tapping
 - tap: get input at times t_tap retaining structure
 - tap_?: do any computations in structured space
 - tap_flat: transpose and flatten structured tap
 - ref.mdl.step: apply to model fit / predict cycle
 - tap_struct: restructure flat prediction

TODO:
 - moving over legacy tapping funcs from :file:`funcs_models.py`
 - move developmental models into smp_base to be able to reuse them
   outside of smp_graphs.

[1] https://arxiv.org/abs/1704.07622
"""

import numpy as np

from smp_base.common import get_module_logger
import logging
logger = get_module_logger(modulename = 'tapping', loglevel = logging.DEBUG)

def tap_tupoff(tup = (), off = 0):
    """block_models.tap_tupoff

    Return the input tuple with constant offset added to both elements
    """
    assert len(tup) == 2, "block_models.py.tap_tupoff wants 2-tuple, got %d-tuple" % (len(tup), )
    return (tup[0] + off, tup[1] + off)

def tap(ref, inkey = None, lag = None, off = 0, source='inputs', outidx=None):
    """tap stuff

    Tap into Block2 inputs or attrs at indices given by lag

    Arguments
    - ref: Reference to model block
    - inkey: input variable or attribute
    - lag: tap indices
    - off: tap offset

    Returns:
    - tapped inputs, structured
    """
    if source == 'inputs': assert inkey in ref.inputs, "block_models.tap needs valid input key, %s not in %s" % (inkey, list(ref.inputs.keys()))
    assert type(lag) in [tuple, list], "block_models.tap needs tapping 'lag' tuple or list, %s given" % (type(lag))

    # handle tapping specs
    if type(lag) is tuple:
        tapping = list(range(lag[0] + off, lag[1] + off))
    elif type(lag) is list:
        tapping = np.array(lag) + off

    if ref.cnt < 2:
        logger.debug(
            '%s.%s tap[%d] %s, %s, %s', ref.__class__.__name__, ref.id, ref.cnt, '%s.%s' % (source, inkey), tapping, outidx)

    # return tapped data
    if source == 'inputs':
        # return tap_inputs(ref, inkey, tapping)
        return ref.inputs[inkey]['val'][...,tapping].copy()
    elif source == 'attr':
        # inkey_ = 'mref_%s' % inkey
        # logger.debug('tap attr = %s, tapping = %s', inkey, tapping)
        # FIXME: make these tapped attrs equal in shape so that tapping spec doesn't break
        #        or fix in some better way
        ref_attr = getattr(ref, inkey)
        return ref_attr[...,tapping].copy()
    else:
        # TODO: consider inputs as attr, attr/key addresses
        return None
    
def tap_flat(tap_struct):
    """block_models.tap_flat

    Return transposed and flattened view of structured input vector
    (aka matrix) 'tap_struct'.
    FIXME: myt transpose / tensor
    """
    return tap_struct.T.reshape((-1, 1))

def tap_unflat(tap_flat, tap_len = 1):
    """block_models..tap_unflat

    Return inverted tap_flat() by reshaping 'tap_flat' into numtaps x
    -1 and transposing
    """
    if tap_len == 1:
        return tap_flat.T
    else:
        return tap_flat.reshape((tap_len, -1)).T

def tap_stack(channels, channel_keys, flat=True):
    channel_keys = fix_channels(channel_keys)
    if flat:
        return np.vstack([channels['%s_flat' % channel_key[0]] for channel_key in channel_keys])
    else:
        return np.vstack([channels[channel_key[0]] for channel_key in channel_keys])
        
################################################################################
# legacy tappings from funcs_models.py

# def tap_imol_fwd(ref):
#     # return tap_imol_fwd_time(ref)
#     tap_pre_fwd, tap_fit_fwd = tap_imol_fwd_time(ref)
#     X_fit_fwd, Y_fit_fwd, X_pre_fwd = tap_imol_fwd_modality(tap_pre_fwd, tap_fit_fwd)
#     return tap_pre_fwd, tap_fit_fwd, X_fit_fwd, Y_fit_fwd, X_pre_fwd
    
# def tap_imol_fwd_time(ref):
#     """tap for imol forward model

#     tap into time for imol forward model

#     Args:
#     - ref(ModelBlock2): reference to ModelBlock2 containing all info

#     Returns:
#     - tapping(tuple): tuple consisting of predict and fit tappings
#     """
#     if ref.mdl['fwd']['recurrent']:
#         tap_pre_fwd = tapping_imol_pre_fwd(ref)
#         # tap_fit_fwd = tapping_imol_recurrent_fit_fwd(ref)
#         tap_fit_fwd = tapping_imol_recurrent_fit_fwd_2(ref)
#     else:
#         tap_pre_fwd = tapping_imol_pre_fwd(ref)
#         tap_fit_fwd = tapping_imol_fit_fwd(ref)

#     return tap_pre_fwd, tap_fit_fwd

# def tap_imol_fwd_modality(tap_pre_fwd, tap_fit_fwd):
#     # collated fit and predict input tensors from tapped data
#     X_fit_fwd = np.vstack((
#         tap_fit_fwd['pre_l0_flat'],  # state.1: motor
#         tap_fit_fwd['meas_l0_flat'], # state.2: measurement
#         tap_fit_fwd['prerr_l0_flat'],# state.3: last error
#     ))
    
#     Y_fit_fwd = np.vstack((
#         tap_fit_fwd['pre_l1_flat'], # state.2: next measurement
#     ))
    
#     X_pre_fwd = np.vstack((
#         tap_pre_fwd['pre_l0_flat'],   # state.1: motor
#         tap_pre_fwd['meas_l0_flat'],  # state.2: measurement
#         tap_pre_fwd['prerr_l0_flat'], # state.3: last error
#     ))
#     return X_fit_fwd, Y_fit_fwd, X_pre_fwd

# def tapping_imol_pre_fwd(ref):
#     """tapping for imol inverse prediction

#     state: pre_l0_{t}   # pre_l0
#     state: pre_l0_{t-1} # meas
#     """
#     mk = 'fwd'
#     rate = 1
    
#     # most recent top-down prediction on the input
#     pre_l1 = ref.inputs['pre_l1']['val'][
#         ...,
#         range(
#             ref.mdl[mk]['lag_past'][0] + ref.mdl[mk]['lag_off_f2p'],
#             ref.mdl[mk]['lag_past'][1] + ref.mdl[mk]['lag_off_f2p'])].copy()

#     # most recent state measurements
#     meas_l0 = ref.inputs['meas_l0']['val'][
#         ...,
#         range(
#             ref.mdl[mk]['lag_past'][0] + ref.mdl[mk]['lag_off_f2p'],
#             ref.mdl[mk]['lag_past'][1] + ref.mdl[mk]['lag_off_f2p'])].copy()
    
#     # most 1-recent pre_l1/meas_l0 errors
#     prerr_l0 = ref.inputs['prerr_l0']['val'][
#         ...,
#         range(ref.mdl[mk]['lag_past'][0] + ref.mdl[mk]['lag_off_f2p'],
#               ref.mdl[mk]['lag_past'][1] + ref.mdl[mk]['lag_off_f2p'])].copy()
    
#     # momentary pre_l1/meas_l0 error
#     prerr_l0 = np.roll(prerr_l0, -1, axis = -1)
#     # FIXME: get my own prediction from that time
#     prerr_l0[...,[-1]] = ref.inputs['pre_fwd_l0']['val'][...,[-ref.mdl[mk]['lag_off_f2p']]] - meas_l0[...,[-1]]

#     # corresponding output k steps in the past, 1-delay for recurrent
#     pre_l0 = ref.inputs['pre_l0']['val'][
#         ...,
#         range(
#             ref.mdl[mk]['lag_past'][0] + ref.mdl[mk]['lag_off_f2p'] - rate,
#             ref.mdl[mk]['lag_past'][1] + ref.mdl[mk]['lag_off_f2p'] - rate)].copy()
    
#     pre_l0 = np.roll(pre_l0, -1, axis = -1)
#     # prerr_l0[...,[-1]] = ref.inputs['pre_l1']['val'][...,[-ref.mdl[mk]['lag_off_f2p']]] - meas_l0[...,[-1]]
#     pre_l0[...,[-1]] = ref.mdl['inv']['pre_l0']
    
#     return {
#         'pre_l1': pre_l1,
#         'meas_l0': meas_l0,
#         'pre_l0': pre_l0,
#         'prerr_l0': prerr_l0,
#         'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
#         'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
#         'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
#         'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 1.0,
#     }

# def tapping_imol_fit_fwd_old(ref):
#     mk = 'fwd'
#     rate = 1
    
#     # Y
#     # Y.1: most recent measurement
#     pre_l1 = ref.inputs['meas_l0']['val'][
#         ...,
#         range(
#             ref.mdl[mk]['lag_future'][0] + 0,
#             ref.mdl[mk]['lag_future'][1] + 0
#         )].copy()

#     # X
#     # X.2 corresponding starting state k steps in the past
#     meas_l0 = ref.inputs['meas_l0']['val'][
#         ...,
#         range(ref.mdl[mk]['lag_past'][0], ref.mdl[mk]['lag_past'][1])].copy()
    
#     # X.3: corresponding error k steps in the past, 1-delay for recurrent connection
#     prerr_l0 = ref.inputs['prerr_l0']['val'][
#         ...,
#         range(
#             ref.mdl[mk]['lag_past'][0] + rate,
#             ref.mdl[mk]['lag_past'][1] + rate)
#         ].copy()
    
#     # X.1: corresponding output k steps in the past, 1-delay for recurrent
#     pre_l0 = ref.inputs['pre_l0']['val'][
#         ...,
#         range(
#             ref.mdl[mk]['lag_past'][0] - 0 + rate,
#             ref.mdl[mk]['lag_past'][1] - 0 + rate
#         )].copy()
    
#     pre_l1_ = tap(ref, 'meas_l0', ref.mdl[mk]['lag_future'])
#     meas_l0_ = tap(ref, 'meas_l0', ref.mdl[mk]['lag_past'])
#     prerr_l0_ = tap(ref, 'prerr_l0', tap_tupoff(ref.mdl[mk]['lag_past'], rate))
#     pre_l0_ = tap(ref, 'pre_l0', tap_tupoff(ref.mdl[mk]['lag_past'], rate))

#     pre_l1_check = np.sum(np.abs(pre_l1 - pre_l1_))
#     meas_l0_check = np.sum(np.abs(meas_l0 - meas_l0_))
#     prerr_l0_check = np.sum(np.abs(prerr_l0 - prerr_l0_))
#     pre_l0_check = np.sum(np.abs(pre_l0 - pre_l0_))

#     if pre_l1_check > 0:
#         logger.debug('tapping_imol_fit_fwd check |pre_l1| = %s', pre_l1_check)
#     if meas_l0_check > 0:
#         logger.debug('tapping_imol_fit_fwd check |meas_l0| = %s', meas_l0_check)
#     if prerr_l0_check > 0:
#         logger.debug('tapping_imol_fit_fwd check |prerr_l0| = %s', prerr_l0_check)
#     if pre_l0_check > 0:
#         logger.debug('tapping_imol_fit_fwd check |pre_l0| = %s', pre_l0_check)
    
#     return {
#         'pre_l1': pre_l1,
#         'meas_l0': meas_l0,
#         'prerr_l0': prerr_l0,
#         'pre_l0': pre_l0,
#         'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
#         'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
#         'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 1.0,
#         'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
#     }

def fix_channels(channels):
    for i, ch in enumerate(channels):
        if type(ch) is str:
            channels[i] = (ch, ch)
    # logger.debug('fix_channels channels = %s', channels)
    return channels

def fix_sources(sources, numsources):
    if sources is None:
        return ['inputs'] * numsources
    elif type(sources) is list:
        # assert len(sources) == numsources, 'Number of source specs needs to be equal the number of sources in the channels'
        return sources
        
def tap_imol(
        ref,
        channels=['meas_l0', 'prerr_l0', 'pre_l0'],
        taps=['lag_past'] * 3,
        offs=[0, 1, 1],
        mk='fwd',
        sources=['inputs'] * 3):
    """tap imol style

    tap from named inputs 'channels' at time indices 'taps' with
    optional offsets 'offs'.

    TODO: factor out mk, collate all imol variables / channels regardless fwd/inv
    TODO: arg: chans (list of names / list of name tuples)
    TODO: arg: times (binary index matrix / lists of indices)
    TODO: arg: offs (list of ints)

    In explicit form it was for example:
    ```
    rate = 1
    ret = {}
    ret['meas_l0'] = tap(ref, 'meas_l0', ref.mdl[mk]['lag_past'])
    ret['prerr_l0'] = tap(ref, 'prerr_l0', tap_tupoff(ref.mdl[mk]['lag_past'], rate))
    ret['pre_l0'] = tap(ref, 'pre_l0', tap_tupoff(ref.mdl[mk]['lag_past'], rate))
    return ret
    ```
    """
    # logger.debug(
    #     'tap_imol channels = %s, taps = %s, offs = %s, mk = %s, srcs = %s',
    #     channels, taps, offs, mk, sources)
    
    assert len(channels) == len(taps) and len(taps) == len(offs), "All input lists need to be same length but are %d %d %d" % (len(channels), len(taps), len(offs))
    channels = fix_channels(channels)
    sources = fix_sources(sources, len(channels))

    # # debug
    # for i, ch in enumerate(channels):
    #     logger.debug('fixed channel[%d] = %s -> %s', i, ch[0], ch[1])
    
    # compile dict of tapped sources
    a = [
        (ch[0], tap(ref, ch[1], ref.mdl[mk][taps[i]], offs[i], sources[i], i)) for i, ch in enumerate(channels)
    ]
    # compile dict of tapped and flattened sources
    b = [
        ('%s_flat' % ch[0], tap_flat(a[i][1])) for i, ch in enumerate(channels)
    ]
    c = dict(a + b)
    # compute stack of flat entries
    c['X'] = tap_stack(c, channels, flat=True)
    return c

# def tap_imol_fwd_fit_X(ref):
#     mk = 'fwd'
#     rate = 1
#     ret = {}
#     ret['meas_l0'] = tap(ref, 'meas_l0', ref.mdl[mk]['lag_past'])
#     ret['prerr_l0'] = tap(ref, 'prerr_l0', tap_tupoff(ref.mdl[mk]['lag_past'], rate))
#     ret['pre_l0'] = tap(ref, 'pre_l0', tap_tupoff(ref.mdl[mk]['lag_past'], rate))
#     return ret
    
# def tap_imol_fwd_fit_Y(ref, mk='fwd'):
#     return {'meas_l0': tap(ref, 'meas_l0', ref.mdl[mk]['lag_future'])}

# def tapping_imol_fit_fwd(ref):
#     return tap_imol_fwd_2(ref, ['meas_l0', 'prerr_l0', 'pre_l0'], ['lag_past'] * 3, [0, 1, 1])

# # imol inverse
# def tap_imol_inv(ref, channels=['pre_l1', 'meas_l0', 'prerr_l0'], taps=['lag_past'] * 3, offs=[0, 1, 1]):
#     return tap_imol_fwd(ref, channels, taps, offs, mk='inv')

def tap_imol_inv_old(ref):
    return tap_imol_inv_time(ref)
    
def tap_imol_inv_time(ref):
    """tap imol inverse

    tapping for imol inverse model

    Args:
    - ref(ModelBlock2): reference to ModelBlock2 containing all info

    Returns:
    - tapping(tuple): tuple consisting of predict and fit tappings
    """
    if ref.mdl['inv']['recurrent']:
        tap_pre_inv = tapping_imol_pre_inv(ref)
        # tap_fit_inv = tapping_imol_recurrent_fit_inv(ref)
        tap_fit_inv = tapping_imol_recurrent_fit_inv_2(ref)
    else:
        tap_pre_inv = tapping_imol_pre_inv(ref)
        tap_fit_inv = tapping_imol_fit_inv(ref)

    return tap_pre_inv, tap_fit_inv

def tap_imol_inv_modality(tap_pre_inv, tap_fit_inv):
    # collated fit and predict input tensors from tapped data
    X_fit_inv = np.vstack((
        tap_fit_inv['pre_l1_flat'],
        tap_fit_inv['meas_l0_flat'],
        tap_fit_inv['prerr_l0_flat'],
        ))
    Y_fit_inv = np.vstack((
        tap_fit_inv['pre_l0_flat'],
        ))
    
    X_pre_inv = np.vstack((
        tap_pre_inv['pre_l1_flat'],
        tap_pre_inv['meas_l0_flat'],
        tap_pre_inv['prerr_l0_flat'],
        ))
    return X_fit_inv, Y_fit_inv, X_pre_inv

################################################################################
# direct forward / inverse model learning via prediction dataset
def tapping_imol_pre_inv(ref):
    """tapping for imol inverse prediction
    """
    
    # most recent top-down prediction on the input
    pre_l1 = ref.inputs['pre_l1']['val'][
        ...,
        list(range(
            ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'],
            ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()

    # most recent state measurements
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        list(range(
            ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'],
            ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
    
    # most 1-recent pre_l1/meas_l0 errors
    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        list(range(
            ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'],
            ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
    
    # momentary pre_l1/meas_l0 error
    prerr_l0 = np.roll(prerr_l0, -1, axis = -1)
    # FIXME: get full tapping
    prerr_l0[...,[-1]] = ref.inputs['pre_l1']['val'][...,[-ref.mdl['inv']['lag_off_f2p']]] - meas_l0[...,[-1]]

    return {
        # 'pre_l1': pre_l1,
        # 'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 1.0,
    }

def tapping_imol_fit_inv(ref):
    rate = 1
    # X
    # last goal prediction with measurement    
    # most recent goal top-down prediction as input
    pre_l1 = ref.inputs['meas_l0']['val'][
        ...,
        list(range(
            ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'],
            ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
    # logger.debug('tapping_imol_fit_inv pre_l1 = %s', pre_l1)
    # corresponding starting state k steps in the past
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        list(range(
            ref.mdl['inv']['lag_past'][0],
            ref.mdl['inv']['lag_past'][1]))].copy()
    # corresponding error k steps in the past, 1-delay for recurrent
    # rate = -1
    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        list(range(
            ref.mdl['inv']['lag_past'][0] + rate,
            ref.mdl['inv']['lag_past'][1] + rate))].copy()
    # logger.debug('tapping_imol_fit_inv prerr_l0 = %s', prerr_l0)
    # rate = 1
    # Y
    # corresponding output k steps in the past, 1-delay for recurrent
    pre_l0 = ref.inputs['pre_l0']['val'][
        ...,
        list(range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'] + rate, ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p'] + rate))].copy()
    # range(ref.mdl['inv']['lag_future'][0], ref.mdl['inv']['lag_future'][1])].copy()
    
    return {
        # 'pre_l1': pre_l1,
        # 'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        # 'pre_l0': pre_l0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 1.0,
        'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
        }

def tapping_imol_recurrent_fit_inv(ref):
    rate = 1
    # X
    # last goal prediction with measurement    
    # FIXME: rate is laglen

    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        list(range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
    
    if ref.cnt < ref.thr_predict:
        # take current state measurement
        pre_l1 = ref.inputs['meas_l0']['val'][
            ...,
            list(range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
        # add noise (input exploration)
        pre_l1 += np.random.normal(0.0, 1.0, pre_l1.shape) * 0.01
    else:
        # take top-down prediction
        pre_l1_1 = ref.inputs['pre_l1']['val'][
            ...,
            list(range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
        # and current state
        pre_l1_2 = ref.inputs['meas_l0']['val'][
            ...,
            list(range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
        mdltr = np.square(max(0, 1.0 - np.mean(np.abs(prerr_l0))))
        # print "mdltr", mdltr
        # explore input around current state depending on pe state
        pre_l1 = pre_l1_2 + (pre_l1_1 - pre_l1_2) * mdltr # 0.05

    # most recent measurements
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        list(range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()

    # update prediction errors
    prerr_l0 = np.roll(prerr_l0, -1, axis = -1)
    prerr_l0[...,[-1]] = pre_l1[...,[-1]] - meas_l0[...,[-1]]
    
    # Y
    # FIXME check - 1?
    pre_l0 = ref.inputs['pre_l0']['val'][
        ...,
        list(range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p']))].copy()
    # range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'] + rate, ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p'] + rate)].copy()

    # print "tapping_imol_recurrent_fit_inv shapes", pre_l1.shape, meas_l0.shape, prerr_l0.shape, pre_l0.shape
    
    return {
        # 'pre_l1': pre_l1,
        # 'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        # 'pre_l0': pre_l0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 0.0,
        'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
        }

def tapping_imol_recurrent_fit_inv_2(ref):
    rate = 1
    # X
    # last goal prediction with measurement    
    # FIXME: rate is laglen

    pre_l1 = ref.inputs['meas_l0']['val'][
        ...,
        list(range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
    
    meas_l0 = ref.inputs['meas_l0']['val'][
        ...,
        list(range(ref.mdl['inv']['lag_past'][0], ref.mdl['inv']['lag_past'][1]))].copy()
    
    prerr_l0 = ref.inputs['prerr_l0']['val'][
        ...,
        list(range(ref.mdl['inv']['lag_past'][0] + ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_past'][1] + ref.mdl['inv']['lag_off_f2p']))].copy()
    # range(ref.mdl['inv']['lag_past'][0] + rate, ref.mdl['inv']['lag_past'][1] + rate)]
    prerr_l0 = np.roll(prerr_l0, -1, axis = -1)
    prerr_l0[...,[-1]] = pre_l1[...,[-1]] - meas_l0[...,[-1]]
    
    # pre_l1 -= prerr_l0[...,[-1]] * 0.1
    # Y
    pre_l0 = ref.inputs['pre_l0']['val'][
        ...,
        list(range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'], ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p']))].copy()
    # range(ref.mdl['inv']['lag_future'][0] - ref.mdl['inv']['lag_off_f2p'] + rate, ref.mdl['inv']['lag_future'][1] - ref.mdl['inv']['lag_off_f2p'] + rate)].copy()
    
    return {
        # 'pre_l1': pre_l1,
        # 'meas_l0': meas_l0,
        'prerr_l0': prerr_l0,
        # 'pre_l0': pre_l0 * 1.0,
        'pre_l1_flat': pre_l1.T.reshape((-1, 1)),
        'meas_l0_flat': meas_l0.T.reshape((-1, 1)),
        'prerr_l0_flat': prerr_l0.T.reshape((-1, 1)) * 0.0,
        'pre_l0_flat': pre_l0.T.reshape((-1, 1)),
    }

