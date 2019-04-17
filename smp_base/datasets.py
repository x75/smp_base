"""smp_base.datasets

Reusable datasets from different sources (input formats) to different
destinations (output formats) [WIP].
"""
from smp_base.impl import smpi

import numpy as np
from scipy.io import wavfile as wavfile
# import mdp

import logging
from smp_base.common import get_module_logger, compose

loglevel_debug = logging.DEBUG
logger = get_module_logger(modulename = 'datasets', loglevel = logging.INFO)

Oger = smpi('Oger')

def get_mackey_glass(sample_len=1000, n_samples=1):
    # check Oger imported by name exists in globals and its value a module
    if 'Oger' in globals() and Oger is not None:
        print('get_mackey_glass Oger = {0}'.format(Oger))
        return Oger.datasets.mackey_glass(sample_len=sample_len, n_samples=n_samples)
    
    logger.warn('datasets.get_mackey_glass: Oger not found, returning zeros')
    # return np.zeros((n_samples, 1, sample_len))
    return [[np.zeros((sample_len, 1))] for _ in list(range(n_samples))]

def wavdataset(
        sample_len, n_samples, filename, n_offset = None,
        numchan = 1):
    """load wav file dataset
    """
    (wavrate, wavdata) = wavfile.read(filename)
    # check channels
    if wavdata.shape[1] > 1:
        wavdata = wavdata[:,0]
    wavdata_max = np.max(np.abs(wavdata))
    wavdata = wavdata.astype(np.float64)/wavdata_max
    
    samples = []

    for _ in range(n_samples):
        if n_offset is None:
            wavoffset = 100000
        else:
            if n_offset < 1:
                wavoffset = np.random.randint(0, (len(wavdata) - (sample_len + 1)))
                # wavoffset = 3000000
            else:
                wavoffset = n_offset
                
        logger.debug("wavoffset = %d" % wavoffset)
        inp = wavdata[wavoffset:(wavoffset+sample_len),np.newaxis]# .tolist()
        samples.append([inp])
    # return list of n_samples with (sample_len, 1) shaped arrays
    return samples

# multi-dimensional MSO
def msond(sample_len=1000, n_samples = 1, dim = 3):
    """msond(sample_len=1000, n_samples = 1) -> input

    Generate the Multiple Sinewave Oscillator time-series, a sum of
    two sines with incommensurable periods. Parameters are:
    - sample_len: length of the time-series in timesteps
    - n_samples: number of samples to generate
    """
    signals = []
    for _ in range(n_samples):
        phase = np.random.rand()
        # x = np.atleast_2d(mdp.numx.arange(sample_len)).T
        x = np.atleast_2d(np.arange(sample_len)).T
        freqs1 = np.abs(np.random.normal(0.2, 0.01, (1, dim)))
        phases1 = np.random.uniform(0, 2 * np.pi, (1, dim))
        freqs2 = np.abs(np.random.normal(0.311, 0.02, (1, dim)))
        phases2 = np.random.uniform(0, 2 * np.pi, (1, dim))
        sin = np.sin(freqs1 * x + phases1) + np.sin(freqs2 * x + phases2)
        logger.debug("sin.shape", sin.shape)
        signals.append([sin])
    return signals

# prepare input
def timeseries_to_prediction_dataset(sample_len, n_samples, data):
    """Convert timeseries to prediction dataset 

    Compute a prediction dataset for supervised learning from a single timeseries.

    Arguments:
     - sample_len: length of the timeseries in timesteps
     - n_samples: number of samples to generate
     - data: timeseries ndarray

    Returns:
     - x: original input timeseries up to the last - 1 observations.
     - y: input duplicate shifted by 1 (was 'prepare_data_flat')
    """
    assert type(data) in [list, np.ndarray], 'Type of data is %s, must be list or ndarray' % (type(data))
    
    if type(data) == list:
        data = data[0][0]
        x = data[:-1,np.newaxis]
        y = data[1:] # np.roll(data[], -1)
    else:
        x = data[:-1,np.newaxis]
        # y = data[1:,np.newaxis] # np.roll(data[], -1)
        y = data[ 1:,np.newaxis] # np.roll(data[], -1)
    # x = x.reshape((x.shape[0], 1))
    # y = y.reshape((y.shape[0], 1)) # target needs to have this shape because keras

    logger.debug("timeseries_to_prediction_dataset input ts.shape = %s" % ( data.shape, ))
    logger.debug("timeseries_to_prediction_dataset output X.shape = %s" % (x.shape,))
    logger.debug("timeseries_to_prediction_dataset output Y.shape = %s" % (y.shape,))
    return (x, y)
    
def timeseries_to_prediction_embedded_dataset(data, maxlen, step):
    """Prepare data for use by .fit function: here we're stacking segments of
    a temporally embedded timeseries, each segment shift by one"""
    # check / compare: keras.preprocessing.sequence.pad_sequences
    # prepare signal by
    # 1. extracting bufsize/maxlen slices
    numbufs = int(data.shape[0]/maxlen)
    logger.debug("smp.datasets.timeseries_to_prediction_embedded_dataset data = %s, numbufs = %s", data.shape, numbufs)
    index = np.asarray([np.arange(i, i+maxlen) for i in range(0, maxlen*(numbufs-2), step)])
    logger.debug("smp.datasets.timeseries_to_prediction_embedded_dataset index = %s/%s", index.shape, index)
    data_x = data[index]
    data_x = data_x.reshape((index.shape[0], maxlen))
    # 2. creating 1D target by shift -1
    # index = np.asarray([np.arange(i, i+maxlen) for i in range(1, maxlen*(numbufs-2), step)])
    data_y = data[maxlen:]
    # data_y = data_y.reshape((index.shape[0], 1))
    # 3. finalize and pack up
    return (data_x, data_y)

def timeseries_to_temporal_embedding(data, embedding_size, pad=False, padchr=0):
    """take a timeseries and produce a temporal embedding of _embedding_size_
    atm only considering 1D conversion, that is, converting a 1D array of size (n, 1) into a 
    2D array of size (n-embedding_size, embedding_size). yuk.
    """
    # check / compare: keras.preprocessing.sequence.pad_sequences
    # prepare signal by
    # 1. extracting bufsize/maxlen slices
    step = 1
    # numbufs = data.shape[0]/embedding_size
    # init data to original length
    if pad:
        # create an index array for fetching the subarrays from the 1D array
        index = np.asarray(
            [np.arange(i, i + embedding_size) for i in range(
                0, data.shape[0] + embedding_size, step
            )]
        )
        # check the index
        logger.debug("index = %s / %s", index.shape, index)
        # # v1
        # # prealloc
        # data_x = np.zeros((data.shape[0], embedding_size))
        # # fetch data according to index
        # data_x[embedding_size:] = data[index].reshape((index.shape[0], embedding_size))

        # v2
        # prealloc
        data_ = np.zeros((data.shape[0] + (2 * embedding_size), 1))
        data_[:embedding_size] = padchr
        data_[embedding_size:-embedding_size] = data
        # fetch data according to index
        data_x = data_[index].reshape((index.shape[0], embedding_size))
    else:
        # create an index array for fetching the subarrays from the 1D array
        index = np.asarray([np.arange(i, i + embedding_size) for i in range(0, data.shape[0]-embedding_size, step)])
        # check the index
        logger.debug("index = %s / %s", index.shape, index)
        # fetch data according to index
        data_x = data[index]
        # shape hygiene
        data_x = data_x.reshape((index.shape[0], embedding_size))
    # return, done
    return data_x

################################################################################
# testing
def test_timeseries_to_temporal_embedding(data):
    data_te = timeseries_to_temporal_embedding(data, 10)
    assert(data.shape == (100, 1))
    assert(data_te.shape == (90, 10))
    logger.debug(data.shape, data_te.shape)

if __name__ == "__main__":
    # data = np.random.uniform(-1., 1., size=(100, 3)
    data = np.random.uniform(-1., 1., size=(100, 1))
    test_timeseries_to_temporal_embedding(data)
