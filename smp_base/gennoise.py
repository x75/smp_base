#!/usr/bin/env python
# Oswald Berthold 2017
# 
# - oneoverfnoise is a python port of gennoise.c by paul bourke
# - levyflight is homegrown (?)



import argparse
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import normalize

import logging
from smp_base.common import get_module_logger

logger = get_module_logger(modulename = 'gennoise', loglevel = logging.INFO)

N = 8192
TWOPOWER = 13
TWOPI = 6.283185307179586476925287

def next_point(prev):
    "choose next destination"
    mode = "np"
    alpha = 1.5
    # angle = random.uniform(0,(2*math.pi))
    if mode == "random":
        angle = random.normalvariate(0,1.8)
        distance = 2 * random.paretovariate(alpha)
        # distance = 2 * random.weibullvariate(1.0, 0.9)
    elif mode == "np":
        angle = np.random.normal(0, 1.8)
        distance = np.random.pareto(alpha)
    # cap distance at DMAX
    #    if distance > DMAX:
    #        distance = DMAX
    point = [(np.sin(angle) * distance)+prev[0], (np.cos(angle) * distance)+prev[1]]
    return np.asarray(point)

class Noise(object):
    def __init__(self):
        pass

    @classmethod
    def oneoverfnoise(self, N, beta, normalize = False):
        """generate 1/f noise

        Arguments:
         - N(int): length
         - beta(float): 1/f**beta

        Returns:
         - tuple(freq, time) aka complex spectrum 'compl' and timeseries 'ts'
        """
        # initialize complex component vectors
        real = np.zeros((N,))
        imag = np.zeros((N,))

        # iterate FFT bands up to nyquist
        # FIXME: vectorize this
        for i in range(1, N/2):
            # spectrum magnitude from eq. ?
            # mag = (i+1)**(-beta/2.) * np.random.normal(0., 1.)
            mag = np.power(i+1, -beta/2.) * np.random.normal(0., 1.)
            # mag = np.power(i+1, -beta/2.) * np.random.uniform(0., 1.)
            # spectrum phase random
            pha = TWOPI * np.random.uniform()

            # convert polar to cartesian
            real[i] = mag * np.cos(pha)
            imag[i] = mag * np.sin(pha)
            
            # fix corner case
            real[N-i] = real[i]
            imag[N-i] = -imag[i]
            imag[N/2] = 0

        # complex array
        compl = real + (imag*1j)
        
        # normalize spectral energy
        if normalize:
            compl /= np.linalg.norm(compl)
            
        # print(compl)
        ts = np.fft.ifft(compl) * N
        logger.debug('timeseries ts is type = %s, shape = %s, var = %s from ifft(compl)' % (type(ts), ts.shape, np.var(ts)))
        # if normalize:
        #     ts /= np.abs(ts)
        logger.debug('timeseries ts is type = %s, shape = %s, var = %s from ifft(compl)' % (type(ts), ts.shape, np.var(ts)))
        return(compl, ts)

    @classmethod
    def levyflight(self, N):
        p = np.asarray([0,0])
        q = p
        qn = p
        numsamp = N
        p_ = np.zeros((numsamp, 2))
        # i = 0
        
        for i in range(numsamp):
            # q = next_point(p)
            # print(p, q)
            # p = q
            # add point
            if np.linalg.norm(qn - q) <= 1:
                q = next_point(p)
                dp = np.asarray(q)-np.asarray(p)
                # print("1", dp)
                dp = normalize(dp[:,np.newaxis], axis=0).ravel()
                # print("2", dp)
            # print(dp)
            qn = p + 1 * dp
            # draw line
            # pygame.draw.line(screen, white, p, qn, 1)
            p = qn
            p_[i,:] = p
        return p_
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--beta", type=float, default=0.)
    parser.add_argument("-l", "--len", type=int, default=8192)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    print("gennoise.main: args = %s" % (args, ))
    
    beta = args.beta
    N = args.len
    seed = args.seed

    np.random.seed(seed)

    (compl, ts) = Noise.oneoverfnoise(N, beta, normalize = True)
    print("gennoise.main: compl = %s, ts = %s" %(compl, ts))

    real = compl.real
    imag = compl.imag
    
    print("gennoise.main: real = %s, imag = %s" %(real, imag))

    fig = plt.figure()
    
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title('Noise spectrum (real/imag)')
    ax1.plot(real, label = "real")
    ax1.plot(imag, label = "imag")
    ax1.legend()
    # np.fft.ifft()

    # print(ts.real)

    ax2 = fig.add_subplot(2,1,2)
    ax2.set_title('Noise timeseries (real)')
    ax2.plot(ts.real)

    fig.show()
    plt.show()
