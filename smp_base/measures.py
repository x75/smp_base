"""smp_base - smp sensorimotor experiments base functions

measures

2017 Oswald Berthold

measures measure things about data like statistical moments, distances, entropy, complexity and so on

the information theoretic measures like in their own file @measures_infth.py
"""

import numpy as np

class meas(object):
    def __init__(self):
        pass

    def step(self, x):
        pass

    @staticmethod
    def identity(x):
        return -x
    
    @staticmethod
    def square(x):
        return np.square(x)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def abs_sqrt(x):
        return np.sqrt(np.abs(x))

    @staticmethod
    def sum_abs(x):
        return np.ones_like(x) * np.sum(np.abs(x))

    @staticmethod
    def sum_square(x):
        return np.ones_like(x) * np.sum(np.square(x))

    @staticmethod
    def sum_sqrt(x):
        return np.ones_like(x) * np.sum(np.sqrt(meas.abs(x)))

    # accel
    @staticmethod
    def abs_accel(x):
        return np.abs(x)

    @staticmethod
    def perf_accel(err, acc):
        """Simple reward: let body acceleration point into reduced error direction"""
        return np.sign(err) * acc
        # self.perf = np.sign(err) * np.sign(acc) * acc**2

    
