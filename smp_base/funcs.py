"""smp_base.funcs

Some simple functions for common use
"""

from numpy import pi, tanh, clip, inf, cos, sin, array, poly1d, sign
from numpy.linalg import norm


def identity(x, *args, **kwargs):
    return x

def linear(x, a = 1.0, b = 0.0):
    return a * x + b

def nonlin_1(x, a = 1.0, b = 0.0):
    return tanh(linear(x, a, b))

def nonlin_2(x, a = 1.0, b = -1.0):
    return clip(linear(x, a, b = 0), 0, inf) + b

def nonlin_3(x, a = pi, b = 0.0):
    return cos(linear(x, a, b = b))

def nonlin_poly(self, u):
    """nonlin_poly

    ip2d.motortransfer_func legacy
    """
    # olimm1 = 0.5
    olim = 2
    # z = array([ 0.27924011,  0.12622341,  0.0330395,  -0.00490162])
    # z = array([ 0.00804775,  0.00223221, -0.1456263,  -0.04297434,  0.74612441,  0.26178644, -0.01953301, -0.00243736])
    # FIXME: somewhere there's a spearate script for generating the coeffs
    z = array([9.46569349e-04,  4.84698808e-03, -1.64436822e-02, -8.76479549e-02,  7.67630339e-02,  4.48107332e-01, -4.53365904e-03, -2.69288039e-04,  1.18423789e-15])
    p3 = poly1d(z)
    # print "pre", self.ip2d.u[ti]
    # self.ip2d.u[ti] = p3(tanh(self.ip2d.u[ti]) * self.olim)
    y = p3(tanh(u) * olim)
    return y

def power4(x):
    # return (x**4 * 2e-3) * sign(x)
    # return (x**2 * 2e-1) * sign(x)
    if norm(x, 2) < 100.:
        return (x**3 * 2e-3)
    else:
        return sign(x) * 1000000.
