"""smp_base: Basic elegibility traces for learnerEH"""

import argparse
import numpy as np
import matplotlib.pylab as pl
from sklearn.preprocessing import normalize

class Eligibility(object):
    """Implement a learning window (eligibility trace) for rate-based correlational learning"""
    def __init__(self, length=10, func=0):
        self.length = length
        self.lengthm1 = 1./self.length
        self.funcindex = func
        if self.funcindex == 0:
            self.efunc_ = self.efunc_rect
        elif self.funcindex == 1:
            self.efunc_ = self.efunc_ramp
        elif self.funcindex == 2:
            self.efunc_ = self.efunc_exp
        elif self.funcindex == 3:
            self.efunc_ = self.efunc_double_exp
        # generate table
        self.gen_efunc_table()

    def gen_efunc_table(self):
        # print "generating efunc table"
        self.domain = np.arange(self.length)
        self.ewin = self.efunc_(self.domain)
        # print "ewin shape", self.ewin.shape
        self.ewin = normalize(np.atleast_2d(self.ewin), axis=1, norm="l1").ravel()
        # print "sum ewin", np.sum(self.ewin)

    def efunc_ramp(self, x):
        """linear ramp window"""
        # x[x >= self.length] = self.length
        return np.clip(1 - (x * 1./self.length), 0, 1)

    def efunc_exp(self, x):
        """exponential decay window"""
        return np.exp(-x * self.lengthm1)
    
    def efunc_rect(self, x):
        """rectangular window"""
        # return x * self.lengthm1
        return np.ones_like(x) * self.lengthm1

    # from woergoetter: Lecture: Learning and Adaptive Algorithms
    # custom copy: pg 122, VL3
    # alpha function (EPSP-like)
    # Damped sinewave
    # Double exponential
    def efunc_double_exp(self, x):
        # print "efunc dbl exp"
        delta = 1.
        ratio = 30 # sharp
        speed = 10 # 
        scale = self.lengthm1
        a = .1
        # b = .05
        b = a * ratio
        la = a * speed
        lb = b * speed
        lx = x # * (1./ self.length)
        return 1/delta * (np.exp(-la * lx * scale) - np.exp(-lb * lx * scale))
    
    def efunc(self, x):
        # print "efunc", x, x.dtype
        return self.ewin[x]


def main(args):
    e = Eligibility(length=args.length)
    if args.mode == "dexp":
        e.efunc_ = e.efunc_double_exp
    elif args.mode == "rect":
        e.efunc_ = e.efunc_rect
    elif args.mode == "ramp":
        e.efunc_ = e.efunc_ramp
    elif args.mode == "exp":
        e.efunc_ = e.efunc_exp
    e.gen_efunc_table()

    x = np.arange(args.length)
    print(x)
    et = e.efunc(x)
    # plot and test with array argument
    cmstr = "ko"
    pl.plot(x, et, cmstr, lw=1.)
    if args.mode == "rect":
        # negative time for readability without lines
        pl.plot(np.arange(-5, x[0]), np.zeros(5,), cmstr, lw=1.)
        # pl.plot([-10, -1, x[0]], [0, 0, et[0]], cmstr, lw=1.)
        pl.plot([x[-1], x[0] + args.length], [et[-1], 0.], cmstr, lw=1.)
        pl.plot(x + args.length, np.zeros((len(et))), cmstr, lw=1.)
        pl.ylim((-0.005, np.max(et) * 1.1))
    # pl.plot(x, et, "k-", lw=1.)
    # pl.yticks([])
    # line at zero
    # pl.axhline(0., c="black")
    pl.xlabel("t [steps]")
    pl.ylabel("Eligibility")
    if args.plotsave:    
        pl.gcf().set_size_inches((6, 2))
        pl.gcf().savefig("eligibility_window.pdf", dpi=300, bbox_inches="tight")
    pl.show()

    # check perf: loop, test with single integer arguments
    import time
    now = time.time()
    for i in range(100):
        for j in range(args.length):
            e.efunc(j)
    print("table took:", time.time() - now)

    now = time.time()
    for i in range(100):
        for j in range(args.length):
            e.efunc_(j)
    print("feval took:", time.time() - now)
        
if __name__ == "__main__":
    modes = ["exp", "dexp", "rect", "ramp"]
    parser = argparse.ArgumentParser(description="Eligibility traces")
    parser.add_argument("-l", "--length", dest="length", help="Length of eligibility window",
                        default=10, type=int)
    parser.add_argument("-m", "--mode", dest="mode", help="Mode, one of " + ", ".join(modes), default="dexp")
    parser.add_argument('-ps', "--plotsave", action='store_true', help='Save plot to pdf?')

    args = parser.parse_args()

    main(args)
