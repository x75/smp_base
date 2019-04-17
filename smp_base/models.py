"""smp_base.models

Base *smp* model class :mod:`smpModel` and wrappers :mod:`smpModelInit`, :mod:`smpModelStep`.

The smpModel interface consists of an (obvious)
:meth:`smpModel.__init__(conf)` and a :meth:`smpModel.step()`
method. Simply put, an smpModel is a map together with some context
that can provide memory of past events. The initialization prepares
the context which can be quite specific for some models and the step
function implements a single full step of the model's update
equation. The composition of the model's step function can again be
quite specific but functionally consists of generating a prediction by
sampling model outputs from its current hidden state. Adaptive models
usually perform a fit using the currently incoming measurement before
the predict step, so that the current prediction is based on the most
recent information.

In addition there are some soft requirements, or conventions,
respectively like a bootstrap method that prepares an untrained
subordinate model for inference, a visualize method that provides some
visualization of the model's current state for debugging and analysis
and a save/load method pair for storing and loading model's which is
useful for large models with long training times.

Things:
 - use abstract method defs, either abc or NotImplementedError
 - merge stray models such as UniformRandomBlock2, iir, CodingBlock2, ...
 - fix visualization
"""
from smp_base.impl import smpi

# req
pickle = smpi('pickle')
np = smpi('numpy')

# smp foo
set_attr_from_dict = smpi('smp_base.common', 'set_attr_from_dict')
set_interactive = smpi('smp_base.plot_utils', 'set_interactive')

################################################################################
# smpModel decorator init
class smpModelInit():
    """smpModelInit wrapper"""
    def __call__(self, f):
        def wrap(xself, *args, **kwargs):
            # print "args", args
            # print "kwargs", kwargs
            assert 'conf' in kwargs, "smpModel needs conf dict"
            conf = kwargs['conf']
            if conf is None:
                conf = xself.defaults
            else:
                for k, v in list(xself.defaults.items()):
                    if k not in conf:
                        # print "models.init setting default", k, v
                        conf[k] = v
            
            f(xself, *args, **kwargs)

            xself.cnt_vis = 0

        return wrap

################################################################################
# smpModel decorator step
class smpModelStep():
    """smpModelStep wrapper"""
    def __call__(self, f):
        def wrap(xself, *args, **kwargs):

            # print "args", args
            # print "kwargs", kwargs

            ret = f(xself, *args, **kwargs)

            if xself.visualize:
                if 'X' in kwargs:
                    X = kwargs['X']
                else:
                    X = args[0]

                if 'Y' in kwargs:
                    Y = kwargs['Y']
                else:
                    Y = args[1]
                    
                # print "X", X
                if Y is not None:
                    xself.Xhist.append(X)
                    xself.Yhist.append(Y)

                    if hasattr(xself, 'Rhist'):
                        xself.Rhist.append(xself.model.r.copy())
                    if hasattr(xself, 'losshist'):
                        xself.losshist.append(np.min(xself.lr.loss))
                    if hasattr(xself, 'Whist'):
                        xself.Whist.append(np.linalg.norm(xself.model.wo))
                
                if xself.cnt_vis % 1000 == 0:
                    xself.visualize_model()

                    plt.draw()
                    plt.pause(1e-9)

            xself.cnt_vis += 1
            return ret
            
        return wrap

class smpModel(object):
    """smpModel

    Base class for function approximator- and regressor-type models
    """
    def __init__(self, conf): # idim = 1, odim = 1, numepisodes = 1, visualize = False):
        """smpModel.__init__

        init
        """
        defaults = {
            'idim': 1,
            'odim': 1,
        }
            
        set_attr_from_dict(self, conf)
        self.model = None
        self.cnt = 0

        # FIXME: variables for  all models
        # X, Y
        # e, perf, loss
        # dw, |W|
        
        # self.idim = idim
        # self.odim = odim
        # self.numepisodes = numepisodes
        # self.visualize = visualize

        if self.visualize:
            set_interactive(True)
            self.visualize_model_init()
        
    def get_params(self, *args, **kwargs):
        # return sane defaults?
        return np.zeros((1,self.odim))
        
    def bootstrap(self):
        """smpModel.bootstrap

        Bootstrap a newborn model so it can be queried
        """
        None

    def predict(self, X):
        """smpModel.predict

        Predict method of model
        """
        if self.model is None:
            print(("%s.predict: implement me" % (self.__class__.__name__)))
            return np.zeros((1, self.odim))
            
    def fit(self, X, Y):
        """smpModel.fit

        Fit method of model
        """
        if self.model is None:
            print(("%s.fit: implement me" % (self.__class__.__name__)))

    def step(self, X, Y):
        """smpModel.fit

        Step the model (fit, predict)
        """
        if self.model is None:
            print(("%s.fit: implement me" % (self.__class__.__name__)))

    def visualize_model_init(self):
        """smpModel.visualize_model_init
        
        model initialization code for visualization: set interactive
        plotting, init model figures and XY history, then call the
        instance's init implementation.
        """
        self.figs = []
        # store data for plotting, another internal model
        self.Xhist = []
        self.Yhist = []
        return
        
    def visualize_model(self):
        """smpModel.visualize

        Visualize the model with whichever methods suits the model
        """
        if self.model is None:
            print(("%s.visualize: implement me" % (self.__class__.__name__)))

    def save(self, filename):
        """smpModel.save

        Save method of model
        """
        pickle.dump(self, open(filename, "wb"))

    @classmethod
    def load(cls, filename):
        """smpModel.load (classmethod)

        Load a model and return the instance
        """
        return pickle.load(open(filename, "rb"))


################################################################################
# simple models
def iir_firstorder_freq_to_coef(f = 0.1):
    """iir_firstorder_freq_to_coef

    Compute the coefficients for a first order IIR filter given a cutoff freqency f

    https://dsp.stackexchange.com/questions/28308/exponential-weighted-moving-average-time-constant/28314#28314
    """
    a = np.cos(f) - 1 - np.sqrt(np.cos(f)**2 - 4 * np.cos(f) + 3)
    a = np.abs(a)
    b = 1 - a
    return b, a

def iir_firstorder_coef_to_freq(b = 0.8, a = 0.2):
    """iir_firstorder_coef_to_freq

    Compute the resulting cutoff frequency for a first order IIR filter given coefficients b, a

    from sympy import symbols
    a, f = symbols('a f')
    expr = sympy.cos(f) - 1 - sympy.sqrt(sympy.cos(f)**2 - 4 * sympy.cos(f) + 3) - a
    sympy.solve(expr, f)

    [acos((a**2/2 + a - 1)/(a - 1))]
    """
    f = np.arccos( (a**2/2 + a - 1) / (a - 1))
    return f

class iir_fo(object):
    # FIXME: make it smpModel
    def __init__(self, b = None, a = 0.2, dim = 1, y_init = 0):
        self.a = a
        self.y_init = y_init
        if b is None:
            self.b = 1 - self.a
        else:
            self.b = b
        self.dim = dim
        self.y = np.ones((self.dim, 1)) * self.y_init

    def predict(self, x):
        self.y = self.b * self.y + self.a * x
        return self.y

if __name__ == '__main__':
    print("testing iir_firstorder coef/freq conversion")
    # loop over frequencies
    for f in np.logspace(0.001, 0.2, 21, base = np.exp(1)) - 1:
        # compute coeffs
        b, a = iir_firstorder_freq_to_coef(f = f)
        # reconstruct f_c
        f_ = iir_firstorder_coef_to_freq(b = b, a = -a)
        # print stuff
        print("f = %f, b = %f, a = %f, f_ = %f" % (f, b, a, f_))
