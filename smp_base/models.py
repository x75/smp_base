import cPickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from smp_base.common import set_attr_from_dict


def make_figure(*args, **kwargs):
    return plt.figure()

def make_gridspec(rows = 1, cols = 1):
    return gridspec.GridSpec(rows, cols)

################################################################################
# smpModel decorator init
class smpModelInit():
    """smpModelInit wrapper"""
    def __call__(self, f):
        def wrap(xself, *args, **kwargs):
            # print "args", args
            # print "kwargs", kwargs
            assert kwargs.has_key('conf'), "smpModel needs conf dict"
            conf = kwargs['conf']
            if conf is None:
                conf = xself.defaults
            else:
                for k, v in xself.defaults.items():
                    if not conf.has_key(k):
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
                # X = kwargs['X']
                # Y = kwargs['Y']
                X = args[0]
                Y = args[1]
                # print "X", X
                xself.Xhist.append(X)
                xself.Yhist.append(Y)

                if hasattr(xself, 'Rhist'):
                    xself.Rhist.append(xself.model.r.copy())
                if hasattr(xself, 'losshist'):
                    xself.losshist.append(np.min(xself.lr.e))
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

        # FIXME: variables for  all models
        # X, Y
        # e, perf, loss
        # dw, |W|
        
        # self.idim = idim
        # self.odim = odim
        # self.numepisodes = numepisodes
        # self.visualize = visualize

        if self.visualize:
            plt.ion()
            self.figs = []
            # store data for plotting, another internal model
            self.Xhist = []
            self.Yhist = []
            
            self.visualize_model_init()
        
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
            print("%s.predict: implement me" % (self.__class__.__name__))
            return np.zeros((1, self.odim))
            
    def fit(self, X, Y):
        """smpModel.fit

        Fit method of model
        """
        if self.model is None:
            print("%s.fit: implement me" % (self.__class__.__name__))

    def visualize_model_init(self):
        return
        
    def visualize_model(self):
        """smpModel.visualize

        Visualize the model with whichever methods suits the model
        """
        if self.model is None:
            print("%s.visualize: implement me" % (self.__class__.__name__))

    def save(self, filename):
        """smpModel.save

        Save method of model
        """
        cPickle.dump(self, open(filename, "wb"))

    @classmethod
    def load(cls, filename):
        """smpModel.load (classmethod)

        Load a model and return the instance
        """
        return cPickle.load(open(filename, "rb"))

def savefig(fig, filename):
    fig_scale_inches = 0.75
    fig.set_size_inches((16 * fig_scale_inches, 9 * fig_scale_inches))
    fig.savefig(filename, dpi = 300, bbox_inches = 'tight')

def plot_nodes_over_data_1d_components_fig(title = 'smpModel', numplots = 1):
    
    fig = plt.figure()
    fig.suptitle("One-dimensional breakdown of SOM nodes per input dimension (%s)" % (title,))
    # fig.suptitle(title)
    # numplots = idim + odim
    gs = gridspec.GridSpec(numplots, 1)
    for i in range(numplots):
        fig.add_subplot(gs[i,0])
    return fig
    
def plot_nodes_over_data_1d_components(fig, X, Y, mdl, e_nodes, p_nodes, e_nodes_cov, p_nodes_cov, saveplot = False):
    """one-dimensional plot of each components of X and Y together with those of SOM nodes for all i and o components"""

    idim = X.shape[1]
    odim = Y.shape[1]
    numplots = idim + odim
    
    for i in range(idim):
        # ax = fig.add_subplot(gs[i,0])
        ax = fig.axes[i]
        ax.clear()
        ax.hist(X[:,i], bins=20)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        yran = ylim[1] - ylim[0]
        offset1 = yran * -0.1
        offset2 = yran * -0.25
        # print("offsets 1,2 = %f, %f" % (offset1, offset2))
        ax.plot(X[:,i], np.ones_like(X[:,i]) * offset1, "ko", alpha=0.33)
        for j,node in enumerate(e_nodes[:,i]):
            myms = 2 + 30 * np.sqrt(e_nodes_cov[i,i,i])
            # print("node", j, node, myms)
            ax.plot([node], [offset2], "ro", alpha=0.33, markersize=10)
            # ax.plot([node], [offset2], "r.", alpha=0.33, markersize = myms)
            # x1, x2 = gmm.
            ax.text(node, offset2, "n%d" % j, fontsize=6)
        # plt.plot(e_nodes[:,i], np.zeros_like(e_nodes[:,i]), "ro", alpha=0.33, markersize=10)
        
    for i in range(idim, numplots):
        # ax = fig.add_subplot(gs[i,0])
        ax = fig.axes[i]
        ax.clear()
        ax.hist(Y[:,i-idim], bins=20)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        yran = ylim[1] - ylim[0]
        offset1 = yran * -0.1
        offset2 = yran * -0.25
        # print("offsets 1,2 = %f, %f" % (offset1, offset2))
        ax.plot(Y[:,i-idim], np.ones_like(Y[:,i-idim]) * offset1, "ko", alpha=0.33)
        for j,node in enumerate(p_nodes[:,i-idim]):
            myms = 2 + 30 * np.sqrt(p_nodes_cov[i-idim,i-idim,i-idim])
            # print("node", j, node, myms)
            ax.plot([node], [offset2], "ro", alpha=0.33, markersize=10)
            # ax.plot([node], [offset2], "r.", alpha=0.33, markersize = myms)
            ax.text(node, offset2, "n%d" % j, fontsize=6)
            
       # plt.plot(p_nodes[:,i-idim], np.zeros_like(p_nodes[:,i-idim]), "ro", alpha=0.33, markersize=10)

    plt.draw()
    plt.pause(1e-9)
            
    if saveplot:
        filename = "plot_nodes_over_data_1d_components_%s.jpg" % (mdl.__class__.__name__,)
        savefig(fig, filename)
        
    fig.show()
    # plt.show()

    
