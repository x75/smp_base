"""smp_base.plot_models

plotting utilities used for visualization of models
"""
import matplotlib.pyplot as plt

from smp_base.plot_utils import make_figure, make_gridspec, savefig

################################################################################
# model visualization code, only used in models_actinf.py
def plot_nodes_over_data_1d_components_fig(title = 'smpModel', numplots = 1):
    
    fig = make_figure
    fig.suptitle("One-dimensional breakdown of SOM nodes per input dimension (%s)" % (title,))
    # fig.suptitle(title)
    # numplots = idim + odim
    gs = make_gridspec(numplots, 1)
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
