"""smp_base - smp sensorimotor experiments base functions

plotting

2017 Oswald Berthold
"""

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import numpy as np

try:
    from pyunicorn.timeseries import RecurrencePlot
    HAVE_PYUNICORN = True
except ImportError, e:
    print "Couldn't import RecurrencePlot from pyunicorn.timeseries, make sure pyunicorn is installed", e
    HAVE_PYUNICORN = False

try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError, e:
    print "Couldn't import seaborn as sns, make sure seaborn is installed", e
    HAVE_SEABORN = False

import pandas as pd
from pandas.tools.plotting import scatter_matrix

def make_axes_from_grid(fig, gs):
    axes = []
    (rows, cols) = gs.get_geometry()
    for row in range(rows):
        axes.append([])
        for col in range(cols):
            axes[-1].append(fig.add_subplot(gs[row, col]))
    return axes

def make_axes_from_spec(fig, gs, axesspec):
    axes = []
    # (rows, cols) = gs.get_geometry()
    for axspec in axesspec:
        axes.append([])
        # for col in range(cols):
        axes[-1].append(fig.add_subplot(gs[axspec]))
    return axes

def makefig(rows = 1, cols = 1, wspace = 0.0, hspace = 0.0, axesspec = None):
    """create figure and subplot structure

return figure handle
"""
    # rows = len(plotitems[0])
    # cols = len(plotitems[0][0])
    fig = plt.figure()
    gs = gridspec.GridSpec(rows, cols)


    if axesspec is None:
        axes = make_axes_from_grid(fig, gs)
    else:
        axes = make_axes_from_spec(fig, gs, axesspec)
        
    # print "fig.axes", fig.axes
    plt.subplots_adjust(wspace = wspace, hspace = hspace)
    # plt.subplots_adjust(wspace=0.1, hspace = 0.3)
    # plt.subplots_adjust(wspace=0.1, hspace = 0.3)
            
    return fig

def set_interactive(interactive = False):
    if interactive:
        plt.ion()
    else:
        plt.ioff()

def timeseries(ax, data, **kwargs):
    """timeseries plot"""
    # marker style
    if kwargs.has_key('marker'):
        marker = kwargs['marker']
    else:
        marker = 'None'
        
    # linestyle
    if kwargs.has_key('linestyle'):
        linestyle = kwargs['linestyle']
    else:
        linestyle = 'solid'

    # labels
    if kwargs.has_key('label'):
        label = kwargs['label']
    else:
        label = None

    # axis title
    if kwargs.has_key('title'):
        title = kwargs['title']
    else:
        title = 'timeseries of %s-shaped data' % data.shape
        
    # explicit xaxis
    if kwargs.has_key('ordinate'):
        ax.plot(kwargs['ordinate'], data, alpha = 0.5, marker = marker, linestyle = linestyle, label = label)
    else:
        ax.plot(data, alpha = 0.5, marker = marker, linestyle = linestyle, label = label)

    ax.legend(fontsize = 6)
    ax.title.set_text(title)
    ax.title.set_fontsize(8.0)

def histogram(ax, data, **kwargs):
    """histogram plot"""
    assert len(data.shape) > 0
    # print "histo data", data
    # style params
    # axis title
    if kwargs.has_key('title'):
        title = kwargs['title']
    else:
        title = 'histogram of %s-shaped data, log-scale' % data.shape

    ax.hist(data, bins = int(np.log(max(3, data.shape[0]/2))), alpha = 0.5)
    ax.set_yscale('log')
    ax.title.set_text(title)
    ax.title.set_fontsize(8.0)

if HAVE_PYUNICORN:
    def rp_timeseries_embedding(ax, data, **kwargs):
        """recurrence plot"""
        emb_del = 1
        emb_dim = 10
        # make data "strictly" one-dimensional
        data = data.reshape((-1, ))
        rp = RecurrencePlot(time_series = data, tau = emb_del, dim = emb_dim, threshold_std = 1.5)

        plotdata = rp.recurrence_matrix()
        length = plotdata.shape[0]
    
        xs = np.linspace(0, length, length)
        ys = np.linspace(0, length, length)
        ax.pcolormesh(xs, ys, plotdata, cmap=plt.get_cmap("Oranges"))
        ax.set_xlabel("$n$")
        ax.set_ylabel("$n$")
else:
    def rp_timeseries_embedding(ax, data, **kwargs):
        print "Dummy, pyunicorn not installed"

# visualization of multi-dimensional data
# check
# smq/plot
# smp/doc/thesis_matrix
# smp/im/im_quadrotor_plot
# smp/playground
# smp/infth
# smp/plot?
# smp/actinf: has dimstack etc ...
# evoplast
# ...?
if HAVE_SEABORN:
    def histogramnd(ax, data, **kwargs):
        scatter_data_raw  = data
        scatter_data_cols = ["x_%d" % (i,) for i in range(data.shape[1])]

        # prepare dataframe
        df = pd.DataFrame(scatter_data_raw, columns=scatter_data_cols)
        
        g = sns.PairGrid(df)
        # g.map_diag(plt.hist)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(plt.hexbin, cmap="gray", gridsize=30, bins="log");

        # print "dir(g)", dir(g)
        # print g.diag_axes
        # print g.axes
    
        # for i in range(data.shape[1]):
        #     for j in range(data.shape[1]): # 1, 2; 0, 2; 0, 1
        #         if i == j:
        #             continue
        #         # column gives x axis, row gives y axis, thus need to reverse the selection for plotting goal
        #         # g.axes[i,j].plot(df["%s%d" % (self.cols_goal_base, j)], df["%s%d" % (self.cols_goal_base, i)], "ro", alpha=0.5)
        #         g.axes[i,j].plot(df["x_%d" % (j,)], df["x_%d" % (i,)], "ro", alpha=0.5)
                
        plt.show()

    
        # run sns scattermatrix on dataframe
        # plot_scattermatrix(df, ax = None)
else:
    def histogramnd(ax, data, **kwargs):
        print "Dummy, seaborn not installed"
        
def plot_scattermatrix(df, **kwargs):
    """plot a scattermatrix of dataframe df"""
    if df is None:
        print "plot_scattermatrix: no data passed"
        return
        
    # df = pd.DataFrame(X, columns=['x1_t', 'x2_t', 'x1_tptau', 'x2_tptau', 'u_t'])
    # scatter_data_raw = np.hstack((np.array(Xs), np.array(Ys)))
    # scatter_data_raw = np.hstack((Xs, Ys))
    # print "scatter_data_raw", scatter_data_raw.shape
    
    plt.ioff()
    # df = pd.DataFrame(scatter_data_raw, columns=["x_%d" % i for i in range(scatter_data_raw.shape[1])])
    sm = scatter_matrix(df, ax = kwargs['ax'], alpha=0.2, figsize=(10, 10), diagonal='hist')
    print type(sm), sm.shape, sm[0,0]
    # fig = sm[0,0].get_figure()
    # if SAVEPLOTS:
    # fig.savefig("fig_%03d_scattermatrix.pdf" % (fig.number), dpi=300)
    # fig.show()
    # plt.show()


def plot_img(ax, data, **kwargs):
    assert ax is not None, "missing axis argument 'ax'"
    vmin = kwargs['vmin']
    vmax = kwargs['vmax']
    cmap = kwargs['cmap']
    title = kwargs['title']
    # FIXME: convert plottype into func: imshow, pcolor, pcolormesh, pcolorfast
    mpl = ax.pcolorfast(data, vmin = vmin, vmax = vmax, cmap = cmap)
    # normalize to [0, 1]
    # mpl = ax.imshow(inv, interpolation = "none")
    # mpl = ax.pcolorfast(data, vmin = vmin, vmax = vmax, cmap = cmap)
    # mpl = ax.pcolorfast(data, vmin = vmins[j], vmax = vmaxs[j], cmap = cmap)
    # mpl = ax.pcolorfast(data, vmin = -2, vmax = 2, cmap = cmap)
    # mpl = ax.pcolormesh(data, cmap = cmap)
    # mpl = ax.pcolor(data)
    # mpl = ax.pcolorfast(data)
    # mpl = ax.imshow(data, interpolation = "none")
    # mpl = ax.pcolormesh(
    #     data,
    #     norm = colors.LogNorm(vmin=data.min(), vmax=data.max())
    
    ax.grid(0)
    if kwargs.has_key('aspect'):
        ax.set_aspect(kwargs['aspect'])

    if kwargs.has_key('colorbar'):
        if kwargs['colorbar']:
            plt.colorbar(mappable = mpl, ax = ax, orientation = "horizontal")

    if kwargs.has_key('title'):
        ax.set_title(title, fontsize=8)
    else:
        ax.set_title("%s" % ('matrix'), fontsize=8)
        
    # if kwargs.has_key('xlabel'):
    ax.set_xlabel("")
        
    # if kwargs.has_key('ylabel'):
    ax.set_ylabel("")
        
    # if kwargs.has_key('xticks'):
    ax.set_xticks([])
        
    # if kwargs.has_key('yticks'):
    ax.set_yticks([])
    
if __name__ == "__main__":
    fig = makefig(2, 3)
