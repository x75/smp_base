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

def makefig(rows = 1, cols = 1, wspace = 0.0, hspace = 0.0, axesspec = None, title = None):
    """makefig

    alias for make_fig, see make_fig?
    """
    return make_fig(rows = rows, cols = cols, wspace = wspace, hspace = hspace, axesspec = axesspec, title = title)

def make_fig(rows = 1, cols = 1, wspace = 0.0, hspace = 0.0, axesspec = None, title = None):
    """make_fig

    create figure, subplot gridspec and axes

    return figure handle
    """
    # rows = len(plotitems[0])
    # cols = len(plotitems[0][0])
    fig = plt.figure()
    gs = gridspec.GridSpec(rows, cols)

    if title is not None:
        fig.suptitle(title)
    
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

def get_ax_size(fig, ax):
    """from stackoverflow"""
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height
        
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

    # linewidth
    if kwargs.has_key('linewidth'):
        linewidth = kwargs['linewidth']
    else:
        linewidth = 2.0
        
    # labels
    if kwargs.has_key('label'):
        label = kwargs['label']
    else:
        label = None

    if kwargs.has_key('xscale'):
        xscale = kwargs['xscale']
    else:
        xscale = 'linear'
        
    if kwargs.has_key('yscale'):
        yscale = kwargs['yscale']
    else:
        yscale = 'linear'

    if kwargs.has_key('xlim'):
        xlim = kwargs['xlim']
    else:
        xlim = None
        
    if kwargs.has_key('ylim'):
        ylim = kwargs['ylim']
    else:
        ylim = None
        
    # axis title
    if kwargs.has_key('title'):
        title = kwargs['title']
    else:
        title = 'timeseries of %s-shaped data' % data.shape
        
    # x-axis shift / bus delay compensation
    if kwargs.has_key('delay'):
        data = np.roll(data, kwargs['delay'], axis = 1)
        
    # explicit xaxis
    if kwargs.has_key('ordinate'):
        ax.plot(
            kwargs['ordinate'], data, alpha = 0.5,
            marker = marker, linestyle = linestyle, linewidth = linewidth,
            label = label)
    else:
        ax.plot(
            data, alpha = 0.5, marker = marker,
            linestyle = linestyle, label = label)

    ax.legend(fontsize = 6)
    
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
        
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
        
    if kwargs.has_key('orientation'):
        orientation = kwargs['orientation']
    else:
        orientation = 'vertical'
        
    if kwargs.has_key('xscale'):
        xscale = kwargs['xscale']
    else:
        xscale = 'linear'
        
    if kwargs.has_key('yscale'):
        yscale = kwargs['yscale']
    else:
        yscale = 'linear'

    if kwargs.has_key('xlim'):
        xlim = kwargs['xlim']
    else:
        xlim = None
        
    if kwargs.has_key('ylim'):
        ylim = kwargs['ylim']
    else:
        ylim = None
        
    ax.hist(
        # data, bins = int(np.log(max(3, data.shape[0]/2))),
        data,
        alpha = 0.5, orientation = orientation)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
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

def interactive():
    """basic example for interactive plotting from GUI interaction via felix stiehler"""

    from functools import partial
    
    set_interactive(1)

    def on_click(event, ax, data):
        """
        Left click: show real size
        Right click: resize
        """
        print 'button pressed', event.button, event.xdata, event.ydata, data.shape
        if event.xdata is not None:
            # data = np.array([[event.xdata, event.ydata],])
            # decoded = decoder.predict(data)
            # decoded = reshape_paths(decoded, flat=False)
            # if args.rel_coords:
            #     decoded = np.cumsum(decoded, axis=1)

            print "ax", ax, "data", data
            ax.clear()

            datarow = int(event.ydata)
            
            ax.plot(data[datarow,:], "k-o", alpha = 0.5)
            plt.pause(1e-6)
            # if event.button == 1:
            #     # left click
            #     if args.rel_coords:
            #         ax.set_xlim([-0.5, 0.5])
            #         ax.set_ylim([-0.5, 0.5])
            #     else:
            #         ax.set_xlim([0.0, 1.0])
            #         ax.set_ylim([0.0, 1.0])
            # else:
            #     ax.autoscale(True)
            # ax.scatter(decoded.T[0], decoded.T[1])
    
    X = np.random.binomial(10, 0.1, size = (30, 10))
    
    fig = plt.figure()

    gs = gridspec.GridSpec(1, 2)

    cmap = plt.get_cmap("Reds")
    
    ax0 = fig.add_subplot(gs[0])
    ax0.pcolormesh(X, cmap = cmap)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(X[0], "k-o", alpha = 0.5)

    plt.pause(1e-6)

    fig.canvas.mpl_connect('button_press_event', partial(on_click, ax = ax1, data = X))

    set_interactive(0)
    
    plt.show()
    
def custom_colorbar():
    """custom_colorbar

    basic example for custom colorbar geometry control
     1. create the colorbar axis with the gridspec, plot the colorbar and set its aspect to match the quadmesh
     2. use inset_axis to create the colorbar axis
    """
    # notes

    # subplot / grid / axes
    # https://matplotlib.org/users/gridspec.html
    # http://matplotlib.org/1.5.3/examples/axes_grid/index.html
    # subplot, subplot2grid, rowspan/colspan, gridspec, axes_grid

    # colorbar
    # custom colorbar with custom axis from grid
    # axes_grid, insets, ...
    # http://matplotlib.org/1.5.3/examples/axes_grid/demo_colorbar_with_inset_locator.html
        
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    numplots = 3
        
    a = np.random.exponential(scale = 1.0)
    b = np.random.exponential(scale = 1.0)
    n = 32
    X = np.random.beta(a = a, b = b, size = (numplots, n, n))
    print "a = %f, b = %f, X = %s" % (a, b, X)

    fig = plt.figure()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6, 3])
    # ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)        

    gs = gridspec.GridSpec(2, 2 * numplots, width_ratios = [9, 1] * numplots)
    gs.hspace = 0.05
    gs.wspace = 0.05
        
    for i in range(numplots):
        cmap = plt.get_cmap("Reds")
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(X[i]))
            
        # ax_im = plt.subplot2grid((1, 2 * numplots), (0, (2 * i)    ))
        # ax_cb = plt.subplot2grid((1, 2 * numplots), (0, (2 * i) + 1))
        # ax = fig.add_subplot(1, numplots, i+1)
        ax_im = fig.add_subplot(gs[0, (2 * i)    ])
        ax_cb = fig.add_subplot(gs[0, (2 * i) + 1])

        img = ax_im.pcolormesh(X[i], norm = norm, cmap = cmap)
        ax_im.set_aspect(1.0)

        cb1 = mpl.colorbar.ColorbarBase(
            ax_cb, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('Some Units')
        ax_cb.set_aspect(9.0/1.0)

        print ax_im.get_position(), ax_im.get_aspect()
        print ax_cb.get_position(), ax_cb.get_aspect()
            
        # cbar = plt.colorbar(mappable = img, orientation = "vertical", cax = ax_cb)
        w_im, h_im = get_ax_size(fig, ax_im)
        w_cb, h_cb = get_ax_size(fig, ax_cb)

        print "w_im = %s, h_im = %s" % (w_im, h_im)
        print "w_cb = %s, h_cb = %s" % (w_cb, h_cb)

    for i in range(numplots):
        cmap = plt.get_cmap("Reds")
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(X[i]))
            
        # ax_im = plt.subplot2grid(gs.get_geometry(), (0, (2 * i)    ), colspan = 2)
        # ax_cb = plt.subplot2grid(gs.get_geometry(), (0, (2 * i) + 1))
        # ax = fig.add_subplot(1, numplots, i+1)
        ax_im = fig.add_subplot(gs[1,(2 * i):(2*i+1)])
        # ax_cb = inset_axes(ax_im,
        #         width="50%",  # width = 10% of parent_bbox width
        #         height="5%",  # height : 50%
        #         loc=1)
        ax_cb = inset_axes(ax_im,
                               width="5%",  # width = 10% of parent_bbox width
                               height = "%d" % (i * 30 + 40, ) + "%",  # height : 50%
                               loc=3,
                               bbox_to_anchor=(1.05, 0., 1, 1),
                               bbox_transform=ax_im.transAxes,
                               borderpad=0,
                               )
        # ax_cb = fig.add_subplot(gs[(numplots * 2) + (2 * i) + 1])

        img = ax_im.pcolormesh(X[i], norm = norm, cmap = cmap)
        ax_im.set_aspect(1.0)

        cb1 = mpl.colorbar.ColorbarBase(
            ax_cb, cmap=cmap, norm=norm, orientation='vertical')
        cb1.set_label('Some Units')
        # ax_cb.set_aspect(9.0/1.0)

        print ax_im.get_position(), ax_im.get_aspect()
        print ax_cb.get_position(), ax_cb.get_aspect()
            
        w_im, h_im = get_ax_size(fig, ax_im)
        w_cb, h_cb = get_ax_size(fig, ax_cb)

        print "w_im = %s, h_im = %s" % (w_im, h_im)
        print "w_cb = %s, h_cb = %s" % (w_cb, h_cb)

            

    fig.show()
    
    plt.show()

# plotting funcs for callback
plotfuncs = {
    'timeseries': timeseries,
    'histogram': histogram,
    'histogramnd': histogramnd,
    'plot_scattermatrix': plot_scattermatrix,
    'plot_img': plot_img,
    }

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", type=str, default = "custom_colorbar", help = "testing mode: [custom_colorbar], interactive")

    args = parser.parse_args()
    # fig = makefig(2, 3)

    if args.mode == "custom_colorbar":
        custom_colorbar()
    elif args.mode == "interactive":
        interactive()
    else:
        print "Unknown mode %s, exiting" % (args.mode)
        sys.exit(1)
        
