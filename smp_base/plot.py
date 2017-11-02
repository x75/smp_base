"""smp_base.plot - plotting functions

.. moduleauthor:: Oswald Berthold, 2017

Depends: numpy, matplotlib, pyunicorn, seaborn, pandas

Includes:
 - config variables: plot_colors, ...
 - utility functions for creating figures and subplot grids and for computing and setting of plot parameters, custom_colorbar
 - low-level kwargs-configurable plotting funcs: timeseries, histogram, histogramnd, rp_timeseries_embedding, plot_scattermatrix, plot_img, ...
 - TODO: custom_colorbar, custom_legend moving smartly out of the way
 - TODO: plotting style configuration: fonts, colors, sizes, formats
 - TODO: sift existing plotting funcs from smp* models, systems, ...
 - TODO: clean up and merge with sift results
"""
from functools import partial
from cycler import cycler

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.colors as mplcolors
import matplotlib.patches as mplpatches
from  matplotlib import rc, rcParams, rc_params

# perceptually uniform colormaps
import colorcet as cc

import numpy as np

from plot_utils import put_legend_out_right

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

# all predefined matplotlib colors
plot_colors = mplcolors.get_named_colors_mapping()
plot_colors_idx = 0

def find_smallest_rectangle(l = 1):
    l_sqrt = int(np.floor(np.sqrt(l)))
    print "sq(l) = %d" % l_sqrt
    # for i in range(1, l/l_sqrt):
    w = l_sqrt
    h = l/w
    while w * h < l:
        w += 1
        # h = l/w
        # print "rect", w*h, l, w, h
    # print "l = %f" % (l_sqrt, )
    return w, h

def test_plot_colors():
    # fig = makefig(rows = 1, cols = 1, title = "Show all colors in smp_base.plot.plot_colors")
    print "plot_colors type = %s, len = %d" % (type(plot_colors), len(plot_colors))
    ncols,nrows = find_smallest_rectangle(len(plot_colors))
    # print "plot_colors dir = %s" % (dir(plot_colors), )
    # print "plot_colors keys = %s" % (plot_colors.keys(), )

    colors = plot_colors
    plot_colors_img = np.zeros((ncols, nrows))
    
    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mplcolors.rgb_to_hsv(mplcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    
    n = len(sorted_names)
    # ncols = 4
    # nrows = n // ncols + 1
    

    print "sorted_names", sorted_names, "n", n
    
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    ywidth = Y / (nrows + 1)
    xwidth = X / ncols
    h = ywidth
    w = xwidth

    for i, name in enumerate(sorted_names):
        col = i % ncols
        row = i // ncols
        
        y = Y - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        ax.text(xi_text, (nrows * h) - y, name, fontsize=(h * 0.4),
                horizontalalignment='left',
                verticalalignment='center')

        # ax.hlines(y + h * 0.1, xi_line, xf_line,
        #         color=colors[name], linewidth=(h * 0.6))

        xpos = col
        ypos = row
        
        # elementary shape without buffersize
        ax.add_patch(
            mplpatches.Rectangle(
                # (30, ypos - (v.shape[0]/2.0) - (yspacing / 3.0)),   # (x,y)
                (xpos * xwidth, ypos * ywidth),   # (x,y)
                xwidth,          # width
                ywidth,          # height
                fill = True,
                color = colors[name],
                # hatch = "|",
                hatch = "-",
            )
        )

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1,
                        top=1, bottom=0,
                        hspace=0, wspace=0)
    plt.show()  

def make_axes_from_grid(fig, gs):
    """Generate 2D array of subplot axes from a gridspec

    Args:
     - fig(matplotlib.figure.Figure): a matplotlib figure handle
     - gs(matplotlib.gridspec.GridSpec): the gridspec

    Returns:
     - list of lists (2D array) of subplot axes
    """
    axes = []
    (rows, cols) = gs.get_geometry()
    for row in range(rows):
        axes.append([])
        for col in range(cols):
            axes[-1].append(fig.add_subplot(gs[row, col]))
    return axes

def make_axes_from_spec(fig, gs, axesspec):
    """Generate 2D array of subplot axes from an irregular axes specification

    Args:
     - fig(matplotlib.figure.Figure): a matplotlib figure handle
     - gs(matplotlib.gridspec.GridSpec): the gridspec
     - axesspec(list): list of gridspec slices

    Returns:
     - list of lists (2D array) of subplot axes
    """
    axes = []
    # (rows, cols) = gs.get_geometry()
    for axspec in axesspec:
        axes.append([])
        # for col in range(cols):
        axes[-1].append(fig.add_subplot(gs[axspec]))
    return axes

def makefig(rows = 1, cols = 1, wspace = 0.0, hspace = 0.0, axesspec = None, title = None):
    """Create a matplotlib figure using the args as spec

    Alias for :func:`make_fig`
    """
    return make_fig(rows = rows, cols = cols, wspace = wspace, hspace = hspace, axesspec = axesspec, title = title)

def make_fig(rows = 1, cols = 1, wspace = 0.0, hspace = 0.0, axesspec = None, title = None):
    """Create a matplotlib figure using the args as spec, create the figure, the subplot gridspec and the axes array

    Args:
     - rows (int): number of rows
     - cols (int):  number of cols
     - wspace (float):  horizontal padding
     - hspace (float):  vertical padding
     - axesspec (list):  list of slices to create irregular grids by slicing the regular gridspec
     - title (str):  the figure's super title (suptitle)
       
    Returns:
     - fig (matplotlib.figure.Figure): the figure handle `fig`
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
    """This function does something.

    Args:
       name (str):  The name to use.

    Kwargs:
       state (bool): Current state to be in.

    Returns:
       int.  The return code::

          0 -- Success!
          1 -- No good.
          2 -- Try again.

    Raises:
       AttributeError, KeyError
     """
    if interactive:
        plt.ion()
    else:
        plt.ioff()

def get_ax_size(fig, ax):
    """Get the size of an axis

    Args:
     - fig: figure handle
     - ax: the axis handle

    Returns:
     - tuple: width, height

    Scraped from stackoverflow, noref.
    """
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

def get_colorcycler(cmap_str = None, cmap_idx = None, c_s = 0, c_e = 255, c_n = 20):
    """get a colorcycler for the cmap 'cmapstr'

    Arguments:
     - cmap_str(str): colormap id string ['rainbow']
     - cmap_idx(array): indices into 0 - 255 to directly select the
       colors of the cycle [None]. If None, cmap_idx will be generated
       with the c_\* parameters.
     - c_s: cmap_idx start
     - c_e: cmap_idx end
     - c_n: cmap_idx numsteps
    """
    if cmap_str is None:
        cmap_str = 'hot'
    cmap = cc.cm[cmap_str] # rainbow

    if cmap_idx is None:
        # cmap_idx = np.linspace(0, 1000, 345, endpoint = False)
        cmap_idx = np.linspace(c_s, c_e, c_n, endpoint = False)
        # print "cmap_idx", cmap_idx, cmap.N
        cmap_idx = [int(i) % cmap.N for i in cmap_idx]
        # print "cmap_idx", cmap_idx
        
    colorcycler = cycler('color', [c for c in cmap(cmap_idx)])
    return colorcycler

def configure_style():
    # colors
    # cmap = plt.get_cmap("Oranges")
    # cmap = cc.cm['isolum'] # isoluminance
    colorcycler = get_colorcycler('rainbow')
    # colorcycler = get_colorcycler('cyclic_mrybm_35_75_c68')
    
    # rc = rc_params()
    rc('axes', prop_cycle = colorcycler)
    
    # print "cc", colorcycler

def kwargs_plot_clean(**kwargs):
    """create kwargs dict from scratch by copying fixed list of item from old kwargs
    """
    kwargs_ = dict([(k, kwargs[k]) for k in ['xticks', 'yticks', 'xticklabels', 'yticklabels'] if kwargs.has_key(k)])
    return kwargs_

def kwargs_plot_clean_hist(**kwargs):
    """create kwargs dict from scratch by copying fixed list of item from old kwargs
    """
    return dict([(k, kwargs[k]) for k in kwargs.keys() if k not in [
        'ordinate', 'title',
        'xticks', 'yticks', 'xticklabels', 'yticklabels',
        'xlim', 'ylim', 'xscale', 'yscale', 'xlabel', 'ylabel',
        ]])

def timeseries(ax, data, **kwargs):
    """Plot data as timeseries

    Args:
     - ax: subplot axis
     - data (numpy.ndarray): n x m ndarray with the data

    Kwargs:
     - marker: mpl marker style
     - linestyle: mpl linestyle
     - linewidth: mpl linewidth
     - ...

    Returns:
     - None

    """
    
    kwargs_ = {
        # style params
        # axis title
        
        'title': 'timeseries of %s-shaped data' % (data.shape,),
        'xscale': 'linear',
        'yscale': 'linear',
        'xlim': None,
        'ylim': None,
        'xlabel': 'time steps [t]',
        'ylabel': 'activity [unit-free]',
        'alpha': 0.5,
        'marker': 'None',
        'linestyle': 'solid',
    }
        
    kwargs_.update(**kwargs)
    kwargs = kwargs_plot_clean_hist(**kwargs_)
    
    # x-axis shift / bus delay compensation
    if kwargs.has_key('delay'):
        data = np.roll(data, kwargs['delay'], axis = 1)

    # clean up kwargs to avoid unintended effects
    # kwargs_ = {} # kwargs_plot_clean(**kwargs)
        
    # explicit xaxis
    if kwargs.has_key('ordinate'):
        ax.plot(
            kwargs['ordinate'], data, **kwargs)
        # alpha = alpha,
        #     marker = marker, linestyle = linestyle, linewidth = linewidth,
        #     label = label,
        #     **kwargs_)
    else:
        ax.plot(
            data, **kwargs)
        # alpha = alpha, marker = marker,
        #     linestyle = linestyle, label = label,
        #     **kwargs_)

    # axis labels
    if kwargs_.has_key('xlabel'):
        ax.set_xlabel('%s' % kwargs_['xlabel'])

    if kwargs_.has_key('ylabel'):
        ax.set_ylabel('%s' % kwargs_['ylabel'])
    
    # axis scale: linear / log
    ax.set_xscale(kwargs_['xscale'])
    ax.set_yscale(kwargs_['yscale'])
    # axis limits: inferred / explicit
    if kwargs_['xlim'] is not None:
        ax.set_xlim(kwargs_['xlim'])
    if kwargs_['ylim'] is not None:
        ax.set_ylim(kwargs_['ylim'])
    # axis ticks
    ax_set_ticks(ax, **kwargs_)
    # axis title and fontsize
    ax.title.set_text(kwargs_['title'])
    ax.title.set_fontsize(8.0) # kwargs_[

def ax_set_ticks(ax, **kwargs):
    if kwargs.has_key('xticks'):
        if not kwargs['xticks']:
            ax.set_xticks([])
            ax.set_xticklabels([])
        # else:
        #     ax.set_xticks(kwargs['xticks'])
        #     ax.set_xticklabels(kwargs['xticks'])
                
    if kwargs.has_key('yticks'):
        # print "timeseries kwargs[yticks]", kwargs['yticks']
        if not kwargs['yticks']:
            ax.set_yticks([])
            ax.set_yticklabels([])
            # print "timeseries disabling yticks"
        # else:
        #     ax.set_yticks(kwargs['yticks'])
            
def histogram(ax, data, **kwargs):
    """histogram plot"""
    assert len(data.shape) > 0
    # print "histo kwargs", kwargs
    kwargs_ = {
        # style params
        # axis title
        'title': 'histogram of %s-shaped data' % (data.shape,),
        'orientation': 'vertical',
        'xscale': 'linear',
        'yscale': 'linear',
        'xlim': None,
        'ylim': None,
    }
    kwargs_.update(**kwargs)
    kwargs = kwargs_plot_clean_hist(**kwargs_)
    
    # if not kwargs.has_key('histtype'):
    #     kwargs_['histtype'] = kwargs['histtype']

    # print "histogram kwargs", kwargs.keys()

    if kwargs_['ylim'] is not None and kwargs_['orientation'] == 'horizontal':
        bins = np.linspace(kwargs_['ylim'][0], kwargs_['ylim'][1], 21)
    elif kwargs_['xlim'] is not None and kwargs_['orientation'] == 'vertical':
        bins = np.linspace(kwargs_['xlim'][0], kwargs_['xlim'][1], 21)
    else:
        bins = 'auto'
    (n, bins, patches) = ax.hist(
        # data, bins = int(np.log(max(3, data.shape[0]/2))),
        data, bins = bins, **kwargs)

    # print "hist bins= %s", bins
    
    ax.set_xscale(kwargs_['xscale'])
    ax.set_yscale(kwargs_['yscale'])
    if kwargs_['xlim'] is not None:
        ax.set_xlim(kwargs_['xlim'])
    if kwargs_['ylim'] is not None:
        ax.set_ylim(kwargs_['ylim'])
    ax.title.set_text(kwargs_['title'])
    ax.title.set_fontsize(8.0) # kwargs_[

    ax_set_ticks(ax, **kwargs_)
    
    # put_legend_out_right(resize_by = 0.8, ax = ax)
    
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
    #     norm = mplcolors.LogNorm(vmin=data.min(), vmax=data.max())
    
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
    # print "a = %f, b = %f, X = %s" % (a, b, X)

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


# class uniform_divergence():
#     def __call__(self, f):
#         def wrap(_self, *args, **kwargs):
#             print "f", f
#             print "args", args
#             print "kwargs", kwargs
#             # return args[0]
#         return wrap

def uniform_divergence(*args, **kwargs):
    """Compute histogram based divergence of bivariate data distribution from prior distribution

    Args:
       args[0](numpy.ndarray, pandas.Series): timeseries X_1
       args[1](numpy.ndarray, pandas.Series): timeseries X_2

    Kwargs:
       color: colors to use
       f: plotting function (hist2d, hexbin, ...)
       xxx: ?

    Returns:
       put image plotting primitve through
    """
    # print "f", f
    # print "args", len(args),
    # for arg in args:
    #     print "arg %s %s" % (type(arg), len(arg))
    # print "kwargs", kwargs.keys()
    f = kwargs['f']
    del kwargs['f']
    color = kwargs['color']
    del kwargs['color']
    # print "f", f
    # return partial(f, args, kwargs) # (args[0], args[1], kwargs)
    # ax = f(args[0].values, args[1].values, kwargs) # (args[0], args[1], kwargs)
    prior = 'uniform'
    if len(args) < 2:
        args = (args[0], args[0])
        prior = 'identity'
    
    h, xe, ye = np.histogram2d(args[0], args[1], normed = True)

    X1 = np.random.uniform(xe[0], xe[-1], len(args[0]))
    if prior == 'identity':
        X2 = X1.copy()
    else:
        X2 = np.random.uniform(ye[0], ye[-1], len(args[1]))
    X2 = np.random.uniform(ye[0], ye[-1], len(args[1]))
    
    h_unif, xe_unif, ye_unif = np.histogram2d(
        X1,
        X2,
        bins = (xe, ye),
        normed = True        
    )

    # really normalize
    h = h / np.sum(h)
    h_unif = h_unif / np.sum(h_unif)
    
    # print "h", h, "xe", xe, "ye", ye
    # print "h_unif", h_unif, "xe_unif", xe_unif, "ye_unif", ye_unif
    # ax = f(*args, **kwargs)
    plt.grid(0)
    X, Y = np.meshgrid(xe, ye)
    # ax = plt.imshow(h - h_unif, origin = 'lower', interpolation = 'none', )
    # difference
    # h_ = (h - h_unif)
    # divergence
    def div_kl(h1, h2):
        # div = np.sum(h1 * np.log(h1/h2))
        print "h1", h1
        print "h2", h2
        log_diff = np.clip(np.log(h1/h2), -20.0, 7.0)
        print "log diff", log_diff
        div = h1 * log_diff
        print "div", div.shape, div
        return div
    h_ = div_kl(h, h_unif)

    ud_cmap = cc.cm['diverging_cwm_80_100_c22'] # rainbow
    # ud_cmap = cc.cm['coolwarm'] # rainbow
    ud_vmin = -1
    ud_vmax =  1
    # ud_cmap = cc.cm['blues'] # rainbow
    # ud_cmap = plt.get_cmap('coolwarm')

    ud_norm = mplcolors.Normalize(vmin = ud_vmin, vmax = ud_vmax)
    
    ax = plt.pcolormesh(X, Y, h_, cmap = ud_cmap, norm = ud_norm)
    # plt.xlim((xe[0], xe[-1]))
    # plt.ylim((ye[0], ye[-1]))
    plt.colorbar()

    # ax_unif = f( bins = ax[1])
    # print "ax", ax
    return ax
    
    
# plotting funcs for callback
plotfuncs = {
    'timeseries': timeseries,
    'histogram': histogram,
    'histogramnd': histogramnd,
    'plot_scattermatrix': plot_scattermatrix,
    'plot_img': plot_img,
    'hexbin': plt.hexbin,
    'hexbin': plt.hexbin,
    'hist2d': plt.hist2d,
    'kdeplot': sns.kdeplot,
    'scatter': plt.scatter,
    'uniform_divergence': uniform_divergence,
    }

# configure, ah, style
configure_style()
    
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", type=str, default = "custom_colorbar", help = "testing mode: [custom_colorbar], interactive, plot_colors")

    args = parser.parse_args()
    # fig = makefig(2, 3)

    if args.mode == "custom_colorbar":
        custom_colorbar()
    elif args.mode == "interactive":
        interactive()
    elif args.mode == "plot_colors":
        test_plot_colors()
    else:
        print "Unknown mode %s, exiting" % (args.mode)
        sys.exit(1)
        
