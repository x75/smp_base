"""smp_base.plot

plotting functions

.. moduleauthor:: Oswald Berthold, 2017

Depends: numpy, matplotlib

Optional: pyunicorn, seaborn, pandas

Includes:
 - config variables: plot_colors, ...
 - utility functions for creating figures and subplot grids and for
   computing and setting of plot parameters, custom_colorbar
 - low-level kwargs-configurable plotting funcs: timeseries,
   histogram, histogramnd, rp_timeseries_embedding,
   plot_scattermatrix, plot_img, ...
 - TODO: custom_colorbar, custom_legend moving smartly out of the way
 - TODO: plotting style configuration: fonts, colors, sizes, formats
 - TODO: sift existing plotting funcs from smp* models, systems, ...
 - TODO: clean up and merge with sift results
"""
from functools import partial, wraps
from collections import OrderedDict
from cycler import cycler
import copy

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.colors as mplcolors
import matplotlib.patches as mplpatches
from matplotlib.table import Table as mplTable
from matplotlib.font_manager import FontManager
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

import logging
from smp_base.common import get_module_logger

loglevel_debug = logging.DEBUG - 1
logger = get_module_logger(modulename = 'plot', loglevel = logging.DEBUG)

from smp_base.measures import div_kl, meas_hist

# all predefined matplotlib colors
plot_colors = mplcolors.get_named_colors_mapping()
plot_colors_idx = 0

# class dec_import_unicorn():
#     """wrapper for failsafe import dependence

#     check if the required import 'import_name' is satisfied by testing
#     `eval(import_name)` and running fallback function substitute
#     """
#     # def __init__(self, **kwargs):
#     #     self.import_name = 'True'
#     #     if kwargs.has_key('import_name'):
#     #         self.import_name = kwargs['import_name']
            
#     def __call__(self, f):
#         # if lib is not None:
            
#         def wrap(ax, data, *args, **kwargs): # _self, 
                
#             # logger.log(loglevel_debug, "f", f)
#             # logger.log(loglevel_debug, "args", args)
#             # logger.log(loglevel_debug, "kwargs", kwargs)
#             # return args[0]
#             # import_flag = eval(self.import_name)
#             import_flag = HAVE_PYUNICORN
            
#             if import_flag:
#                 f_ = f
#             else:
#                 # f_ = lambda f: logger.log(loglevel_debug, "Required import %s not satisfied, requested func %s not executed" % (import_name, f))
#                 def f_(ax, data, *args, **kwargs):
#                     logger.log(loglevel_debug, "Required import %s not satisfied, requested func %s not executed" % (import_name, f))
#                 # return f_()
                
#             f_(ax, data, args, kwargs)
#         return wrap

def find_smallest_rectangle(l = 1):
    l_sqrt = int(np.floor(np.sqrt(l)))
    logger.log(loglevel_debug, "sq(l) = %d" % l_sqrt)
    # for i in range(1, l/l_sqrt):
    w = l_sqrt
    h = l/w
    while w * h < l:
        w += 1
        # h = l/w
        # logger.log(loglevel_debug, "rect", w*h, l, w, h)
    # logger.log(loglevel_debug, "l = %f" % (l_sqrt, ))
    return w, h

def test_plot_colors():
    # fig = makefig(rows = 1, cols = 1, title = "Show all colors in smp_base.plot.plot_colors")
    logger.log(loglevel_debug, "plot_colors type = %s, len = %d" % (type(plot_colors), len(plot_colors)))
    ncols,nrows = find_smallest_rectangle(len(plot_colors))
    # logger.log(loglevel_debug, "plot_colors dir = %s" % (dir(plot_colors), ))
    # logger.log(loglevel_debug, "plot_colors keys = %s" % (plot_colors.keys(), ))

    colors = plot_colors
    plot_colors_img = np.zeros((ncols, nrows))
    
    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mplcolors.rgb_to_hsv(mplcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    
    n = len(sorted_names)
    # ncols = 4
    # nrows = n // ncols + 1
    

    logger.log(loglevel_debug, "sorted_names", sorted_names, "n", n)
    
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

    Gridspec slice example by slicing parts out of the maximum resolution grid:

    .. code::

        # regular
        axesspec = [(0, 0), (0,1), (1, 0), (1,1)]
        # row 1 colspan is 3
        axesspec = [(0, 0), (0, 1), (0, 2), (0, slice(3, None))]
    
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

    Arguments:
     - rows (int): number of rows
     - cols (int):  number of cols
     - wspace (float):  horizontal padding
     - hspace (float):  vertical padding
     - axesspec (list): list of slices to create irregular grids by
       slicing the regular gridspec
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
        
    # logger.log(loglevel_debug, "fig.axes", fig.axes)
    fig.subplots_adjust(wspace = wspace, hspace = hspace)
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
        # logger.log(loglevel_debug, "cmap_idx", cmap_idx, cmap.N)
        cmap_idx = [int(i) % cmap.N for i in cmap_idx]
        # logger.log(loglevel_debug, "cmap_idx = %s", cmap_idx)
        
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
    
    # logger.log(loglevel_debug, "cc", colorcycler)

# def kwargs_plot_clean_plot(**kwargs):
#     """create kwargs dict from scratch by copying fixed list of item from old kwargs
#     """
#     # kwargs_ = dict([(k, kwargs[k]) for k in ['xticks', 'yticks', 'xticklabels', 'yticklabels'] if kwargs.has_key(k)])
#     # return kwargs_
#     return dict([(k, kwargs[k]) for k in kwargs.keys() if k not in [
#         'delay', 'ordinate',
#         'title', 'title_pos', 'aspect',
#         'labels',
#         'xticks', 'yticks', 'xticklabels', 'yticklabels', 'xinvert', 'yinvert',
#         'xlim', 'ylim', 'xscale', 'yscale', 'xlabel', 'ylabel',
#         # specific
#         'orientation',
#     ]])

# def kwargs_plot_clean_hist(**kwargs):
#     """create kwargs dict from scratch by copying fixed list of item from old kwargs
#     """
#     return dict([(k, kwargs[k]) for k in kwargs.keys() if k not in [
#         'delay', 'ordinate',
#         'title', 'aspect',
#         'labels',
#         'xticks', 'yticks', 'xticklabels', 'yticklabels', 'xinvert', 'yinvert',
#         'xlim', 'ylim', 'xscale', 'yscale', 'xlabel', 'ylabel',
#     ]])

# def kwargs_plot_clean_histogram(**kwargs):
#     """create kwargs dict from scratch by copying fixed list of item from old kwargs
#     """
#     return dict([(k, kwargs[k]) for k in kwargs.keys() if k not in [
#         'delay', 'ordinate',
#         'title', 'aspect',
#         'labels',
#         'xticks', 'yticks', 'xticklabels', 'yticklabels', 'xinvert', 'yinvert',
#         'xlim', 'ylim', 'xscale', 'yscale', 'xlabel', 'ylabel',
#         # specific
#         'alpha', 'marker', 'linestyle',
#         'orientation',
#     ]])

# def kwargs_plot_clean_bar(**kwargs):
#     """create kwargs dict from scratch by copying fixed list of item from old kwargs
#     """
#     return dict([(k, kwargs[k]) for k in kwargs.keys() if k not in [
#         'delay', 'ordinate',
#         'title', 'aspect',
#         'labels',
#         'xticks', 'yticks', 'xticklabels', 'yticklabels', 'xinvert', 'yinvert',
#         'xlim', 'ylim', 'xscale', 'yscale', 'xlabel', 'ylabel',
#         # specific
#         'orientation',
#         'marker',
#         ]])

def plot_clean_kwargs(clean_type = None, **kwargs):
    clean = {'common': [
        'aspect',
        'delay',
        'labels',
        'ordinate',
        'title',
        'title_pos',
        'xinvert',
        'xlabel',
        'xlim',
        'xscale',
        'xticklabels',
        'xticks',
        'xtwin',
        'yinvert',
        'ylabel',
        'ylim',
        'yscale',
        'yticklabels',
        'yticks',
        'ytwin',
        'lineseg_val', 'lineseg_idx',
        'density',
        'bins',
    ]}
    clean['plot'] = ['orientation']
    clean['histogram'] = ['orientation', 'marker', 'alpha', 'linestyle', 'histtype', 'color']
    clean['bar'] = ['orientation', 'marker', 'histtype', 'normed']
    clean['pcolor'] = ['colorbar']
    # clean['linesegments'] = ['lineseg_val', 'lineseg_idx']

    clean_keys = clean['common']
    if clean_type is not None:
        if clean.has_key(clean_type):
            clean_keys += clean[clean_type]
    
    return dict([(k, kwargs[k]) for k in kwargs.keys() if k not in clean_keys])
    

def ax_set_title(ax, **kwargs):
    ax.title.set_text(kwargs['title'])
    ax.title.set_alpha(0.65)

    # axis title and fontsize
    if kwargs['title_pos'] == 'top_in':
        # ax.text(
        #     0.5, 0.9,
        #     kwargs['title'],
        #     horizontalalignment = 'center',
        #     transform = ax.transAxes,
        #     alpha = 0.65,
        #     # bbox = dict(facecolor='red', alpha=0.5),
        # )
        ax.title.set_position((0.5, 0.9))
    elif kwargs['title_pos'] in ['bottom', 'bottom_out']:
        # ax.title.set_text(kwargs['title'], alpha = 0.65)
        ax.title.set_position((0.5, -0.1))
    elif kwargs['title_pos'] in ['bottom_in']:
        # ax.title.set_text(kwargs['title'], alpha = 0.65)
        ax.title.set_position((0.5, 0.1))
    else: # top_out
        ax.title.set_position((0.5, 1.05))
        # ax.title.set_text(kwargs['title'], alpha = 0.65)

class plotfunc(object):
    def __call__(self, f):
        _loglevel = loglevel_debug + 0

        @wraps(f)
        def wrap(ax, data, *args, **kwargs):
            kwargs_ = {
                # style params
                # axis title
                'alpha': 0.5,
                'linestyle': 'solid',
                'marker': 'None',
                'orientation': 'horizontal',
                'title': '%s of %s-shaped data' % (f.func_name, data.shape,),
                'title_pos': 'top_in',
                'xinvert': None,
                'xlabel': 'steps [n]',
                'xlim': None,
                'xscale': 'linear',
                'yinvert': None,
                'ylabel': 'activity [x]',
                'ylim': None,
                'yscale': 'linear',
            }

            # update defaults with incoming kwargs
            kwargs_.update(**kwargs)

            kwargsf = {}
            kwargsf.update(**kwargs_)

            logger.log(_loglevel, 'plotfunc f = %s' % (f.func_name, ))
            
            # set axis title
            ax_set_title(ax, **kwargs_)
    
            # x-axis shift / bus delay compensation
            if kwargs_.has_key('delay'):
                data = np.roll(data, kwargs['delay'], axis = 1)

            logger.log(_loglevel, 'plotfunc kwargs_ = %s' % (kwargs_.keys(), ))
            # call plotfunc
            fval = f(ax, data, *args, **kwargsf)

            if fval is not None:
                kwargs_.update(**fval)

            logger.log(_loglevel, 'plotfunc kwargs_ = %s' % (kwargs_.keys(), ))
            
            # axis labels
            if kwargs_.has_key('xlabel') and kwargs_['xlabel'] and kwargs_['xlabel'] is not None:
                ax.set_xlabel('%s' % kwargs_['xlabel'])

            if kwargs_.has_key('ylabel') and kwargs_['ylabel'] and kwargs_['ylabel'] is not None:
                ax.set_ylabel('%s' % kwargs_['ylabel'])
    
            # axis scale: linear / log
            ax.set_xscale(kwargs_['xscale'])
            ax.set_yscale(kwargs_['yscale'])
            # axis limits: inferred / explicit
            if kwargs_['xlim'] is not None:
                ax.set_xlim(kwargs_['xlim'])
            if kwargs_['ylim'] is not None:
                ax.set_ylim(kwargs_['ylim'])

            # # axis aspect
            # if kwargs_.has_key('aspect'):
            #     ax.set_aspect(kwargs_['aspect'])

            # axis ticks
            ax_set_ticks(ax, **kwargs_)

            return fval
    
        return wrap

@plotfunc()
def table(ax, data, **kwargs):
    kwargs_ = {
        # style params
        # axis title
        'alpha': 0.5,
        'linestyle': 'solid',
        'marker': 'None',
        'orientation': 'horizontal',
        'title': '%s of %s-shaped data' % ('table', data.shape,),
        'xinvert': None,
        'xlabel': 'steps [n]',
        'xlim': None,
        'xscale': 'linear',
        'yinvert': None,
        'ylabel': 'activity [x]',
        'ylim': None,
        'yscale': 'linear',
    }
    _loglevel = loglevel_debug + 1
    
    # logger.log(_loglevel, '    table ax = %s, data = %s, kwargs = %s' % (ax.title.get_text(), data, kwargs.keys()))
    # logger.log(_loglevel, '    table ' % (ax.title.get_text(), data, kwargs.keys()))

    ax.axis('tight')
    ax.axis('off')
    
    colors = None # ['r','g','b']
    colLabels = None # ['Measure']
    rowLabels = kwargs['labels'] # ['Average', 'Service Average', 'Benchmark']
    cellText = [['%.04f' % (_, )] for _ in data.T] # .T # [overall, svc_avg, benchmark]
    # logger.log(_loglevel, '    running ax.table on cellText = %s' % (cellText, ))
    
    font = FontManager(size = 8)
    the_table = ax.table(
        cellText = cellText,
        rowLoc = 'right',
        rowColours = colors,
        rowLabels = rowLabels,
        colWidths = [.5,.5],
        colLabels = None, # colLabels,
        colLoc = 'center',
        loc = 'center',
        # fontsize = 6,
    )

    # table position tuning
    # left bottom, width, height
    # ax.set_position([0.0, 0.0, 1.0, 0.9])
    # the_table.set_position([0.0, 0.0, 1.0, 0.9])
    # table = mplTable()
    bbox = ax.get_position()
    # logger.log(_loglevel, '    ax position = %s' % (bbox, ))
    bbox.x0 *= 1.2
    bbox.y0 *= 0.7

    ax.set_position(bbox)
    bbox2 = ax.get_position()
    # logger.log(_loglevel, '    ax position = %s' % (bbox2, ))
    
    # EDIT: Thanks to Oz for the answer-- Looping through the properties of the table allows easy modification of the height property:

    table_props = the_table.properties()
    table_cells = table_props['child_artists']
    for cell in table_cells: cell.set_height(0.1)

@plotfunc()
def linesegments(ax, data, **kwargs):
    """plotfunc(linesegments)

    Plot line segments between pairs (tuples) of x,y points.
    """
    kwargs_ = {}
    kwargs_.update(**kwargs)
    logger.info('kwargs = %s', kwargs)

    lineseg_idx = np.array(kwargs['lineseg_idx'])
    lineseg_x   = np.array(kwargs['lineseg_val'][0])

    kwargs = plot_clean_kwargs('plot', **kwargs_)
    
    logger.info('data = %s', data.shape)
    for lineseg_i in lineseg_idx:
        x = list(lineseg_i)
        y = np.hstack((data[lineseg_i[0],[0]], data[lineseg_i[1],[1]]))

        # timeseries()
        ax.plot(x, y, **kwargs)
        
@plotfunc()
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
    
    # kwargs_ = {
    #     # style params
    #     # axis title
        
    #     'alpha': 0.5,
    #     'linestyle': 'solid',
    #     'marker': 'None',
    #     'orientation': 'horizontal',
    #     'title': 'timeseries of %s-shaped data' % (data.shape,),
    #     'xlabel': 'time steps [t]',
    #     'xlim': None,
    #     'xscale': 'linear',
    #     'xinvert': None,
    #     'ylabel': 'activity [x]',
    #     'ylim': None,
    #     'yscale': 'linear',
    #     'yinvert': None,
    # }
    _loglevel = loglevel_debug + 0 # 1
    
    kwargs_ = {}
    kwargs_.update(**kwargs)
    
    # clean up kwargs to avoid unintended effects
    # kwargs_ = {} # kwargs_plot_clean(**kwargs)

    # x axis (ordinate)
    if kwargs_.has_key('ordinate'):
        x = kwargs_['ordinate']
    else:
        x = np.arange(0, data.shape[0])
    # y axis (abscissa)
    y = data

    logger.log(_loglevel, "    timeseries x = %s, y = %s" % (x.shape, y.shape))
    # orientation
    if kwargs_.has_key('orientation') and kwargs_['orientation'] != 'horizontal':
        x_ = x.copy()
        x = y
        y = x_
        logger.log(_loglevel, "    timeseries orientation x = %s, y = %s" % (x.shape, y.shape))
        axis_keys = ['label', 'scale', 'lim', 'ticks', 'ticklabels']
        for ax_key in axis_keys:
            # for ax_name in ['x', 'y']:
            ax_key_x = '%s%s' % ('x', ax_key)
            ax_key_y = '%s%s' % ('y', ax_key)
            if kwargs_.has_key(ax_key_x):
                if kwargs_.has_key(ax_key_y):
                    bla_ = copy.copy(kwargs_[ax_key_y])
                    kwargs_[ax_key_y] = kwargs_[ax_key_x]
                    kwargs_[ax_key_x] = bla_
                else:
                    kwargs_[ax_key_y] = copy.copy(kwargs_[ax_key_x])
                    kwargs_.pop(ax_key_x)
            elif kwargs_.has_key(ax_key_y):
                kwargs_[ax_key_x] = copy.copy(kwargs_[ax_key_y])
                kwargs_.pop(ax_key_y)

    # prepare timeseries kwargs
    # kwargs = kwargs_plot_clean_plot(**kwargs_)
    kwargs = plot_clean_kwargs('plot', **kwargs_)
    
    logger.log(_loglevel, "    timeseries pre plot x = %s, y = %s" % (x.shape, y.shape))
    # plot
    ax.plot(x, y, **kwargs)

    return kwargs_
    
def ax_invert(ax, **kwargs):
    kwargs_ = kwargs
    # logger.log(loglevel_debug, "    plot.ax_invert kwargs_ = %s" % (kwargs_, ))
    # axis invert?
    if kwargs_.has_key('xinvert') and kwargs_['xinvert']: # is not None:
        logger.log(loglevel_debug, "    plot.ax_invert inverting xaxis with xinvert = %s" % (kwargs_['xinvert'], ))
        ax.invert_xaxis()
    if kwargs_.has_key('yinvert') and kwargs_['yinvert']: # is not None:
        logger.log(loglevel_debug, "    plot.ax_invert inverting yaxis with yinvert = %s" % (kwargs_['yinvert'], ))
        ax.invert_yaxis()

def ax_set_ticks(ax, **kwargs):
    ax_xticks = ax.get_xticks()
    ax_yticks = ax.get_yticks()
    logger.log(loglevel_debug, "    plot.ax_set_ticks ax = %s, xticks = %s" % (ax.get_title(), ax_xticks,))
    logger.log(loglevel_debug, "    plot.ax_set_ticks ax = %s, yticks = %s" % (ax.get_title(), ax_yticks,))

    # logger.log(loglevel_debug, "ax_set_ticks kwargs = %s" % (kwargs.keys(),))
    if kwargs.has_key('xticks'):
        if kwargs['xticks'] is None:
            pass
        elif kwargs['xticks'] is False:
            ax.set_xticks([])
            ax.set_xticklabels([])
        elif type(kwargs['xticks']) is list or type(kwargs['xticks']) is tuple:
            ax.set_xticks(kwargs['xticks'])
            ax.set_xticklabels(kwargs['xticks'])
        else:
            ax.set_xticks(ax_xticks)
        logger.log(loglevel_debug, "    plot.ax_set_ticks     kwargs[xticks] = %s, xticks = %s" % (kwargs['xticks'], ax.get_xticks(),))
                
    if kwargs.has_key('xticklabels'):
        if kwargs['xticklabels'] is None:
            pass
        elif kwargs['xticklabels'] is False:
            ax.set_xticklabels([])
        elif type(kwargs['xticklabels']) is list or type(kwargs['xticklabels']) is tuple:
            ax.set_xticklabels(kwargs['xticklabels'])
        logger.log(loglevel_debug, "    plot.ax_set_ticks     kwargs[xticklabels] = %s, xticklabels = %s" % (kwargs['xticklabels'], ax.get_xticklabels(),))
        
    if kwargs.has_key('yticks'):
        # logger.log(loglevel_debug, "timeseries kwargs[yticks]", kwargs['yticks'])
        if kwargs['yticks'] is None:
            pass
        elif kwargs['yticks'] is False:
            ax.set_yticks([])
            ax.set_yticklabels([])
        elif type(kwargs['yticks']) in [list, tuple, np.ndarray]:
            ax.set_yticks(kwargs['yticks'])
            ax.set_yticklabels(kwargs['yticks'])
        else:
            ax.set_yticks(ax_yticks)
        logger.log(loglevel_debug, "    plot.ax_set_ticks     kwargs[yticks] = %s, yticks = %s" % (kwargs['yticks'], ax.get_yticks(),))
        
    if kwargs.has_key('yticklabels'):
        if kwargs['yticklabels'] is None:
            pass
        elif kwargs['yticklabels'] is False:
            ax.set_yticklabels([])
        elif type(kwargs['yticklabels']) is list or type(kwargs['yticklabels']) is tuple:
            ax.set_yticklabels(kwargs['yticklabels'])
        logger.log(loglevel_debug, "    plot.ax_set_ticks     kwargs[yticklabels] = %s, yticklabels = %s" % (kwargs['yticklabels'], ax.get_yticklabels(),))
            
@plotfunc()
def histogram(ax, data, **kwargs):
    """smp_base.plot.histogram

    Plot the histogram
    """
    assert len(data.shape) > 0, 'Data has bad shape = %s' % (data.shape, )
    
    _loglevel = loglevel_debug + 0
    
    # logger.log(_loglevel, "    plot.histogram histo kwargs", kwargs)
    # init local kwargs
    kwargs_ = {}
    kwargs_.update(**kwargs)
    # kwargs = kwargs_plot_clean_histogram(**kwargs_)
    kwargs = plot_clean_kwargs('histogram', **kwargs_)
    
    # if not kwargs.has_key('histtype'):
    #     kwargs_['histtype'] = kwargs['histtype']

    logger.log(_loglevel, "    plot.histogram kwargs .keys = %s" % (kwargs.keys()))
    logger.log(_loglevel, "    plot.histogram kwargs_.keys = %s" % (kwargs_.keys()))

    # explicit limits and bins configuration
    if kwargs_['ylim'] is not None and kwargs_['orientation'] == 'horizontal':
        bins = np.linspace(kwargs_['ylim'][0], kwargs_['ylim'][1], 21 + 1)
        # logger.log(_loglevel, "    plot.histogram setting bins = %s for orientation = %s from ylim = %s" % (bins, kwargs_['orientation'], kwargs_['ylim']))
    elif kwargs_['xlim'] is not None and kwargs_['orientation'] == 'vertical':
        bins = np.linspace(kwargs_['xlim'][0], kwargs_['xlim'][1], 21 + 1)
        # logger.log(_loglevel, "    plot.histogram setting bins = %s for orientation = %s from xlim = %s" % (bins, kwargs_['orientation'], kwargs_['xlim']))
    elif 'bins' in kwargs_:
        bins = kwargs_['bins']
    else:
        bins = 'auto'
        
    logger.log(
        _loglevel,
        "    plot.histogram setting bins = %s for orientation = %s from xlim = %s",
        bins, kwargs_['orientation'], kwargs_['xlim'])
    
    # FIXME: decouple compute histogram; incoming data is bar data
    # already (def bar(...))
    logger.log(_loglevel, "    plot.histogram data = %s", data.shape)
    # if data.shape[-1] > 1:
    for i in range(data.shape[-1]):
        # compute the histogram for each variable (columns) in the input data
        # (n, bins) = np.histogram(data, bins = bins, **kwargs)
        # (n, bins) = meas_hist(data, bins = bins, **kwargs)
        (n, bins_i) = meas_hist(data[:,[i]], bins = bins, **kwargs)

        binwidth = np.mean(np.abs(np.diff(bins_i)))
        bincenters = bins_i[:-1] + binwidth/2.0
        n = n / float(np.sum(n))
    
        logger.log(_loglevel, "    plot.histogram[%d] n = %s/%s", i, n.shape, n)
        logger.log(_loglevel, "    plot.histogram[%d] binwidth = %s", i, binwidth)
        logger.log(_loglevel, "    plot.histogram[%d] bincenters = %s/%s", i, bincenters.shape, bincenters)
    
        # kwargs = kwargs_plot_clean_bar(**kwargs_)
        kwargs_b = plot_clean_kwargs('bar', **kwargs_)
        logger.log(_loglevel, "    plot.histogram[%d] kwargs = %s", i, kwargs_b.keys())
    
        # orientation
        if kwargs_['orientation'] == 'vertical':
            axbar = ax.bar
            kwargs_b['width'] = binwidth
        elif kwargs_['orientation'] == 'horizontal':
            axbar = ax.barh
            kwargs_b['height'] = binwidth

        # plot the pre-computed histogram with bar plot
        patches = axbar(bincenters, n, **kwargs_b)

@plotfunc()
def bar(ax, data, **kwargs):
    """bar plot
    """
    assert len(data.shape) > 0
    
    _loglevel = loglevel_debug + 0
    
    # setting default args
    kwargs_ = {}
    kwargs_.update(**kwargs)

    # cleaning smp keywords for mpl plot func
    kwargs = plot_clean_kwargs('bar', **kwargs_)
    logger.log(_loglevel, "kwargs = %s", kwargs.keys())
    
    # explicit coordinates
    if kwargs_.has_key('ordinate'):
        bincenters = kwargs_['ordinate']
        binwidth = np.ones_like(bincenters) * np.mean(np.abs(np.diff(bincenters)))
    # implicit coordinates
    else:
        bincenters = np.arange(data.shape[0])
        binwidth = 1.

    # orientation
    if kwargs_['orientation'] == 'vertical':
        axbar = ax.bar
        kwargs['width'] = binwidth
    elif kwargs_['orientation'] == 'horizontal':
        axbar = ax.barh
        kwargs['height'] = binwidth

    # debug
    logger.log(_loglevel, "bar bincenters.shape = %s, bincenters = %s" % (bincenters.shape, bincenters, ))
    
    # iterate data columns bar only does single array
    for i in range(data.shape[1]):
        n = data[:,i]
        patches = axbar(bincenters, n, **kwargs)

    # logger.log(_loglevel, "hist n    = %s" % ( n.shape, ))
    # logger.log(_loglevel, "hist bins = %s, len(bins) = %d" % ( bins.shape, len(bins)))

def ax_set_aspect(ax, **kwargs):
    """set mpl axis aspect
    """
    _loglevel = loglevel_debug + 0
    kwargs_ = kwargs
    logger.log(_loglevel, "   ax_set_aspect ax = %s, kwargs.keys = %s" % (ax.title.get_text(), kwargs_.keys(),))
    
    # axis aspect
    if kwargs_.has_key('aspect'):
        ax_aspect = ax.get_aspect()
        ax_xlim = ax.get_xlim()
        ax_ylim = ax.get_ylim()
        logger.log(_loglevel, "   ax_set_aspect aspect = %s, xlim = %s, ylim = %s" % (ax_aspect, ax_xlim, ax_ylim))
        if kwargs_['aspect'] == 'shared': # means square proportions
            xlim_range = 2.2 #  FIXME: hardcoded constant, np.abs(ax_xlim[1] - ax_xlim[0])
            ylim_range = np.abs(ax_ylim[1] - ax_ylim[0])
            ax_aspect = xlim_range/ylim_range
        else:
            ax_aspect = kwargs_['aspect']
            
        logger.log(_loglevel, "   ax_set_aspect ax_aspect = %s" % (ax_aspect, ))
        ax.set_aspect(ax_aspect)
    
    
if HAVE_PYUNICORN:
    # dec_import_unicorn = partial(dec_import, import_name = 'HAVE_PYUNICORN')
    # @dec_import_unicorn()
    def rp_timeseries_embedding(ax, data, **kwargs):
        """recurrence plot using pyunicorn
        """
        emb_del = 1
        emb_dim = 10
        # logger.log(loglevel_debug, "rp_timeseries_embedding data", data)
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
        logger.log(loglevel_debug, "Dummy, pyunicorn not installed")

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
        """n-dimensional histogram seaborn based
        """
        scatter_data_raw  = data
        scatter_data_cols = ["x_%d" % (i,) for i in range(data.shape[1])]

        # prepare dataframe
        df = pd.DataFrame(scatter_data_raw, columns=scatter_data_cols)
        
        g = sns.PairGrid(df)
        # g.map_diag(plt.hist)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(plt.hexbin, cmap="gray", gridsize=30, bins="log");

        # logger.log(loglevel_debug, "dir(g)", dir(g))
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
        logger.log(loglevel_debug, "Dummy, seaborn not installed")
        
def plot_scattermatrix(df, **kwargs):
    """plot a scattermatrix from dataframe
    """
    if df is None:
        logger.log(loglevel_debug, "plot_scattermatrix: no data passed")
        return
        
    # df = pd.DataFrame(X, columns=['x1_t', 'x2_t', 'x1_tptau', 'x2_tptau', 'u_t'])
    # scatter_data_raw = np.hstack((np.array(Xs), np.array(Ys)))
    # scatter_data_raw = np.hstack((Xs, Ys))
    # logger.log(loglevel_debug, "scatter_data_raw", scatter_data_raw.shape)
    
    plt.ioff()
    # df = pd.DataFrame(scatter_data_raw, columns=["x_%d" % i for i in range(scatter_data_raw.shape[1])])
    sm = scatter_matrix(df, ax = kwargs['ax'], alpha=0.2, figsize=(10, 10), diagonal='hist')
    print type(sm), sm.shape, sm[0,0]
    # fig = sm[0,0].get_figure()
    # if SAVEPLOTS:
    # fig.savefig("fig_%03d_scattermatrix.pdf" % (fig.number), dpi=300)
    # fig.show()
    # plt.show()

@plotfunc()
def plot_img(ax, data, **kwargs):
    """plot an image using imshow, pcolor, pcolormesh
    """
    assert ax is not None, "missing axis argument 'ax'"
    vmin = kwargs['vmin']
    vmax = kwargs['vmax']
    cmap = kwargs['cmap']
    title = kwargs['title']
    cax = None
    if 'cax' in kwargs:
        cax = kwargs['cax']
    
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
            if 'colorbar_orientation' in kwargs:
                orientation = kwargs['colorbar_orientation']
            else:
                orientation = 'horizontal'
            cb = plt.colorbar(mappable = mpl, ax = ax, cax=cax, orientation = orientation)
            kwargs['cax'] = cb.ax

    if kwargs.has_key('title'):
        ax.set_title(title) # , fontsize=8)
    else:
        ax.set_title("%s" % ('matrix')) # , fontsize=8)
        
    # return ax, cax
    return kwargs
        
def interactive():
    """basic example for interactive plotting and GUI interaction

    via felix stiehler
    """
    from functools import partial
    
    set_interactive(1)

    def on_click_orig(event, ax, data):
        """
        Left click: show real size
        Right click: resize
        """
        logger.log(loglevel_debug, 'button pressed', event.button, event.xdata, event.ydata, data.shape)
        if event.xdata is not None:
            # data = np.array([[event.xdata, event.ydata],])
            # decoded = decoder.predict(data)
            # decoded = reshape_paths(decoded, flat=False)
            # if args.rel_coords:
            #     decoded = np.cumsum(decoded, axis=1)

            logger.log(loglevel_debug, "ax", ax, "data", data)
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

def fig_interaction(fig, ax, data):
    """fig_interaction

    Expanded example for interactive plotting and GUI interaction.

    Args:
    - fig(figure): figure handle
    - ax(ax): axis handle
    - data(ndarray/dict): plotdatad dict
    """
    do_interaction = True
    
    def on_click(event, ax, data):
        """event handler (on_click, ...) for opening a subplot in a
        separate figure to satisfy advanced zooming desires.

        Left click: show real size
        Right click: resize
        """
        logger.log(
            loglevel_debug, 'ax.inaxes = %s button %s pressed, xdata = %s, ydata = %s, data.shape = %s',
            event.inaxes, event.button, event.xdata, event.ydata, type(data))
        
        # if not do_interaction: return

        if ax is None: return
        if data is None: return
        if event.inaxes != ax: return
            
        if event.xdata is not None:
            # data = np.array([[event.xdata, event.ydata],])
            # decoded = decoder.predict(data)
            # decoded = reshape_paths(decoded, flat=False)
            # if args.rel_coords:
            #     decoded = np.cumsum(decoded, axis=1)

            # logger.log(loglevel_debug + 1, "event type = %s,  dir = %s", type(event), dir(event))
            # logger.log(loglevel_debug + 1, "        ax = %s, data = %s", ax, data.shape)
            
            # ax.clear()

            # datarow = int(event.ydata)
            
            # ax.plot(data[datarow,:], "k-o", alpha = 0.5)
            # plt.pause(1e-6)
            
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

            fig_ = makefig(rows = 1, cols = 1, title = ax.title.get_text())
            # fig_.suptitle(ax.title.get_text())
            # logger.log(loglevel_debug + 1, "        fig_.axes = %s", fig_.axes)
            # fig_.axes.append(ax)
            # fig_.add_axes(ax)
            # ax_ = fig_.add_subplot(1,1,1)
            ax_ = fig_.axes[0]
            # logger.log(loglevel_debug + 1, "        fig_.axes = %s", fig_.axes)
            ax_.clear()
            # ax_.cla()
            # ax_.autoscale()

            # logger.log(loglevel_debug + 1, "    post clear")
            # logger.log(loglevel_debug + 1, "        ax_.get_lines() = %s", ax_.get_lines())
            # logger.log(loglevel_debug + 1, "        ax_.get_legend_handles_labels() = %s", ax_.get_legend_handles_labels())
            # # ax_.draw()
            
            # for l in ax.get_lines():
            #     ax_.add_line(copy.deepcopy(l))
            
            # ax_.plot(np.random.uniform(0, 1, (data.shape[0], )))
            cmap_str = 'rainbow'

            num_cgroups = 5
            num_cgroup_color = 5
            num_cgroup_dist = 255/num_cgroups

            if type(data) in [dict, OrderedDict]:
                for i, k in enumerate(data):
                    inkc = i
                    ax_.set_prop_cycle(
                        get_colorcycler(
                            cmap_str = cmap_str, cmap_idx = None,
                            c_s = inkc * num_cgroup_dist, c_e = (inkc + 1) * num_cgroup_dist, c_n = num_cgroup_color
                        )
                    )
            
                    ax_.plot(data[k]['t'], data[k]['data'], label = 'data_%d' % (i, ), alpha = 0.5, linestyle = '-', marker = '.')
            else:
                for i in range(data.shape[1]):
                    inkc = i
                    ax_.set_prop_cycle(
                        get_colorcycler(
                            cmap_str = cmap_str, cmap_idx = None,
                            c_s = inkc * num_cgroup_dist, c_e = (inkc + 1) * num_cgroup_dist, c_n = num_cgroup_color
                        )
                    )
            
                    ax_.plot(data[:,[i]], label = 'data_%d' % (i, ), alpha = 0.5, linestyle = '-', marker = '.')
            # ax.get_legend_handles_labels()
            # lg = ax.get_legend()
            # ax_.legend(['x%d' % (i, ) for i in range(data.shape[1])])
            ax_.legend()

            # logger.log(loglevel_debug + 1, "        ax_.get_lines() = %s", ax_.get_lines())
            # logger.log(loglevel_debug + 1, "        ax.get_legend_handles_labels() = %s", ax.get_legend_handles_labels())
            # logger.log(loglevel_debug + 1, "        ax_.get_legend_handles_labels() = %s", ax_.get_legend_handles_labels())
            
            # fig_.draw()
            fig_.show()
            plt.draw()
            plt.pause(1e-9)

        
    def on_click_zoom(event, ax, data):
        """Enlarge or restore the selected axis."""
    
        logger.log(
            loglevel_debug, 'on_click_zoom ax.inaxes = %s button %s pressed, xdata = %s, ydata = %s, data.shape = %s',
            event.inaxes, event.button, event.xdata, event.ydata, type(data))

        # if not do_interaction: return
        
        inax = event.inaxes
        if inax is None:
            # Occurs when a region not in an axis is clicked...
            return
        if event.button is 1:
            # On left click, zoom the selected axes
            inax._orig_position = inax.get_position()
            inax.set_position([0.1, 0.1, 0.85, 0.85])
            for axis in event.canvas.figure.axes:
                # Hide all the other axes...
                if axis is not inax:
                    axis.set_visible(False)
        elif event.button is 3:
            # On right click, restore the axes
            try:
                inax.set_position(inax._orig_position)
                for axis in event.canvas.figure.axes:
                    axis.set_visible(True)
            except AttributeError:
                # If we haven't zoomed, ignore...
                pass
        else:
            # No need to re-draw the canvas if it's not a left or right click
            return
        
        event.canvas.draw()
        plt.draw()
        plt.pause(1e-9)
        
    fig.canvas.mpl_connect('button_press_event', partial(on_click, ax = ax, data = data))
    # fig.canvas.mpl_connect('button_press_event', partial(on_click_zoom, ax = ax, data = data))
    
    # def on_key(event, ax, data): # , block, fig
    #     print('you pressed', event.key, event.xdata, event.ydata, do_interaction)
    #     # do_interaction = not do_interaction

    #     # cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # fig.canvas.mpl_connect('key_press_event', partial(on_key, ax = ax, data = data))
    # # fig.canvas.mpl_connect('button_press_event', partial(on_click_zoom, ax = ax, data = data))

def custom_colorbar_demo():
    """custom_colorbar_demo
    
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
    # logger.log(loglevel_debug, "a = %f, b = %f, X = %s" % (a, b, X))

    fig = plt.figure()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6, 3])
    # ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)        

    gs = gridspec.GridSpec(2, 2 * numplots, width_ratios = [9, 1] * numplots)
    gs.hspace = 0.05
    gs.wspace = 0.05

    # plot the first row
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

        logger.log(loglevel_debug, "w_im = %s, h_im = %s" % (w_im, h_im))
        logger.log(loglevel_debug, "w_cb = %s, h_cb = %s" % (w_cb, h_cb))

    # plot the second row
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

        logger.log(loglevel_debug, "w_im = %s, h_im = %s" % (w_im, h_im))
        logger.log(loglevel_debug, "w_cb = %s, h_cb = %s" % (w_cb, h_cb))

    fig.show()
    
    plt.show()

def uniform_divergence(*args, **kwargs):
    """Compute histogram based divergence of bivariate data
    distribution from prior distribution

    Args:
       args[0](numpy.ndarray, pandas.Series): timeseries X_1
       args[1](numpy.ndarray, pandas.Series): timeseries X_2

    Kwargs:
       color: colors to use
       f: plotting function (hist2d, hexbin, ...)
       xxx: ?

    Returns:
       pass through image plotting primitive
    """
    # logger.log(loglevel_debug, "f", f)
    # logger.log(loglevel_debug, "args", len(args),)
    for arg in args:
        logger.log(loglevel_debug, "arg %s %s" % (type(arg), len(arg)))
    # logger.log(loglevel_debug, "kwargs", kwargs.keys())
    f = kwargs['f']
    del kwargs['f']
    color = kwargs['color']
    del kwargs['color']
    # logger.log(loglevel_debug, "f", f)
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
    
    # logger.log(loglevel_debug, "h", h, "xe", xe, "ye", ye)
    # logger.log(loglevel_debug, "h_unif", h_unif, "xe_unif", xe_unif, "ye_unif", ye_unif)
    # ax = f(*args, **kwargs)
    plt.grid(0)
    X, Y = np.meshgrid(xe, ye)
    # ax = plt.imshow(h - h_unif, origin = 'lower', interpolation = 'none', )
    # difference
    # h_ = (h - h_unif)
    # divergence
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
    # logger.log(loglevel_debug, "ax", ax)
    return ax

# plot functions of this module
plotfuncs = {
    'hexbin': plt.hexbin,
    'hexbin': plt.hexbin,
    'hist2d': plt.hist2d,
    'histogram': histogram,
    'histogramnd': histogramnd,
    'kdeplot': sns.kdeplot,
    'partial': partial,
    'plot_img': plot_img,
    'plot_scattermatrix': plot_scattermatrix,
    'scatter': plt.scatter,
    'table': table,
    'timeseries': timeseries,
    'linesegments': linesegments,
    'uniform_divergence': uniform_divergence,
}

# configure, ah, style
configure_style()
    
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", type=str, default = "custom_colorbar_demo", help = "testing mode: [custom_colorbar_demo], interactive, plot_colors")

    args = parser.parse_args()
    # fig = makefig(2, 3)

    if args.mode == "custom_colorbar_demo":
        custom_colorbar_demo()
    elif args.mode == "interactive":
        interactive()
    elif args.mode == "plot_colors":
        test_plot_colors()
    else:
        print "Unknown mode %s, exiting" % (args.mode)
        sys.exit(1)
