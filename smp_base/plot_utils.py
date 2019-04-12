"""smp_base.plot_utils

basic plotting utilities
"""

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib import gridspec
from matplotlib.font_manager import FontProperties

import logging
from smp_base.common import get_module_logger

loglevel_debug = logging.DEBUG - 1
logger = get_module_logger(modulename = 'plot_utils', loglevel = logging.DEBUG)

def set_interactive(interactive = False):
    """smp_base.plot_utils.set_interactive

    set interactive plotting to `interactive`, defaults to `False`.
    """
    if interactive:
        plt.ion()
    else:
        plt.ioff()

# from models
def make_figure(*args, **kwargs):
    return plt.figure()

def make_gridspec(rows = 1, cols = 1):
    return gridspec.GridSpec(rows, cols)

def savefig(fig, filename):
    fig_scale_inches = 0.75
    fig.set_size_inches((16 * fig_scale_inches, 9 * fig_scale_inches))
    fig.savefig(filename, dpi = 300, bbox_inches = 'tight')

def set_latex_header():
    # plotting parameters
    rc('text', usetex=True)
    rc('font', serif="Times New Roman")
    rc('font', family='sans')
    rc('font', style="normal",
    variant="normal", weight=700,
    stretch="normal", size=10.0)
    rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}\usepackage{latexsym}\usepackage{bm}"

def set_fontprops():
    fontP = FontProperties()
    fontP.set_size('small')
    return fontP

def ax_check(ax = None):
    if ax is None:
        ax = plt.gca()
    return ax

def resize_panel_vert(resize_by = 0.8, ax = None, shift_by=0.2):
    ax = ax_check(ax)
    box = ax.get_position()
    ax.set_position([box.x0 + (box.width * shift_by), box.y0, box.width * resize_by, box.height])
    
def resize_panel_horiz(resize_by = 0.8, ax = None, shift_by=0.2):
    ax = ax_check(ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * resize_by])

def custom_legend(labels = None, handles = None, resize_by = 0.8, ax = None, loc = 'right', lg = None):
    """custom_legend

    customize legend position outside of axis
    """
    logger.debug('custom_legend resize_by = %s, loc = %s', resize_by, loc)
    if loc == 'right' or loc == 'left':
        put_legend_out_right(labels = labels, handles=handles, resize_by = resize_by, ax = ax, right = loc, lg = lg)
    elif loc in ['upper', 'top'] or loc == 'lower':
        put_legend_out_top(labels = labels, handles=handles, resize_by = resize_by, ax = ax, top = loc, lg = lg)
    elif type(loc) is tuple:
        put_legend_out(labels = labels, handles=handles, resize_by = resize_by, ax = ax, loc = loc, lg = lg)

def put_legend_out(labels = None, handles=None, resize_by = 0.8, ax = None, loc = None, lg = None):
    logger.debug('put_legend_out resize_by = %s, loc = %s', resize_by, loc)
    ax = ax_check(ax)
    shift_by = 1 - resize_by
    if loc[0] < 0.1 or loc[0] > 0.9:
        resize_panel_vert(resize_by = resize_by, ax = ax, shift_by=shift_by)
    if loc[1] < (0.1 - 1) or loc[1] > (0.9 - 1):
        resize_panel_horiz(resize_by = resize_by, ax = ax, shift_by=shift_by)

    if labels is None and handles is None:
        ax.legend(loc = loc, ncol=1)
    elif labels is not None and handles is None:
        if len(labels) < 1: return
        ax.legend(loc = loc, ncol=1, labels = labels)
    else:
        if len(labels) < 1: return
        ax.legend(loc = loc, ncol=1, labels = labels, handles=handles)
        
def put_legend_out_right(labels = None, handles=None, resize_by = 0.8, ax = None, right = 'left', lg = None):
    ax = ax_check(ax)
    if right == 'right':
        shift_by = 1 - resize_by
    else:
        shift_by = 0
        
    resize_panel_vert(resize_by = resize_by, ax = ax, shift_by=shift_by)
    loc = 'upper %s' % (right, )
    if lg is None:
        bboxy = 0.95
    else:
        bboxy = 0.5
        
    if right == 'left':
        bbox = (1.1, bboxy)
    elif right == 'right':
        bbox = (-0.15, bboxy)
    # loc = 'center left'
    # bbox = (1.0, 0.5)
    if labels is None and handles is None:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=1)
    elif labels is not None and handles is None:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=1, labels = labels)
    else:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=1, labels = labels, handles=handles)

def put_legend_out_top(labels = None, handles=None, resize_by = 0.8, ax = None, top = 'lower', lg = None):
    ax = ax_check(ax)
    resize_panel_horiz(resize_by = resize_by, ax = ax)
    loc = '%s center' % (top, )
    if lg is None:
        bboxx = 0.1
    else:
        bboxx = 0.5

    if top == 'upper':
        bbox = (bboxx, -0.1) #
    elif top == 'lower':
        bbox = (bboxx, 1.1) #
    if labels is None and handles is None:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=10)        
    elif labels is not None and handles is None:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=10, labels = labels)
    else:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=10, labels = labels, handles=handles)
