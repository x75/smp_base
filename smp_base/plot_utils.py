"""Some plotting utils"""

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

def set_latex_header():
    # plotting parameters
    rc('text', usetex=True)
    rc('font', serif="Times New Roman")
    rc('font', family='serif')
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

def resize_panel_vert(resize_by = 0.8, ax = None):
    ax = ax_check(ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * resize_by, box.height])
    
def resize_panel_horiz(resize_by = 0.8, ax = None):
    ax = ax_check(ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * resize_by])

def custom_legend(labels = None, resize_by = 0.8, ax = None, loc = 'right'):
    if loc == 'right' or loc == 'left':
        put_legend_out_right(labels = labels, resize_by = resize_by, ax = ax, right = loc)
    elif loc == 'upper' or loc == 'lower':
        put_legend_out_top(labels = labels, resize_by = resize_by, ax = ax, top = loc)

def put_legend_out_right(labels = None, resize_by = 0.8, ax = None, right = 'left'):
    ax = ax_check(ax)
    resize_panel_vert(resize_by = resize_by, ax = ax)
    loc = 'upper %s' % (right, )
    if right == 'left':
        bbox = (0.95, 0.95)
    elif right == 'right':
        bbox = (-0.15, 0.95)
    # loc = 'center left'
    # bbox = (1.0, 0.5)
    if labels is None:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=1)
    else:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=1, labels = labels)

def put_legend_out_top(labels = None, resize_by = 0.8, ax = None, top = 'lower'):
    ax = ax_check(ax)
    resize_panel_horiz(resize_by = resize_by, ax = ax)
    loc = '%s center' % (top, )
    if top == 'upper':
        bbox = (0.5, -0.1) #
    elif top == 'lower':
        bbox = (0.5, 1.1) #
    if labels is None:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=10)        
    else:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=10, labels = labels)

