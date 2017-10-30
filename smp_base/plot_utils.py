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
    
def put_legend_out_right(labels = None, resize_by = 0.8, ax = None):
    ax = ax_check(ax)
    resize_panel_vert(resize_by = resize_by, ax = ax)
    loc = 'upper left'
    bbox = (0.95, 1.0)
    # loc = 'center left'
    # bbox = (1.0, 0.5)
    if labels is None:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=1)
    else:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=1, labels = labels)

def put_legend_out_top(labels = None, resize_by = 0.8, ax = None):
    ax = ax_check(ax)
    resize_panel_horiz(resize_by = resize_by, ax = ax)
    loc = 'lower center'
    bbox = (0.5, 1.1) #
    if labels is None:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=10)        
    else:
        ax.legend(loc = loc, bbox_to_anchor = bbox, ncol=10, labels = labels)

