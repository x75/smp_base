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
    resize_panel_vert(resize_by = resize_by)
    if not labels is None:
        ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=8)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=8)

def put_legend_out_top(labels = None, resize_by = 0.8, ax = None):
    ax = ax_check(ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * resize_by])
    if not labels is None:
        ax.legend(labels, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=10, fontsize=8)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=10, fontsize=8)        

