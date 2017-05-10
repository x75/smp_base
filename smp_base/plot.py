"""smp_base - smp sensorimotor experiments base functions

plotting

2017 Oswald Berthold
"""

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import numpy as np

from pyunicorn.timeseries import RecurrencePlot

def makefig(rows = 1, cols = 1):
    """create figure and subplot structure

return figure handle
"""
    # rows = len(plotitems[0])
    # cols = len(plotitems[0][0])
    fig = plt.figure()
    gs = gridspec.GridSpec(rows, cols)

    axes = []

    for row in range(rows):
        axes.append([])
        for col in range(cols):
            axes[-1].append(fig.add_subplot(gs[row, col]))
    # print "fig.axes", fig.axes
    plt.subplots_adjust(wspace=0, hspace = 0.3)
            
    return fig

def timeseries(ax, data, **kwargs):
    """timeseries plot"""
    # style params
    if kwargs.has_key('marker'):
        marker = kwargs['marker']
    else:
        marker = 'None'
    if kwargs.has_key('linestyle'):
        linestyle = kwargs['linestyle']
    else:
        linestyle = 'solid'
    if kwargs.has_key('ordinate'):
        ax.plot(kwargs['ordinate'], data, alpha = 0.5, marker = marker, linestyle = linestyle)
    else:
        ax.plot(data, alpha = 0.5, marker = marker, linestyle = linestyle)

def histogram(ax, data, **kwargs):
    """histogram plot"""
    # style params
    ax.hist(data, bins = int(np.log(data.shape[0]/20)), alpha = 0.5)
    ax.set_yscale('log')

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
        
if __name__ == "__main__":
    fig = makefig(2, 3)
