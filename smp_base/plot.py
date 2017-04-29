
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

def makefig(rows = 1, cols = 1):
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
            
    return fig

def timeseries(ax, data):
    # style params
    ax.plot(data, alpha = 0.5)

def histogram(ax, data):
    # style params
    ax.hist(data, alpha = 0.5)
    
if __name__ == "__main__":
    fig = makefig(2, 3)
