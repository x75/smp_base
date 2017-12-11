"""smp_base.dimstack

.. moduleauthor:: Oswald Berthold, Eugenio Piasini <e.piasini@ucl.ac.uk>

First version of dimensional stacking core worker function.

Added digitize pointcloud function to convert pointclouds into
grid spaces by binning the points and averaging the function
values per bin.
"""

import numpy as np

def dimensional_stacking(data, x_dims, y_dims):
    """dimensional_stacking

    Stack an n-dimensional ndarray in two dimensions according to the
    dimensional ordering expressed by x_dims and y_dims.

    See LeBlanc, Ward, Wittels 1990, 'Exploring N-Dimensional
    Databases'.

    Arguments:
     - data(ndarray): n-dimensional ndarray (e.g. data.shape=[4,5,6])
     - x_dims(list): dimensions to be stacked on the x axis,
       big-endian style ('slowest' dimension first, 'fastest'
       dimension last.). e.g. x_dims=[2,0]
     - y_dims(list): dimensions to be stacked on the y axis,
       big-endian. e.g. y_dims = [1]

    Returns:
     - stacked_data(ndarray)
    """
    new_x_length = reduce(lambda x, y: x * y,
			  [data.shape[idx] for idx in x_dims])
    new_y_length = reduce(lambda x, y: x * y,
			  [data.shape[idx] for idx in y_dims])
    new_shape = (new_y_length, new_x_length)
    dim_order = y_dims + x_dims
    stacked_data = data.transpose(*dim_order).reshape(*new_shape)
    #print data.shape, new_shape, dim_order
    #print(stacked_data)
    return stacked_data


def digitize_pointcloud(data, argdims = [0], numbins = 3, valdims = 1, f_fval = np.mean):
    """digitize_pointcloud

    Digitize a pointcloud given as n x k matrix data, with argument dimensions k_arg < k,
    and value dimension k_val < k, k_arg + k_val = k, computing the function value in a
    given bin via the f_fval function

    Arguments:
    - data: n x k matrix of arguments and values from sampling a function
    - argdims: dimension indices of function arguments
    - numbins: number of bins for the digitization
    - valdims: dimension indices (index actually) of function values

    Returns:
    - A matrix with shape (numbins x numbins x ... x numbins) and shape length = k_arg and scalar function value entries

    """
    # for getting the product space
    import itertools
    # print "data", data.shape
    # H = np.histogramdd(data, bins=3, normed = True)
    
    # argument data
    space = data[:,argdims]
    # print "space", space.shape, space

    # vals  = data[:,len(argdims):]
    # print "vals", vals

    # float-valued argument data digitized to bin indices
    space_digitized = np.digitize(space, bins = np.linspace(np.min(space), np.max(space), numbins + 1))
    # print "space_digitized", space_digitized.shape, space_digitized

    # initialize return matrix
    plotdata_new = np.zeros(tuple([numbins for k in range(len(argdims))]))
    # print "plotdata_new", plotdata_new.shape

    # enumerate matrix indices
    # np.indices((3,3,3,3,3,3)) # this should work as well
    for idx_ in [np.array([k]) for k in itertools.product(*[range(1, numbins + 1)] * len(argdims))]:
        # print "idx_", idx_

        # get indices where digitized space equal to current index element-wise
        sidx = idx_ == space_digitized

        # if all elements equal then vector equal
        sidx = np.sum(sidx, axis = 1) > (len(argdims) - 1)
        # print "sidx", sidx.shape, np.sum(sidx)

        # get the functions value at vector equal indices
        v___ = data[sidx, valdims]
        # print "v___", v___

        # if there are samples for this bin, compute the mean function value for the bin
        # plotdata_new[idx_[0],idx_[1],idx_[2],idx_[3],idx_[4],idx_[5]] = np.mean(v___)
        if len(v___) > 0:
            # print "digitize_pointcloud found", v___
            plotdata_new[tuple(idx_.T-1)] = f_fval(v___)

    # return the digitized function
    return plotdata_new
