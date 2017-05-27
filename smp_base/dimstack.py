"""
Author: Eugenio Piasini <e.piasini@ucl.ac.uk>
Date:   Tue Jun 19 10:12:40 2012 +0100

    First version of core dimensional stacking function.
"""

import numpy as np

def dimensional_stacking(data, x_dims, y_dims):
    """
    Stack an n-dimensional ndarray in two dimensions according to the
    dimensional ordering expressed by x_dims and y_dims.

    See LeBlanc, Ward, Wittels 1990, 'Exploring N-Dimensional
    Databases'.
     * data: n-dimensional ndarray (e.g. data.shape=[4,5,6])
     * x_dims: dimensions to be stacked on the x axis,
     big-endian style ('slowest' dimension first, 'fastest'
     dimension last.). e.g. x_dims=[2,0]
     * y_dims: dimensions to be stacked on the y axis,
     big-endian. e.g. y_dims = [1]
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
