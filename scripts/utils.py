import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import os
    
def pca_proj(array, dim=1):
    '''
    Gives the `dim`-dimensional PCA representation of the input array,
    computed via the SVD.

    Parameters:
        array (numpy.array): A nxd data array (n datapoints, d dimensions). 
        dim (int): A number < d, the number of dimensions to reduce to.

    Returns:
        proj (numpy.array): The nxdim data array projected onto the PC(s).
    '''
    # Center columns i.e. axes.
    array = array - array.mean(axis=0)
    # Compute the SVD.
    svd = np.linalg.svd(array, full_matrices=False)
    svd[1][dim:] = 0
    proj = svd[0].dot(np.diag(svd[1])).dot(svd[2])
    return proj

def round_to_nearest_value(array, values, get_index=False):
    '''
    Rounds a numerical array elementwise to values in `values`. 

    Parameters:
        array (numpy.array): A mxn numerical array.
        values (numpy.array): A q-length numerical array.
        get_index (bool): Do you want the output matrix to be populated by
            indices of the nearest elements in `values`, instead of 
            their actual value?

    Returns:
        out (numpy.array): A mxn `array`. The elements are the rounded values
            (get_index=False) or the indices of the matching values (get_index=True).
    '''
    values = np.array(values)
    array = np.array(array)
    # This will return the highest index, among ties.
    if get_index:
        out = np.zeros(np.shape(array), dtype=int)
        for i in range(1, len(values)):
            v = values[i]
            out[np.abs(array - values[out]) > np.abs(array - v)] = i
    else:
        out = values[np.zeros(np.shape(array), dtype=int)]
        for v in set(values[1:]):
            out[np.abs(array - out) > np.abs(array - v)] = v
    return out

def image_to_grid(image, W_OVER_H, grid_nx=None, grid_ny=None, interpolation=np.median):
    '''
    Imposes onto an image a grid whose cell dimensions' proportions are equal to
    `W_OVER_H`. Then, converts each cell to a single value using `interpolation`.
    Essentially, this scales down an image's resolution and uses rectangular pixels.
    The grid will not have the exact same proportions as the image, because an integer
    number of cells will be used for the grid (but, this change of ratio is minimized).
    Parameters:
        image (numpy.array): A mxn numerical array.
        W_OVER_H (float): The width-over-height ratio of each cell. 
        grid_nx (int): The number of cells in each row of the grid. Only one of
            `grid_nx` and `grid_ny` can be provided, since one is determined by
            the other (and the image dimensions and `W_OVER_H`).
        grid_ny (int): The number of cells in each column of the grid. See `grid_nx`.
        interpolation (function): A function which takes in a pxq numerical array
            and outputs a single number. For example, np.mean or np.median.
        color_weight (float):
        values (numpy.array): A q-length numerical array.
        get_index (bool): Do you want the output matrix to be populated by
            indices of the nearest elements in `values`, instead of 
            their actual value?

    Returns:
        out (numpy.array): A mxn `array`. The elements are the rounded values
            (get_index=False) or the indices of the matching values (get_index=True).
    '''

    # Exactly one of output_width and output_height should be given.
    if not np.logical_xor(grid_nx is None, grid_ny is None):
        raise ValueError('Exactly one of grid dimensions must be given.')

    (IMAGE_HEIGHT, IMAGE_WIDTH) = np.shape(image)
    
    # Compute the scale and grid dimensions.
    # (W_OVER_H) * (grid_nx / grid_ny) =: (IMAGE_WIDTH)/(IMAGE_HEIGHT)
    if grid_nx is not None:
        scale = IMAGE_WIDTH/grid_nx
        x_divs = [int(d*scale) for d in range(grid_nx+1)] #Should change to int(round())?
        grid_ny = int(round(IMAGE_HEIGHT/scale * W_OVER_H))
        y_divs = [int(h*IMAGE_HEIGHT/grid_ny) for h in range(grid_ny+1)]
    elif grid_ny is not None:
        scale = IMAGE_HEIGHT/grid_ny
        y_divs = [int(h*scale) for h in range(grid_ny+1)]
        grid_nx = int(round(IMAGE_WIDTH/scale / W_OVER_H))
        x_divs = [int(d*IMAGE_WIDTH/grid_nx) for d in range(grid_nx+1)]

    # Superimpose grid and interpolate.
    grid = np.zeros((grid_ny, grid_nx))
    for x in range(grid_nx):
        for y in range(grid_ny):
            image_cell = image[y_divs[y]:y_divs[y+1], x_divs[x]:x_divs[x+1]]
            grid[y,x] = interpolation(image_cell)
    return grid