import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from PIL import Image, ImageFont, ImageDraw # pip install pillow

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
    number of cells is required for the grid (but, this trimming is minimized).

    Parameters:
        image (numpy.array): A mxn numerical array.
        W_OVER_H (float): The width-over-height ratio of each cell.
        grid_nx (int): The number of cells in each row of the grid. Only one of
            `grid_nx` and `grid_ny` can be provided, since one is determined by
            the other (and the image dimensions and `GLYPH_W_OVER_H`).
        grid_ny (int): The number of cells in each column of the grid. See `grid_nx`.
        interpolation (function): A function which takes in a pxq numerical array
            and outputs a single number. For example, np.mean or np.median.
        color_weight (float): [not implemented]
        values (numpy.array): [not implemented] A q-length numerical array.
        get_index (bool): [not implemented] Do you want the output matrix to be
            populated by indices of the nearest elements in `values`, instead of
            their actual value?

    Returns:
        out (numpy.array): A grid_nx x grid_ny `numpy.array`. The elements are the rounded values
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
        x_divs = [int(d*scale) for d in range(int(grid_nx+1))] #Should change to int(round())?
        grid_ny = int(round(IMAGE_HEIGHT/scale * W_OVER_H))
        y_divs = [int(h*IMAGE_HEIGHT/grid_ny) for h in range(grid_ny+1)]
    elif grid_ny is not None:
        scale = IMAGE_HEIGHT/grid_ny
        y_divs = [int(h*scale) for h in range(int(grid_ny+1))]
        grid_nx = int(round(IMAGE_WIDTH/scale / W_OVER_H))
        x_divs = [int(d*IMAGE_WIDTH/grid_nx) for d in range(grid_nx+1)]

    # Superimpose grid and interpolate.
    grid = np.zeros((int(grid_ny), int(grid_nx)))
    for x in range(int(grid_nx)):
        for y in range(int(grid_ny)):
            image_cell = image[y_divs[y]:y_divs[y+1], x_divs[x]:x_divs[x+1]]
            if image_cell.size == 0:
                image_cell = image[y_divs[y], x_divs[x]]
            grid[y,x] = interpolation(image_cell)
    return grid

def write_glyph_array(glyph_array, out_fname, out_fmt='img',
    font=None, GLYPH_WIDTH=None, GLYPH_HEIGHT=None):
    '''
    Writes a text-as-pixel array of glyphs/characters into a file.
    Parameters:
        glyph_array (numpy.array): The text-as-pixel array of length-one strings.
        out_fname (str): The file to save to. Should be compatible with `out_fmt`.
        out_fmt (str): Either 'text' for a text file, or 'img' for an image file.
        font (PIL.ImageFont font): The font to use for an image output.
        GLYPH_WIDTH (int): The width of each glyph, to use for an image output.
        GLYPH_HEIGHT (int): The height of each glyph, including the line space,
        	to use for an image output.
    Returns:
        None.
    '''
    newlines = np.array([0x000a for _ in range(len(glyph_array))]).reshape((len(glyph_array), 1))
    glyph_array = np.append(glyph_array, newlines, axis = 1)
    shape = np.shape(glyph_array)
    out = u''.join([chr(i) for i in list(glyph_array.flatten())])

    if out_fmt == 'text':
        with open(out_fname, "w") as f: print(out, file=f)

    if out_fmt == 'img':
        WIDTH = GLYPH_WIDTH * (shape[1] - 1)
        HEIGHT = GLYPH_HEIGHT * shape[0]
        image = Image.new('1', (WIDTH, HEIGHT), color=1)
        draw = ImageDraw.Draw(image)
        draw.text((0,0), out, font=font)
        image.save(out_fname)

    return

def make_glyph_map(font_df, unicodes=None, lum_div=None, blackpoint=0, whitepoint=1,
    save_spectrum=False, spectrum_fname=None, cmds_axis=None):
    '''
    font_df (pandas.DataFrame): table of glyph data for a font. See `analyze_font()`.
    unicodes (list): Unicodes for the glyphs to use. If none, all in `font_df` will be used.
    lum_div (int): An integer from 1 to 100. If not None, the luminosity percentiles of
        the glyphs will be rounded to this many values. A low `lum_div` will have lower
        luminosity resolution, but higher color resolution since more glyphs will be
        available for a given luminosity. A high `lum_div` (or None value) will have
        higher luminosity resolution but lower color resolution. I find a `lum_div` of
        ~8-10 is best for most images.
    whitepoint (float): A number between `blackpoint` and 1.
        Image regions with luminosity above this value will be represented
        by the least-dark glyph.
    blackpoint (float): A number between 0 and `whitepoint`.
        Image regions with luminosity below this value will be represented
        by the most-dark glyph.
    save_spectrum (bool): Do you want to save the spectrum (glyph mapping)
        to a file?
    spectrum_fname (str): The name of the file to save the spectrum to.
    cmds_axis (float): 0 to 1.
    '''
    df = font_df

    # Collect the set of glyphs to use.
    if unicodes is not None:
        df = df.loc[df['unicode'].isin([g for g in set(unicodes)])].reset_index(drop=True)

    # Set the luminosity percentile of each glyph relative to the others.
    ## The darkest will be 0, and lightest will be 100.
    font_df['lum_pct'] = (whitepoint - font_df['darkness']/np.max(font_df['darkness'])*(whitepoint-blackpoint))*100

    # Group luminosity percentiles into `lum_div` groups. Sort by luminosity percentile.
    if lum_div is not None: font_df['lum_pct'] = np.round(np.round(font_df['lum_pct'] /100 *lum_div) *100 /lum_div)
    font_df['lum_pct'] = np.clip(font_df['lum_pct'], a_min=0, a_max=99).astype(int)
    font_df = font_df.iloc[np.argsort(font_df['lum_pct']),:].reset_index(drop=True)

    # Count the number of glyphs at each luminosity percentile value.
    font_df['lum_unique'] = True
    unique_v, unique_i = np.unique(font_df['lum_pct'], return_index=True)
    divs = pd.Series(index=font_df.loc[unique_i,'lum_pct'], name='chars', dtype=object)
    font_df['PCA'] = np.nan
    pca = PCA(n_components=1)

    for i in range(len(unique_i)):
        if i == len(unique_i)-1:
            start = unique_i[i]
            end = font_df.index[-1]
        else:
            start = unique_i[i]
            end = unique_i[i+1]-1
        # If there are more than one glyph at this percentile...
        if end - start != 0:
            font_df.loc[start:end,'lum_unique'] = False
            # Order them using their CMDS embedding: Center them and project onto the highest
            # PC axis. Then, record their order from left to right (lowest to highest CMDS1).
            font_df.loc[start:end,'PCA'] = pca_proj(font_df.loc[start:end,['CMDS1','CMDS2']], 1)[:,0]
            pca_order = np.argsort(font_df.loc[start:end,'PCA']).values
            divs[font_df.at[unique_i[i],'lum_pct']] = np.array(font_df.loc[start:end, 'unicode'].values[pca_order])
        else:
            divs[font_df.at[unique_i[i],'lum_pct'] ]= np.atleast_1d(font_df.at[unique_i[i],'unicode'])

    # Create a spectrum mapping where the (x,y) value is the glyph which will be used
    ## For a given (luminosity, color) value.
    glyph_map = np.zeros((100,100)).astype(int)
    lum_pct_ramp = round_to_nearest_value(range(100), font_df['lum_pct'][unique_i])
    for lum in range(100):
        x = np.array(divs.at[lum_pct_ramp[lum]])
        lum_clrs = np.clip(np.round(np.array(range(100))/100*len(x)), a_min=0, a_max=len(x) - 1).astype(int)
        glyph_map[lum,:] = x[lum_clrs]

    return glyph_map

def normalize_values(M, M_min, M_max):
    M = (M - np.min(M))/(np.max(M) - np.min(M))
    M = (M*M_max) + M_min
    M = np.clip(M, M_min, M_max)
    return M
