import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from decolorize import decolorize
from analyze_font import load_font, analyze_font
from utils import *

def pixel_to_glyph(image, typeface_fname, out_fname=None, out_fmt='img',
    FONT_SIZE=48, unicodes=range(0x0020, 0x007E), lum_div=10, 
    font_analysis_fname=None, linespace=4, grid_nx=None, grid_ny=None,
    interpolation=np.median,
    exposure=0, contrast=0,
    whitepoint=1, blackpoint=0,
    save_spectrum=False, spectrum_fname=None):
    '''
    Performs the pixel-to-glyph algorithm.

    Parameters:
        image (str or numpy.array): The name of the image file
            to convert, or an mxnx3 RGB image array itself.
        typeface_fname (str): The name of the typeface file to use.
        out_fname (str): The file to save the result to. If None,
            no file is saved (the Unicode array is still returned).
        out_fmt (str): The format of the result: 'img' or 'txt'.
        FONT_SIZE (int): The font size to use.
        unicodes (tuple): Unicodes for the glyphs to use.
            If None, all ASCII glyphs (0x0020 to 0x007E) will be used.
        lum_div (int): An integer from 1 to 100. If not None, the
            luminosity percentiles of the glyphs will be rounded to
            this many values. A low `lum_div` will have lower
            luminosity resolution, but higher color resolution since
            more glyphs will be available for a given luminosity.
            A high `lum_div` (or None value) will have higher
            luminosity resolution but lower color resolutionn.
            I find a `lum_div` of ~8-10 is best for most images.
        font_analysis_fname (str): The name of a font analysis file for
            `font_analysis_fname` and `FONT_SIZE`. If provided, this file
            will be loaded and the font is not analyzed, saving time.
            TODO: Check that this is a correct analysis for the given
            font and font size.
        linespace (int): The number of pixels between each line of text.
        grid_nx (int): The number of glyphs per line of text. Only one of
            `grid_nx` and `grid_ny` can be provided, since one is determined by
            the other (and the image, glyph dimensions and line space).
        grid_ny (int): The number of lines of text. See `grid_nx`.
        interpolation (function): A function which takes in a pxq numerical array
            and outputs a single number. For example, np.mean or np.median.
            This is the method used to re-scale the image.
        whitepoint (float): A number between `blackpoint` and 1.
            Image regions with luminosity above this value will be represented
            by the least-dark glyph.
        exposure (float): A number between -3 and 3.
        contrast (float): A number between -3 and 3.
        blackpoint (float): A number between 0 and `whitepoint`.
            Image regions with luminosity below this value will be represented
            by the most-dark glyph.
        save_spectrum (bool): Do you want to save the spectrum (glyph mapping)
            to a file?
        spectrum_fname (str): The name of the file to save the spectrum to.
    Returns:
        out (numpy.array): The text-as-pixel representation of the image. Each
            element of the array is a glyph/glyph, a new textual "pixel".
    '''

    #Load and analyze font, using a pre-existing analysis file if possible:
    font = load_font(typeface_fname, FONT_SIZE)
    if font_analysis_fname is None:
        typeface_name = ''.join(typeface_fname.split('/')[-1].split('.')[:-1])
        font_analysis_fname = typeface_name + '_' + str(FONT_SIZE) + 'pt.tsv'
    if os.path.isfile(font_analysis_fname):
        font_df = pd.read_csv(font_analysis_fname, sep='\t', index_col=0)
        missing_unicodes = set(unicodes) - set(font_df['unicode'])
        if len(missing_unicodes) > 0:
            print('Warning: these unicode values were not analyzed:')
            print(', '.join([str(i) for i in missing_unicodes]))
            if len(missing_unicodes) == len(unicodes): raise ValueError('Empty unicode set.')
    else: font_df = analyze_font(font, font_analysis_fname = font_analysis_fname, unicodes=unicodes, verbose=True)

    # Load image.
    if isinstance(image, str): image = mpimg.imread(image)

    # Decompose into a luminosity channel (L) and color channel (C).
    ## Citation:
    ## Decolorize: Fast, Contrast Enhancing, Color to Grayscale Conversion
    ## Mark Grundland and Neil A. Dodgson
    ## Pattern Recognition, vol. 40, no. 11, pp. 2891-2896, (2007). ISSN 0031-3203.
    ## http://www.eyemaginary.com/Portfolio/TurnColorsGray.html
    ## http://www.eyemaginary.com/Rendering/decolorize.m
    decolorized = decolorize(image)
    [G, L, C] = [decolorized[:,:,i] for i in range(3)]

    GLYPH_HEIGHT = np.min(font_df['height']) + linespace
    GLYPH_WIDTH = font_df.at[0,'width']
    GLYPH_W_OVER_H = GLYPH_WIDTH / GLYPH_HEIGHT

    if np.logical_and(grid_nx is not None, grid_ny is not None):
        raise ValueError('Only one of the grid dimensions must be given.')
    if np.logical_and(grid_nx is None, grid_ny is None):
        grid_nx = 100 #default.

    # Re-scale image such that each array element is suitable for a single glyph.
    # TODO: This ignores how the bottom text line is less tall than the rest by `linespace`...
    L = image_to_grid(L, W_OVER_H=GLYPH_W_OVER_H, grid_nx=grid_nx, grid_ny=grid_ny, interpolation=interpolation)
    C = image_to_grid(C, W_OVER_H=GLYPH_W_OVER_H, grid_nx=grid_nx, grid_ny=grid_ny, interpolation=interpolation)

    L = normalize_values(L, 0, 100)
    C = normalize_values(C, 0, 100)

    # EXPOSURE: vary between +1/3 and +3
    if exposure != 0:
        if exposure < 0: exposure = -1/exposure
        L = np.power(L/100, 1/exposure) * 100 #gamma

    # CONTRAST: vary between -3 and +3
    if contrast != 0:
        contrast = contrast + 1*np.sign(contrast)
        if contrast > 0:
            L = normalize_values(L, -1, 1)
            L = 1/(1 + np.exp(-1 * L * contrast)) #sigmoid
            L = normalize_values(L, 0, 100)
        else:
            contrast = min(contrast, -1 * 1e-5)
            contrast = .5 * np.power(10., contrast)
            L = normalize_values(L, contrast, 1-contrast)
            L = np.log(L/(1-L)) #logit
            L = normalize_values(L, 0, 100)

    # Make glyph map.
    glyph_map = make_glyph_map(font_df, unicodes=unicodes, lum_div=lum_div,
        blackpoint=blackpoint, whitepoint=whitepoint,
        save_spectrum=save_spectrum, spectrum_fname=spectrum_fname, cmds_axis=None)
    if save_spectrum:
        write_glyph_array(glyph_map, out_fname=spectrum_fname,
            out_fmt='img', font=font, GLYPH_WIDTH=GLYPH_WIDTH, GLYPH_HEIGHT=GLYPH_HEIGHT)

    # Replace each pixel in the re-scaled image with a glyph using the spectrum mapping.
    L = np.clip(np.round(L).astype(int), a_min=0, a_max=99)
    C = np.clip(np.round(C).astype(int), a_min=0, a_max=99)
    glyph_vec = [glyph_map[(lum,clr)] for (lum,clr) in zip(L.flatten(),C.flatten())]
    glyph_arr = np.array(glyph_vec).reshape(np.shape(L))

    # Write output.
    if out_fname is None: out_fname = 'pixel_to_glyph_art.jpg'
    write_glyph_array(glyph_arr, out_fname, out_fmt=out_fmt, font=font,
        GLYPH_HEIGHT = GLYPH_HEIGHT, GLYPH_WIDTH = GLYPH_WIDTH)

    return
