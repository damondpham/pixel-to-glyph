import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFont, ImageDraw
from decolorize import decolorize
from sklearn.decomposition import PCA
from analyze_font import load_font, analyze_font
from utils import pca_proj, round_to_nearest_value, image_to_grid

def write_text_as_pixel(str_array, out_fname, fmt='img', 
    font=None, CHAR_WIDTH=None, CHAR_HEIGHT=None):
    '''
    Writes a text-as-pixel array of glyphs/characters into a file.

    Parameters:
        str_array (numpy.array): The text-as-pixel array of length-one strings.
        out_fname (str): The file to save to. Should be compatible with `fmt`.
        fmt (str): Either 'text' for a text file, or 'img' for an image file. 
        font (PIL.ImageFont font): The font to use for an image output.
        CHAR_WIDTH (int): The width of each glyph, to use for an image output.
        CHAR_HEIGHT (int): The height of each glyph, including the line space,
        	to use for an image output.

    Returns:
        None. 
    '''
    newlines = np.array(['\n' for _ in range(len(str_array))]).reshape((len(str_array), 1))
    str_array = np.append(str_array, newlines, axis = 1)
    shape = np.shape(str_array)
    out = ''.join(str_array.flatten())
    
    if fmt == 'text':
        with open(out_fname, "w") as f: print(out, file=f)
        
    if fmt == 'img':
        WIDTH = CHAR_WIDTH * (shape[1] - 1)
        HEIGHT = CHAR_HEIGHT * shape[0]
        image = Image.new('1', (WIDTH, HEIGHT), color=1)
        draw = ImageDraw.Draw(image)
        draw.text((0,0), out, font=font)
        image.save(out_fname)

def do_text_as_pixel(image_fname, out_fname, font_fname, fmt='img',
	FONT_SIZE=36, glyphs=None, lum_div=5,
    analysis_fname=None, linespace=4,
    grid_nx=None, grid_ny=None, interpolation=np.median,
    whitepoint=1, blackpoint=0,
    save_spectrum=False, spectrum_fname=None):
    '''
    Performs the text-as-pixel algorithm.

    Parameters:
        image_fname (str): The name of the image file to convert.
        out_fname (str): The file to save the result to.
        font_fname (str): The name of the font file to use.
        fmt (str): The format of the result: 'img' or 'txt'.
        FONT_SIZE (int): The font size to use.
        glyphs (str): A string containing the posssible characters 
        	to use. If none, all glyphs will be considered.
        lum_div (int): An integer from 1 to 100. If not None, the 
        	luminosity percentiles of the glyphs will be rounded to 
        	this many values. A low `lum_div` will have lower 
        	luminosity resolution, but higher color resolution since
        	more glyphs will be available for a given luminosity. 
        	A high `lum_div` (or None value) will have higher
        	luminosity resolution but lower color resolutionn.
        	I find a `lum_div` of ~8-10 is best for most images.
        analysis_fname (str): The name of a font analysis file for
        	`font_fname` and `FONT_SIZE`. If provided, this file
        	will be loaded and the font is not analyzed, saving time.
        	TODO: Check that this is a correct analysis for the given
        	font and font size.
        linespace (int): The number of pixels between each line of text.
        grid_nx (int): The number of characters per line of text. Only one of
            `grid_nx` and `grid_ny` can be provided, since one is determined by
            the other (and the image, glyph dimensions and line space).
        grid_ny (int): The number of lines of text. See `grid_nx`.
        interpolation (function): A function which takes in a pxq numerical array
            and outputs a single number. For example, np.mean or np.median.
            This is the method used to re-scale the image.
        whitepoint (float): A number between `blackpoint` and 1.
        	Image regions with luminosity above this value will be represented
        	by the least-dark glyph.
        blackpoint (float): A number between 0 and `whitepoint`.
       		Image regions with luminosity below this value will be represented
       		by the most-dark glyph.
        save_spectrum (bool): Do you want to save the spectrum (character mapping)
            to a file?
        spectrum_fname (str): The name of the file to save the spectrum to.

    Returns:
        out (numpy.array): The text-as-pixel representation of the image. Each
        	element of the array is a glyph/character, a new textual "pixel".
    '''
    font = load_font(font_fname, FONT_SIZE)
    
    # Analyze the font, using a pre-existing analysis file if possible.
    if analysis_fname is None:
        font_name = ''.join(font_fname.split('/')[-1].split('.')[:-1])
        analysis_fname = os.path.join(os.getcwd(), '..', os.fspath('fonts'),
                              os.fspath('analysis'),
                              font_name + '_' + str(FONT_SIZE) + 'pt.tsv')        
    if os.path.isfile(analysis_fname):
        df = pd.read_csv(analysis_fname, sep='\t', index_col=0)
    else: df = analyze_font(font_fname, FONT_SIZE=FONT_SIZE, analysis_fname = analysis_fname)
    
    CHAR_HEIGHT = np.min(df['height']) + linespace
    CHAR_WIDTH = df.at[0,'width']

    CHAR_W_OVER_H = CHAR_WIDTH / CHAR_HEIGHT
    
    # Load image.
    image = mpimg.imread(image_fname)
    # Decompose into a luminosity channel (L) and color channel (C).
    ## Citation:
    ## Decolorize: Fast, Contrast Enhancing, Color to Grayscale Conversion   
	## Mark Grundland and Neil A. Dodgson   
	## Pattern Recognition, vol. 40, no. 11, pp. 2891-2896, (2007). ISSN 0031-3203.   
	## http://www.eyemaginary.com/Portfolio/TurnColorsGray.html   
	## http://www.eyemaginary.com/Rendering/decolorize.m   
    decolorized = decolorize(image)
    [G, L, C] = [decolorized[:,:,i] for i in range(3)]

    # Re-scale image such that each pixel is suitable for a single glyph.
    # TODO: This ignores how the bottom text line is less tall than the rest by `linespace`...
    L = image_to_grid(L, W_OVER_H=CHAR_W_OVER_H, grid_nx=grid_nx, grid_ny=grid_ny, interpolation=interpolation)
    C = image_to_grid(C, W_OVER_H=CHAR_W_OVER_H, grid_nx=grid_nx, grid_ny=grid_ny, interpolation=interpolation)
    
    # Coerce values to be between [0, 1] inclusive.
    L = (L - np.min(L))/(np.max(L) - np.min(L))
    C = (C - np.min(C))/(np.max(C) - np.min(C))

    # Scale to [0, 100] inclusive and round down.
    L = (L*100 - 1e-8).astype(int)
    C = (C*100 - 1e-8).astype(int)

    # Cutoff non-permissible values, just in case weird rounding happened.
    L[L < 0] = 0
    L[L > 100] = 100
    C[C < 0] = 0
    C[C > 100] = 100

    # Collect the set of glyphs to use.
    if glyphs is not None: df = df.loc[df['glyph'].isin([g for g in set(glyphs)])].reset_index(drop=True)

    # Set the luminosity percentile of each glyph relative to the others.
    ## The darkest will be 0, and lightest will be 100.
    df['lum_pct'] = ((whitepoint - df['darkness']/np.max(df['darkness'])*(whitepoint-blackpoint))*100).astype(int)

    # Group luminosity percentiles into `lum_div` groups. Sort by luminosity percentile.
    if lum_div is not None: df['lum_pct'] = np.round(np.round(df['lum_pct'] /100 *lum_div) *100 /lum_div)
    df = df.iloc[np.argsort(df['lum_pct']),:].reset_index(drop=True)

    # Count the number of glyphs at each luminosity percentile value. 
    df['lum_unique'] = True
    unique_v, unique_i = np.unique(df['lum_pct'], return_index=True)
    divs = pd.Series(index=df.loc[unique_i,'lum_pct'], name='chars')
    df['PCA'] = np.nan
    pca = PCA(n_components=1)
    for i in range(len(unique_i)):
        if i == len(unique_i)-1:
            start = unique_i[i]
            end = df.index[-1]
        else:
            start = unique_i[i]
            end = unique_i[i+1]-1
        # If there are more than one glyph at this percentile...
        if end - start != 0:
            df.loc[start:end,'lum_unique'] = False
            # Order them using their CMDS embedding: Center them and project onto the highest
            # PC axis. Then, record their order from left to right (lowest to highest CMDS1).
            df.loc[start:end,'PCA'] = pca_proj(df.loc[start:end,['CMDS1','CMDS2']], 1)[:,0]
            pca_order = np.argsort(df.loc[start:end,'PCA']).values
            divs[df.at[unique_i[i],'lum_pct']] = df.loc[start:end, 'glyph'].values[pca_order]
        else:
            divs[df.at[unique_i[i],'lum_pct'] ]= np.atleast_1d(df.at[unique_i[i],'glyph'])
    
    # Create a spectrum mapping where the (x,y) value is the character which will be used
    ## For a given (luminosity, color) value.
    char_map = np.full((100,100), ' ')
    lum_pct_ramp = round_to_nearest_value(range(100), df['lum_pct'][unique_i])
    for lum in range(100):
        for clr in range(100):
            x = divs.at[lum_pct_ramp[lum]]
            char_map[lum, clr] = x[int(np.floor(clr/100*len(x)))]
    
    if save_spectrum:
        write_text_as_pixel(char_map, spectrum_fname, fmt = fmt, font = font,
                            CHAR_HEIGHT = CHAR_HEIGHT, CHAR_WIDTH = CHAR_WIDTH)

    # Replace each pixel in the re-scaled image with a glyph using the spectrum mapping. 
    out = [char_map[(lum,clr)] for (lum,clr) in zip(L.flatten(),C.flatten())]
    out = np.array(out).reshape(np.shape(L))

    # Write output.
    write_text_as_pixel(out, out_fname, fmt=fmt, font=font, 
    	CHAR_HEIGHT = CHAR_HEIGHT, CHAR_WIDTH = CHAR_WIDTH)
    
    return out

'''
# DEMO:
result = do_text_as_pixel(image_fname='../img/demo6.png', out_fname = '../results/demo_result.jpg',
                       font_fname='../fonts/PTM55FT.ttf', FONT_SIZE=16, 
                       glyphs=None,
                       analysis_fname=None, linespace=4, lum_div=15,
                       grid_nx=200, grid_ny=None, interpolation=np.median,
                       whitepoint=1, blackpoint=.00,
                       save_spectrum=True, spectrum_fname='../results/demo_spectrum.jpg')
'''
