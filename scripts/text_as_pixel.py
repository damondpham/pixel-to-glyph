import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageFont, ImageDraw
from decolorize import decolorize
from sklearn.decomposition import PCA
from analyze_font import load_font, analyze_font

image_fname='../img/apple.jpg'
font_fname='../fonts/cour.ttf'
FONT_SIZE=36
grid_nx=120
grid_ny=None
analysis_fname=None
linespace=4
interpolation=np.median

def write_output(str_array, out_fname, format='text', 
                 font=None, df=None, linespace=4):
    newlines = np.array(['\n' for _ in range(len(str_array))]).reshape((len(str_array), 1))
    str_array = np.append(str_array, newlines, axis = 1)
    shape = np.shape(str_array)
    out = ''.join(str_array.flatten())
    
    if format == 'text':
        with open(out_fname, "w") as f: print(out, file=f)
        
    if format == 'img':
        WIDTH = (df.at[0, 'width']-1) * shape[1]
        HEIGHT = (np.min(df['height']) + linespace) * shape[0]
        image = Image.new('1', (WIDTH, HEIGHT), color=1)
        draw = ImageDraw.Draw(image)
        draw.text((0,0), out, font=font)
        image.save(out_fname)
    
def pca_proj(array, dim=1):
    svd = np.linalg.svd(array, full_matrices=False)
    svd[1][dim:] = 0
    proj = svd[0].dot(np.diag(svd[1])).dot(svd[2])
    return proj

def round_to_nearest_value(array, values, get_index=False):
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

def round_down(array, values, get_index=False):
    values = np.sort(set(values))
    array = np.array(array)
    # This will return the highest index, among ties.
    if get_index:
        out = np.zeros(np.shape(array), dtype=int)
        for i in range(1, len(values)):
            v = values[i]
            out[array - v > 0] = i
    else:
        out = values[np.zeros(np.shape(array), dtype=int)]
        for v in values[1:]:
            out[array - v > 0] = v
    return out

def image_to_grid(image, CHAR_W_OVER_H, 
                  grid_nx=None, grid_ny=None, interpolation=np.median,
                  color_weight = .5):

    # Exactly one of output_width and output_height should be given.
    if not np.logical_xor(grid_nx is None, grid_ny is None):
        raise ValueError('Exactly one of grid dimensions must be given.')

    (IMAGE_HEIGHT, IMAGE_WIDTH) = np.shape(image)
    
    # Compute the scale, grid dimensions, and trim size.
    # (grid_nx * CHAR_WIDTH)/(grid_ny * CHAR_HEIGHT) = (IMAGE_WIDTH)/(IMAGE_HEIGHT)
    # This ignores how the bottom row of text is less tall than the rest by linespace...
    if grid_nx is not None:
        scale = IMAGE_WIDTH/grid_nx
        x_divs = [int(d*scale) for d in range(grid_nx+1)] #changed from int to round?
        grid_ny = int(round(IMAGE_HEIGHT/scale * CHAR_W_OVER_H))
        y_divs = [int(h*IMAGE_HEIGHT/grid_ny) for h in range(grid_ny+1)]
    elif grid_ny is not None:
        scale = IMAGE_HEIGHT/grid_ny
        y_divs = [int(h*scale) for h in range(grid_ny+1)]
        grid_nx = int(round(IMAGE_WIDTH/scale / CHAR_W_OVER_H))
        x_divs = [int(d*IMAGE_WIDTH/grid_nx) for d in range(grid_nx+1)]

    grid = np.zeros((grid_ny, grid_nx))
    for x in range(grid_nx):
        for y in range(grid_ny):
            image_cell = image[y_divs[y]:y_divs[y+1], x_divs[x]:x_divs[x+1]]
            grid[y,x] = interpolation(image_cell)
            
    return grid

def text_as_pixel(image_fname, out_fname,
                  font_fname, FONT_SIZE=36, glyphs=None, color_weight=0,
                  analysis_fname=None, linespace=4,
                  grid_nx=None, grid_ny=None, interpolation=np.median,
                  whitepoint=1, blackpoint=0):
  
    font = load_font(font_fname, FONT_SIZE)
    
    # Analyze the font, using a pre-existing analysis file if possible.
    if analysis_fname is None:
        font_name = ''.join(font_fname.split('/')[-1].split('.')[:-1])
        analysis_fname = os.path.join(os.getcwd(), '..', os.fspath('fonts'),
                              os.fspath('analysis'),
                              font_name + '_' + str(FONT_SIZE) + 'pt.tsv')        
    if os.path.isfile(analysis_fname):
        df = pd.read_csv(analysis_fname, sep='\t', index_col=0)
    else: df = analyze_font(font_fname, analysis_fname = analysis_fname)
    
    CHAR_HEIGHT = np.min(df['height'])
    CHAR_WIDTH = df.at[0,'width']
    CHAR_W_OVER_H = CHAR_WIDTH / (CHAR_HEIGHT + linespace)
    
    image = mpimg.imread(image_fname)
    decolorized = decolorize(image)
    [G, L, C] = [decolorized[:,:,i] for i in range(3)]
    L = image_to_grid(L, CHAR_W_OVER_H, grid_nx=grid_nx, grid_ny=grid_ny, interpolation=interpolation)
    C = image_to_grid(C, CHAR_W_OVER_H, grid_nx=grid_nx, grid_ny=grid_ny, interpolation=interpolation)
    
    C = (C - np.min(C))/(np.max(C) - np.min(C))
    
    L = (L*100 - 1e-8).astype(int)
    C = (C*100 - 1e-8).astype(int)
    
    if glyphs is not None: df = df.loc[df['glyph'].isin([g for g in glyphs])].reset_index(drop=True)
    df['lum_pct'] = ((whitepoint - df['darkness']/np.max(df['darkness'])*(whitepoint-blackpoint))*100).astype(int)
    df = df.iloc[np.argsort(df['lum_pct']),:].reset_index(drop=True)

    #upper divide: choose this row's value if the luminosity is less than lum_upper
    #df['lum_upper'] = np.append(
    #        (df['lum_pct'][0:-1].to_numpy() + df['lum_pct'][1:].to_numpy() )/2,
    #        df.at[np.shape(df)[0]-1,'lum_pct'])
    df['lum_unique'] = True
    unique_v, unique_i = np.unique(df['lum_pct'], return_index=True)
    divs = pd.Series(index=df.loc[unique_i,'lum_pct'], name='chars')
    df['PCA'] = np.nan
    pca = PCA(n_components=1)
    # For each unique luminosity percentile
    for i in range(len(unique_i)):
        # Get the set of corresponding characters
        if i == len(unique_i) -1:
            start = unique_i[i]
            end = df.index[-1]
        else:
            start = unique_i[i]
            end = unique_i[i+1]-1
        # If the set size is greater than zero, do this:
        if end - start != 0:
            df.loc[start:end,'lum_unique'] = False
            #df.loc[start:end,'lum_upper'] = np.max(df.loc[start:end,'lum_upper'])
            df.loc[start:end,'PCA'] = pca_proj(df.loc[start:end,['CMDS1','CMDS2']], 1)[:,0]
            pca_order = np.argsort(df.loc[start:end,'PCA']).values
            divs[df.at[unique_i[i],'lum_pct']] = df.loc[start:end, 'glyph'].values[pca_order]
        else:
            divs[df.at[unique_i[i],'lum_pct'] ]= np.atleast_1d(df.at[unique_i[i],'glyph'])
    
    char_map = np.full((100,100), ' ')
    lum_pct_ramp = round_to_nearest_value(range(100), df['lum_pct'][unique_i])
    for lum in range(100):
        for clr in range(100):
            x = divs.at[lum_pct_ramp[lum]]
            char_map[lum, clr] = x[int(np.floor(clr/100*len(x)))]

    # IF MERGE: LUM_PCT NEEDS TO BECOME AVERAGE OF THE TOTAL SET

    # whitechar_index = np.argmin(df['darkness'])
    # blackchar_index = np.argmax(df['darkness'])

    out = [char_map[(lum,clr)] for (lum,clr) in zip(L.flatten(),C.flatten())]
    out = np.array(out).reshape(np.shape(L))
    if color_weight == 0: 
        write_output(out, out_fname, format='img', font=font, df=df, linespace=4)
        return out
    
    # TO DO: clustering for larger color_weights
    
    return out

# bug: mpimg.imread() rotates some pictures...
result = text_as_pixel(image_fname='../img/apple.jpg', out_fname = 'temp.jpg',
                       font_fname='../fonts/cour.ttf', FONT_SIZE=36, glyphs=None, color_weight=0,
                       analysis_fname=None, linespace=4,
                       grid_nx=200, grid_ny=None, interpolation=np.median,
                       whitepoint=1, blackpoint=0)