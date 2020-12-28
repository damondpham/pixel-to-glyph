import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFont, ImageDraw # pip install pillow
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from utils import normalize_values

def load_font(typeface_fname, FONT_SIZE=36):
    '''
    Load up and return a PIL.ImageFont object

    Parameters:
        typeface_fname (str): Path to a bitmap, TrueType, or OpenType typeface file.
        FONT_SIZE (int): The font size (for ttf/otf). Default 12.

    Returns:
        font: The PIL.ImageFont font object
    '''

    try: font = ImageFont.load(typeface_fname, FONT_SIZE) # bitmap
    except:
        try: font= ImageFont.truetype(typeface_fname, FONT_SIZE) #ttf/otf
        except: raise ValueError(typeface_fname + ' is not bitmap, ttf, or otf.')
    return font

def analyze_font(font, unicodes=range(0x0020, 0x007E), evenly_space_darkness=0.5,
    save=True, font_analysis_fname="font_analysis.tsv", verbose=True):
    '''
    Collects information about the characters in a font to use later when building images.
    If the glyphs for the given unicode values are not monospace, the largest monospace set will be used.

    Parameters:
    	font (PIL.ImageFont object).
        unicodes (iterable): The unicode values for the glyphs to analyze.
        evenly_space_darkness: The darkness score depends on two measurements.
            The first is the proportion of a glyph's area which its shape 
            covers. This measurement directly corresponds to the amount of
            pixels "colored-in," but it may not have a wide range for the glyph
            set. The second is the rank of the first measurement divided by
            the total number of glyphs. This measurement will be evenly-spaced.
            Choose a value of 0 to only use the first measurement, a value of 
            one to only use the second, and a value of 0.5 to use both equally.
        save (bool): Should the analysis be saved to a tsv file, `font_analysis_fname`?
        font_analysis_fname (str): The file to save the analysis to.
        verbose (bool): Print occasional updates?

    Returns:
        df (pandas.DataFrame): A table of information about each glyph in the monospace set:
        	glyph (str): The glyph itself
        	unicode (int): The unicode value of the glyph (base 10). See e.g. https://unicode-table.com/en/
        	width (int): The width, in pixels. This will be the same for all glyphs in the table.
        	height (int): THe height, in pixels. This can differ between glyphs.
        		Note: the lowest height will be the "true" height for a line of text.
        	darkness (float): The proportion of the glyph's area which its shape covers. Range: [0,1].
        	CMDS1 (float): First similarity dimension. 
            
                The lowest Jaccard distance (i.e. lowest dissimilarity) between
        		horizontal translations of each pair of glyphs is computed. Then, CMDS is performed
        		on the dissimilarity matrix to make a 2D embedding of the glyphs. CMDS1 is the first axis of
        		this embedding.
        	CMDS2 (float): Second similarity dimension.
    '''

    # # Stop if combining characters are in the glyph set.
    # combining = [i for j in (range(0x0300, 0x036F), range(0x1AB0, 0x1AFF),
    #     range(0x1DC0, 0x1DFF), range(0x20D0, 0x20FF), range(0xFE20, 0xFE2F)) for i in j]

    # if np.intersect1d(np.array(unicodes), np.array(combining)).shape > 0:
    #     raise ValueError('cannot use combining characters.')

    # Get the glyphs for each unicode value.
    df = pd.DataFrame([np.array([str(chr(i)) for i in unicodes]), np.array(unicodes)]).transpose()
    df.columns=['glyph', 'unicode']

    # Get the standard width and maximum height.
    image = Image.new('1', (0, 0), color=1)
    draw = ImageDraw.Draw(image)
    glyph_dims = [draw.textsize(str(chr(g)), font=font) for g in df['unicode']] #libraqm: features='-liga-kern'
    df['width'], df['height'] = zip(*glyph_dims)

    # Handle non-identical widths
    if len(set(df['width'])) != 1:
        # Throw out characters of different widths to make a monospaced set.
        print('Warning: this font is not exactly monospace')
        mode_width = max(set(df['width']), key=list(df['width']).count)
        removed = df.loc[df['width'] != mode_width, :].reset_index(drop=True)
        df = df.loc[df['width'] == mode_width, :].reset_index(drop=True)
        print('These glyphs do not have the mode width %d and will be removed:'%mode_width)
        print(removed)

    # Define the final set of glyphs and their dimensions.    
    GLYPHS = [str(chr(i)) for i in df['unicode']]
    WIDTH = df.at[0,'width']
    HEIGHT = max(df['height'])
    MAX_DIM = (WIDTH, HEIGHT)

    # Check for variable-width glyphs (e.g. combining characters).
    image = Image.new('1', (0, 0), color=1)
    draw = ImageDraw.Draw(image)
    # TODO: the below line assumes "a" is in the glyph set. 
    dim_of_a = draw.textsize('a', font=font)
    glyph_dims_btwn = [draw.textsize('a' + str(chr(g)) + 'a', font=font) for g in df['unicode']] #libraqm: features='-liga-kern'
    okay = np.array([glyph_dims_btwn[i][0] == dim_of_a[0]*2 + df.at[i, 'width'] for i in range(len(glyph_dims_btwn))])
    if not np.all(okay):
        removed = df.loc[list(~okay),:].reset_index(drop=True)
        print('These glyphs are combining or variable width and will be removed:')
        print(removed)
        df = df.loc[okay,:].reset_index(drop=True)
        GLYPHS = [str(chr(i)) for i in df['unicode']]

    if df.shape[0] == 0: raise ValueError('No valid glyphs.')

    # Get the "darkness" of each glyph.
    ## First, measure the proportion of pixels that are covered by the glyph/colored in.
    image = Image.new('1', (WIDTH*len(GLYPHS), HEIGHT), color=1)
    draw = ImageDraw.Draw(image)
    draw.text((0,0), ''.join(GLYPHS), font=font)
    all_glyphs = np.array(image.getdata()).reshape((HEIGHT, WIDTH*len(GLYPHS)))
    glyphs_cropped = [all_glyphs[:,(WIDTH*i):(WIDTH*(i + 1))] for i in range(len(GLYPHS))]
    glyphs_pixels = [np.sum(1-g) for g in glyphs_cropped]
    df['pixels'] = glyphs_pixels
    df['darkness'] = df['pixels']/np.prod(MAX_DIM)
    # Second, rank the first measurement, scale these measurements, and add them up.
    if evenly_space_darkness != 0:
        darkness2 = np.argsort(df['darkness'])/df.shape[0]
        df['darkness'] = ((1 - evenly_space_darkness) * df['darkness']) + (evenly_space_darkness * darkness2)
        df['darkness'] = normalize_values(df['darkness'], M_min=0, M_max=1)

    # Compute pairwise dissimilarity via Jaccard distance. Record
    ## the lowest Jaccard distance among all horizontal translations.
    # TODO: speed up via matrix ops.
    dissim = np.zeros((len(GLYPHS), len(GLYPHS)))
    glyphs_mat = np.tile(np.pad(np.vstack(glyphs_cropped), ((0,0), (0, 1)), 'constant', constant_values=1), WIDTH)
    first_segments = [slice(i*(WIDTH+1), (i+1)*(WIDTH+1) -(i+1)) for i in range(WIDTH)]
    second_segments = [slice((i+1)*(WIDTH+1)-i, (i+1)*(WIDTH+1)) for i in range(WIDTH)]
    segments = first_segments + second_segments

    for i in range(len(GLYPHS)-1):
        i_glyph = glyphs_cropped[i]
        i_glyph_mat = np.tile(i_glyph, (len(GLYPHS)-(i+1), WIDTH+1))
        bg_glyph_mat = glyphs_mat[(i+1)*HEIGHT:,:]

        AND_ = np.logical_and(bg_glyph_mat < 1, i_glyph_mat < 1)

        for j in range(len(GLYPHS)-i-1):
            r = slice(j*HEIGHT, (j+1)*HEIGHT)
            this_and = np.array([np.sum(AND_[r,segments[k]]) for k in range(len(segments))])
            this_or = glyphs_pixels[i] + glyphs_pixels[i+j+1] - this_and
            empty = np.equal(this_or, 0)
            if np.any(empty):
                this_and = this_and[~empty]
                this_or = this_or[~empty]
            if np.sum(this_or) == 0:
                best_Jaccard = 0
            else:
                Jaccard = [(this_or[k] - this_and[k])/this_or[k] for
                           k in range(len(this_or))]
                best_Jaccard = np.min(Jaccard)

            dissim[i,j+i+1] = best_Jaccard
            dissim[j+i+1,i] = best_Jaccard

    dissim = np.abs(dissim)

    # Embed dissimilarities in 2D space via multidimensional scaling.
    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    CMDS = embedding.fit_transform(dissim)
    df['CMDS1'] = CMDS[:,0]
    df['CMDS2'] = CMDS[:,1]
    df['CMDS1'] = (df['CMDS1'] - np.min(df['CMDS1'])) / (np.max(df['CMDS1']) - np.min(df['CMDS1']))
    df['CMDS2'] = (df['CMDS2'] - np.min(df['CMDS2'])) / (np.max(df['CMDS2']) - np.min(df['CMDS2']))

    # Save analysis to file if requested.
    if save:
        if font_analysis_fname is None: font_analysis_fname = 'demo_font_analysis.tsv'
        df.to_csv(font_analysis_fname, sep='\t')

    return df

def plot_font_CMDS(font, font_analysis, plot_fname):
    '''
    Display the CMDS embedding of a font analysis.

    Parameters:
    	font (PIL.ImageFont object).
    	font_analysis (pandas.DataFrame object): an analysis of the font made by
            `analyze_font`
    '''

    WIDTH = font_analysis.at[0,'width']
    HEIGHT = max(font_analysis['height'])
    cmds_dim = WIDTH * 50
    image = Image.new('L', (cmds_dim + WIDTH*3, cmds_dim + HEIGHT*3), color='white')
    draw = ImageDraw.Draw(image)

    for i in range(font_analysis.shape[0]):
        draw.text(
            (font_analysis.at[i,'CMDS1']*cmds_dim + WIDTH, font_analysis.at[i,'CMDS2']*cmds_dim + HEIGHT), 
            chr(font_analysis.at[i, 'unicode']), font=font
        )

    image.save(plot_fname)