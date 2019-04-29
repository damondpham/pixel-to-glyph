import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFont, ImageDraw
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

def load_font(font_fname, FONT_SIZE):
    ftype = font_fname.split('.')[-1]
    if ftype == 'pil': #bitmap
        font = ImageFont.load(font_fname, FONT_SIZE)
    if ftype == 'ttf': #TrueType or OpenType
        font = ImageFont.truetype(font_fname, FONT_SIZE)
    else: raise ValueError(ftype + ' is not a recognized font file extension')
    return font

# Currently, this will only work with fonts which are monospace (and without kerning!).
# For near-monospaced fonts, it will remove glyphs whose widths are not equal to the mode.  
def analyze_font(font_fname, FONT_SIZE=36, save=True, analysis_fname=None, visualize=True):
    font = load_font(font_fname, FONT_SIZE)
    
    # The font must have the ASCII 32 to 126 (decimal) glyphs:
    glyphs_all =  ' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
    df = pd.DataFrame([g for g in glyphs_all], columns = ['glyph'])
    df['ASCII_code'] = [i+32 for i in range(126-32+1)]
    
    # Get the standard width and maximum height
    image = Image.new('1', (0, 0), color=1)
    draw = ImageDraw.Draw(image)
    df['width'] = [draw.textsize(g, font=font)[0] for g in glyphs_all]
    df['height'] = [draw.textsize(g, font=font)[1] for g in glyphs_all]
    
    if len(set(df['width'])) != 1: 
        # Throw out characters of different widths to make a monospaced set
        print('Warning: ' + font_fname + ' ' + str(FONT_SIZE) + 'pt is not monospace')
        mode_width = max(set(df['width']), key=list(df['width']).count)
        remove = df['width'] != mode_width
        print('Removing these glyphs:' + ''.join([glyphs_all[i] for i,x in enumerate(remove) if x]))
        
        GLYPHS = ''.join([glyphs_all[i] for i,x in enumerate(remove) if not x])
        df = df.loc[np.invert(remove), :].reset_index(drop=True)
    else: GLYPHS = glyphs_all
        
    WIDTH = df.at[0,'width']
    MAX_DIM = (WIDTH, max(df['height']))
    
    # Get the "darkness" of each glyph,
    ## the proportion of the glyph's area which its shape covers.
    image = Image.new('1', (WIDTH*len(GLYPHS), MAX_DIM[1]), color=1)
    draw = ImageDraw.Draw(image)
    draw.text((0,0), GLYPHS, font=font)
    image_as_array = np.array(image.getdata()).reshape((MAX_DIM[1], MAX_DIM[0]*len(GLYPHS)))
    for i in range(len(GLYPHS)):
        glyph_as_array = image_as_array[:,(WIDTH*i):(WIDTH*(i + 1))]
        df.at[i,'darkness'] = np.sum(1-glyph_as_array) / np.prod(MAX_DIM)
    
    # Compute pairwise dissimilarity via Hamming distance, i.e.
    ## the best overlap among all horizontal translations.
    dissim = np.zeros((len(GLYPHS), len(GLYPHS)))
    for i in range(len(GLYPHS)):
        print('computing dissimilarities for ' + df.at[i, 'glyph'])
        i_array = image_as_array[:,(WIDTH*i):(WIDTH*(i + 1))]
        template = np.pad(i_array, ((0,0), (WIDTH-1, WIDTH-1)), 'constant')
        for j in range(len(GLYPHS)):
            min_Hamming = 1
            if i >= j: next
            j_array = image_as_array[:,(WIDTH*j):(WIDTH*(j + 1))]
            for t in range(WIDTH*2-2):
                XOR = np.logical_xor(template, np.pad(j_array, ((0,0), (t, WIDTH*2-2-t)), 'constant'))
                Hamming = np.sum(XOR) / (np.sum(i_array) + np.sum(j_array))
                min_Hamming = min(Hamming, min_Hamming)
                
            dissim[i,j] = min_Hamming
            dissim[j,i] = min_Hamming
    dissim = np.abs(dissim)
    
    # Embed dissimilarities in 2D space via multidimensional scaling.
    embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    CMDS = embedding.fit_transform(dissim)
    df['CMDS1'] = CMDS[:,0]
    df['CMDS2'] = CMDS[:,1]
    
    if visualize:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.scatter(CMDS[:,0], CMDS[:,1], s=0)
        
        for i in range(len(GLYPHS)):
            ax.annotate(GLYPHS[i], (CMDS[i,0], CMDS[i,1]))
    
    # Save analysis.
    if save: 
        if analysis_fname is None:
            path = os.path.join(os.getcwd(), 
                                '..', os.fspath('fonts'), os.fspath('analysis'))
            if not os.path.exists(path): os.makedirs(path)
            analysis_fname = os.path.join(path, ''.join(font_fname.split('.')[:-1]) + 
                                          '_' + str(FONT_SIZE) + 'pt.tsv')
        df.to_csv(analysis_fname, sep='\t')
    
    return df

# demo:
# font_fname = 'cour.ttf'
# font_fname = os.path.join(os.getcwd(), '..', os.fspath('fonts'), font_fname)
# analysis = analyze_font(font_fname)