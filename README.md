#text-as-pixel

An ASCII art generator: it makes a block of text which resembles an input image.

Unlike many other existing ASCII art generators, it takes into account the color content of the image as well. It does this by first projecting the 2D colorspace of the image onto a 1D subspace using predominant component analysis. It then creates a mapping between glyph shape and this color subspace for a given luminosity value.

Future modifications to the algorithm will allow for an increase in the "weight" of color information.


### Citation

Decolorize: Fast, Contrast Enhancing, Color to Grayscale Conversion   
Mark Grundland and Neil A. Dodgson   
Pattern Recognition, vol. 40, no. 11, pp. 2891-2896, (2007). ISSN 0031-3203.   
http://www.eyemaginary.com/Portfolio/TurnColorsGray.html   
http://www.eyemaginary.com/Rendering/decolorize.m   