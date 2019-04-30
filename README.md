#text-as-pixel

An ASCII art generator: it makes a block of text which resembles an input image.

Unlike many other existing ASCII art generators, it takes into account the color content of the image as well. 

##### Features 
text-as-pixel allows you to choose
* any monospace font in bitmap/TrueType/OpenType format. (For non-monospace fonts, it will use the largest monospace subset.)
* the glyphs to use for constructing the image (e.g. all of them, or a smaller subset like "MYNAMEmyname !\*.,\~")
* the balance between luminosity resolution vs. color resolution through the `lum_divs` parameter.
* the image dimensions (in number of characters).
* the font size.
* the spacing between each line of text in the image.
* an interpolation method to use when re-scaling the image.

##### Drawbacks
* The glyph set is limited to ASCII 32 to 126:
 !"#$%&\'()\*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^\_\`abcdefghijklmnopqrstuvwxyz{|}~
* The font file MUST have all of the above glyphs (TODO: check beforehand and prune to remove this constraint)
* Kerning is not taken into account: you should use monospace fonts.

### How to use it

TODO

### How it works

TODO

### Citation

My method of processing the color content directly uses this algorithm:

Decolorize: Fast, Contrast Enhancing, Color to Grayscale Conversion   
Mark Grundland and Neil A. Dodgson   
Pattern Recognition, vol. 40, no. 11, pp. 2891-2896, (2007). ISSN 0031-3203.   
http://www.eyemaginary.com/Portfolio/TurnColorsGray.html   
http://www.eyemaginary.com/Rendering/decolorize.m   