## Name: João Victor Garcia Coelho NUSP: 10349540
## Course Code: SCC0251 - 2020/1


## Assignment 1 : intensity transformations


import numpy as np
import imageio

# Definitions of transformations

# Inversion Transformation
def inversion(z):
    return 255 - z

# Contrast Modulation Tranform
# Transform the intensity level range of a image into a new range
def contrast_modulation(z, c, d):
    imin =  np.min(z) # getting the lowest intensity value of image z
    imax = np.max(z) # getting the highest intensity value of image z
    new_range = d - c # the difference of the new range

    # applying the transform on image z
    # divide by 255 because the input image z has always a range of [0,255] in this assingment
    return z*(new_range/255) + c

# Logarithmic Transformation
def logarithmic(z):
    imax =  np.max(z) # getting the highest intensity value of image z
    c_scale =  255/np.log2(imax+1) # constant scalar 
    
    # applying the transform on image z
    # sum 1 in log2 to avoid log2(0) that doesn't exists
    return (c_scale* np.log2(z.astype(np.int32)+1))

# Gamma Adjustment Transform
def gamma_adjustment(z, W, lambd):
    return (W*np.power(z, lambd))

# Reading Basic Inputs


filename = str(input()).rstrip()
input_img = imageio.imread(filename)
method = int(input())
save = int(input())

# Selecting the given method and apply the transformation to the image
if method == 1:
    output_img = inversion(input_img)

if method == 2:
    # Reading another parameters of constrast modulation transform
    c = int(input())
    d = int(input())
    output_img = contrast_modulation(input_img, c, d)
if method == 3:
    output_img = logarithmic(input_img)

if method == 4:
     # Reading another parameters of gamma adjustment transform
    W = int(input())
    lambd = float(input())
    output_img = gamma_adjustment(input_img, W, lambd)

# Saving the image if asked so
if save == 1:
    imageio.imwrite('output_img.png',output_img)

##################### Applying the Root Squared Error(RSE) to input and output images

################### Explaining the condition to the method equals 1

# The inversion transformation transforms the darker pixels in brighter ones and vice versa
# So in brighter images, the difference between an inversion image and the original gives us
# negative numbers. Because the input image in of the type 'uint8', it does not permit negative numbers,
# so we transform the images in 'int32' to embrace these numbers and calculate the correct error

# Mathematically, we have:

# RSE = Inverted_Image - Original Image =  255 -z -z =  255 -2z
# If we have a pixel that has a value greater than 127, this difference will result in a negative number

if method == 1:
    rse = np.sqrt(np.sum((output_img.astype(np.int32)-input_img.astype(np.int32))**2))
else:
    rse = np.sqrt(np.sum((output_img - input_img)**2))  

# Printing the error rounding to 4 decimal places.
print('{:.4f}'.format(rse))