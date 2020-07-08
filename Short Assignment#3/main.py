import numpy as np
import imageio
import skimage
from skimage import morphology

def scaling(image):
    return (image - np.min(image)) * 255 / (np.max(image) - np.min(image))

def opening(input_image, k):
    ## Perform only opening
    struct_elem = morphology.disk(k)
    output_image = input_image.copy()
    
    output_image[:,:,0] =  morphology.erosion(output_image[:,:,0], struct_elem)
    output_image[:,:,0] = morphology.dilation(output_image[:,:,0], struct_elem)
    output_image[:,:,1] =  morphology.erosion(output_image[:,:,1], struct_elem)
    output_image[:,:,1] = morphology.dilation(output_image[:,:,1], struct_elem)
    output_image[:,:,2] =  morphology.erosion(output_image[:,:,2], struct_elem)
    output_image[:,:,2] = morphology.dilation(output_image[:,:,2], struct_elem)
    
    return output_image

def morphological_gradient(image, struct_elem):
    gradient_h = morphology.dilation(normalized_h, struct_elem)- morphology.erosion(normalized_h, struct_elem)
    return gradient_h

filename = str(input()).rstrip()
k = int(input())
option = int(input())
##
input_image =  imageio.imread(filename)

if option == 1:
    output_image = opening(input_image, k)
elif option == 2:
    
    hsv_image = skimage.color.rgb2hsv(input_image)
    normalized_h = scaling(hsv_image[:,:,0])
    struct_elem = morphology.disk(k)
    gradient_h = morphological_gradient(normalized_h, struct_elem)
    
    output_image = input_image.copy()
    output_image[:,:,0] = gradient_h
    output_image[:,:,1] = morphology.dilation(morphology.erosion(normalized_h, struct_elem), struct_elem)
    output_image[:,:,2] = morphology.erosion(morphology.dilation(normalized_h, struct_elem), struct_elem)
elif option == 3:
    output_image = opening(input_image, 2*k)

    hsv_image = skimage.color.rgb2hsv(output_image)
    normalized_h = scaling(hsv_image[:,:,0])
    struct_elem = morphology.disk(k)
    gradient_h = morphological_gradient(normalized_h, struct_elem)
    
    output_image = input_image.copy()
    output_image[:,:,0] = gradient_h
    output_image[:,:,1] = morphology.dilation(morphology.erosion(normalized_h, struct_elem), struct_elem)
    output_image[:,:,2] = morphology.erosion(morphology.dilation(normalized_h, struct_elem), struct_elem)



n = input_image.shape[0]*input_image.shape[1]
mrse = np.sqrt((np.sum((output_image.astype(np.int32)-input_image.astype(np.int32))**2))/n)
print('{:.4f}'.format(mrse))