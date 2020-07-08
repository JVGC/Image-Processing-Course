## Name: João Victor Garcia Coelho  NUSP:   10349540
##       Paulo André de Oliveira Carneiro   10295304
## Course Code: SCC0251 - 2020/1
## Github: https://github.com/JVGC/Assignment-2-Image-Enhancement-and-Filtering


## Assignment 1 : Image Enhancement and Filtering

import numpy as np
import imageio

# Gaussian Function
def gaussian_kernel(x, sigma):
    return (1 / (2 * np.pi * (sigma ** 2))) * np.exp((-x ** 2) / (2 * (sigma ** 2)))


## Calculating the Spatial Gaussian Component
def spatial_gaussian_component(n, sigma):
    w = np.zeros([n, n])  # creating the kernel
    a = int((n - 1) / 2)
    for x in range(n):
        for y in range(n):
            ## Euclidian Distance between this pixel and the center pixel
            euclidian = np.sqrt((x - a) ** 2 + (y - a) ** 2)
            w[x, y] = gaussian_kernel(euclidian, sigma)

    return w


def bilateral_filter(input_img, kernel_size, sigma_s, sigma_r):

    ## Adding padding zero in the input image
    nrows, ncols = input_img.shape
    a = int((n - 1) / 2)
    input_img = np.concatenate((input_img, np.zeros([a, ncols])), axis=0)
    input_img = np.concatenate((np.zeros([a, ncols]), input_img), axis=0)
    input_img = np.concatenate((np.zeros([nrows + 2 * a, a]), input_img), axis=1)
    input_img = np.concatenate((input_img, np.zeros([nrows + 2 * a, a])), axis=1)

    ## Calculating the spatial component centered at the pixel (x,y)
    spatial_gaussian = spatial_gaussian_component(n, sigma_s)

    ## Creating the output image as a zero matriz
    output_img = np.zeros(input_img.shape, dtype=np.uint8)
    nrows, ncols = input_img.shape

    for x in range(a, nrows - a):
        for y in range(a, ncols - a):

            # for every pixel, the normalization factor (Wp)
            # and the the Intensity of the output pixel
            # are initialized as 0
            Wp = 0
            If = 0

            # gets subImage
            sub_img = input_img[x - a : x + a + 1, y - a : y + a + 1]

            ## Calculating the range gaussian for every neighboor of pixel (x,y)
            range_gaussian = gaussian_kernel(sub_img - input_img[x, y], sigma_r)

            ## Final value of the filter
            w = np.multiply(range_gaussian, spatial_gaussian)

            ## Calculating the normalization factor by summing the filter's values
            Wp = np.sum(w)

            ## Multiplying the filter's value of each pixel by its intensity in the input image
            ## And summing all these values
            If = np.sum(np.multiply(w, sub_img))

            ## Finally, calculating the output value of the pixel (x,y)
            ## by dividing it by the normalization value
            output_img[x, y] = If / Wp
    return input_img, output_img


def scaling(image):
    return (image - np.min(image)) * 255 / (np.max(image) - np.min(image))


def kernel(sigma, n):
    kernel = np.zeros((1, n))
    a = int((n - 1) / 2)
    for x in range(n):
        kernel[0, x] = gaussian_kernel(x - a, sigma)

    return kernel


def vignette_filter(input_img, sigma_row, sigma_col):
    Wrow = kernel(sigma_col, input_img.shape[1])
    Wcol = kernel(sigma_row, input_img.shape[0])
    W = np.transpose(Wcol) * Wrow
    output_img = scaling(W * input_img)
    return output_img


def padding(input_img, a, b):
    nrows, ncols = input_img.shape
    input_img = np.concatenate((input_img, np.zeros([a, ncols])), axis=0)
    input_img = np.concatenate((np.zeros([a, ncols]), input_img), axis=0)
    input_img = np.concatenate((np.zeros([nrows + 2 * b, b]), input_img), axis=1)
    input_img = np.concatenate((input_img, np.zeros([nrows + 2 * b, b])), axis=1)

    return input_img


def unpadding(img, a, b):
    return img[a : img.shape[0] - a, b : img.shape[1] - b]


def conv(kernel, image):
    # Applying the convolutional filter on the image
    a = int((kernel.shape[0] - 1) / 2)
    b = int((kernel.shape[1] - 1) / 2)

    # padding the input image
    image = padding(image, a, b)
    new_image = np.zeros(image.shape)

    for x in range(a, image.shape[0] - a):
        for y in range(b, image.shape[1] - b):
            neighborhood = image[x - a : x + a + 1, y - b : y + b + 1]
            value = np.sum(np.multiply(neighborhood, kernel))
            new_image[x, y] = value

    return unpadding(image, a, b), unpadding(new_image, a, b)


def laplacian_filter(kernel, image, c):
    # Convolving the original image
    image, new_image = conv(kernel, image)

    # Scaling the filtered image If , using normalization(0 - 255)
    new_image = scaling(new_image)

    # Adding the filtered image, multiplied by the parameter c, back to the original image.
    new_image = new_image * c + image

    # Scaling the filtered image If , using normalization(0 - 255)
    new_image = scaling(new_image)

    return image, new_image.astype(np.uint8)


filename = str(input()).rstrip()
method = int(input())
save = int(input())

input_img = imageio.imread(filename)

if method == 1:
    n = int(input())
    sigma_s = float(input())
    sigma_r = float(input())
    input_img, output_img = bilateral_filter(input_img, n, sigma_s, sigma_r)
elif method == 2:
    kernels = {}
    kernels[1] = np.matrix([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernels[2] = np.matrix([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    c = float(input())
    k = int(input())
    input_img, output_img = laplacian_filter(kernels[k], input_img, c)
elif method == 3:
    sigma_row = float(input())  # Caso5: 50
    sigma_col = float(input())  # Caso5: 50
    output_img = vignette_filter(input_img, sigma_row, sigma_col)
else:
    print("invalid method")
# ERROR
rse = np.sqrt(np.sum((input_img - output_img) ** 2))
# Printing the error rounding to 4 decimal places.
print("{:.4f}".format(rse))

if save:
    imageio.imwrite("output_img.png", output_img)
