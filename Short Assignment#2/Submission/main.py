import numpy as np
import imageio
from scipy.fftpack import fftn, ifftn, fftshift

def gaussian_filter(k=3, sigma=1.0):
   arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
   x, y = np.meshgrid(arx, arx)
   filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma) )
   return filt/np.sum(filt)

def denoise_gaussian(input_image, k, sigma):
   
    gaussian = gaussian_filter(k, sigma)
    pad1 = (input_image.shape[0]//2)-gaussian.shape[0]//2
    gaussian_padded = np.pad(gaussian, (pad1,pad1-1), "constant",  constant_values=0)
    
    max_g = np.amax(input_image)
    
    FFT_input_image = fftn(input_image)
    FFT_gaussian =  fftn(gaussian_padded)
    
    G = np.multiply(FFT_input_image, FFT_gaussian)
    r = np.real(fftshift(ifftn(G)))
    r = (r - np.min(r)) * max_g / (np.max(r) - np.min(r))
    
    return r

def CLSF(input_image, k, sigma, gamma):
    gaussian = gaussian_filter(k, sigma)
    pad1 = (input_image.shape[0]//2)-gaussian.shape[0]//2
    gaussian_padded = np.pad(gaussian, (pad1,pad1-1), "constant",  constant_values=0)
    
    max_d =  np.amax(input_image)
    
    laplacian =  np.array([[ 0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    pad1 = (input_image.shape[0]//2)-laplacian.shape[0]//2
    laplacian_padded = np.pad(laplacian, (pad1,pad1-1), "constant",  constant_values=0)
    
    FFT_denoised_image = fftn(denoised_image)
    FFT_gaussian =  fftn(gaussian_padded)
    FFT_laplacian = fftn(laplacian_padded)
    
    f = np.divide(FFT_gaussian.conjugate(), 
                  (np.abs(FFT_gaussian)**2 + gamma*np.abs(FFT_laplacian)**2 )) * FFT_denoised_image
    
    r = np.real(fftshift(ifftn(f)))
    deblurred_image = (r - np.min(r)) * max_d / (np.max(r) - np.min(r))
    
    return deblurred_image

filename = str(input()).rstrip()
k = int(input())
sigma = float(input())
gamma =  float(input())

## LOCAL
#input_image =  imageio.imread('../images_sa2/'+filename)

# TO SUBMISSION
input_image =  imageio.imread(filename)

denoised_image =  denoise_gaussian(input_image, k, sigma)

deblurred_image =  CLSF(denoised_image, k, sigma, gamma)

std =  np.std(deblurred_image[:])
print("{:.1f}".format(std))