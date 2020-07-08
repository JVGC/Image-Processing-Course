## Name: João Victor Garcia Coelho NUSP: 10349540
## Course Code: SCC0251 - 2020/1


## Short Assignment 1 : Filtering in Fourier Domain


import numpy as np
import imageio

def DFT(f):
    
    # create empty array of complex coefficients
    F = np.zeros(f.shape, dtype=np.complex64)
    n,m = f.shape
    
    # creating indices for x, to compute multiplication using numpy (f*exp)
    x = np.arange(n)
    # for each frequency 'u,v'
    for u in np.arange(n):
        for v in np.arange(m):
            for y in np.arange(m):
                F[u,v] += np.sum(f[:,y] * np.exp( (-1j*2*np.pi) * (((u*x)/n)+((v*y)/m)) ))
    
    return F/np.sqrt(n*m)

def inverse_DFT(F):
    
    # create empty array of complex coefficients
    f = np.zeros(F.shape, dtype=np.complex64)
    n,m = F.shape
    
    # creating indices for u, to compute multiplication using numpy (f*exp)
    u = np.arange(n)
    # for each frequency 'x,y'
    for x in np.arange(n):
        for y in np.arange(m):
            for v in np.arange(m):
                f[x,y] += np.sum(F[:,v] * np.exp( (1j*2*np.pi) * (((u*x)/n)+((v*y)/m)) ))
    
    return np.abs(f)/np.sqrt(n*m)

def filter_threshold(f, threshold, fourier_spectrum, second_peak):
    # create empty array of complex coefficients
    n,m = f.shape
    count = 0 # count of coefficients below second_peak
    F = np.zeros(f.shape, dtype=np.complex64)
    for x in np.arange(n):
        for y in np.arange(m):
            # Setting to 0 all coefficients for which the Fourier Spectrum is below
            # p2*threshold
            if((fourier_spectrum[x,y] < (second_peak*threshold))):
                count +=1
                F[x,y] = 0
            else:
                F[x,y] = f[x,y]
    return F, count


# Reading Inputs

filename = str(input()).rstrip()
threshold = float(input())

input_img =  imageio.imread(filename)


## Performing the algorithm
output_img  =DFT(input_img)

fourier_spectrum = np.abs(output_img)

second_peak = max(np.amax(fourier_spectrum[1:,:]), np.amax(fourier_spectrum[:,1:]))

output_img1, count = filter_threshold(output_img, threshold, fourier_spectrum, second_peak)

output_img =  inverse_DFT(output_img1)


## Calculating the Mean of the images and printing
original_mean =  input_img.mean()
new_mean = output_img.mean()

print("Threshold="+"{:.4f}".format(second_peak*threshold))
print("Filtered Coefficients="+"{}" .format(count))
print("Original Mean="+"{:.2f}".format(original_mean))
print("New Mean="+"{:.2f}".format(new_mean.real))
