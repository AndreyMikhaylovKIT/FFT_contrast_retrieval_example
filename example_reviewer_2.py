"""
Created by Mikhaylov Andrey. 
Last update 15.07.2022.

andrey.mikhaylov@kit.edu

based on Wen, H. H., Bennett, E. E., Kopace, R., Stein, A. F., & Pai, V. (2010). 
Single-shot x-ray differential phase-contrast and diffraction imaging using two-dimensional transmission gratings. 
Optics letters, 35(12), 1932-1934.
doi: https://doi.org/10.1364/OL.35.001932
"""





import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave
import scipy.ndimage

# rotational angle, iHartmann mask wasnt aligned perfectly 
angle = 0.366

# Specify paths to the sample, dark and flat images
sample = scipy.ndimage.rotate(imread(r'I:\example\data\frame_0000.tif').astype(np.float32),angle)[248:1844,245:1841]
dark = scipy.ndimage.rotate(imread(r'I:\example\data\dark.tiff').astype(np.float32),angle)[248:1844,245:1841]
flat = scipy.ndimage.rotate(imread(r'I:\example\data\flat.tif').astype(np.float32),angle)[248:1844,245:1841]

# number of periods in the field of view
fft_00 = 248
fft_shape_half = int(fft_00/2)
# image size in pixels
shape = sample.shape

# real space hanning window to avoid artifacts on the borders
rsw = np.hanning(sample.shape[0]) * (np.ones(sample.shape) * np.hanning(sample.shape[1])).T 

# Positions of the harminoics in the frequency domain
pos_00 = int(shape[0]/2 - fft_shape_half),int(shape[0]/2 + fft_shape_half)
pos_01 = int(shape[0]/2 - 3*fft_shape_half),int(shape[0]/2 - fft_shape_half)
pos_10 = int(shape[0]/2 - 3*fft_shape_half),int(shape[0]/2 - fft_shape_half)



# FFT of the sample and flat images after dark field correction
fft_sample = np.fft.fftshift(np.fft.fft2((sample-dark)*rsw))
fft_flat = np.fft.fftshift(np.fft.fft2((flat-dark)*rsw))


# Exact crops of the harmonics for sample (_s) and flat
I_0_s = fft_sample[pos_00[0]:pos_00[1],pos_00[0]:pos_00[1]]
I_0 = fft_flat[pos_00[0]:pos_00[1],pos_00[0]:pos_00[1]]

I_01_s = fft_sample[pos_00[0]:pos_00[1],pos_01[0]:pos_01[1]]
I_01 = fft_flat[pos_00[0]:pos_00[1],pos_01[0]:pos_01[1]]

I_10_s = fft_sample[pos_10[0]:pos_10[1],pos_00[0]:pos_00[1]]
I_10 = fft_flat[pos_10[0]:pos_10[1],pos_00[0]:pos_00[1]]


# fourier space hanning window to avoid artifacts on the borders
fftw = np.hanning(I_0_s.shape[0]) * (np.ones(I_0_s.shape) * np.hanning(I_0_s.shape[1])).T

# Transmission
T = np.fft.ifft2(np.fft.fftshift(I_0_s*fftw))/np.fft.ifft2(np.fft.fftshift(I_0*fftw))

# minus log of scattering
S_log_01 = -np.log((np.fft.ifft2(np.fft.fftshift(I_01_s*fftw))/np.fft.ifft2(np.fft.fftshift(I_01*fftw)))/(T))
S_log_10 = -np.log((np.fft.ifft2(np.fft.fftshift(I_10_s*fftw))/np.fft.ifft2(np.fft.fftshift(I_10*fftw)))/(T))

# scattering before logarithm for reference
S_01 = (np.fft.ifft2(np.fft.fftshift(I_01_s*fftw))/np.fft.ifft2(np.fft.fftshift(I_01*fftw)))/(T)
S_10 = (np.fft.ifft2(np.fft.fftshift(I_10_s*fftw))/np.fft.ifft2(np.fft.fftshift(I_10*fftw)))/(T)

# phase contrast
P_01 = np.unwrap(np.unwrap(np.angle(np.fft.ifft2(np.fft.fftshift(I_01_s*fftw)))-np.angle(np.fft.ifft2(np.fft.fftshift(I_01*fftw))),axis=1),axis=0)
P_10 = np.unwrap(np.unwrap(np.angle(np.fft.ifft2(np.fft.fftshift(I_10_s*fftw)))-np.angle(np.fft.ifft2(np.fft.fftshift(I_10*fftw))),axis=0),axis=1)



# Visualization

plt.imshow(np.abs(T), cmap='Greys_r')
plt.title('Transmission')
plt.colorbar()
plt.show()

plt.imshow(np.abs(S_log_01), cmap='Greys_r')
plt.title('Log scattering 01')
plt.colorbar()
plt.show()

plt.imshow(np.abs(S_log_10), cmap='Greys_r')
plt.title('Log scattering 10')
plt.colorbar()
plt.show()

plt.imshow(np.abs(S_01), cmap='Greys_r',vmax=1)
plt.title('Scattering no log 01')
plt.colorbar()
plt.show()

plt.imshow(np.abs(S_10), cmap='Greys_r',vmax=1)
plt.title('Scattering no log 10')
plt.colorbar()
plt.show()

plt.imshow(np.abs(P_01), cmap='Greys_r')
plt.title('DPhase 01')
plt.colorbar()
plt.show()

plt.imshow(np.abs(P_10), cmap='Greys_r')
plt.title('DPhase 10')
plt.colorbar()
plt.show()



# =0.5*(-log(x1) + (-log(x2))) = 0.5*log(1/(x1*x2)) - half sum of the scattering.
# Background is not zero. Fitting of the background gives 0.332 of flat value.
# After that all negative values are broken pixels.

s_sum = 0.5*np.log(1/(np.abs(S_01)*np.abs(S_10)))

plt.imshow(s_sum-0.332, cmap='Greys_r',vmin=0)
plt.title('Scatt. sum')
plt.colorbar()
plt.show()

# uncomment if you want to save data and specify path

# imsave(r'I:\example\T.tif', np.abs(T).astype(np.float32))
# imsave(r'I:\example\s01.tif', np.abs(S_01).astype(np.float32))
# imsave(r'I:\example\s10.tif', np.abs(S_10).astype(np.float32))
# imsave(r'I:\example\p01.tif', np.abs(P_01).astype(np.float32))
# imsave(r'I:\example\p10.tif', np.abs(P_10).astype(np.float32))
# imsave(r'I:\example\s_log_01.tif', np.abs(S_log_01).astype(np.float32))
# imsave(r'I:\example\s_log_10.tif', np.abs(S_log_10).astype(np.float32))
# imsave(r'I:\example\s_sum.tif', s_sum.astype(np.float32))




