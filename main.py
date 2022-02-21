"""
Author: Emmett Lawlor
Student ID: 18238831
Module: RE4012/RE4017
"""

from PIL import Image
import numpy as np
from skimage.transform import rotate
import scipy.fftpack as fft 

pil_im = Image.open('sinogram.png')
print(type(pil_im))
channels = pil_im.split()

r = np.array(channels[0])
g = np.array(channels[1])
b = np.array(channels[2])

def ramp_transform(channel):
    return fft.rfft(channel, axis=1)

def ramp_filter(ffts):
    # Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts*ramp

def inverse_ramp(channel):
    return fft.irfft(channel, axis=1)

"Reconstruct the image by back projecting the filtered projections"
def back_project(operator):
    laminogram = np.zeros((operator.shape[1],operator.shape[1]))
    dTheta = 180.0 / operator.shape[0]
    for i in range(operator.shape[0]):
        temp = np.tile(operator[i],(operator.shape[1],1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram

def channels_rescaling(ch):
    chi, clo = ch.max(), ch.min()
    chnorm = 255 * ((ch - clo) / (chi - clo))
    ch8bit = np.floor(chnorm).astype(np.uint8)
    return ch8bit

print("Reconstructed image as is")
r, b, g = (back_project(r), back_project(b), back_project(g))
reconstructed_image = np.dstack( (r.astype(np.uint8) * 255, b.astype(np.uint8) * 255, g.astype(np.uint8) * 255) )
Image.fromarray(reconstructed_image).show()

print("Ramp transfer freq domain")
r, b, g = (ramp_transform(r), ramp_transform(b), ramp_transform(g))
Image.fromarray(r).show()

print('Ramp filtered freq domain')
r, b, g = (ramp_filter(r), ramp_filter(b), ramp_filter(g))
Image.fromarray(r).show()

print('Inverse ramp filter domain')
r, b, g = (inverse_ramp(r), inverse_ramp(b), inverse_ramp(g))
Image.fromarray(r).show()

print('Back projected inverse ramp filtered')
r, b, g = (back_project(r), back_project(b), back_project(g))
r, b, g = (channels_rescaling(r), channels_rescaling(b), channels_rescaling(g))
reconstructed_image = np.dstack( (r, b, g) )
Image.fromarray(reconstructed_image).show()