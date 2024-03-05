import numpy as np 
from numpy.fft import fftn

r'''Parameters of Potential vorticity equation with total variation.
'''

#Define parameters, units
g = 10 #m/s^2
# avera latitude
theta = 40
# Coriolis parameter, 1/s
f0 = 2 * (2 * np.pi / (24 * 3600)) * np.sin(np.radians(theta))
beta = 2 * (2 * np.pi / (24 * 3600)) * np.cos(np.radians(theta))/6400000
# Rosby radius of deformation,
LR = 25000 # in meters,
# pixel lenght
dx = 111000/60
dy = 111000/60




def vorticity_y(ngrid_lon,ngrid_lat):

    a = np.ones((1, ngrid_lon))
    at = np.transpose(a)
    b = np.arange(0, ngrid_lat)
    bt = np.transpose(b)


    y = np.transpose(np.kron(at, b))
    return y 


def psf2otf(psf, output_shape=None):
    psfsize = np.array(psf.shape)
    if output_shape is not None:
        outsize = np.array(output_shape)
        padsize = outsize - psfsize
    else:
        padsize = np.zeros(psfsize.shape, dtype=np.int64)
        outsize = psfsize

    psf = np.pad(psf, ((0, padsize[0]), (0, padsize[1])), 'constant')

    for i in range(len(psfsize)):
        psf = np.roll(psf, -int(psfsize[i] / 2), i)

    otf = fftn(psf)
    nElem = np.prod(psfsize)
    nOps = 0

    for k in range(len(psfsize)):
        nffts = nElem / psfsize[k]
        nOps = nOps + psfsize[k] * np.log2(psfsize[k]) * nffts

    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)

    return otf



def laplace(SSH):
    kernel = np.array([[0, (1/dy)**2, 0], [1/dx**2, -(2/dx**2)  - (2/dy**2), 1/dy**2], [0, 1/dy**2, 0]])
    otfA = psf2otf(kernel, SSH.shape)
    return otfA 


