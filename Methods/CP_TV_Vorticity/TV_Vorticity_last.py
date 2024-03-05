import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftn
import sys 
sys.path.append('..')
print(sys.path)
from utils.toolbox import *


r'''Fixing the values for the potential vorticity, and defining the physical equations. 
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
#Laplacian kernel 
kernel = np.array([[0, (1/dy)**2, 0], [1/dx**2, -(2/dx**2)  - (2/dy**2), 1/dy**2], [0, 1/dy**2, 0]])

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


def optA(x, otfA):
    Ax = np.copy(x)
    Ax[:, :] = np.real(np.fft.ifft2(otfA * np.fft.fft2(x[:, :])))
    return Ax


def optAT(x, otfA):
    Ax = np.copy(x)
    Ax[:, :] = np.real(np.fft.ifft2(np.conj(otfA) * np.fft.fft2(x[:, :])))
    return Ax






def B_new(x, otf):

    vorticity = (g/f0) * optA(x, otf)

    q = vorticity - (g / f0) * (1/LR)**2*x 

    return grad(q)



def B_star_new(Y, otf):
    temp = grad_T(Y)
    return ((g / f0) * optAT(temp, otf) - (g / f0) * (1/LR)**2 * temp)


def adjoint_test(B, B_star, otfA, rx=600, cx=600):
    u = np.random.randn(rx, cx)
    v = np.random.rand(2, rx, cx)
    s1 = np.sum(B(u, otfA) * v)
    s2 = np.sum(B_star(v, otfA) * u)
    return np.isclose(s1 - s2, 0)


def operator_norm(A, A_star, otf, rx=600, cx=600):
    b0 = np.random.random([rx, cx])
    n = 25
    beta = 0
    for i in range(n):
        bk = A_star(A(b0, otf), otf)
        lambda0 = np.linalg.norm(bk, 2) / np.linalg.norm(b0, 2)
        b0 = bk
    return lambda0
#
# otfA = psf2otf(kernel, SSH_obs0.shape)



# otfA = psf2otf(kernel, SSH_obs.shape)


# norm_B = operator_norm(B_new, B_star_new, otfA, rx=601,cx=601)   


# SSH_obs0, mask0 = mask_and_filling(SSH_obs)

# ngrid_lat = SSH_obs0.shape[0]
# # ngrid_lat = 600

# ngrid_lon = SSH_obs0.shape[1]
# # ngrid_lon = 600




# vorticity = (g / f0) * optA(SSH_obs0, otfA)


# def y_q(ngrid_lon, ngrid_lat):

#     a = np.ones((1, ngrid_lon))
#     at = np.transpose(a)
#     b = np.arange(0, ngrid_lat)
#     bt = np.transpose(b)
#     y = np.transpose(np.kron(at, b))

#     return y 




# a = np.ones((1, ngrid_lon))
# at = np.transpose(a)
# b = np.arange(0, ngrid_lat)
# bt = np.transpose(b)


# y = np.transpose(np.kron(at, b))

if __name__=='__main__':

    SSH_obs = np.load('../datasets/np_array/2012-10-13_image_data.npy')

    kernel = np.array([[0, (1/dy)**2, 0], [1/dx**2, -(2/dx**2)  - (2/dy**2), 1/dy**2], [0, 1/dy**2, 0]])
    otfA = psf2otf(kernel, SSH_obs.shape)

    
    norm_B = operator_norm(B_new, B_star_new, otfA, rx=601,cx=601)   


    SSH_obs0, mask0 = mask_and_filling(SSH_obs)

    ngrid_lat = SSH_obs0.shape[0]
    # ngrid_lat = 600

    ngrid_lon = SSH_obs0.shape[1]
    # ngrid_lon = 600

   


    vorticity = (g / f0) * optA(SSH_obs0, otfA)

    a = np.ones((1, ngrid_lon))
    at = np.transpose(a)
    b = np.arange(0, ngrid_lat)
    bt = np.transpose(b)


    y = np.transpose(np.kron(at, b))

    print(y.shape)
    #u = np.random.randn(601, 601)
    #v = np.random.rand(2, 601, 601)
    #s1 = np.sum(optA(u, otfA) * v)
    #s2 = np.sum(optAT(v, otfA) * u)

    #u = np.random.randn(601, 601)
    #v = np.random.rand(2, 601, 601)
    #s1 = np.sum(B_new(u, otfA) * v)
    #s2 = np.sum(B_star_new(v, otfA) * u)
    #print(s1)
    #print(s2)


# plt.imshow(beta*dy*y, origin='lower')
# #
# q = vorticity - (g / f0) * (1/LR)**2*SSH_obs0 + beta*dy*y
#
# temp1=beta*dy*y
# print(grad(temp1).shape)
# plt.figure(figsize=(20,20))
# plt.subplot(2,2,1)
# plt.imshow(vorticity, cmap='gist_stern', interpolation='none', origin='lower')
# plt.subplot(2,2,2)
# plt.imshow((g / f0) * (1/LR)**2*SSH_obs0, cmap='gist_stern', interpolation='none', origin='lower')
# plt.subplot(2,2,3)
# plt.imshow(beta*dy*y, origin='lower')
# plt.colorbar()

# plt.show()