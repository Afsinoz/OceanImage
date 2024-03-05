import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt 

r'''Main functions of the algorithms. 
Functions: 
- grad : Discrete Gradient 
- grad_T : Discrete Divergence 
- prox_f : Proximal operator of L2 function 
- prox_g : Proximal operator of absolute value function
- prox_g_star : Proximal operator of Absolute value function.
- prox_g_star_translated: Shifted Proximal operator of convex conjugate of absolute value function. 
- Crit_TV : Criterion of the total variation regularization minimization problem. 
- Crit_B: Criterion of the total variation regularization minimization problem when the inside function is Potential Vorticity. 
- Crit_Conda_Vu: 'Criterion of the total variation regularization minimization problem when there is two regularization terms, TV and TV with potential vorticity. 
- degradation: To produce randomly masked images to check the performance under the algorithms.
'''


def mask_and_filling(image, filling=0):
    r'''This function fills the sparse points in the pictures, with a desired value 
    arg: image(np.array) 
    filling: float 
    '''
    image1 = image.copy()

    image1[np.isnan(image1)] = filling
    mask = np.where(image1 == filling, 0, 1)
    return image1, mask

def nan_percentage(image):
    nan_count = np.sum(np.isnan(image))
    total_elemets = image.size

    return (nan_count/total_elemets) *100

def grad(X):
    r''' Discrete Gradient of an image
    Args:
        X (numpy.narray): 2-D image 
    Return: 
        G (numpy.narray): 2-Dx2-D gradient of the image
    '''
    G = np.zeros_like([X, X])
    # G[0, :, :-1, :] = X[:, 1:, :] - X[:, :-1, :] # Horizontal Direction
    # G[1, :-1, :, :] = X[1:, :, :] - X[:-1, :, :] # Vertical Direction
    G[0, :, :-1] = X[:, 1:] - X[:, :-1]  # Horizontal Direction
    G[1, :-1, :] = X[1:, :] - X[:-1, :]  # Vertical Direction
    return G


def grad_T(Y):
    r'''Discrete divergence operator
    Args: 
        Y (numpy.narray): 2-Dx2-D gradient of the image.
    Returns:
        -G_T (numpy.narray): 2-D image, divergence of the gradient.
    '''
    G_T = np.zeros_like(Y[0])
    G_T[:, :-1] += Y[0, :, :-1]  # Corresponds to c[0]
    G_T[:-1, :] += Y[1, :-1, :]  # Corresponds to c[1]
    G_T[:, 1:] -= Y[0, :, :-1]  # Corresponds to c[0]
    G_T[1:, :] -= Y[1, :-1, :]  # Corresponds to c[1]
    # G_T[:, :-1,:] += Y[0, :, :-1,:] # Corresponds to c[0]
    # G_T[:-1, :,:] += Y[1, :-1, :,:] # Corresponds to c[1]
    # G_T[:, 1:,:] -= Y[0, :, :-1,:] # Corresponds to c[0]
    # G_T[1:, :,:] -= Y[1, :-1, :,:] # Corresponds to c[1]
    return -G_T


def prox_f(z, y, lambd, mask):
    '''
    Proximal operator for the function f(x) = 1/2||Ax-z||^2
    Variables:
    mask:           masked matrix
    z:              the values that we know, masked picture
    y:              the variable of the proximal operator
    lambd:          the hyperparameter
    '''
    return (lambd * mask * z + y) / (mask * lambd + np.ones(z.shape))


def prox_g(x, tau):
    r''' Proximal operator of Absolute value function. Returns, SoftThresholding function. 
    '''
    return (x < -tau) * (x + tau) + (x > tau) * (x - tau)


def prox_g_star(x, tau, lambd):
    r''' Proximal operator of convex conjugate of absolute value function. 
    '''
    # moreau identity
    return x - tau * prox_g(x / tau, lambd * 1 / tau)

def prox_g_star_translated(x, tau, lambd,c):
    r'''Shifted Proximal operator of convex conjugate of absolute value function by c. 
    '''
    # moreau identity
    return x - tau * (c + prox_g(x / tau - c , lambd * 1 / tau))

def crit_TV(X, y, lambd, mask):
    r'''Criterion of the total variation regularization minimization problem. 
    '''
    return np.linalg.norm(mask * X - y) ** 2 + lambd * np.sum(np.abs(grad(X)))

def crit_B(X, y, lambd, mask,B, otfA, c):
    r'''Criterion of the total variation regularization minimization problem when the inside function is Potential Vorticity. 
    '''
    return np.linalg.norm(mask * X - y) ** 2 + lambd * np.sum(np.abs(B(X,otfA)+c))

def crit_Conda_Vu(X, y, lambd,chi, mask,B, otfA, c):
    r'''Criterion of the total variation regularization minimization problem when there is two regularization terms, TV and TV with potential vorticity. 
    '''
    return np.linalg.norm(mask * X - y) ** 2 +chi* np.linalg.norm(np.abs(B(X,otfA)+c)) +lambd * np.sum(np.abs(grad(X)))


def degradation(SSH_ref, th=0.):
    r'''To produce randomly masked images to check the performance under the algorithms.
    '''
    (rx,cx) = SSH_ref.shape
    rand  = np.random.normal(0,1,(rx,cx))
    ind   = np.where(rand>th)

    mask  = np.zeros((rx,cx))
    mask[ind] = 1
    z         = np.zeros((rx,cx))
    z[ind]    = SSH_ref[ind]
    masked_values = ind[0].shape[0]
    total_values = rx*cx 
    percantage = int((1-masked_values/total_values)*100)

    return z, mask, percantage 

# def degradation_percentage(SSH_ref, percentange):
#     (rx,cx) = SSH_ref.shape
#     # rand = np.random.normal(0,1,(rx,cx))
#     SSH_ref_flat = SSH_ref.flatten()

#     array_flat = SSH_ref.flatten()
#     num_samples = array_flat.size

#     selected_indexes = np.where(np.random.rand(num_samples) < percentange / 100)

# # Step 4: Finally, select the elements at those indexes from the original array.
#     masked_value = np.nan # Replace with any value you want to use for masking (e.g., np.nan, -999, etc.)
#     array_flat[selected_indexes] = masked_value

# # Reshape the modified random_array_flat back to the original shape (600, 6000)
#     array_masked = array_flat.reshape(SSH_ref.shape)

#     return array_masked
    


 



def main():
    SSH_obs_mat = loadmat('../datasets/data_obs/2012-11-04_SSH.mat', simplify_cells=True)
    SSH_obs = SSH_obs_mat['SSH_obs']
    SSH_ref = SSH_obs_mat['SSH_ref']

    Nan_per = nan_percentage(SSH_obs)
    # nan_count = np.sum(np.isnan(SSH_obs))
    # print(nan_count)

    # print(Nan_per)
    # print(SSH_obs.size)
    # threshold = -0.1
    # _,_,percentage = degradation(SSH_ref,th=threshold)
    # print(percentage)



    return None



if __name__=='__main__':
    main()