import numpy as np 
import sys 
import xarray as xr

sys.path.append('..')
from utils.toolbox import *  
from Methods.CP_TV_Vorticity.TV_Vorticity_last import *

r'''This script is dedicated to display the potential vorticity field from a sea surface height. 
'''


SSH_obs = np.load('../../datasets/np_array/2012-10-13_image_data.npy')

SSH_obs0, mask  = mask_and_filling(SSH_obs) 


# SSH_est = loadmat('../../results/Satellite/CV_TV_V/2012-10-13/2012-10-13_sigma=100_lambda=1000_chi=10_it=50000_rho=1/2012-10-13_sigma=100_lambda=1000_chi=10_it=50000_rho=1.mat',simplify_cells=True)
dates = ['2012-10-13','2012-10-24','2012-11-04']

date = dates[0]
sigma = 0.9
chi = 100
lambd = 0.1
it = 100_000
# file_name = f'{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1' #This is for CV_TV
file_name = f'{date}_sigma={sigma}_lambda={lambd}_it={it}_rho=Nan'

# SSH_est = loadmat(f'../../results/Satellite/CV_TV_V/{date}/{file_name}/{file_name}.mat',simplify_cells=True)
SSH_est = loadmat(f'../../results/Satellite/CP_TV_V/{date}/{file_name}/{file_name}.mat',simplify_cells=True)
# PnP = loadmat('../../results/PnP/2012-10-13_SSH_truemask_PnP.mat', simplify_cells=True)

date_np = np.datetime64(date)
delta_t = np.timedelta64(5,'D')
SSH_ref = xr.open_mfdataset('../../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
SSH_ref = SSH_ref[0]



kernel = np.array([[0, (1/dy)**2, 0], [1/dx**2, -(2/dx**2)  - (2/dy**2), 1/dy**2], [0, 1/dy**2, 0]])
otfA = psf2otf(kernel, SSH_ref.shape)

otfB = psf2otf(kernel, (600,600))


ngrid_lat = 600

ngrid_lon = 600

dy = 111000/60

beta = 2 * (2 * np.pi / (24 * 3600)) * np.cos(np.radians(theta))/6400000

a = np.ones((1, ngrid_lon))
at = np.transpose(a)
b = np.arange(0, ngrid_lat)
bt = np.transpose(b)


y = np.transpose(np.kron(at, b))

def q(x, otf):

    vorticity = (g/f0) * optA(x, otf) 


    q = vorticity - (g / f0) * (1/LR)**2*x + beta*dy*y


    return q

# B_SSH = B_new(SSH_obs0, otfA)[0,:,:]
# Q_SSH = q(SSH_ref,otfA)

# Q_SSH_est = q(SSH_est['SSH_est'][:600,:600], otfB) 


# Q_N_SSH = q(SSH_obs0,otfA)
def mask_image_with_nan(image, mask):
    # Create a copy of the image
    masked_image = np.copy(image)

    # Set 'nan' values in the masked_image where the mask is 0
    masked_image[mask == 0] = np.nan

    return masked_image

# B_SSH = mask_image_with_nan(B_SSH,mask)

# Q_N_SSH = mask_image_with_nan(Q_SSH,mask)
# plt.imshow(Q_SSH_est[1:-1,1:-1], cmap='gist_stern',vmin=-2e-4,vmax=2e-4,interpolation='None',origin='lower') 
# plt.imshow(Q_N_SSH[1:-1,1:-1], cmap='gist_stern',interpolation='None',origin='lower') 

# plt.colorbar()
# plt.savefig('Q_SSH_est.png')
# plt.show()

OI = np.load('../../Data_challange/2012-10-13_OI.npy')

def plot_pv(SSH_ref, SSH_est):


    Q_SSH = q(SSH_ref,otfA)
    # np.save(f'../../results/PV_Field/{date}_reference_PV_field',Q_SSH)
    # Q_SSH_est = q(SSH_est['SSH_est'][:600,:600], otfB) 
    # temp = SSH_est['SSH_est'][:600,:600]
    temp = OI
    Q_SSH_est = q(temp, otfB) 
    method_name = 'OI'

    np.save(f'../../results/PV_Field/{method_name}_PV_field',Q_SSH_est )

    
    fig, axs = plt.subplots(2,2,figsize=(10,10))

    # axs[0,0].imshow(SSH_ref,cmap='gist_stern',vmin=-2e-4,vmax=2e-4,interpolation='None',origin='lower') 

    axs[0,0].imshow(SSH_ref,cmap='gist_stern',interpolation='None',origin='lower') 
    axs[0,0].set_title('Reference')
    axs[1,0].imshow(Q_SSH[1:-1,1:-1], cmap='gist_stern',vmin=-2e-4,vmax=2e-4,interpolation='None',origin='lower') 
    axs[1,0].set_title('Potential Vorticity of Reference')
    # axs[0,1].imshow(SSH_ref,cmap='gist_stern',vmin=-2e-4,vmax=2e-4,interpolation='None',origin='lower') 

    axs[0,1].imshow(temp,cmap='gist_stern',interpolation='None',origin='lower') 
    axs[0,1].set_title('SSH Estimation')
    
    axs[1,1].imshow(Q_SSH_est[1:-1,1:-1], cmap='gist_stern',vmin=-2e-4,vmax=2e-4,interpolation='None',origin='lower') 
    axs[1,1].set_title('Potential Vorticity of Estimation')

    # plt.colorbar()
    # plt.savefig('Q_SSH_est.png')
    plt.show()

plot_pv(SSH_ref, SSH_est)
    