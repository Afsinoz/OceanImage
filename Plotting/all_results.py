import numpy as np 
import xarray as xr
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  # Import the formatter


date = '2012-10-13'
date_np = np.datetime64(date)
delta_t = np.timedelta64(5,'D')

## REF
SSH_ref = xr.open_mfdataset('../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
SSH_ref = SSH_ref[0]

SSH_pv = np.load('../results/PV_Field/2012-10-13_reference_PV_field.npy')  


## OI, Optimal Interpolation Method
OI = np.load('../Data_challange/2012-10-13_OI.npy')

OI_pv = np.load('../results/PV_Field/OI_PV_field.npy')

## TV, Total Variation Regularization
file_name_tv = f'{date}_sigma=1_lambda=1_it=15000_rho=Nan'
SSH_TV_mat = loadmat(f'../results/Satellite/CP_TV/{date}/{file_name_tv}/{file_name_tv}.mat',simplify_cells=True)
SSH_TV_est = SSH_TV_mat['SSH_est']
SSH_tv_pv = np.load('../results/PV_Field/TV_lambda=1_PV_field.npy')

## CP_TV, Putting Vorticity equation into Total variation Term

file_name_tv_pv = f'{date}_sigma=0.9_lambda=0.1_it=100000_rho=Nan'
SSH_TV_V_mat = loadmat(f'../results/Satellite/CP_TV_V/{date}/{file_name_tv_pv}/{file_name_tv_pv}.mat',simplify_cells=True)
SSH_TV_V_est = SSH_TV_V_mat['SSH_est']
SSH_tv_V_pv = np.load('../results/PV_Field/TV_PV_lambda=0.1_PV_field.npy')

##Hybrid, Hybrid method of putting total variation and vorticity equation into the equation
file_name_hybrid = f'{date}_sigma=100_lambda=1.0_chi=100_it=80000_rho=1' #This is for CV_TV
SSH_Hybrid = loadmat(f'../results/Satellite/CV_TV_V/{date}/{file_name_hybrid}/{file_name_hybrid}.mat',simplify_cells=True)
SSH_Hybrid_est = SSH_Hybrid['SSH_est']
SSH_Hybrid_pv = np.load('../results/PV_Field/Hybrid_PV_field.npy')

## PnP, Plug and Play method, it was executed by Nelly. 
PnP = loadmat('../results/PnP/2012-10-13_SSH_truemask_PnP.mat', simplify_cells=True)
PnP_SSH = PnP['xrec']

PnP_pv = np.load('../results/PV_Field/PnP_PV_field.npy')


fig, axs = plt.subplots(2,4,figsize=(20, 10))


# # # Reference
# axs[0,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
# axs[0,0].set_title('Reference')
# # axs[0,0].text(0.1, 0.5, 'Left Text', transform=axs[0,0].transAxes, ha='left', va='center')
# # plt.figtext(0.1, 0.5, 'Left Text', ha='left', va='center')
# axs[1,0].imshow(SSH_pv, cmap='gist_stern',vmin=-2e-4,vmax=2e-4,interpolation='None',origin='lower',extent=[-65, -55, 33, 43]) 
# axs[1,0].set_title('Reference Vorticity Field')
## OI

axs[0,0].imshow(OI, cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
axs[0,0].set_title('Optimal Interpolation', fontsize=20)
axs[1,0].imshow(OI_pv, cmap='gist_stern',vmin=-2e-4,vmax=2e-4, origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
axs[1,0].set_title('OI Vorticity Field', fontsize=20)
## TV
axs[0,1].imshow(SSH_TV_est[:600,:600], cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
axs[0,1].set_title('TV',fontsize=20)
axs[1,1].imshow(SSH_tv_pv, cmap='gist_stern',vmin=-2e-4,vmax=2e-4, origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
axs[1,1].set_title('TV Vorticity Field', fontsize=20)
## TV_V
axs[0,2].imshow(SSH_TV_V_est[:600,:600], cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
axs[0,2].set_title('TV _PV', fontsize=20)
axs[1,2].imshow(SSH_tv_V_pv, cmap='gist_stern',vmin=-2e-4,vmax=2e-4, origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
axs[1,2].set_title('TV_PV Vorticity Field',fontsize=20)

## Hybrid
axs[0,3].imshow(SSH_Hybrid_est[:600,:600], cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
axs[0,3].set_title('Hybrid',fontsize=20)
axs[1,3].imshow(SSH_Hybrid_pv, cmap='gist_stern',vmin=-2e-4,vmax=2e-4, origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
axs[1,3].set_title('Hybrid Vorticity Field',fontsize=20)

# ##PnP
# axs[2,2].imshow(PnP_SSH[:600,:600], cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
# axs[2,2].set_title('PnP')
# axs[3,2].imshow(PnP_pv, cmap='gist_stern',vmin=-2e-4,vmax=2e-4, origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
# axs[3,2].set_title('PnP Vorticity Field')


# plt.imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43]) 
# plt.imshow(SSH_pv, cmap='gist_stern',vmin=-2e-4,vmax=2e-4,interpolation='None',origin='lower',extent=[-65, -55, 33, 43]) 
# plt.axis('off')
plt.show()