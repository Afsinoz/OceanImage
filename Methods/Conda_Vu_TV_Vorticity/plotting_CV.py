import numpy as np 
import matplotlib.pyplot as plt 
import os
from scipy.io import savemat, loadmat 
# import CV 
import sys 
sys.path.append('..')
from utils.toolbox import *

SSH_ref = np.load('../../datasets/data_ref/np_array_ref/2012-10-13_image_ref.npy')

SSH_obs = np.load('../../datasets/np_array/2012-10-13_image_data.npy')


SSH_obs0, mask0 = mask_and_filling(SSH_obs)


SSH_est_1 = loadmat('../..//Conda_Vu/results/Results/sigma=0.3_lambda=0.01_it=1000', simplify_cells=True)
SSH_est_2 = loadmat('../../Conda_Vu/results/Results/sigma=0.3_lambda=0.001_it=1000', simplify_cells=True)
SSH_est_3 = loadmat('../../Conda_Vu/results/Results/sigma=0.9_lambda=1_it=5000', simplify_cells=True)
SSH_est_4 = loadmat('../../Conda_Vu/results/Results/sigma=0.3_lambda=0.1_it=10000', simplify_cells=True)
SSH_est_5 = loadmat('../../Conda_Vu/results/Results/sigma=0.6_lambda=0.1_it=10000', simplify_cells=True)
SSH_est_6 = loadmat('../../Conda_Vu/results/Results/sigma=0.9_lambda=0.1_it=10000', simplify_cells=True)



SSH_results = [SSH_est_1,SSH_est_2,SSH_est_3,SSH_est_4,SSH_est_5,SSH_est_6]

num_subplots = len(SSH_results) + 1
# Create the figure
fig, axs = plt.subplots(3, num_subplots, figsize=(5*num_subplots, 15))



axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None')
axs[0,0].set_title(f'SSH Observations')
axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
axs[1,0].set_title('Reference')
axs[2,0].axis('off')



for i in range(1,num_subplots):
    axs[0, i].imshow(SSH_results[i-1]['SSH_est'], cmap='gist_stern', origin='lower', interpolation='None')
    axs[0, i].set_title('SSH_est_CV, sigma={sigma},\nlambda={lamd}, it={iter}'.format(
        sigma=SSH_results[i-1]['params']['sigma'],
        lamd=SSH_results[i-1]['params']['lambd'], 
        iter=SSH_results[i-1]['params']['max_iter']), fontsize=18)
    axs[0, i].axis('off')

    axs[1,i].set_title('Performance ')
    axs[1,i].grid('on')
    l1, = axs[1,i].plot(SSH_results[i-1]['psnr_list'], color='green') 
    axs[1,i].set_ylabel('PSNR', color='green')
    ax2 = axs[1,i].twinx()
    ax2.set_ylabel('RMSE', color='red')
    l2, = ax2.plot(SSH_results[i-1]['rmseb'], color='red')
    axs[1,i].legend([l1, l2], ['PSNR',"RMSE_Based"])

    axs[2, i].plot(SSH_results[i-1]['crit'] )
    axs[2,i].grid('on')
    axs[2, i].set_title('Crit')
    # axs[1, i].set_ylim([0, 1])

# Adjust the spacing between subplots
fig.tight_layout()

filename = f'results/all_figures/result_1.png'

# Check if the file already exists
if os.path.isfile(filename):
    # If the file exists, create a new filename with a number suffix
    idx = 1
    while os.path.isfile(f"{filename[:-4]}_{idx}.png"):
        idx += 1
    filename = f"{filename[:-4]}_{idx}.png"


plt.savefig(filename)
# Show the figure
plt.show()
