import numpy as np 
import xarray as xr
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.best_results import * 



dates = ['2012-10-13','2012-10-24','2012-11-04']

date = dates[0]

data_set = ['Degraded', 'Satellite']

method_names = ["CP_TV", 'CP_TV_V', 'CV_TV_V']

method = method_names[1]

SSH_CP_TV = []

SSH_CP_TV_V = []

SSH_CV_TV_V1 = []
SSH_CV_TV_V2 = []


BASE_DIR = Path('../')

RESULTS_DIR = BASE_DIR / 'results'

SAT_RESULTS_DIR = RESULTS_DIR / data_set[1]

METHOD_SAT_RESULTS_DIR = SAT_RESULTS_DIR / method_names[0]

DATE_M_S_RESULTS_DIR = METHOD_SAT_RESULTS_DIR/ dates[0]

# for lambd in lambds1:
#     file_name = f'2012-10-13_sigma=0.9_lambda={lambd}_it=15000_rho=1'
#     SSH_result = loadmat(f'../results/Satellite/CP_TV/2012-10-13/{file_name}/{file_name}.mat', simplify_cells=True)
#     SSH_CP_TV.append(SSH_result)
    
# SSH_results_all.append(SSH_CP_TV)
# for lambd in lambds2:
#     try:
#         folder_name= f'2012-12-13_tau=0.9_sigma_0.9_lambda={lambd}_maxiter=5005'
#         SSH_result=loadmat(f'../Old_results/new_reg/2012-12-13_SSH/{folder_name}/test.mat',simplify_cells=True)
#         SSH_result['name'] = method_names[1]
#         print(SSH_result['param_B'].keys())
#         temp = SSH_result['param_B']
#         SSH_result['params'] = temp
#         temp2 = SSH_result['params']['lambda']
#         SSH_result['params']['lambd'] = temp2

#         SSH_CP_TV_V.append(SSH_result)
#     except FileNotFoundError:
#         print('No Experiment Found')

# SSH_results_all.append(SSH_CP_TV_V)
SSH_results = []
sigmas = [100]
lambds1 = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000]

SSH_results_all = []
it  = 50000
for lambd in lambds1:
    try:
        file_name = f'../results/Satellite/CV_TV_V/Old_2023-09-07/2012-10-13/2012-10-13_sigma=100_lambda={lambd}_chi=1_it={it}_rho=1/2012-10-13_sigma=100_lambda={lambd}_chi=1_it={it}_rho=1.mat'
        SSH_result_sigma1=loadmat(f'{file_name}',simplify_cells=True)
        SSH_result_sigma1['name'] = method_names[2]
        SSH_results.append(SSH_result_sigma1)
    except FileNotFoundError:
        print('No Experiment Found')



# for lambd in lambds1:
#     try:
#         file_name= f'sigma=0.6_lambda={lambd}_it=50000_rho=1.mat'
#         SSH_result_sigma2=loadmat(f'../Conda_Vu_TV_Vorticity/results/Results/{file_name}',simplify_cells=True)
#         SSH_result_sigma2['name'] = method_names[2]
#         SSH_CV_TV_V2.append(SSH_result_sigma2)
#     except FileNotFoundError:
#         print('No Experiment Found')

# SSH_results_all.append(SSH_CV_TV_V2)

# print(SSH_results[0].keys())

num_subplots = len(SSH_results) + 1
# Create the figure

fig, axs = plt.subplots(3, num_subplots, figsize=(5*num_subplots, 15))


SSH_mat = loadmat(f'../datasets/data_obs/{date}_SSH.mat',simplify_cells=True)


SSH_obs = SSH_mat['SSH_obs']

date_np = np.datetime64(date)
delta_t = np.timedelta64(5,'D')


SSH_ref = xr.open_mfdataset('../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
SSH_ref = SSH_ref[0]


axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None')
axs[0,0].set_title(f'Satellite')
axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
axs[1,0].set_title('Reference')
axs[2,0].axis('off')

j=0
for i in range(1,num_subplots):
    
    axs[j, i].imshow(SSH_results[i-1]['SSH_est'], cmap='gist_stern', origin='lower', interpolation='None')
    axs[j, i].set_title('SSH_{name}_est, sigma={sigma},\nlambda={lamd}, it={iter}'.format(
        name=SSH_results[i-1]['name'],
        sigma=SSH_results[i-1]['params']['sigma'],
        lamd=SSH_results[i-1]['params']['lambd'], 
        iter=SSH_results[i-1]['params']['max_iter']), fontsize=18)
    axs[j, i].axis('off')
    axs[1,i].set_title('Performance of {name}'.format(name=SSH_results[i-1]['name']))
    axs[1,i].grid('on')
    l1, = axs[1,i].plot(SSH_results[i-1]['psnr_list'], color='green') 
    axs[1,i].set_ylabel('PSNR', color='green')
    ax2 = axs[1,i].twinx()
    ax2.set_ylabel('RMSE', color='red')
    l2, = ax2.plot(SSH_results[i-1]['rmseb'], color='red')
    axs[1,i].legend([l1, l2], ['PSNR',"RMSE_Based"])

    axs[2, i].plot(SSH_results[i-1]['crit'] )
    axs[2,i].grid('on')
    axs[2, i].set_title('Crit {name}'.format(name=SSH_results[i-1]['name']))

    fig.tight_layout()
plt.show()



