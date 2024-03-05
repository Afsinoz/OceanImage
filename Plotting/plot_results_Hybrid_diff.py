import numpy as np 
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import sys 
sys.path.append('..')
from utils.toolbox import *
from utils.eval import * 


dates = ['2012-10-13','2012-10-24','2012-11-04']

date = dates[0]


data_set = ['Degraded', 'Satellite']

method_names = ["CP_TV", 'CP_TV_V', 'CV_TV_V']

SSH_CP_TV = []

SSH_CP_TV_V = []

SSH_CV_TV_V1 = []
SSH_CV_TV_V2 = []
SSH_CV_TV_V3 = []

SSH_Hybrid = dict()

BASE_DIR = Path('../')

RESULTS_DIR = BASE_DIR / 'results'

SAT_RESULTS_DIR = RESULTS_DIR / data_set[1]

METHOD_SAT_RESULTS_DIR = SAT_RESULTS_DIR / method_names[0]

DATE_M_S_RESULTS_DIR = METHOD_SAT_RESULTS_DIR/ dates[0]




date = dates[0]
chis = [1, 10 ,100]
lambdas = [0.001, 0.1, 1, 10, 100, 1000, 10000]
it = 50_000

SSH_results_all = []
# for lambd in lambdas:
#     for chi in chis:
#         try:
#             file_name = f'{date}_sigma=100_lambda={lambd}_chi={chi}_it={it}_rho=1'
#             SSH_result_sigma1=loadmat(f'../results/Satellite/CV_TV_V/Old_2023-09-07/2012-10-13/{file_name}/{file_name}.mat',simplify_cells=True)
#             SSH_result_sigma1['name'] = method_names[2]
#             SSH_Hybrid[chi] = SSH_result_sigma1 
#         except FileNotFoundError:
#             print('No Experiment Found')


# print(len(SSH_Hybrid[1])

for lambd in lambdas:
        try:
            file_name = f'{date}_sigma=100_lambda={lambd}_chi=1_it={it}_rho=1'
            SSH_result_sigma1=loadmat(f'../results/Satellite/CV_TV_V/Old_2023-09-07/2012-10-13/{file_name}/{file_name}.mat',simplify_cells=True)
            SSH_result_sigma1['name'] = method_names[2]
            SSH_CV_TV_V1.append(SSH_result_sigma1)
        except FileNotFoundError:
            print('No Experiment Found')

SSH_results_all.append(SSH_CV_TV_V1)

for lambd in lambdas:
        try:
            file_name = f'{date}_sigma=100_lambda={lambd}_chi=10_it={it}_rho=1'
            SSH_result_sigma2=loadmat(f'../results/Satellite/CV_TV_V/Old_2023-09-07/2012-10-13/{file_name}/{file_name}.mat',simplify_cells=True)
            SSH_result_sigma2['name'] = method_names[2]
            SSH_CV_TV_V2.append(SSH_result_sigma2)
        except FileNotFoundError:
            print('No Experiment Found')

SSH_results_all.append(SSH_CV_TV_V2)


for lambd in lambdas:
        try:
            file_name = f'{date}_sigma=100_lambda={lambd}_chi=100_it={it}_rho=1'
            SSH_result_sigma3=loadmat(f'../results/Satellite/CV_TV_V/Old_2023-09-07/2012-10-13/{file_name}/{file_name}.mat',simplify_cells=True)
            SSH_result_sigma3['name'] = method_names[2]
            SSH_CV_TV_V3.append(SSH_result_sigma3)
        except FileNotFoundError:
            print('No Experiment Found')

SSH_results_all.append(SSH_CV_TV_V3)


date_np = np.datetime64(date)
delta_t = np.timedelta64(5,'D')

n_methods = 3

num_subplots = np.max(np.array((len(SSH_CV_TV_V1),len(SSH_CV_TV_V2), len(SSH_CV_TV_V3))))+1

fig, axs = plt.subplots(3, num_subplots, figsize=(5*num_subplots, 15))


SSH_ref = xr.open_mfdataset('../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
SSH_ref = SSH_ref[0]

SSH_obs = SSH_CV_TV_V1[0]['SSH_obs']



axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None')
axs[0,0].set_title(f'Satellite')
axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
axs[1,0].set_title('Reference')
axs[2,0].axis('off')



for j in range(3):
    for i in range(1,len(SSH_results_all[j])+1):
        
        axs[j, i].imshow(SSH_results_all[j][i-1]['SSH_est'], cmap='gist_stern', origin='lower', interpolation='None')
        axs[j, i].set_title('SSH_{name}_est, \nlambda={lamd}, PSNR={PSNR}'.format(
            name=SSH_results_all[j][i-1]['name'],
            PSNR=round(PSNR(SSH_ref, SSH_results_all[j][i-1]['SSH_est'][:600,:600]),2),
            lamd=SSH_results_all[j][i-1]['params']['lambd'], 
            iter=SSH_results_all[j][i-1]['params']['max_iter']), fontsize=18)
        axs[j, i].axis('off')

fig.tight_layout()
plt.show()
