import numpy as np 
import xarray as xr
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  # Import the formatter


dates = ['2012-10-13','2012-10-24','2012-11-04']

date = dates[0]

data_set = ['Degraded', 'Satellite']

method_names = ["CP_TV", 'CP_TV_V', 'Hybrid']

method = method_names[2]


SSH_CV_TV_la01 = []

SSH_CV_TV_la1 = []

SSH_CV_TV_la10 = []

SSH_CV_TV_la100 = []

SSH_results_all = []



BASE_DIR = Path('../')

RESULTS_DIR = BASE_DIR / 'results'

SAT_RESULTS_DIR = RESULTS_DIR / data_set[1]

METHOD_SAT_RESULTS_DIR = SAT_RESULTS_DIR / method

DATE_M_S_RESULTS_DIR = METHOD_SAT_RESULTS_DIR/ dates[0]



SSH_results = []
sigma = 100
chi_list = [0.1, 1, 10, 100]
lambds_list = [0.1, 1.0, 10]
it = 80_000
for chi in chi_list:
    lambd = lambds_list[0]
    try:
        file_name = f'../results/Satellite/CV_TV_V/{date}/{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1/{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1.mat'
        SSH_result_sigma0=loadmat(f'{file_name}',simplify_cells=True)
        SSH_result_sigma0['params']['chi'] = chi
        SSH_result_sigma0['name'] = method_names[2]
        SSH_CV_TV_la01.append(SSH_result_sigma0)
    except FileNotFoundError:
        print(f'No Experiment Found {chi}')
SSH_results_all.append(SSH_CV_TV_la01)


for chi in chi_list:
    lambd = lambds_list[1]
    try:
        file_name = f'../results/Satellite/CV_TV_V/{date}/{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1/{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1.mat'
        SSH_result_sigma2=loadmat(f'{file_name}',simplify_cells=True)
        SSH_result_sigma2['params']['chi'] = chi
        SSH_result_sigma2['name'] = method_names[2]
        SSH_CV_TV_la1.append(SSH_result_sigma2)
    except FileNotFoundError:
        print(f'No Experiment Found {chi}')
SSH_results_all.append(SSH_CV_TV_la1)


for chi in chi_list:
    lambd = lambds_list[2]
    try:
        file_name = f'../results/Satellite/CV_TV_V/{date}/{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1/{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1.mat'
        SSH_result_sigma1=loadmat(f'{file_name}',simplify_cells=True)
        SSH_result_sigma1['params']['chi'] = chi
        SSH_result_sigma1['name'] = method_names[2]
        SSH_CV_TV_la10.append(SSH_result_sigma1)
    except FileNotFoundError:
        print(f'No Experiment Found {chi}')
SSH_results_all.append(SSH_CV_TV_la10)



# for chi in chi_list:
#     lambd = lambds_list[3]
#     try:
#         file_name = f'../results/Satellite/CV_TV_V/{date}/{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1/{date}_sigma={sigma}_lambda={lambd}_chi={chi}_it={it}_rho=1.mat'
#         SSH_result_sigma3=loadmat(f'{file_name}',simplify_cells=True)
#         SSH_result_sigma3['params']['chi'] = chi
#         SSH_result_sigma3['name'] = method_names[2]
#         SSH_CV_TV_la100.append(SSH_result_sigma3)
#     except FileNotFoundError:
#         print(f'No Experiment Found {chi}')
# SSH_results_all.append(SSH_CV_TV_la100)




n_methods = 3

num_subplots = np.max(np.array((len(SSH_CV_TV_la01), len(SSH_CV_TV_la1))))+1

fig, axs = plt.subplots(len(chi_list)-1, num_subplots, figsize=(5*num_subplots, 15))


date_np = np.datetime64(date)
delta_t = np.timedelta64(5,'D')


SSH_mat = loadmat('../datasets/data_obs/2012-10-13_SSH.mat',simplify_cells=True)

SSH_ref = xr.open_mfdataset('../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
SSH_ref = SSH_ref[0]


SSH_obs = SSH_mat['SSH_obs']

axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43])
axs[0,0].set_title('Satellites', fontsize=25)
axs[0,0].set_xlabel('Latitude',fontsize=25)
axs[0,0].set_ylabel('Longitude',fontsize=25)
# axs[0,0].get_xticklabels('major')
axs[0,0].get_yaxis().set_major_formatter(FormatStrFormatter('%g°N'))
axs[0,0].get_xaxis().set_major_formatter(FormatStrFormatter('%g°W'))
axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43])
axs[1,0].set_title('Reference', fontsize=25)
axs[1,0].set_xlabel('Latitude',fontsize=25)
axs[1,0].set_ylabel('Longitude',fontsize=25)
axs[1,0].get_yaxis().set_major_formatter(FormatStrFormatter('%g°N'))
axs[1,0].get_xaxis().set_major_formatter(FormatStrFormatter('%g°W'))
axs[2,0].axis('off')
# axs[3,0].axis('off')


for j in range(3):
    for i in range(1,len(SSH_results_all[j])+1):
        
        axs[j, i].imshow(SSH_results_all[j][i-1]['SSH_est'], cmap='gist_stern', origin='lower', interpolation='None')
        axs[j, i].set_title('Hybrid \n χ={chi}, λ={lamd}'.format(
            name=SSH_results_all[j][i-1]['name'],
            chi=SSH_results_all[j][i-1]['params']['chi'],
            # sigma=SSH_results_all[j][i-1]['params']['sigma'],
            lamd=SSH_results_all[j][i-1]['params']['lambd'], 
            iter=SSH_results_all[j][i-1]['params']['max_iter']), fontsize=22)
        axs[j, i].axis('off')

fig.tight_layout()
plt.show()


