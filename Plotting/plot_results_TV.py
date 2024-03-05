import numpy as np 
import xarray as xr
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  # Import the formatter



dates = ['2012-10-13','2012-10-24','2012-11-04']

date = dates[0]

data_sets = ['Degraded', 'Satellite']

data_set = data_sets[1]

method_names = ["CP_TV", 'CP_TV_V', 'CV_TV_V']

method = method_names[0]

SSH_CP_TV = []

SSH_CP_TV_V = []

SSH_CV_TV_V1 = []
SSH_CV_TV_V2 = []

BASE_DIR = Path('../')

RESULTS_DIR = BASE_DIR / 'results'

SAT_RESULTS_DIR = RESULTS_DIR / data_set[0]

METHOD_SAT_RESULTS_DIR = SAT_RESULTS_DIR / method

DATE_M_S_RESULTS_DIR = METHOD_SAT_RESULTS_DIR/ dates[0]



SSH_results = []
sigma = 1
chi = 10
# lambds1 = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10_000]
lambds1 =  [0.1, 1, 10]

SSH_results_all = []
it  = 15000
for lambd in lambds1:
    try:
        file_name = f'../results/{data_set}/{method}/{date}/{date}_sigma={sigma}_lambda={lambd}_it={it}_rho=Nan/{date}_sigma={sigma}_lambda={lambd}_it={it}_rho=Nan.mat'
        SSH_result_sigma1=loadmat(f'{file_name}',simplify_cells=True)
        SSH_result_sigma1['name'] = method
        SSH_results.append(SSH_result_sigma1)
    except FileNotFoundError:
        print('No Experiment Found')
num_subplots = len(SSH_results) + 1
# Create the figure


fig, axs = plt.subplots(3, num_subplots, figsize=(5*num_subplots, 15))


SSH_mat = loadmat(f'../datasets/data_obs/{date}_SSH.mat',simplify_cells=True)
# SSH_mat = loadmat(f'../datasets/ref_degraded/84_masked/2012-10-14_SSH_ref_84_masked.mat',simplify_cells=True)
# 
# SSH_obs = SSH_mat['SSH_masked']
SSH_obs = SSH_mat['SSH_obs']
# SSH_ref = SSH_mat['SSH_ref']

date_np = np.datetime64(date)
delta_t = np.timedelta64(5,'D')


SSH_ref = xr.open_mfdataset('../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
SSH_ref = SSH_ref[0]


axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43])
axs[0,0].set_title('Satellites', fontsize=25)
axs[0,0].set_xlabel('Latitude',fontsize=25)
axs[0,0].set_ylabel('Longitude',fontsize=25)
axs[0,0].get_yaxis().set_major_formatter(FormatStrFormatter('%g°N'))
axs[0,0].get_xaxis().set_major_formatter(FormatStrFormatter('%g°W'))
axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None',extent=[-65, -55, 33, 43])
axs[1,0].set_title('Reference', fontsize=25)
axs[1,0].set_xlabel('Latitude',fontsize=25)
axs[1,0].set_ylabel('Longitude',fontsize=25)
axs[1,0].get_yaxis().set_major_formatter(FormatStrFormatter('%g°N'))
axs[1,0].get_xaxis().set_major_formatter(FormatStrFormatter('%g°W'))
axs[2,0].axis('off')


name1 = 'TV'
j=0
for i in range(1,num_subplots):
    
    axs[j, i].imshow(SSH_results[i-1]['SSH_est'], cmap='gist_stern', origin='lower', interpolation='None')
    axs[j, i].set_title('TV,\nλ={lamd}'.format(
        name=SSH_results[i-1]['name'],
        sigma=SSH_results[i-1]['params']['sigma'],
        lamd=SSH_results[i-1]['params']['lambd'], 
        iter=SSH_results[i-1]['params']['max_iter']), fontsize=22)
    axs[j, i].axis('off')
    axs[1,i].set_title('Performance'.format(name=SSH_results[i-1]['name']),fontsize=22)
    axs[1,i].grid('on')
    l1, = axs[1,i].plot(SSH_results[i-1]['psnr_list'], color='green') 
    axs[1,i].set_ylabel('PSNR', color='green',fontsize=18)
    ax2 = axs[1,i].twinx()
    ax2.set_ylabel('RMSE', color='red',fontsize=18)
    l2, = ax2.plot(SSH_results[i-1]['rmseb'], color='red')
    axs[1,i].legend([l1, l2], ['PSNR',"RMSE_Based"],prop = { "size": 20 })

    axs[2, i].plot(SSH_results[i-1]['crit'])
    axs[2,i].grid('on')
    axs[2, i].set_title('Criterion',fontsize=25)

    fig.tight_layout()
plt.show()


