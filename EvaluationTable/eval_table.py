import numpy as np 
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate  # Import tabulate
import sys 
import xarray as xr
sys.path.append('..')
from utils.eval import *

r'''This script is dedicated to create a table of RMSE score of the results.
'''

dates = ['2012-10-13','2012-10-24','2012-11-04']

date = dates[0]

data_set = ['Degraded', 'Satellite']

method_names = ["CP_TV", 'CP_TV_V', 'CV_TV_V']

SSH_CP_TV = []

SSH_CP_TV_V = []

SSH_CV_TV_V = []

BASE_DIR = Path('../')

RESULTS_DIR = BASE_DIR / 'results'

SAT_RESULTS_DIR = RESULTS_DIR / data_set[1]

METHOD_SAT_RESULTS_DIR = SAT_RESULTS_DIR / method_names[0]

DATE_M_S_RESULTS_DIR = METHOD_SAT_RESULTS_DIR/ dates[0]

lambds1 = [0.001, 0.01, 0.1, 1 , 1.0 , 10, 100]
lambds2 = [1, 10, 100]
SSH_results_all = []

for lambd in lambds1:
    try:
        file_name = f'{date}_sigma=1_lambda={lambd}_it=15000_rho=Nan'
        SSH_result = loadmat(f'{BASE_DIR}/results/Satellite/CP_TV/{date}/{file_name}/{file_name}.mat', simplify_cells=True)
        SSH_CP_TV.append(SSH_result)
    except FileNotFoundError:
        print(f'No Experiment Found TV, {lambd}')
SSH_results_all.append(SSH_CP_TV)

for lambd in lambds1:
    try:
        folder_name= f'{date}_sigma=0.9_lambda={lambd}_it=100000_rho=Nan'
        SSH_result=loadmat(f'{BASE_DIR}/Ocean_Image/results/Satellite/CP_TV_V/{date}/{folder_name}/{folder_name}.mat',simplify_cells=True)
        SSH_result['name'] = method_names[1]
        SSH_CP_TV_V.append(SSH_result)
    except FileNotFoundError:
        print(f'No Experiment Found CP_TV {lambd}')

SSH_results_all.append(SSH_CP_TV_V)

chi = 100
it = 80_000
for lambd in lambds1:
    try:
        file_name = f'{date}_sigma=100_lambda={lambd}_chi={chi}_it={it}_rho=1'
        SSH_result=loadmat(f'{BASE_DIR}/Ocean_Image/results/Satellite/CV_TV_V/{date}/{file_name}/{file_name}.mat',simplify_cells=True)
        SSH_result['name'] = method_names[2]
        SSH_CV_TV_V.append(SSH_result)
    except FileNotFoundError:
        print(f'No Experiment Found Hybrid {lambd}')

SSH_results_all.append(SSH_CV_TV_V)


rmse_values_lists = [] 

date_np = np.datetime64(date)
delta_t = np.timedelta64(5,'D')

SSH_ref = xr.open_mfdataset(f'{BASE_DIR}/Ocean_Image/datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
SSH_ref = SSH_ref[0]


# Loop through the SSH results and calculate PSNR for each set of images
for results_set in SSH_results_all:
    rmse_values = []
    for i in range(len(results_set)):
        ref_image = SSH_ref
        obs_image = results_set[i]['SSH_est'][:600,:600]
        
        # Calculate PSNR using your custom function
        psnr = PSNR(ref_image, obs_image)
        rmse_values.append(psnr)
    
    rmse_values_lists.append(rmse_values)

# Create a list of headers for the table
table_headers = ["Image Set", "Image", "RMSE-Base"]

# Create a list of data rows for the table
table_data = []

for i, psnr_values in enumerate(rmse_values_lists):
    for j, psnr in enumerate(psnr_values):
        table_data.append([f"Set {i + 1}", f"Image {j + 1}", psnr])

# Print the PSNR values as a table
table = tabulate(table_data, headers=table_headers, tablefmt="grid")
print(table)
