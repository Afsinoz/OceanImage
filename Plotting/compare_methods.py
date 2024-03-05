import numpy as np 
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt

'File Path should carefully be decided here'

dates = ['2012-10-13','2012-10-24','2012-11-04']

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

lambds1 = [0.001, 0.01, 0.1, 1]
lambds2 = [1, 10, 100]
SSH_results_all = []

for lambd in lambds1:
    file_name = f'2012-10-13_sigma=0.9_lambda={lambd}_it=15000_rho=1'
    SSH_result = loadmat(f'../results/Satellite/CP_TV/2012-10-13/{file_name}/{file_name}.mat', simplify_cells=True)
    SSH_CP_TV.append(SSH_result)
    
SSH_results_all.append(SSH_CP_TV)
for lambd in lambds2:
    try:
        folder_name= f'2012-12-13_tau=0.9_sigma_0.9_lambda={lambd}_maxiter=5005'
        SSH_result=loadmat(f'../Old_results/new_reg/2012-12-13_SSH/{folder_name}/test.mat',simplify_cells=True)
        SSH_result['name'] = method_names[1]
        print(SSH_result['param_B'].keys())
        temp = SSH_result['param_B']
        SSH_result['params'] = temp
        temp2 = SSH_result['params']['lambda']
        SSH_result['params']['lambd'] = temp2

        SSH_CP_TV_V.append(SSH_result)
    except FileNotFoundError:
        print('No Experiment Found')

SSH_results_all.append(SSH_CP_TV_V)


for lambd in lambds1:
    try:
        file_name= f'sigma=0.9_lambda={lambd}_it=50000_rho=1.mat'
        SSH_result=loadmat(f'../Conda_Vu_TV_Vorticity/results/Results/{file_name}',simplify_cells=True)
        SSH_result['name'] = method_names[2]
        SSH_CV_TV_V.append(SSH_result)
    except FileNotFoundError:
        print('No Experiment Found')

SSH_results_all.append(SSH_CV_TV_V)


n_methods = 3

num_subplots = np.max(np.array((len(SSH_CP_TV),len(SSH_CP_TV_V), len(SSH_CV_TV_V))))+1

fig, axs = plt.subplots(3, num_subplots, figsize=(5*num_subplots, 15))


SSH_ref = SSH_CP_TV[0]['SSH_ref']
SSH_obs = SSH_CP_TV_V[0]['SSH_obs']

print(SSH_CP_TV[0].keys())


axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None')
axs[0,0].set_title(f'Satellite')
axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
axs[1,0].set_title('Reference')
axs[2,0].axis('off')

SSH_result = []
SSH_result.append(SSH_CP_TV)

for j in range(3):
    for i in range(1,len(SSH_results_all[j])+1):
        
        axs[j, i].imshow(SSH_results_all[j][i-1]['SSH_est'], cmap='gist_stern', origin='lower', interpolation='None')
        axs[j, i].set_title('SSH_{name}_est, sigma={sigma},\nlambda={lamd}, it={iter}'.format(
            name=SSH_results_all[j][i-1]['name'],
            sigma=int(SSH_results_all[j][i-1]['params']['sigma']),
            lamd=SSH_results_all[j][i-1]['params']['lambd'], 
            iter=SSH_results_all[j][i-1]['params']['max_iter']), fontsize=18)
        axs[j, i].axis('off')

fig.tight_layout()
plt.show()



