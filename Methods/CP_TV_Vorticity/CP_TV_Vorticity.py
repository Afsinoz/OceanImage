
# import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
# from new_reg_B_update import *
from scipy.io import savemat
import json 
import sys #ignore 
sys.path.append('..')
from utils.toolbox import *
from utils.eval import * 
from TV_Vorticity_last import *
from utils.fix_params_TV_vorticity import *


r'''This script is dedicated to perform non-smooth optimization algorithm called the Chambolle Pock algorithm with two regulazation terms, L2 norm of the total variation and L1 of the variations of potential vorticity.

At the end it is looping over different hyperparameters, applying the method and it is saving, .json, .mat and .png files in the subjected folder. If folder exist it doesn't repeate the process. 
'''


 

def run_Chambolle_Pock_B(x_masked, mask, lon_y, otf, tau, sigma, lambd, max_iter, X_ref=None):
    r'''
    Perform the Chambolle Pock algorithm https://hal.science/hal-00490826/document to solve non-smooth convex optimization problem, which is 
    ..math::
        arg\min_s || o  -\mathbf{H}s||_2^2 + \frac{\chi}{2} || D s ||_2^2 + \lambda || D(\mathcal{L}\frac{g}{f_0} s - \frac{1}{L_d^2} \frac{g}{f_0} s + \beta y) ||_1.
    where:
    - s is the optimization variable,
    - H is a linear operator,
    - D is an operator associated with the input image and mask,
    - o is the input image,
    - chi, lambda, tau, sigma, rho are hyperparameters,
    - L is a linear operator,
    - g and f_0 are constants,
    - y is another variable,
    - beta is a constant,
    - L_d is a parameter,
    - max_iter is the maximum number of iterations,
    - X_ref is an optional reference.

    Args:
        x_masked (np.array): Input image.
        mask (np.array): Calculated mask.
        tau (float): Hyperparameter.
        sigma (float): Hyperparameter.
        rho (float): Hyperparameter.
        lambd (float): Hyperparameter.
        max_iter (int): Maximum number of iterations.
        X_ref (np.array, optional): Optional reference.
    Returns: 
        x_est (np.array): Reconstructed image using the Conda-Vu algorithm.
        crit (np.array): Convergence criteria values over iterations.
        err (np.array): Relative error values over iterations.
        rmseb (np.array): Root mean square error values over iterations.
        psnr_list (np.array): Peak signal-to-noise ratio values over iterations.
    '''
    x0 = x_masked

    v0 = B_new(x0, otf)

    crit = 1e5 * np.ones(max_iter)
    err = 1e5 * np.ones(max_iter)
    rmseb = 1e5 * np.ones(max_iter)
    psnr_list = 1e5*np.ones(max_iter)
    # Set Variables X_est
    x_est = np.copy(x_masked)

    # Update
    def update(x0, v0):
        xnew = prox_f(x_masked, x0 - tau * B_star_new(v0, otf), tau, mask)
        ynew = 2 * xnew - x0
        vnew = prox_g_star_translated(v0 + sigma * B_new(ynew, otf), sigma, lambd, grad(beta*dy*lon_y))
        # Vnew = prox_gstar(V0 + sigma*grad(Ynew), sigma)
        return xnew, vnew

    for i in tqdm(range(max_iter)):
        x_prev = x0
        x0, v0 = update(x0, v0)
        crit[i] = crit_B(x0, x_masked, lambd, mask, B_new , otf, beta*dy*lon_y)
        err[i] = np.linalg.norm(x0 - x_prev) / np.linalg.norm(x_masked)
        rmseb[i] = rmse_based_numpy(X_ref, x0[:600,:600])

        psnr_list[i] = PSNR(original=X_ref, estimate=x0[:600,:600])
        # Stores the image if it is supposed to be a measured iteration
        # print(crit[i])
        if i + 1 == max_iter:
            x_est = x0

    return x_est, crit, err, rmseb, psnr_list




def main(date='2012-10-13',data_set='Satellite'):
    date_np = np.datetime64(date)
    delta_t = np.timedelta64(5, 'D')

    methods_name = 'CP_TV_V'


    BASE_DIR = Path('../../')

    RESULTS_DIR = BASE_DIR / 'results'

    DATASET_RESULTS = RESULTS_DIR / data_set

    CP_TV_V_RESULTS_DIR = DATASET_RESULTS / methods_name

    CP_TV_V_RESULTS_DIR_DATE = CP_TV_V_RESULTS_DIR /f'{date}'

    CP_TV_V_RESULTS_DIR_DATE.mkdir(parents=True, exist_ok=True)

    

    #SSH_ref = np.load(f'../../datasets/data_ref/np_array_ref/{date}_image_ref.npy')
    SSH_ref = xr.open_mfdataset('../../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
    SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
    SSH_ref = SSH_ref[0]
    if data_set=='Satellite':
        SSH_obs_mat = loadmat(f'../../datasets/data_obs/{date}_SSH.mat',simplify_cells=True)
        # SSH_masked, mask, percantage = degradation(SSH_ref, th=threshold)
        SSH_obs = SSH_obs_mat['SSH_obs']
        percentage = nan_percentage(SSH_obs)
        SSH_masked, mask = mask_and_filling(SSH_obs)
    elif data_set=='Degraded':
        threshold = -0.1
        SSH_masked, mask, percentage = degradation(SSH_ref, th=threshold)
        SSH_obs = SSH_masked.copy()
    
    otfA = psf2otf(kernel, SSH_masked.shape)

    

    lon_y = vorticity_y(SSH_masked.shape[0],SSH_masked.shape[1])
    if adjoint_test(B_new, B_star_new, otfA,rx=SSH_masked.shape[0],cx=SSH_masked.shape[1]):
        norm_B = operator_norm(B_new, B_star_new, otfA,rx=SSH_masked.shape[0],cx=SSH_masked.shape[1])

        # Different values of lambda
        #for v_tau 0.1 0.5 0.9

        #tau = v_tau/(np.sqrt(norm_B))
        #sigma = v_sigma/ (np.sqrt(norm_B))
        # taus = [0.1, 0.5, 0.9]
        # sigmas = [0.9]

        sigmas = [0.9]
        lambds = [0.001, 0.01, 0.1, 1, 10, 100]
        # lambds = [10]
            
        # lambds = [0.1]

        # max_iters = [1000, 5000, 10_000, 50_000]
        max_iters = [80_000]
        # max_iters = [10]
        # max_iters = [12]

        rho = 'Nan'
        for max_iter in max_iters:
            for lambd in lambds:
                for v_sigma in sigmas:
                    tau = v_sigma/(np.sqrt(norm_B))
                    sigma = v_sigma/ (np.sqrt(norm_B))
                    # # folder_name = f"{central_date}_tau={v_tau}_sigma_{v_tau}_lambda={lamd}_maxiter={max_iter}_new"
                    # # path_est = f'../results/new_reg/{central_date}_SSH/{folder_name}/est'
                    # # path_figure = f'../results/new_reg/{central_date}_SSH/{folder_name}/figures'
                    # paths = [path_est, path_figure]
                    # for path in paths:
                    #     isExist = os.path.exists(path)
                    #     if not isExist:
                    #         os.makedirs(path)
                    params ={
                    "lambd" : lambd,
                    "sigma" :v_sigma ,
                    "max_iter" : max_iter,
                    "rho" : rho,
                    'date':date
                    }
                    FOLDER_DIR = CP_TV_V_RESULTS_DIR_DATE/f'{date}_sigma={v_sigma}_lambda={lambd}_it={max_iter}_rho={rho}'
                    
                    if os.path.isdir(f'{FOLDER_DIR}'):
                        print('Experiment already done')

                    else:
                        FOLDER_DIR.mkdir(parents=True, exist_ok=True)

                        file_name = FOLDER_DIR/f'{date}_sigma={v_sigma}_lambda={lambd}_it={max_iter}_rho={rho}'

                        SSH_est, crit, err, rmseb, psnr_list = run_Chambolle_Pock_B(SSH_masked, mask,lon_y, otfA, tau, sigma, lambd, max_iter,SSH_ref)

                        # param_B = {'v_tau':v_tau, 
                        #            'tau':tau, 
                        #            'sigma':sigma, 
                        #            'lambda':lambd, 
                        #            'max_iter':max_iter }
                        # res = {"SSH_ref" : SSH_ref,'SSH_obs':SSH_obs, "Mask":mask ,'SSH_est':SSH_est, "crit": crit, "err": err,"param_B":param_B}
                        results = {"SSH_ref":SSH_ref,
                            "SSH_obs":SSH_obs,
                            'SSH_est':SSH_est,
                            "mask":mask,
                            'crit':crit,
                            'rmseb':rmseb,
                            'psnr_list':psnr_list ,
                            "params":params,
                            'name':methods_name}


                        with open(f'{file_name}.json', 'w') as json_file:
                            json.dump(params, json_file, indent=4)
                        savemat(f"{file_name}.mat", results)
                        
                        

                        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                        axs[2,0].axis('off')

                        axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None')
                        if data_set=='Degraded':
                            axs[0,0].set_title(f'{data_set} SSH {percentage}% mask')
                        elif data_set=='Satellite':
                            axs[0,0].set_title(f'{data_set} SSH{percentage}% NaN')
                        axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
                        axs[1,0].set_title('Reference')
                        axs[0,1].imshow(SSH_est, cmap='gist_stern', origin='lower', interpolation='None')
                        axs[0,1].set_title('Estimation_CV')
                        axs[1,1].set_title('Performance')
                        axs[1,1].set_ylabel('PSNR', color='green')
                        l1, = axs[1,1].plot(psnr_list, color='green') 
                        ax2 = axs[1,1].twinx()
                        ax2.set_ylabel('RMSE', color='red')
                        ax2.set_ylim(0,1)
                        l2, = ax2.plot(rmseb, color='red')

                        # axs[1,1].loglog(critTV)
                        axs[1,1].legend([l1, l2], ['PSNR',"RMSE_Based" ])
                        # axs[1,1].legend(labels=['Criterion','RMSE', 'PSNR'])
                        axs[1,1].grid('on')
                        axs[2,1].plot(crit)
                        # del axs[2, 0]
                        axs[2,1].set_title('Criterion')
                        axs[2,1].grid('on')
                        plt.suptitle('{date}_Method={method_name}sigma={sigma}_rho={rho}_lambda={lambd}_max_iter={max_iter}'.format(date=date,sigma=int(sigma),rho=rho,method_name=methods_name,lambd=lambd,max_iter=max_iter))
                        fig.tight_layout()
                        plt.savefig(f'{file_name}.png')
                        plt.show()
                        plt.close()


def best_stepsize(date, data_set):
    date_np = np.datetime64(date)
    delta_t = np.timedelta64(5,'D')

    
    # SSH_ref = np.load(f'../../datasets/data_ref/np_array_ref/{date}_image_ref.npy')
    SSH_ref = xr.open_mfdataset('../../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
    SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
    SSH_ref = SSH_ref[0]
    if data_set=='Satellite':
        SSH_obs_mat = loadmat(f'../../datasets/data_obs/{date}_SSH.mat',simplify_cells=True)
        # SSH_masked, mask, percantage = degradation(SSH_ref, th=threshold)
        SSH_obs = SSH_obs_mat['SSH_obs']
        percentage = nan_percentage(SSH_obs)
        SSH_masked, mask = mask_and_filling(SSH_obs)
    elif data_set=='Degraded':
        threshold = -0.1
        SSH_masked, mask, percentage = degradation(SSH_ref, th=threshold)
        SSH_obs = SSH_masked.copy()
    
    otfA = psf2otf(kernel, SSH_masked.shape)

    lon_y = vorticity_y(SSH_masked.shape[0],SSH_masked.shape[1])

    norm_L = np.sqrt(8)


    norm_B = operator_norm(B_new, B_star_new, otfA,rx=SSH_masked.shape[0],cx=SSH_masked.shape[1])


    sigma_list = [0.001, 0.01, 0.1, 1., 10, 100]
    max_iter = 10_000
    lambd = 10

    chi= 1
    rho = 1 # between 0 and 2 

    losses = []

    state_cv = dict()
    i = 0 
    for sigma in sigma_list:
        i =+ 1
        tau = 1/((chi/2)*norm_L**2 + sigma*norm_B)

        SSH_est, crit, err, rmseb, psnr_list =  run_Chambolle_Pock_B(SSH_masked, 
                                                                    mask,
                                                                    lon_y,
                                                                    otfA, 
                                                                    tau, 
                                                                    sigma, 
                                                                    lambd, 
                                                                    max_iter,
                                                                    SSH_ref)
        
        state_cv[f'{sigma}']= SSH_est, crit, max_iter, err, rmseb, psnr_list 

    file_name = f'{date}_sigma=different_lambda={lambd}_chi={chi}_it={max_iter}_rho={rho}'

    savemat(f"{file_name}.mat",  state_cv)
        

    plt.figure()
    plt.title("Visualising convergence of different $\sigma$ values")

    for sigma in sigma_list:

        _, losses, _, _, _, _, = state_cv[f'{sigma}']
        losses = losses / losses[0]
        plt.loglog(losses, label=f"sigma={sigma:.3f}")

    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()
    plt.savefig(f'Convergence {max_iter}')
    plt.show()
        










if __name__=='__main__':
    
    dates = ['2012-10-13','2012-10-24','2012-11-04']
    
    # dates = ['2012-10-13']

    # data_sets = ['Degraded', 'Satellite']
    data_sets =['Satellite']

    for date0 in dates:
        for data_set0 in data_sets:
            # best_stepsize(date0, data_set0)
            main(date0, data_set=data_set0)
            
