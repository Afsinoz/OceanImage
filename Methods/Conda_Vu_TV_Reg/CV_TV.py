import numpy as np 
from tqdm import tqdm
from pathlib import Path
import json 
import sys
import os 
from scipy.io import savemat, loadmat
sys.path.append('..')
from utils.toolbox import *
from utils.eval import * 
from utils.fix_params_TV_vorticity import * 
from CP_TV_Vorticity.TV_Vorticity_last import *


r'''This script is dedicated to perform non-smooth optimization algorithm called Conda-Vu algorithm with total variation regularization term.

At the end it is looping over different hyperparameters, applying the method and it is saving, .json, .mat and .png files in the subjected folder. If folder exist it doesn't repeate the process. 
'''

date = '2012-10-13'

methods_name = 'CV_TV'

data_set = ['Degraded', 'Satellite']

BASE_DIR = Path('../')

RESULTS_DIR = BASE_DIR / 'results'

DATASET_RESULTS = RESULTS_DIR / data_set[1]

CV_TV_RESULTS_DIR = DATASET_RESULTS / methods_name

CV_TV_RESULTS_DIR_DATE = CV_TV_RESULTS_DIR /f'{date}'

CV_TV_RESULTS_DIR_DATE.mkdir(parents=True, exist_ok=True)






    
SSH_ref = np.load('../../datasets/data_ref/np_array_ref/2012-10-13_image_ref.npy')



ngrid_lat = 601
ngrid_lon = 601

a = np.ones((1, ngrid_lon))
at = np.transpose(a)
b = np.arange(0, ngrid_lat)
bt = np.transpose(b)


y = np.transpose(np.kron(at, b))


threshold = 1.

SSH_masked, mask, percantage = degradation(SSH_ref, th=threshold)



def run_Conda_Vu_TV(x_masked, mask, tau, sigma, rho, lambd, max_iter, X_ref=None):
    r'''
    Perform the Conda-Vu algorithm https://hal.in2p3.fr/GIPSA-AGPIG/hal-01120544v1 to solve non-smooth convex optimization problem, which is 
    ..math::
        arg\min_s || o  -\mathbf{H}s||_2^2 + \frac{\lambd}{2} || D s ||_2^2   
        
    where:
    - s is the optimization variable,
    - H is a linear operator,
    - D is an operator associated with the input image and mask,
    - o is the input image,
    - lambda, tau, sigma, rho are hyperparameters,
    - g and f_0 are constants,
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
    y0 = grad(x0)

    crit = 1e5 * np.ones(max_iter)
    err = 1e5 * np.ones(max_iter)
    rmseb = 1e5 * np.ones(max_iter)
    psnr_list = 1e5*np.ones(max_iter)
    # Set Variables X_est
    x_est = np.copy(x_masked)

    def update(x0, y0):
        temp1 = x0 - tau * B_star_new(B_new(x0, otfA) + grad(beta*dy*y),otfA) - tau * grad_T(y0)
        xnew = prox_f(x_masked, temp1, tau, mask)
        temp2 = (2 * xnew - x0)
        ynew = prox_g_star(y0 + sigma * grad(temp2), sigma, lambd)
        # Vnew = prox_gstar(V0 + sigma*grad(Ynew), sigma)
        # print((rho*xnew).shape, ((1-rho)*x0).shape)
        xnew = rho*xnew + (1-rho)*x0
        ynew = rho*ynew + (1-rho)*y0
        return xnew, ynew
        # Iteration


    for i in tqdm(range(max_iter)):
        x_prev = x0
        x0, y0 = update(x0, y0)
        crit[i] = crit_Conda_Vu(x0, x_masked, lambd, mask, B_new , otfA, beta*dy*y)
        err[i] = np.linalg.norm(x0 - x_prev) / np.linalg.norm(x_masked)
        rmseb[i] = rmse_based_numpy(X_ref, x0[:600,:600])
        #         snr_list[i]= SNR(original=X_ref, estimate=X0)
        psnr_list[i] = PSNR(original =X_ref, estimate = x0[:600,:600])
        # Stores the image if it is supposed to be a measured iteration
        if i + 1 == max_iter:
            x_est = x0



    return x_est, crit, err, rmseb, psnr_list 


if __name__=='__main__':




    norm_B = operator_norm(B_new, B_star_new, otfA,rx=601,cx=601)

    norm_L = np.sqrt(8)
    # lambds = [0.001,0.01,0.1]
    lambds = [0.01]
    # sigmas = [0.6,0.9]
    sigmas = [0.9]
    # max_iters = [1000,5000,10000,15000]
    max_iters = [10]
    rho = 1 # between 0 and 2 


    for sigma in sigmas:
        tau = 1/(sigma*norm_L**2 + norm_B/2)
        for lambd in lambds:
            for max_iter in max_iters:
                params ={
                    "lambd" : lambd,
                "sigma" :sigma ,
                "max_iter" : max_iter,
                "rho" : rho,
                'date':date
                }
                


                SSH_est, crit, err, rmseb, psnr_list = run_Conda_Vu_TV(SSH_obs0,
                                                                    mask0,
                                                                    tau=tau,
                                                                    sigma=sigma,
                                                                    rho=rho,
                                                                    lambd=lambd,
                                                                    max_iter=max_iter,
                                                                    X_ref=SSH_ref)

                results = {"SSH_ref":SSH_ref,
                           "SSH_obs":SSH_obs,
                           'SSH_est':SSH_est,
                           "mask":mask0,
                           'crit':crit,
                           'rmseb':rmseb,
                           'psnr_list':psnr_list ,
                           "params":params,
                           'name':methods_name}


                FOLDER_DIR = CV_TV_RESULTS_DIR_DATE/f'{date}_sigma={sigma}_lambda={lambd}_it={max_iter}_rho={rho}'
                FOLDER_DIR.mkdir(parents=True, exist_ok=True)

                file_name = FOLDER_DIR/f'{date}_sigma={sigma}_lambda={lambd}_it={max_iter}_rho={rho}'
                with open(f'{file_name}.json', 'w') as json_file:
                    json.dump(params, json_file, indent=4)

                savemat(f'{file_name}.mat', results)
                fig, axs = plt.subplots(3, 2, figsize=(10, 15))
                axs[2,0].axis('off')

                axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None')
                axs[0,0].set_title(f'Degraded SSH')
                axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
                axs[1,0].set_title('Reference')
                axs[0,1].imshow(SSH_est, cmap='gist_stern', origin='lower', interpolation='None')
                axs[0,1].set_title('Estimation_CV')
                axs[1,1].set_title('Performance')
                axs[1,1].set_ylabel('PSNR', color='green')
                l1, = axs[1,1].plot(psnr_list, color='green') 
                ax2 = axs[1,1].twinx()
                ax2.set_ylabel('RMSE', color='red')
                l2, = ax2.plot(rmseb, color='red')

                # axs[1,1].loglog(critTV)
                axs[1,1].legend([l1, l2], ['PSNR',"RMSE_Based" ])
                # axs[1,1].legend(labels=['Criterion','RMSE', 'PSNR'])
                axs[1,1].grid('on')
                axs[2,1].plot(crit)
                # del axs[2, 0]
                axs[2,1].set_title('Criterion')
                axs[2,1].grid('on')
                plt.suptitle('{date}_sigma={sigma}_rho={rho}_lambda={lambd}_max_iter={max_iter}'.format(date=date,sigma=sigma,rho=rho,lambd=lambd,max_iter=max_iter))
                fig.tight_layout()
                plt.savefig(f'{file_name}.png')
                plt.show()
                plt.close()
