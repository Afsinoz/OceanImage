import numpy as np 
import matplotlib.pyplot as plt
import os 
from scipy.io import savemat, loadmat
import sys
sys.path.append('..')
from utils.eval import *





def plot_one_est_degraded(SSH_masked, SSH_ref, SSH_est, crit, err, rmseb, psnr_list, percantage=0,name=None, save_dir=False):

    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    axs[2,0].axis('off')

    axs[0,0].imshow(SSH_masked, cmap='gist_stern', origin='lower', interpolation='None')
    axs[0,0].set_title(f'Degraded SSH, %{percantage} mask')
    axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
    axs[1,0].set_title('Reference')
    axs[0,1].imshow(SSH_est, cmap='gist_stern', origin='lower', interpolation='None')
    axs[0,1].set_title('Estimation')
    axs[1,1].set_title('Performance')
    l1, = axs[1,1].plot(psnr_list, color='green') 
    ax2 = axs[1,1].twinx()
    l2, = ax2.plot(rmseb, color='red')

    # axs[1,1].loglog(critTV)
    axs[1,1].legend([l1, l2], ["RMSE_Based", 'PSNR'])
    # axs[1,1].legend(labels=['Criterion','RMSE', 'PSNR'])
    axs[1,1].grid('on')
    axs[2,1].plot(crit)
    # del axs[2, 0]
    axs[2,1].set_title('Criterion')
    axs[2,1].grid('on') 
    plt.suptitle(f'{name}')
    plt.savefig()
    plt.show()
    plt.close()





def plot(SSH_results, SSH_ref, SSH_degraded, percantage=0):
    num_subplots = len(SSH_results) + 1
    # Create the figure

    fig, axs = plt.subplots(3, num_subplots, figsize=(5*num_subplots, 15))

    axs[0,0].imshow(SSH_degraded, cmap='gist_stern', origin='lower', interpolation='None')
    axs[0,0].set_title(f'Degraded SSH, %{percantage} mask')
    axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
    axs[1,0].set_title('Reference')
    axs[2,0].axis('off')



    for i in range(1,num_subplots):
        axs[0, i].imshow(SSH_results[i-1]['SSH_est'], cmap='gist_stern', origin='lower', interpolation='None')
        axs[0, i].set_title('SSH_{name}_est, tau={tau},\nlambda={lamd}, it={iter}'.format(
            name=SSH_results[i-1]['name'],
            tau=SSH_results[i-1]['params']['tau'],
            lamd=SSH_results[i-1]['params']['lambda'], 
            iter=SSH_results[i-1]['params']['max_iter']), fontsize=18)
        axs[0, i].axis('off')

        axs[1,i].set_title('Performance of {name}'.format(name=SSH_results[i-1]['name']))
        axs[1,i].grid('on')
        l1, = axs[1,i].plot(SSH_results[i-1]['PSNR'], color='green') 
        axs[1,i].set_ylabel('PSNR', color='green')
        ax2 = axs[1,i].twinx()
        ax2.set_ylabel('RMSE', color='red')
        l2, = ax2.plot(SSH_results[i-1]['rmse_b'], color='red')
        axs[1,i].legend([l1, l2], ['PSNR',"RMSE_Based"])

        axs[2, i].plot(SSH_results[i-1]['crit'] )
        axs[2,i].grid('on')
        axs[2, i].set_title('Crit {name}'.format(name=SSH_results[i-1]['name']))
        # axs[1, i].set_ylim([0, 1])

    # Adjust the spacing between subplots
    fig.tight_layout()

    filename = f'Degradation_Results/{percantage}_mask/output_{percantage}.png'

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

def main():
    SSH_ref = loadmat('../datasets/data_obs/2012-10-13_SSH.mat')
    SSH_PnP = loadmat('../results/PnP/2012-10-13_SSH_truemask_PnP.mat')
    SSH_CP_TV = loadmat('../results/Satellite/CP_TV/2012-10-13/2012-10-13_sigma=0.9_lambda=0.1_it=15000_rho=Nan/2012-10-13_sigma=0.9_lambda=0.1_it=15000_rho=Nan.mat',simplify_cells=True)
    Score_PnP = rmse_based_numpy(SSH_ref['SSH_ref'],SSH_PnP['xrec'][:600,:600])
    Score_CP_TV = rmse_based_numpy(SSH_ref['SSH_ref'],SSH_CP_TV['SSH_est'][:600,:600])
    print(Score_PnP)
    print(Score_CP_TV)


    SSH_PnP['SSH_est'] = SSH_PnP['xrec'][:600,:600]
    SSH_results = [SSH_PnP['xrec'][:600,:600], SSH_CP_TV['SSH_est'][:600,:600]]

    # plot(SSH_results, SSH_ref['SSH_ref'], SSH_ref['SSH_obs'])

    # plt.imshow(SSH_PnP['xrec'][:600,:600],cmap='gist_stern',origin='lower')

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(SSH_ref['SSH_ref'][:600,:600],cmap='gist_stern',origin='lower')
    plt.title('Reference')
    plt.subplot(2,2,2)
    plt.imshow(SSH_PnP['xrec'][:600,:600],cmap='gist_stern',origin='lower')
    plt.title('PnP Results Score:{Score:.5f}'.format(Score=Score_PnP))
    plt.subplot(2,2,3)
    plt.imshow(SSH_ref['SSH_obs'][:600,:600], cmap='gist_stern',interpolation='None',origin='lower')
    plt.title('Observations')
    plt.subplot(2,2,4)
    plt.imshow(SSH_CP_TV['SSH_est'][:600,:600],cmap='gist_stern',origin='lower')
    plt.title('CP_TV_Results Score:{Score:.5f}'.format(Score=Score_CP_TV))
    
def PnP_results():
    SSH_ref = loadmat('../datasets/data_obs/2012-10-13_SSH.mat')
    SSH_PnP = loadmat('../results/PnP/PnP_FB_unfolded_ISTA_ver2_2012-10-13_SSH_ref_50_masked_truemask_sigma_0.05_gamma_1.99_data_vary_alpha_0.9_normalized.mat')
    # SSH_PnP = loadmat('../results/PnP/2012-10-13_SSH_ref_84_masked.mat')
    SSH_CP_TV = loadmat('../results/Degraded/CP_TV/2012-10-13/2012-10-13_sigma=0.9_lambda=0.1_it=15000_rho=Nan/2012-10-13_sigma=0.9_lambda=0.1_it=15000_rho=Nan.mat',simplify_cells=True)
    Score_PnP = rmse_based_numpy(SSH_ref['SSH_ref'],SSH_PnP['xrec'][:600,:600])
    Score_CP_TV = rmse_based_numpy(SSH_ref['SSH_ref'],SSH_CP_TV['SSH_est'][:600,:600])

    print(SSH_PnP.keys())
    SSH_obs_mat = loadmat('../datasets/ref_degraded/84_masked/2012-10-13_SSH_ref_84_masked.mat')
    SSH_obs = SSH_obs_mat['SSH_masked']
    print(
        SSH_obs_mat.keys()
    )
    # print(Score_PnP)
    # print(Score_CP_TV)


    SSH_PnP['SSH_est'] = SSH_PnP['xrec'][:600,:600]
    SSH_results = [SSH_PnP['xrec'][:600,:600], SSH_CP_TV['SSH_est'][:600,:600]]

    # plot(SSH_results, SSH_ref['SSH_ref'], SSH_ref['SSH_obs'])

    # plt.imshow(SSH_PnP['xrec'][:600,:600],cmap='gist_stern',origin='lower')

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(SSH_ref['SSH_ref'][:600,:600],cmap='gist_stern',origin='lower')
    plt.title('Reference')
    plt.subplot(2,2,2)
    plt.imshow(SSH_PnP['xrec'][:600,:600],cmap='gist_stern',origin='lower')
    plt.title('PnP Results Score:{Score:.5f}'.format(Score=Score_PnP))
    plt.subplot(2,2,3)
    plt.imshow(SSH_obs, cmap='gist_stern',interpolation='None',origin='lower')
    plt.title('Observations')
    plt.subplot(2,2,4)
    plt.imshow(SSH_CP_TV['SSH_est'][:600,:600],cmap='gist_stern',origin='lower')
    plt.title('CP_TV_Results Score:{Score:.5f}'.format(Score=Score_CP_TV))

    






if __name__=='__main__':
    # main()
    SSH = loadmat('../results/Satellite/CV_TV_V/2012-10-13/2012-10-13_sigma=100_lambda=0.001_it=20000_rho=1/2012-10-13_sigma=100_lambda=0.001_chi=1_it=20000_rho=1.mat',simplify_cells=True)
    SSH_ref = SSH['SSH_ref']
    SSH_obs = SSH['SSH_obs']
    

    print(SSH.keys())
    plot_one_est_degraded(SSH_obs,SSH_ref,SSH_est=SSH['SSH_est'],crit=SSH['crit'],err=SSH['err'],rmseb=SSH['rmseb'],psnr_list=SSH['psnr_list'])
    # PnP_results()