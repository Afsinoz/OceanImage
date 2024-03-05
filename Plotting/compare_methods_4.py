import numpy as np 
from scipy.io import loadmat, savemat
from pathlib import Path
import matplotlib.pyplot as plt
import sys 
sys.path.append('..')
from utils.best_results import * 


def plot_best_5(folder_path):
    best_5 = find_best_5(folder_path)
    SSH_results = [loadmat(elements[0],simplify_cells=True) for elements in best_5]
    scores = [elements[1] for elements in best_5]
    num_subplots = len(SSH_results) + 1
    print(num_subplots)
    for score in scores:
        print(score)
    # # Create the figure

    fig, axs = plt.subplots(3, num_subplots, figsize=(5*num_subplots, 15))
    
    SSH_ref = SSH_results[0]['SSH_ref'] 
    SSH_obs = SSH_results[0]['SSH_obs']


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
        axs[1,i].set_title('Performance of {name}, best {score}'.format(name=SSH_results[i-1]['name'],score=scores[i-1]))
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



def plot_compare_best(M1_path,M2_path,M3_path):
    best_5_M1 = find_best_5(M1_path)
    best_5_M2 = find_best_5(M2_path)
    best_5_M3 = find_best_5(M3_path)
    SSH_results_all = []
    scores_all = []

    SSH_results_M1 = [loadmat(elements[0],simplify_cells=True) for elements in best_5_M1]
    scores_M1 = [elements[1] for elements in best_5_M1]

    SSH_results_all.append(SSH_results_M1)
    scores_all.append(scores_M1)

    SSH_results_M2 = [loadmat(elements[0],simplify_cells=True) for elements in best_5_M2]
    scores_M2 = [elements[1] for elements in best_5_M2]

    SSH_results_all.append(SSH_results_M2)
    scores_all.append(scores_M2)

    SSH_results_M3 = [loadmat(elements[0],simplify_cells=True) for elements in best_5_M3]
    scores_M3 = [elements[1] for elements in best_5_M3]

    SSH_results_all.append(SSH_results_M3)
    scores_all.append(scores_M3)

    n_methods = 3

    num_subplots = np.max(np.array((len(SSH_results_M1),len(SSH_results_M2), len(SSH_results_M3))))+1

    fig, axs = plt.subplots(3, num_subplots, figsize=(5*num_subplots, 15))

    SSH_ref = SSH_results_M1[0]['SSH_ref']
    SSH_obs = SSH_results_M1[0]['SSH_obs']

    axs[0,0].imshow(SSH_obs, cmap='gist_stern', origin='lower', interpolation='None')
    axs[0,0].set_title(f'Satellite')
    axs[1,0].imshow(SSH_ref, cmap='gist_stern', origin='lower', interpolation='None')
    axs[1,0].set_title('Reference')
    axs[2,0].axis('off')


    for j in range(n_methods):
        for i in range(1,len(SSH_results_all[j])+1):
            
            axs[j, i].imshow(SSH_results_all[j][i-1]['SSH_est'], cmap='gist_stern', origin='lower', interpolation='None')
            axs[j, i].set_title('SSH_{name}_est, Score:{score:.5f}, sigma={sigma:.5f},\nlambda={lamd}, it={iter}'.format(
                name=SSH_results_all[j][i-1]['name'],
                score=scores_all[j][i-1],
                sigma=SSH_results_all[j][i-1]['params']['sigma'],
                lamd=SSH_results_all[j][i-1]['params']['lambd'], 
                iter=SSH_results_all[j][i-1]['params']['max_iter']),
                 fontsize=12)
            axs[j, i].axis('off')

    fig.tight_layout()
    plt.show()




    return None




if __name__=='__main__':
    CP_TV_SAT = '../results/Satellite/CP_TV/2012-10-13'
    CP_TV_V_SAT = '../results/Satellite/CP_TV_V/2012-10-13'
    CV_TV_V_SAT = '../results/Satellite/CV_TV_V/2012-10-13'



    CP_TV_DEG = '../results/Degraded/CP_TV/2012-10-13'
    CP_TV_V_DEG = '../results/Degraded/CP_TV_V/2012-10-13'
    CV_TV_V_DEG = '../results/Degraded/CV_TV_V/2012-10-13'


    # plot_best_5(CP_TV)
    plot_compare_best(CP_TV_SAT,CP_TV_V_SAT,CV_TV_V_SAT)

    plot_compare_best(CP_TV_DEG,CP_TV_V_DEG,CV_TV_V_DEG)
