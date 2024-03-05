import numpy as np 
import scipy.io 
import os
from toolbox import *
import matplotlib.pyplot as plt

path = '../datasets/data_obs'

isExist = os.path.exists(path)
if not isExist:
    os.mkdir(path)

date = '2012-11-04'

SSH_ref = np.load(f'../datasets/data_ref/np_array_ref/{date}_image_ref.npy')

SSH_obs = np.load(f'../datasets/np_array/{date}_image_data.npy')

SSH_obs0, mask = mask_and_filling(SSH_obs)

SSH = {'SSH_ref':SSH_ref, 'SSH_obs':SSH_obs, "SSH_obs0":SSH_obs0,'mask':mask }




scipy.io.savemat(f'{path}/{date}_SSH.mat', SSH)

if __name__=='__main__':
    SSH = scipy.io.loadmat(f'{path}/{date}_SSH.mat',simplify_cells=True)
    
    fig = plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(SSH['SSH_ref'], cmap='gist_stern',interpolation='None', origin='lower')
    plt.subplot(2,2,2)
    plt.imshow(SSH['SSH_obs'], cmap='gist_stern',interpolation='None', origin='lower')
    plt.subplot(2,2,3)
    plt.imshow(SSH['mask'], cmap='gist_stern',interpolation='None', origin='lower')
    plt.show()




