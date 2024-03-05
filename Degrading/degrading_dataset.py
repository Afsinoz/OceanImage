import numpy as np 
from scipy.io import loadmat, savemat
import os
r'''This script is dedicated to degradation of the subjected days. It takes the numpy.narrays of the date and save them as .mat file.
Purpose is to make it suitable for matlab environment as well. 
'''


def degradation(SSH_ref, rx=600, cx=600, th=0.):
    rand  = np.random.normal(0,1,(rx,cx))
    ind   = np.where(rand>th)

    # burada sadece 0.5'ten buyuk olan noktalari aliyoruz, geri kalani maskleniyor
    #ind   = np.where(np.ones((rx,cx))==1)
    mask  = np.zeros((rx,cx))
    mask[ind] = 1
    z         = np.zeros((rx,cx))
    z[ind]    = SSH_ref[ind]
    masked_values = ind[0].shape[0]
    total_values = rx*cx 
    percantage = int((1-masked_values/total_values)*100)

    return z, mask, percantage 

folder_path = '/datasets/data_ref/np_array_ref'

elements = os.listdir(folder_path)
threshold = -0.5
for element in elements:
    date = element[:10]
    file_path = os.path.join(folder_path, element)
    if os.path.isfile(file_path):
        SSH_ref = np.load(file_path)
        SSH_masked, mask, percantage = degradation(SSH_ref, th=threshold)
        path = f'../datasets/ref_degraded/{percantage}_masked'
        isExist = os.path.exists(path)
        if not isExist:
            os.mkdir(path)
        SSH_degraded = {'SSH_ref':SSH_ref,'SSH_masked':SSH_masked, 'mask':mask, 'date':date}
        savemat(f'{path}/{date}_SSH_ref_{percantage}_masked.mat',SSH_degraded)

print(f'DONE,%{percantage} masked')