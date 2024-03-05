import numpy as np


r'''Evaluation functions

'''

def PSNR(original, estimate):
    error = (original - estimate)
    mse = np.nanmean(np.square(error))
    # max_pixel = 1.0
    psnr = 20 * np.log10(np.max(original) / np.sqrt(mse))
    return psnr


def rmse_numpy(y_true, y_pred):
    error = y_pred - y_true
    squared_error = np.square(error)
    mean_squared_error = np.nanmean(squared_error)
    rmse = np.sqrt(mean_squared_error)
    return rmse


def rms_numpy(array):
    squared_values = np.square(array)
    mean_squared = np.mean(squared_values)
    rms = np.sqrt(mean_squared)
    return rms


def rmse_based_numpy(y_true, y_pred):
    return 1 - (rmse_numpy(y_true, y_pred)/rms_numpy(y_true))

# image_ref = np.load('../datasets/data_ref/np_array_ref/2012-10-13_image_ref.npy')

# image_TV = np.load('../results/CP_TV/est/2012-10-13_image_est_TV.npy')[:600, :600]

# image_B = np.load('../results/new_reg/2012-12-13_SSH/2012-12-13_tau=0.1_sigma_0.1_lambda=1_maxiter=15000')[:600, :600]


# result1 = rmse_based_numpy(image_ref, image_TV)
# result2 = rmse_based_numpy(image_ref, image_B)
# print("RMSE:", result1, result2)
# def rmse_torch(y_true, y_pred):
#     error = y_true - y_pred
#     squared_error = torch.square(error)
#     mean_squared_error = torch.nanmean(squared_error)
#     rmse = torch.sqrt(mean_squared_error)
#     return rmse

# def rms_torch(tensor):
#     squared_values = torch.square(tensor)
#     mean_squared = torch.nanmean(squared_values)
#     rms = torch.sqrt(mean_squared)
#     return rms
