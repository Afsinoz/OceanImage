import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter  # Import the formatter
from scipy.io import loadmat, savemat

date = '2012-10-24'

# file_name = f'../datasets/ref_degraded/84_masked/{date}_SSH_ref_84_masked.mat'

file_name = f'../datasets/data_obs/{date}_SSH.mat'
SSH_mat = loadmat(file_name,simplify_cells=True)
print(SSH_mat.keys())


# SSH_obs = SSH_mat['mask'][:600,:600]
SSH_obs = SSH_mat['SSH_obs']

name =f'Mask  {date}'

date_np = np.datetime64(date)
delta_t = np.timedelta64(5,'D')


SSH_ref = xr.open_mfdataset('../datasets/Original/dc_ref/*.nc', combine='nested', concat_dim='time')
SSH_ref = SSH_ref.sel(time=slice(date_np - delta_t, date_np + delta_t)).mean(dim='time').to_array().values
SSH_ref = SSH_ref[0]

# Assuming 'average_da' is your DataArray with shape (600, 600)
plt.imshow(SSH_obs, cmap='gist_stern',norm='linear',interpolation='none',origin='lower',extent=[-65, -55, 33, 43])  # You can specify the colormap you prefer
# plt.colorbar()  # Add a colorbar to the plot
# # plt.title(name)  # Set a title for the plot
plt.xlabel('Latitude')  # Label the x-axis (customize as needed)
plt.ylabel('Longitude')  # Label the y-axis (customize as needed)
# plt.scatter([55, 33], [33, 43], color='red', marker='o', label='Points a and c')
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g°N'))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g°W'))
plt.axis('off')
plt.show()  # Display the plot

