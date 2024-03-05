import xarray as xr
import numpy as np
import torch
import os
# import matplotlib.pyplot as plt
from tqdm import tqdm

r''' Main functions of Data Processing from the .nc type continuous time series to nxn image like discrete data.
Functions: 
- date_interval: Extracting the time window with the date as a center 
- pre_arrays: Preprocess of the xarray.Dataset object to numpy.narray. It only extract the points exist. Ignore the parts with 'Nan' data 

- discretization: Discretization of previous process. It takes the medium and mean of the data points in a corresponded grid and return 1-D arrays of tuples of coordinates,
1-D mean sea surface height value, and 1-D median sea surface height value.

- fill_missing_values:Discretization of previous process. It takes the medium and mean of the data points in a corresponded grid and return 1-D arrays of tuples of coordinates,
    1-D mean sea surface height value, and 1-D median sea surface height value.
- image_like_data: Turns the ordered discrete time series into r0xr0 grided data. 

- mask_and_filling: Fill the 'NaN' points of the image.

- data_to_image: Execute the whole data processing from continuous time series to image like discrete data.
'''
def date_interval(ds, date, int_length=5):
    r''' Extracting the time window with the date as a center 
    Args: 
        ds (xarray.Dataset): Data set of the time series 
        date (numpy.datetime64): Choosen date to be the center of the interval 
        int_length (int): time window scope 
    Return:
        ds_sel (xarray.Dataset): Selected xarray dataset 
        central_date (numpy.datetime64): Choosen date
        int_length (int): Number of day included the time window
    '''

    central_date = date
    delta_t = np.timedelta64(int_length, 'D')

    t_min = central_date - delta_t
    t_max = central_date + delta_t
    ds_sel = ds.sel(time=slice(t_min, t_max))
    return ds_sel, central_date, int_length


def pre_arrays(ds_sel):
    r'''Preprocess of the xarray.Dataset object to numpy.narray. It only extract the points exist. Ignore the parts with 'Nan' data 


    Arg:
        ds_sel (xarray.Dataset): Selected xarray.dataset
    Return:
        arr_lon (numpy.narray): 1-D array, longitude dataset the ds_sel
        arr_lat (numpy.narray): 1-D array, latitute dataset the ds_sel
        arr_ssh_obs (numpy.narray): 1-D array, Sea Surface observations of the satellites
    '''
    arr_lon = ds_sel.lon.values
    arr_lat = ds_sel.lat.values

    arr_ssh_obs = ds_sel.ssh_obs.values

    arr_lon = arr_lon[~np.isnan(arr_lon)]
    arr_lat = arr_lat[~np.isnan(arr_lat)]
    arr_ssh_obs = arr_ssh_obs[~np.isnan(arr_ssh_obs)]
    return arr_lon, arr_lat, arr_ssh_obs


def discretization(arr_lon, arr_lat, arr_ssh_obs, r0=60):
    r'''Discretization of previous process. It takes the medium and mean of the data points in a corresponded grid and return 1-D arrays of tuples of coordinates,
    1-D mean sea surface height value, and 1-D median sea surface height value.

    Arg:
        arr_lon (numpy.narray): 1-D array, longitude dataset the ds_sel
        arr_lat (numpy.narray): 1-D array, latitute dataset the ds_sel
        arr_ssh_obs (numpy.narray): 1-D array, Sea Surface height observations of the satellites
        r0 (int): How many times to dive an interval of length 1. 
    Return:
        coords (numpy.narray): 1-D array of coordinate values (longitude, latitude)
        mean_ssh (numpy.narray): 1-D array of mean sea surface height values of corresponded coordinates
        median_ssh (numpy.narray): 1-D array of median sea surface height values of corresponded coordinates
        r0 (int): values of grid, it will be used it in image_like_data function
    '''
    # how fine we want to divide an interval, for example, if r=5 we are having interval length 0.2.

    # round the number is the longitude values
    rounded_arr_lon = np.round(arr_lon * r0) / r0
    r_lon = rounded_arr_lon

    # round the number is the latitude values

    rounded_arr_lat = np.round(arr_lat * r0) / r0
    r_lat = rounded_arr_lat
    # create the coordinates, for gridded structure
    coords = np.column_stack((r_lon, r_lat))

    # order the longitude and latitude values respectively
    idx = np.lexsort((r_lat, r_lon))
    # order the coords in the above order
    coords = coords[idx]
    # take out unique values and the number of unique values
    unique_coordinates, counts = np.unique(coords, axis=0, return_counts=True)

    # ordering according the ssh values in lex order that we had previously
    ord_arr_ssh_obs = arr_ssh_obs[idx]

    # calculate mean values of ssh, for the same coordinates
    ind1 = 0
    mean_ssh = np.array([])
    for i in range(len(counts)):
        c = counts[i]
        m_vls = np.ones(c) * np.mean(ord_arr_ssh_obs[ind1:ind1 + c])
        mean_ssh = np.append(mean_ssh, m_vls)
        ind1 += c

    # calculate median values of ssh, for the same coordinates
    ind0 = 0
    median_ssh = np.array([])
    for i in range(len(counts)):
        c = counts[i]
        m_vls = np.ones(c) * np.median(ord_arr_ssh_obs[ind0:ind0 + c])
        median_ssh = np.append(median_ssh, m_vls)
        ind0 += c
    return coords, mean_ssh, median_ssh, r0


def fill_missing_values(coords, mean_ssh, r, rounding=3):
    r'''It take the coordinate and mean sea surgace height values, round them up to third decimal and fill the empty coordinate values. This is for specific region on the ocean which is
    between (295,305) longitude and (33,43) latitude, one part of the Gulf stream. 
    Arg:
        coords (numpy.narray): 1-D array of coordinate values (longitude, latitude)
        mean_ssh (numpy.narray): 1-D array of mean sea surface height values of corresponded coordinates
        r (int): How many times to dive an interval of length 1.
        rounding (int): Rounding the numerical values up to a decimal.
    Return:
        total_data_ordered (numpy.array): totally ordered 1-D discrete time series. 
    '''



    coords = np.around(coords, decimals=rounding + 1)
    # create a np list with coords and mean_ssh
    mean_data = np.column_stack((coords, mean_ssh))
    mean_data_copy = mean_data.copy()

    unique, indeces = np.unique(mean_data_copy, axis=0, return_index=True)
    sorted_indices = np.argsort(indeces)
    unique_mean_data = unique[sorted_indices]

    lonmin = 295.
    lonmax = 305.
    dx = 1 / r

    latmin = 33.
    latmax = 43.
    dy = 1 / r
    # grid values for lon and lat. 1/60 is smt we wouldn't like !! I need to find a way for that
    glon = np.arange(lonmin, lonmax + dx, dx)
    glat = np.arange(latmin, latmax + dy, dy)
    # there is a numerical error about the lenght of the list, so I am just extracting the last values which is 43.02
    # when r = 50

    # Creating the grid
    X, Y = np.meshgrid(glon, glat)

    # Creating the set for the pseudo gridded structure
    glist = np.column_stack((X.ravel(), Y.ravel()))

    glist = [tuple((np.around(row[0], decimals=rounding), np.around(row[1], decimals=rounding))) for row in
             glist.tolist()]
    gset = set(glist)

    # set for the original coordinates
    original_coords_list = unique_mean_data[:, :2].tolist()
    original_coords_tuples = [tuple((round(row[0], rounding), round(row[1], rounding))) for row in original_coords_list]
    original_coords_set = set(original_coords_tuples)

    dif_set = gset.difference(original_coords_set)

    dif_list = list(dif_set)
    dif_arr = np.array(dif_list)

    nan_ssh = np.ones(len(dif_arr)) * np.nan

    nan_data = np.column_stack((dif_arr, nan_ssh))

    total_data = np.concatenate((unique_mean_data, nan_data))

    idxx = np.lexsort((total_data[:, 1], total_data[:, 0]))

    total_data_ordered = total_data[idxx]

    return total_data_ordered


# total_data_ordered = fill_missing_values(coords,mean_ssh,r0)


def image_like_data(total_data_ordered, r0, rounding=3):
    r'''Turns the ordered discrete time series into r0xr0 grided data. 
    '''

    arr = total_data_ordered.copy()

    min_lon, max_lon = np.min(arr[:, 0]), np.max(arr[:, 0])
    min_lat, max_lat = np.min(arr[:, 1]), np.max(arr[:, 1])

    grid_size = r0 * 10 + 1
    #     grid_size = 401
    lon_res = (max_lon - min_lon) / (grid_size - 1)
    lon_res = np.around(lon_res, decimals=rounding)
    lat_res = (max_lat - min_lat) / (grid_size - 1)
    lat_res = np.around(lat_res, decimals=rounding)

    # create empty image
    D = np.ones((grid_size, grid_size)) * np.nan

    for i in range(arr.shape[0]):
        lon, lat, ssh = arr[i]
        x = int((lon - min_lon) / lon_res)
        y = int((lat - min_lat) / lat_res)
        if y == 501:
            continue
        D[y, x] = ssh
    image = D
    return image, grid_size


def mask_and_filling(image, filling=0):
    r'''Fill the 'NaN' points of the image
    Args:
        image (numpy.narray): 2-D array (an image)
        filling (float64): desired values of filling.
    Return:
        image1 (numpy.narray): 2-D filled image data
        mask (numpy.narray): 2-D coordinates of 'NaN' values
    '''
    image1 = image.copy()

    image1[np.isnan(image1)] = filling
    mask = np.where(image1 == filling, 0, 1)
    return image1, mask


def data_to_image(ds, date, int_length=5, r0=60, rounding=3, filling=0):
    r'''Execute the whole data processing from continuous time series to image like discrete data. 
    Arg:
        ds (xarray.Dataset): Data set of the time series 
        date (numpy.datetime64): Choosen date to be the center of the interval 
        int_length (int): time window scope 
        r0 (int): How many times to dive an interval of length 1.
        rounding (int): Rounding the numerical values up to a decimal.
        filling (float64): desired values of filling.
    Return: 
        image_filled (numpy.narray): 2-D filled image data
        mask (numpy.narray): 2-D coordinates of 'NaN' values
        central_date (numpy.datetime64): Choosen date
        grid_size (int): r0x10 x r0x10 
        image_data: 2-D Non filled image data.
    '''


    ds_sel, central_date, interval = date_interval(ds, date, int_length)

    # preprocessing the data
    arr_lon, arr_lat, arr_ssh_obs = pre_arrays(ds_sel)

    filling_ssh = np.nanmean(arr_ssh_obs)

    # obtaining mean_ssh for the gridded structure
    coords, mean_ssh, median_ssh, r = discretization(arr_lon, arr_lat, arr_ssh_obs, r0)

    # fillin the values with 'nan'
    total_data_ordered = fill_missing_values(coords, mean_ssh, r, rounding)

    # image-like data and grid_size
    image_data, grid_size = image_like_data(total_data_ordered, r, rounding)

    # obtaining the mask
    image_filled, mask = mask_and_filling(image_data, filling)

    return image_filled, mask, central_date, grid_size, image_data


if __name__ == '__main__':
    path_main = '/Users/afsinozdemir/Desktop/Ocean_Data_Challenge/Ocean_Data_SSH_Notebooks/dc_obs/2020a_SSH_mapping_NATL60_karin_swot.nc'

    ds = xr.open_dataset(path_main)

    path_np = '../datasets/np_array'
    path_torch = '../datasets/torch_tensor'
    paths = [path_np, path_torch]
    for path in paths:
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        else:
            print('Path already exists')
    date0 = np.datetime64('2012-10-13')

    day_interval = 5
    r = 60
    steps = 11
    rounding = 6
    # image_sel, mask, central_date, grid_size, image_data = data_to_image(ds, date0, day_interval, r,rounding)

    for i in tqdm(range(0, 20)):
        date = date0 + i * steps
        # image_sel, mask, central_date, grid_size, image_data = data_to_image(ds, date, day_interval, r, rounding)
        file_name = f'datasets/np_array/{date}_image_data.npy'
        isExist_file = os.path.exists(file_name)
        if not isExist_file:
            image_sel, mask, central_date, grid_size, image_data = data_to_image(ds, date, day_interval, r, rounding)
            np.save(file_name, image_data)
            image_data_tensor = torch.from_numpy(image_data)
            torch.save(image_data_tensor, f'datasets/torch_tensor/{date}_image_data.pt')
        # np.save(f'../../datasets/data_numpy/mask_dataset/{date}_mask.pt',mask)
