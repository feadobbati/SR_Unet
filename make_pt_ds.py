import netCDF4 as nc
import torch
import numpy as np
import os
import sys
from torch.utils.data import TensorDataset
from alive_progress import alive_bar
from typing import List, Dict
import json


def get_xy_single_var(data_path:str, cms2ogs_map:Dict[str, str], cms_name: str, is_test:bool):
    '''
        Returns the input and target values corresponding to a given variable in the data folder.

        Args:
            data_path (str): path to the data directory
            cms2ogs_map: dictionary associating Copernicus Marine names to CADEAU names
            cms_name (str): variable name in the Copernicus Marine notation.
            is_test (bool): True if we are considering test dataset, False otherwise
        Returns:
            x_arr (numpy array): input of the DL model for the given variable
            y_arr (numpy array): target of the DL model for the given variable
    '''
    ogs_name = cms2ogs_map[cms_name]

    main_folder_cms = "nc_iCMS_test" if is_test else "iCMS_nc"
    main_folder_ogs = "nc_OGS_test" if is_test else "NARF_nc"

    cms_path = os.path.join(data_path, main_folder_cms, cms_name)
    ogs_path = os.path.join(data_path, main_folder_ogs, ogs_name)

    x_list = []; y_list=[]

    cms_filenames = sorted(os.listdir(cms_path))
    ogs_filenames = sorted(os.listdir(ogs_path))

    ds_type = "test" if is_test else "train"

    with alive_bar(len(cms_filenames), title=f"Processing CMS {ds_type} data for {cms_name}...") as bar:
        for filename in cms_filenames:

            file_path = os.path.join(cms_path, filename)
            file_ds = nc.Dataset(file_path)
            x_list.append(file_ds[cms_name][:].data)
            bar()

    with alive_bar(len(ogs_filenames), title=f"Processing OGS {ds_type} data for {ogs_name}...") as bar:
        for filename in ogs_filenames:

            file_path: str = os.path.join(ogs_path, filename)
            file_ds = nc.Dataset(file_path)
            y_list.append(file_ds[ogs_name][:].data)
            bar()
    return np.array(x_list), np.array(y_list)


def get_river_vector(data_path:str, is_test:bool):
    """data_path need to have inside a 'rivers' directory, with vector and vector_test
    subdirectory including files with normalized vectors of river data flow-rates
    """
    river_path = os.path.join(data_path, "rivers", "vector_test" if is_test else "vector")
    x_list = []

    cms_filenames = sorted(os.listdir(river_path))

    ds_type = "test" if is_test else "train"

    with alive_bar(len(cms_filenames), title=f"Processing river {ds_type} data ...") as bar:
        for filename in cms_filenames:
            file_path = os.path.join(river_path, filename)
            x=np.loadtxt(file_path)
            x_list.append(x)
            bar()
    return np.array(x_list)


def make_rivers_dataset(data_path:str, train_only: bool = False, test_only:bool = False):
    """Construct river training and test torch dataset.
    """

    if not test_only:
        rivers_train = get_river_vector(data_path, is_test=False)
        rivers_train_save_path = os.path.join(data_path, "rivers", "rivers_train.pt")
        print("Saving the pytorch river datasets...")
        torch.save(torch.Tensor(rivers_train), rivers_train_save_path)
        print("Saved!")
    if not train_only:
        rivers_test = get_river_vector(data_path, is_test=True)
        rivers_test_save_path = os.path.join(data_path, "rivers", "rivers_test.pt")
        print("Saving the pytorch river datasets...")
        torch.save(torch.Tensor(rivers_test), rivers_test_save_path)
        print("Saved!")


def get_mean_std(ds:np.array, mask:np.array):
    """Compute mean and standard deviation for the dataset ds.
    """

    mask = np.repeat(mask, ds.shape[0], axis=0)
    ds=np.ma.masked_array(ds, mask)
    data_shape = ds.shape[2:]
    axis = (0,) + tuple(range(2, 2+len(data_shape)))
    var_means = np.ma.mean(ds, axis=axis)
    var_stds = np.ma.std(ds, axis=axis)
    return var_means, var_stds

def normalize(ds:np.array, means:np.array, stds:np.array, mask:np.array):
    """Compute normalization of the dataset, given its mean
       and standard deviation.
    """

    mask = np.repeat(mask, ds.shape[0], axis=0)
    ds = np.ma.masked_array(ds, mask)
    normalized_ds = np.zeros_like(ds, dtype=np.float32)
    for i in range(ds.shape[0]):  # Iterate over each data sample
        for v in range(ds.shape[1]):  # Iterate over each channel
            if ds.ndim == 4:  # Check if the image is 2D or 3D
                normalized_ds[i, v, :, :] = np.ma.masked_invalid((ds[i, v, :, :] - means[v]) / stds[v]).filled(10e6)
            else:
                normalized_ds[i, v, :, :, :] = np.ma.masked_invalid((ds[i, v, :, :, :] - means[v]) / stds[v]).filled(10e6)
    return normalized_ds.data


def make_var_dataset(data_path:str, var_list:List[str], cms2ogs_map:Dict[str, str], stat:bool = True, train_only: bool = False, test_only:bool = False):
    '''
        Saves the Pytorch dataset corresponding to a set of given variables in the data folder.

        Args:
            data_path (str): inside the path must be a directory with 3D data. In addition, there may be 'statistics' directory, with mean and standard deviations
            of the datasets.
            var_list (List[str]): list of variable names in the Copernicus Marine notation.
            cms2ogs_map: dictionary associating to Copernicus Marine names CADEAU names
            stat (opt, bool): True (default) if we have already computed means and standard deviations for the variables, False if we need to do it
    '''

    cms_im_train = []
    ogs_im_train = []
    cms_im_test = []
    ogs_im_test = []

    x_means = []
    x_stds = []
    y_means = []
    y_stds = []

    for var in var_list:
        if not test_only:
            varx_list, vary_list = get_xy_single_var(data_path, cms2ogs_map, var, False)
            cms_im_train.append(varx_list)
            ogs_im_train.append(vary_list)
        if not train_only:
            varx_list, vary_list = get_xy_single_var(data_path, cms2ogs_map, var, True)
            cms_im_test.append(varx_list)
            ogs_im_test.append(vary_list)
        if stat:
            with open(f'{data_path}/statistics/cms/stat_cms_{var}.txt', 'r') as file:
                line_elements = []
                for line in file:
                    if line.strip():
                        line_elements.append(np.float32(line))
            x_means.append(line_elements[0])
            x_stds.append(line_elements[1])


            with open(f'{data_path}/statistics/ogs/stat_ogs_{var}.txt', 'r') as file:
                line_elements = []
                for line in file:
                    if line.strip():
                        line_elements.append(np.float32(line))
            y_means.append(line_elements[0])
            y_stds.append(line_elements[1])


    x_means = np.array(x_means)
    x_stds = np.array(x_stds)
    y_means = np.array(y_means)
    y_stds = np.array(y_stds)

    cms_im_train = list(zip(*cms_im_train))
    ogs_im_train = list(zip(*ogs_im_train))
    cms_im_test = list(zip(*cms_im_test))
    ogs_im_test = list(zip(*ogs_im_test))

    x_train = np.array(cms_im_train)
    y_train = np.array(ogs_im_train)
    x_test = np.array(cms_im_test)
    y_test = np.array(ogs_im_test)

    mask = x_train[0] > 100000

    if not stat:
        x_full = np.concatenate((x_train, x_test), axis=0)
        y_full = np.concatenate((y_train, y_test), axis=0)
        x_means, x_stds = get_mean_std(x_full, mask)
        y_means, y_stds = get_mean_std(y_full, mask)

    if not test_only:
        x_train = normalize(x_train, x_means, x_stds, mask)
        y_train = normalize(y_train, y_means, y_stds, mask)
        train_torch_ds = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    if not train_only:
        x_test = normalize(x_test, x_means, x_stds, mask)
        y_test = normalize(y_test, y_means, y_stds, mask)
        test_torch_ds = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    if len(var_list) == len(cms2ogs_map):
        name = "all_"
    else:
        name = ""
        for var in var_list:
            name = name + f"{var}_"

    dest_path = data_path

    if not test_only:
        train_save_path = os.path.join(dest_path, name + "train_dataset.pt")
        print("Saving the pytorch datasets...")
        torch.save(train_torch_ds, train_save_path)
        print("Saved!")
    if not train_only:
        test_save_path =  os.path.join(dest_path, name + "test_dataset.pt")
        print("Saving the pytorch datasets...")
        torch.save(test_torch_ds, test_save_path)
        print("Saved!")



def help():
    print("Script to produce the torch dataset")
    print("")
    print("Example of usage:")
    print("python make_pt_ds.py -dp /data/NASea_data -v no3 po4")
    print("In addition you can add -stat if you previously computed avg and std")
    print("-r if you want to create river dataset")
    sys.exit(0)


if __name__== "__main__":
    """Script to construct the training and the test dataset
    (after that test and training data have already been divided).
    Inputs:
        -dp (str): data path for training and test. 

        -stat (bool, optional): flag denoting the presence inside the data path of a
        directory with mean and variance of each variable, both for Copernicus and OGS
        dataset (cfr compute_avgstd.py)

        -v (list): list of variables that we want to put inside THE SAME dataset file.

        -r (bool, optional): to use for the construction of river dataset

    """
    i = 1
    var_list = []
    data_path = None
    rivers = False
    read_stat = False
    train_only = False
    test_only = False 

    while i < len(sys.argv):
        if sys.argv[i] == "-dp":
            if data_path != None: raise ValueError("Repeated input for data path")
            data_path = sys.argv[i+1]; i+= 2
        elif sys.argv[i] == "-stat": # To use if we saved means and stds on files
            if read_stat: raise ValueError("Repeated input for stat")
            else:
                read_stat = True
                i += 1
        elif sys.argv[i] == "-v":
            if var_list != []: raise ValueError("Repeated input for variable")
            if sys.argv[i+1] == "all":
                var_list = "all"
                i+= 2
            else:
                while i < len(sys.argv)-1:
                    if not sys.argv[i+1].startswith('-'):
                        var_list.append(sys.argv[i+1]) ; i+= 1
                    else:
                        break
                i+= 1
        elif sys.argv[i] == "-r":
            rivers = True; i+=1
        elif sys.argv[i] == "-train_only":
            train_only = True; i+= 1
        elif sys.argv[i] == "-test_only":
            test_only = True; i+= 1
        elif sys.argv[i] == "-h" or sys.argv[i] == "--help":
            help()
            i+=1
        else:
            i+=1

    if data_path is None:
        help()
    if var_list == []:
        print("[WARNING] No variables provided with -v. No variable dataset will be created.")

    map_path = os.path.join(data_path, "cms2ogs.json") # dictionary that assign CADEAU name of variables to Copernicus Marine names
    with open(map_path, 'r') as f:
        cms2ogs_map = json.load(f)
    if var_list == "all":
        var_list = list(cms2ogs_map.keys())  # ??

    print(f"[make_dataset for variables {var_list}] Starting execution")
    if var_list != []:
        make_var_dataset(data_path, var_list, cms2ogs_map, read_stat, train_only, test_only)
    if rivers:
        make_rivers_dataset(data_path, train_only, test_only)
    print(f"[make_dataset for variable {var_list}] Ending execution")
    