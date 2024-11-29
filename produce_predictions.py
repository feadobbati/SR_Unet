import pytorch_lightning as pl
import sys
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os
from functools import reduce
import netCDF4 as nc
import numpy.ma as ma
import numpy as np

sys.path.insert(0, "./..")
from utils.data_module import ICDataModule
from models.convolutional.conv_model import ConvModel

pl.seed_everything(0, workers=True)

accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(accelerator)

class obj(object):
    def __init__(self, d):
        self._len_d = len(d)
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)

    def __len__(self):
        return self._len_d


def new_ncFile(mask_tensor, prediction, image_file, var):
    """ Create the netCDF file associated to the prediction of a given variable

        Args:
            mask (torch tensor)
            prediction data
            name of the file (str)
            variable (str): name of the variable in CMS notation

    """
    destination = prediction_dir + "/" + var + "/"
    filename = os.path.basename(image_file)
    nc_file = nc.Dataset(os.path.join(data_path, "original", "nc_iCMS_test", var, filename))
    #nc_file = nc.Dataset(data_path + "nc_iCMS_test/" + var + "/" + filename)
    reference_name = var + filename[len(var):].replace("-", "_")

    if not os.path.exists(destination):
        os.makedirs(destination)

    new_file = nc.Dataset(os.path.join(destination, reference_name), "w", format="NETCDF4")
    print(os.path.join(destination, reference_name))
    lat_variable = nc_file.variables['latitude']
    lon_variable = nc_file.variables['longitude']
    dep_variable = nc_file.variables['depth']

    # Get the lon-lat values
    lat_values = lat_variable[:]
    lon_values = lon_variable[:]
    dep_values = dep_variable[:]

    new_file.createDimension("longitude", len(lon_values))
    new_file.createDimension("latitude", len(lat_values))
    new_file.createDimension("depth", len(dep_values))

    # Create latitude and longitude variables in the new file
    dest_longitudes = new_file.createVariable("longitude", lon_values.dtype, ("longitude",))
    dest_latitudes = new_file.createVariable("latitude", lat_values.dtype, ("latitude",))
    dest_depth = new_file.createVariable("depth", dep_values.dtype, ("depth",))

    # Write the new latitude and longitude values
    dest_latitudes[:] = lat_values
    dest_longitudes[:] = lon_values
    dest_depth[:] = dep_values

    # Create a variable in the new file and write the interpolated array
    dest_array = new_file.createVariable(var, nc_file[var][:].dtype, ("depth", "latitude", "longitude"))
    prediction = prediction.cpu().detach().numpy()
    mask_tensor = mask_tensor.numpy()
    #print("p", prediction)
    #print("m", nc_file[var].data)
    masked_prediction = ma.masked_array(prediction, mask=nc_file[var][:].mask)#mask=mask_tensor)
    #print("shape masked pred", masked_prediction.shape)
    dest_array[:] = masked_prediction
    nc_file.close()

def predict(prediction_dir, test_path, weight_file, data_path, var, riv):

    x_means = 0
    x_stds = 0
    ds = torch.load(test_path)
    ds = np.array(ds)

    input_test = torch.tensor(ds[:, 0, :, :, :, :])
    mask = torch.tensor(input_test > 1000000)

    file_names = []
    for v in var:
        filenames = sorted(os.listdir(os.path.join(data_path, "original", "nc_iCMS_test", v)))
        file_names.append(filenames)

    if riv == True:
        riv_test_path = os.path.join(data_path, "rivers/rivers_test.pt")
        riv_tensor = torch.load(riv_test_path)

    model = ConvModel.load_from_checkpoint(weight_file)
    model.eval()

    stats = []
    for i in range(len(var)):
        with open(f'{data_path}/statistics/ogs/stat_ogs_{var[i]}.txt', 'r') as file:
            line_elements = []
            for line in file:
                if line.strip():
                    line_elements.append(np.float32(line))
        x_means = line_elements[0]
        x_stds = line_elements[1]
        stats.append([x_means, x_stds])

    for i in range(len(input_test)):
        input_image = input_test[i, :, :, :]#.unsqueeze(0).cuda()
        mask_image = mask[i, :, :, :]#.unsqueeze(0)
        input_image[mask_image] = 0
        #print("input_image", torch.max(input_image))
        if riv == True:
            river = riv_tensor[i]
            prediction = model(input_image.unsqueeze(0).cuda(), river.cuda(0))
        else:
            prediction = model(input_image.unsqueeze(0).cuda())
        #print("prediction", torch.max(prediction))

        for v in range(len(var)):
            pred_var = (prediction[:, v, :, :, :] * stats[v][1]) + stats[v][0]

            filename = file_names[v][i]

            new_ncFile(mask_image.squeeze(), pred_var.squeeze(), filename, var[v])



if __name__== "__main__":
    i = 1
    weight_file = None
    data_path = None
    main_net = None
    riv = False
    var = []
    #loss = None
    test_path = None
    n_var = None

    while i < len(sys.argv):
        if sys.argv[i] == "-wf":
            if weight_file != None: raise ValueError("Repeated input for configuration path")
            weight_file = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-dp":
            if data_path != None: raise ValueError("Repeated input for output path")
            data_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-tep":
            test_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-v":
            while i < len(sys.argv)-1:
                if not sys.argv[i+1].startswith('-'):
                    var.append(sys.argv[i+1]) ; i+= 1
                else:
                    break
            i+= 1

        #elif sys.argv[i] == "-loss":
        #    loss = sys.argv[i+1]
        #    i += 2
        elif sys.argv[i] == "-net":
            if main_net != None: raise ValueError("Repeated input for network")
            main_net = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-r":
            riv = True
            i += 1
        else:
            i +=1
    print("var", var)
    if main_net is None: raise TypeError("Missing value for network")

    prediction_dir = os.path.join(data_path, "original", "predictions")
    if not os.path.exists(prediction_dir):
        os.mkdir(prediction_dir)

    predict(prediction_dir, test_path, weight_file, data_path, var, riv)
