import pytorch_lightning as pl
import sys
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cmo
import seaborn as sns
import numpy as np
from functools import reduce
import scipy.stats

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




def save_distribution_plot(data, filename='distribution_plot.png'):
    # Check if the data contains exactly 100 elements
    if len(data) != 100:
        raise ValueError("The input array must contain exactly 100 elements.")

    # Set plot size
    plt.figure(figsize=(10, 6))

    # Plot distribution using seaborn
    sns.histplot(data, bins=10, kde=True)

    # Add labels and title
    plt.title('Distribution of 100 Elements')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Save the plot to a file
    plt.savefig(filename)

    # Close the plot to free memory
    plt.close()

    print(f"Plot saved as {filename}")



def run_network(data_module:ICDataModule, main_net:str, riv_net:bool, conf:obj, output_path:str, test_path:str, weight_path:str, n_var=None, loss=None, data_path=None):

    if loss == None:
        loss = conf.training.loss
    if n_var == None:
        n_var = conf.n_var

    ds = torch.load(test_path)
    ds = np.array(ds)

    input_test = torch.tensor(ds[:, 0, :, :, :, :])
    mask = torch.tensor(input_test > 100000)

    var_name = os.path.basename(test_path).split('_')[0]
    file_names = sorted(os.listdir(os.path.join(data_path, var_name)))

    if riv == True:
        riv_test_path = os.path.join(output_path, "rivers/rivers_test.pt")
        riv_tensor = torch.load(riv_test_path)

    model = ConvModel.load_from_checkpoint(weight_path)
    model.eval()

    #for i in range(len(input_test)):

        #with torch.no_grad():
        #    no_dropout_prediction = model(input_test[i, :, :, :, :].unsqueeze(0).cuda())
        #    mask_image = mask[i, :, :, :, :]


    #### Dropout layers
    model.main_net.drop1.train()
    model.main_net.drop2.train()
    model.main_net.drop3.train()
    model.main_net.drop4.train()

    model.main_net.drop_up1.train()
    model.main_net.drop_up2.train()
    model.main_net.drop_up3.train()
    model.main_net.drop_up4.train()
    ####

    unc_gen = []
    for i in range(len(input_test)):
        single_image = []
        for _ in range(100):
            with torch.no_grad():
                input_image = input_test[i, :, :, :]#.unsqueeze(0).cuda()
                mask_image = mask[i, :, :, :]#.unsqueeze(0)
                input_image[mask_image] = 0
                river = riv_tensor[i]
                prediction = model(input_image.unsqueeze(0).cuda(), river.cuda())
                prediction = torch.squeeze(prediction, dim=1)
                prediction[mask_image] = 0
                prediction = prediction.cpu().numpy()
                single_image.append(prediction)
        single_image = np.array(single_image)
        std_im = np.std(single_image, axis=0)
        mean_im = np.mean(single_image, axis=0)
        mean_x_map = np.mean(single_image, axis=(1,2,3,4))
        np.save(f"uncertainty/mean_matrix_{file_names[i].split('.')[0]}.npy", mean_im)
        np.save(f"uncertainty/std_matrix_{file_names[i].split('.')[0]}.npy", std_im)
        print(mean_x_map.shape)
        q_norm = scipy.stats.shapiro(mean_x_map)
        print("stat", q_norm.statistic)
        print("pval", q_norm.pvalue)

        save_distribution_plot(mean_x_map, f'distribution_plot_{file_names[i].split('.')[0]}.png')

    mask = mask_image.cpu().numpy()
    np.save("mask.npy", mask)




if __name__== "__main__":
    i = 1
    conf_path = None
    output_path = None
    main_net = None
    riv = False
    loss = None
    test_path = None
    n_var = None
    stat_path = None
    data_path = None

    while i < len(sys.argv):
        if sys.argv[i] == "-cp":
            if conf_path != None: raise ValueError("Repeated input for configuration path")
            conf_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-op":
            if output_path != None: raise ValueError("Repeated input for output path")
            output_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-tep":
            test_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-trp":
            train_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-wf":
            weight_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-sp":
            stat_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-loss":
            loss = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-dp":
            data_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-net":
            if main_net != None: raise ValueError("Repeated input for network")
            main_net = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-r":
            riv = True
            i += 1
        else:
            i +=1

    if conf_path is None: raise TypeError("Missing value for configuration path")
    if output_path is None: raise TypeError("Missing value for output path")
    if main_net is None: raise TypeError("Missing value for network")

    with open(conf_path, 'r') as f:
        conf = json.load(f)
    conf = obj(conf)

    if test_path == None:
        test_path = conf.var_test_path
    if not loss:
        loss = conf.training.loss

    data_module = ICDataModule(
            train_path = train_path,
            test_path = test_path,
            river_train_path = conf.river_train_path if riv else None,
            river_test_path = conf.river_test_path if riv else None,
            batch_size = conf.training.batch_size
    )

    n_var = data_module.get_numchannels()

    print(f"[testing for dataset '{os.path.basename(test_path)}'] Starting execution")
    print(
        f"[testing for dataset '{os.path.basename(test_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
        \n[testing for variable '{os.path.basename(test_path)}'] \t-- model = {main_net} \
        \n[testing for variable '{os.path.basename(test_path)}'] \t-- loss = {loss} \
        \n[testing for variable '{os.path.basename(test_path)}'] \t-- river_info = {str(riv)} \
        \n[testing for variable '{os.path.basename(test_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    run_network(data_module=data_module, main_net=main_net, riv_net=riv, conf=conf, output_path=output_path, test_path=test_path, weight_path=weight_path, n_var=n_var, loss=loss, data_path=data_path)
    print(f"[testing for variable '{os.path.basename(test_path)}'] Ending execution")
    #plot_nparray()
