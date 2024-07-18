import pytorch_lightning as pl
import sys
import torch
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import os
from functools import reduce

from utils.data_module import ICDataModule
from models.convolutional.conv_model import ConvModel

pl.seed_everything(0, workers=True)

accelerator = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(accelerator)

class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)



def train(data_module:ICDataModule, main_net:str, riv_net:bool, conf:obj, output_path:str, train_path:str, n_var=None, loss=None):

    if loss == None:
        loss = conf.training.loss
    if n_var == None:
        n_var = conf.n_var
    
    model = ConvModel(
                    main_net=main_net, 
                    n_dimensions=len(vars(conf.data_dim)), 
                    riv_net = riv_net, 
                    loss = loss, 
                    num_channels=n_var, 
                    riv_in_dim = conf.n_riv, 
                    riv_out_dim = reduce(lambda x, y: x * y, vars(conf.data_dim).values()),
                    lr = conf.training.lr
            )

    print("output in train", reduce(lambda x, y: x * y, vars(conf.data_dim).values()))
    print(vars(conf.data_dim))
    train_file = os.path.splitext(os.path.basename(train_path))[0]
    best_filename = f'best_{model.name}_{train_file}'

    tb_logger = loggers.TensorBoardLogger(save_dir="./")

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=conf.training.patience,
        verbose=False,
        mode="min")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,  # Save the model with the lowest validation loss
        dirpath=output_path,
        filename=best_filename
    )
    trainer = pl.Trainer(detect_anomaly=True, accelerator=accelerator, strategy="ddp_find_unused_parameters_true", log_every_n_steps=20, max_epochs=conf.training.max_epochs, callbacks=[early_stop_callback, checkpoint_callback], logger=tb_logger, check_val_every_n_epoch=1)


    trainer.fit(model, datamodule=data_module)



if __name__== "__main__":
    """File for the training of the neural network.
    It must be used after the dataset construction.
    Inputs:
        -cp (str): complete path to the configuration file, giving the parameters
        to use for the learning (num of epochs, patience...)
        -op (str): output path to save weights of the network
        -tep (str): path to the test dataset 
        -trp (str): path to the training dataset
        -loss (str): loss function, can be 'mse', 'rmse', 'perceptual'
        -net (str): can be 'srcnn', 'unet', 'unet_mcd'
        -r (bool, optional): to use if we want to include river data in the training
    """

    i = 1
    conf_path = None
    output_path = None
    main_net = None
    riv = False
    loss = None
    train_path = None
    test_path = None
    n_var = None
    
    while i < len(sys.argv): 
        if sys.argv[i] == "-cp":
            if conf_path != None: raise ValueError("Repeated input for configuration path")
            conf_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-op":
            if output_path != None: raise ValueError("Repeated input for output path")
            output_path = sys.argv[i+1]
            i += 2
        #elif sys.argv[i] == "-tep":  #maybe removed if we keep test file separated!
        #    test_path = sys.argv[i+1]
        #    i += 2
        elif sys.argv[i] == "-trp":
            train_path = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "-loss":
            loss = sys.argv[i+1]
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

    if train_path == None:
        train_path = conf.var_train_path
    #if test_path == None:
    #    test_path = conf.var_test_path
    if not loss:
        loss = conf.training.loss

    data_module = ICDataModule(
            train_path = train_path, 
            test_path = test_path, 
            river_train_path = conf.river_train_path if riv else None, 
            #river_test_path = conf.river_test_path if riv else None, 
            batch_size = conf.training.batch_size
    )

    n_var = data_module.get_numchannels()

    print(f"[training for dataset '{os.path.basename(train_path)}'] Starting execution")
    print(
        f"[training for dataset '{os.path.basename(train_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
        \n[training for variable '{os.path.basename(train_path)}'] \t-- max_epochs = {conf.training.max_epochs}\
        \n[training for variable '{os.path.basename(train_path)}'] \t-- model = {main_net} \
        \n[training for variable '{os.path.basename(train_path)}'] \t-- patience = {conf.training.patience} \
        \n[training for variable '{os.path.basename(train_path)}'] \t-- lr = {conf.training.lr} \
        \n[training for variable '{os.path.basename(train_path)}'] \t-- loss = {loss} \
        \n[training for variable '{os.path.basename(train_path)}'] \t-- river_info = {str(riv)} \
        \n[training for variable '{os.path.basename(train_path)}'] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    )
    train(data_module=data_module, main_net=main_net, riv_net=riv, conf=conf, output_path=output_path, train_path=train_path, n_var=n_var, loss=loss)
    print(f"[training for variable '{os.path.basename(train_path)}'] Ending execution")
