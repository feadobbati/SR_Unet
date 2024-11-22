import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset
from typing import Optional

def make_shape_even(image_tensor):
    '''
    Utility to use when either height or width of images is not even and the architecture works just for images with even shape values.
    '''
    channel, original_height, original_width = image_tensor.shape

    new_height = original_height + 1 if original_height % 2 != 0 else original_height
    new_width = original_width + 1 if original_width % 2 != 0 else original_width


    increased_image_tensor = torch.zeros(channel, new_height, new_width)

    increased_image_tensor[:, :original_height, :original_width] = image_tensor
    increased_image_tensor[:, :original_height, -1] = image_tensor[:, :, -1]
    increased_image_tensor[:, -1, :original_width] = image_tensor[:, -1, :]
    increased_image_tensor[:, -1, -1] = increased_image_tensor[:, -2, -2]

    return increased_image_tensor


class ICDataset(Dataset):
    '''
    Data coming from this custom dataset are returned as:
    (raw_data, river_data, target_value)

    '''
    def __init__(self, tensor_dataset, rivers_dataset=None, resize_to_even = False):
        self.ds = tensor_dataset
        self.rivers = rivers_dataset
        self.resize_to_even = resize_to_even

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x_sample, y_sample = self.ds[idx]
        if self.resize_to_even and (x_sample.shape[-2] % 2 != 0 or x_sample.shape[-1] % 2 != 0):
            x_sample = make_shape_even(x_sample)
            y_sample = make_shape_even(y_sample)
        if self.rivers != None:
            river_sample = self.rivers[idx]
            return x_sample, river_sample, y_sample
        return x_sample, y_sample


class ICDataModule(pl.LightningDataModule):

    def __init__(self, train_path:str, test_path:str, river_train_path:Optional[str]=None, river_test_path:Optional[str]=None, batch_size:int=32, resize_to_even:bool=False):
        super(ICDataModule, self).__init__()
        self.train_path=train_path
        self.test_path=test_path
        self.river_train_path=river_train_path
        self.river_test_path=river_test_path
        self.batch_size=batch_size
        self.resize_to_even = resize_to_even

    def setup(self, stage:str):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            #load the dataset
            tensor_ds = torch.load(self.train_path)
            if self.river_train_path != None:
                rivers_ds = torch.load(self.river_train_path)
            else:
                rivers_ds= None
            ds = ICDataset(tensor_dataset=tensor_ds, rivers_dataset=rivers_ds, resize_to_even=self.resize_to_even)
            #split in train and validation sets
            train_size = int(0.9 * len(tensor_ds))
            val_size = len(tensor_ds) - train_size
            self.train_ds, self.val_ds = random_split(ds, [train_size, val_size])

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            test_ds = torch.load(self.test_path)
            if self.river_test_path != None:
                rivers_ds = torch.load(self.river_test_path)
            else:
                rivers_ds = None
            self.test_ds = ICDataset(tensor_dataset=test_ds, rivers_dataset=rivers_ds, resize_to_even=self.resize_to_even)


    def get_numchannels(self):
        test_dataset = torch.load(self.test_path)
        test_ds = ICDataset(tensor_dataset=test_dataset)
        return np.array(test_ds).shape[2]


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds,
                                           batch_size=self.batch_size,
                                           num_workers=4,
                                           pin_memory=True,
                                           shuffle = True
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds,
                                           batch_size=self.batch_size,
                                           num_workers=4,
                                           pin_memory=True,
                                           shuffle = False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds,
                                           batch_size=1,
                                           num_workers=4,
                                           pin_memory=True,
                                           shuffle = False)
