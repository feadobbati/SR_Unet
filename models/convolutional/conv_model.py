import torch
from torch import nn
import pytorch_lightning as pl
from models.convolutional.losses import Masked_MSELoss, Masked_RMSELoss, VGGPerceptualLoss, masked_psnr, masked_ssim, masked_rmse
from models.convolutional.networks import UNet3D_MCD, RiverNet_MCD


class ConvModel(pl.LightningModule):
    '''
        Loss function can be either rmse, mse, perceptual.
        The number of channels must consider just the variables (i.e., not the river channel)
    '''
    def __init__(self, main_net, n_dimensions, riv_net = False, loss = 'rmse', num_channels=1, lr = 1e-3, stats = None):
        super(ConvModel, self).__init__()

        self.save_hyperparameters()
        input_channels = num_channels + 1 if riv_net else num_channels

        self.main_net = UNet3D_MCD(input_channels = input_channels, output_channels = num_channels)

        if loss == 'mse':
            self.loss = Masked_MSELoss()
        elif loss == 'rmse':
            self.loss = Masked_RMSELoss()
        elif loss == 'perceptual':
            self.loss = VGGPerceptualLoss(n_dimensions=n_dimensions)
        else:
            raise ValueError('Invalid argument for the loss function')

        # MC Dropout added - to decide whether to keep
        self.river_net = True if riv_net else None
        self.name = f"conv_model_{main_net}_{loss}"
        self.n_dimensions = n_dimensions
        self.lr = lr
        self.stats = stats

    def forward(self, x, riv=None, riv_mask=None):
        x = self.main_net(x, riv)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters())
        return optimizer

    def training_step(self, train_batch, batch_idx):

        if self.river_net is not None:
            x, riv, y = train_batch
        else:
            x, y = train_batch

        mask = (x > 10e3).detach()
        x[mask] = 0
        x = x.detach()

        if self.river_net is not None:
            riv_mask = mask[:, 0:1, :, :].detach() #extract mask of shape (bs, 1, h, w)
            pred = self.forward(x, riv, riv_mask)
        else:
            pred = self.forward(x)

        loss = self.loss(pred, y, mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        if self.river_net is not None:
            x, riv, y = val_batch
        else:
            x, y = val_batch

        mask = (x > 10e3)
        x[mask] = 0

        if self.river_net is not None:
            riv_mask = mask[:, 0:1, :, :] #extract mask of shape (bs, 1, h, w)
            pred = self.forward(x, riv, riv_mask)
        else:
            pred = self.forward(x)

        loss = self.loss(pred, y, mask)
        psnr_score = masked_psnr(pred, y, mask)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_psnr', psnr_score, sync_dist=True)


    def test_step(self, test_batch, batch_idx):

        if self.river_net is not None:
            x, riv, y = test_batch
        else:
            x, y = test_batch

        mask = (x > 10e4)
        x[mask] = 0

        if self.river_net is not None:
            riv_mask = mask[:, 0:1, :, :] #extract mask of shape (bs, 1, h, w)
            pred = self.forward(x, riv, riv_mask)
        else:
            pred = self.forward(x)

        loss = self.loss(pred, y, mask)
        psnr_score = masked_psnr(pred, y, mask)
        ssim_score = masked_ssim(pred, y, mask)
        if self.stats is not None:
            rmse_score = masked_rmse(pred, y, mask, self.stats)
            self.log('test_rmse', rmse_score, sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_psnr', psnr_score, sync_dist=True)
        self.log('test_ssim', ssim_score, sync_dist=True)
