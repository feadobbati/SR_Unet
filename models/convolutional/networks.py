import torch
import torch.nn as nn
from torch.nn.functional import relu



class UNet3D_MCD(nn.Module):
    def __init__(self, input_channels = 1, output_channels=1):
        super().__init__()
        # Encoder
        # In the encoder, convolutional layers with the Conv3d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception 
        # of the last block which does not include a max-pooling layer.
        # -------
        # input: 27 x 300 x 494
        self.e11 = nn.Conv3d(input_channels, 64, kernel_size=(2, 3, 3), padding=(1, 1, 1))  # output: 28 x 300 x 494
        self.e12 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding='same')
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # output: 14 x 75 x 124
        self.drop1 = nn.Dropout(p=0.2)

        self.e21 = nn.Conv3d(64, 128, kernel_size=3, padding='same')
        self.e22 = nn.Conv3d(128, 128, kernel_size=(3, 3, 2), padding=(1, 1, 1))  # output: 14 x 150 x 248
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # output: 7 x 38 x 62
        self.drop2 = nn.Dropout(p=0.2)

        self.e31 = nn.Conv3d(128, 256, kernel_size=(2, 2, 3), padding=(1, 1, 1))  # output: 8 x 76 x 124
        self.e32 = nn.Conv3d(256, 256, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # output: 4 x 19 x 31
        self.drop3 = nn.Dropout(p=0.2)

        self.e41 = nn.Conv3d(256, 512, kernel_size=3, padding='same')  # output: 4 x 38 x 62
        self.e42 = nn.Conv3d(512, 512, kernel_size=3, padding='same')
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(p=0.2)

        self.e51 = nn.Conv3d(512, 1024, kernel_size=3, padding='same')  # output: 2 x 17 x 31
        self.e52 = nn.Conv3d(1024, 1024, kernel_size=3, padding='same')

        # Decoder
        self.upconv1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.drop_up1 = nn.Dropout(p=0.2)
        self.d11 = nn.Conv3d(1024, 512, kernel_size=3, padding='same')
        self.d12 = nn.Conv3d(512, 512, kernel_size=3, padding='same')  # output: 4 x 38 x 62

        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)  # output: 8 x 76 x 124
        self.drop_up2 = nn.Dropout(p=0.2)
        self.d21 = nn.Conv3d(512, 256, kernel_size=(2, 2, 3), padding=(0, 0, 1))  # output: 7 x 75 x 124
        self.d22 = nn.Conv3d(256, 256, kernel_size=3, padding='same')

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.drop_up3 = nn.Dropout(p=0.2)
        self.d31 = nn.Conv3d(256, 128, kernel_size=(3, 3, 2), padding=(1, 1, 0))  # output: 14 x 150 x 247
        self.d32 = nn.Conv3d(128, 128, kernel_size=3, padding='same')

        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)  # output: 28 x 300 x 494
        self.drop_up4 = nn.Dropout(p=0.2)
        self.d41 = nn.Conv3d(128, 64, kernel_size=(2, 3, 3), padding=(0, 1, 1))  # output: 27 x 300 x 494
        self.d42 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # Output layer
        self.outconv = nn.Conv3d(64, output_channels, kernel_size=1)

        self.river_net = RiverNet_MCD(input_size=19, output_size=300*494*27)

    def forward(self, x, riv=None):
        if riv != None:
            riv_chan = self.river_net(riv).reshape((-1,) + x.shape[-3:])
            riv_chan = riv_chan.unsqueeze(1)
            x = torch.cat((x, riv_chan), dim=1)
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.drop1(self.pool1(xe12))

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.drop2(self.pool2(xe22))

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.drop3(self.pool3(xe32))

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xu2 = self.drop_up2(self.upconv2(xe42))
        xu22 = torch.cat([xu2, xe32], dim=1)

        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.drop_up3(self.upconv3(xd22))
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))
        xu4 = self.drop_up4(self.upconv4(xd32))
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        output = self.outconv(xd42)

        return output


class RiverNet_MCD(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2):
        super(RiverNet_MCD, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.initial_hidden_size = 30 

        # Calculate the size increase step
        hidden_size_step = 10 

        # Define the layers
        self.layers = nn.ModuleList()
        in_size = input_size
        hidden_size = self.initial_hidden_size
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
            if i < num_layers - 2:
                hidden_size += hidden_size_step
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.5))
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x.clone())
        return x