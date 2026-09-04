import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import functions as functions

# UNET --------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------

#UNET
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1,  features=[64, 128, 256, 512], 
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride =2)

        #Down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[int(idx/2)]

            ###TODO: resizing to handle wrong image sizes due to flooring in maxpool layer

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# Loss
def L1L2Loss(pred, target, l1_weight = 1, l2_weight = 1):
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    return l1_weight * l1(pred, target) + l2_weight * l2(pred, target)

# UNET - xavier initialization + sigmoid final layer (for normalization)

class UNET_normalization(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1,  features=[64, 128, 256, 512], 
    ):
        super(UNET_normalization, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride =2)

        #Down
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #Up
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.final_conv.weight, gain=0.1)   
        nn.init.zeros_(self.final_conv.bias)                        
        self.final_activation = nn.Sigmoid()

    def forward(self, x):

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[int(idx/2)]

            ###TODO: resizing to handle wrong image sizes due to flooring in maxpool layer

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        x = self.final_conv(x)

        return self.final_activation(x)

    
# GAN --------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

# Losses

loss_object = nn.BCEWithLogitsLoss()

# directly from GAN paper
def generator_loss(disc_generated_output, gen_output, target):
    # GAN loss
    gan_loss = loss_object(disc_generated_output, torch.ones_like(disc_generated_output))

    # Mean absolute error (L1 loss)
    l1_loss = F.l1_loss(gen_output, target)

    # Euclidean (L2) loss
    r = target - gen_output
    l2_loss = torch.norm(r, p=2)

    # Total generator loss
    total_gen_loss = gan_loss + 100*l1_loss + l2_loss

    return total_gen_loss

# directly from GAN paper
def discriminator_loss(disc_real_output, disc_generated_output):
    # Real loss
    real_loss = loss_object(disc_real_output, torch.ones_like(disc_real_output))

    # Generated loss
    generated_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))

    # Total discriminator loss
    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

# Generator (UNET)

# Discriminator

# class Discriminator(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         self.disc = nn.Sequential(
#             nn.Linear(in_features, 256),
#             nn.LeakyReLU(0.1),
#             nn.Linear(256, 128),
#             nn.LeakyReLU(0.1),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.disc(x)

class Downsample(nn.Module):
    def __init__(self, in_channels,out_channels, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        initializer = nn.init.normal_

        self.apply_batchnorm = apply_batchnorm

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=size, stride=1, padding='same', bias=False)
        initializer(self.conv1.weight, 0.0, 0.02)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=out_channels, kernel_size=size, stride=1, padding='same', bias=False)
        initializer(self.conv2.weight, 0.0, 0.02)

        if self.apply_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)

        if self.apply_batchnorm:
            x = self.batchnorm(x)

        x = self.leaky_relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels,out_channels ,size, apply_dropout=False,apply_batchnorm=True):
        super(Upsample, self).__init__()
        initializer = nn.init.normal_

        self.conv_transpose = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=size, stride=2, padding=1, output_padding=0, bias=False)
        initializer(self.conv_transpose.weight, 0.0, 0.02)
        self.apply_batchnorm=apply_batchnorm
        if self.apply_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels)

        self.apply_dropout = apply_dropout
        if self.apply_dropout:
            self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x)

        if self.apply_dropout:
            x = self.dropout(x)

        x = self.relu(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, nside, inly, outly):
        super(Discriminator, self).__init__()

        self.down1 = Downsample(inly+outly,4, 4, apply_batchnorm=False)  # (batch_size, 512, 512, 8)
        self.down2 = Downsample(4,8, 4)  # (batch_size, 256, 256, 8)
        self.down3 = Downsample(8,16, 4)  # (batch_size, 128, 128, 16)
        self.down4 = Downsample(16,32, 4)  # (batch_size, 64, 64, 16)

        self.zero_pad1 = nn.ZeroPad2d(1)  # Padding to ensure the dimensions match
        self.conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.zero_pad2 = nn.ZeroPad2d(1)  # Padding to ensure the dimensions match
        self.last = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=0)

    def forward(self, inp, tar):
        x = torch.cat([inp, tar], dim=1)  # (batch_size, 1024, 1024, channels*2)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.zero_pad1(x)
        x = self.conv(x)
        x = self.batchnorm1(x)
        x = self.leaky_relu(x)

        x = self.zero_pad2(x)
        x = self.last(x)

        return x

