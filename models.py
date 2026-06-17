import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import cmbNN_codes.functions as functions

# #Hyperparameters
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE = 16
# NUM_EPOCHS = 3
# NUM_WORKERS = 1
# IM_HEIGHT = 64
# IM_WIDTH = 64

# n = 1000                                                #number of images
# filename = "../data/1k_64_multicircle_10xs20xy20"       #directory of data

# UNET --------------------------------------------------------------------------------------------------------------------------------------

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
    
# GAN --------------------------------------------------------------------------------------------------------------------------------------

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

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
    


