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
    """
    Double Convolutional layer as main building block of UNET.
    Convolutional layers are sublimented by BatchNorms and ReLUs.
    """
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

# ResNet --------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Double convolutional block. The output of the downsampling is summed with the initial input to create a residual arcitecture.
    """
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        return self.relu(out+identity)
    
def make_res_stage(in_channels, out_channels, num_blocks):
    layers = [ResBlock(in_channels, out_channels)]
    for _ in range(num_blocks - 1):
        layers.append(ResBlock(out_channels, out_channels))
    return nn.Sequential(*layers)
    
class ResUNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features = [64, 128, 256, 512, 1024], blocks_per_level = 4):
        super(ResUNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down
        for feature in features:
            self.downs.append(make_res_stage(in_channels, feature, blocks_per_level))
            in_channels = feature

        # Up

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(make_res_stage(feature*2, feature, blocks_per_level))

        self.bottleneck = make_res_stage(features[-1], features[-1] * 2, blocks_per_level)
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
            skip_connection = skip_connections[idx // 2]

            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
    
        return self.final_conv(x)

# GAN2 -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

# The Gan class does not inherit from nn.Module. Therefor additional features are added below 

def downsample_block(in_channels, out_channels):
     return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1), 
                          nn.BatchNorm2d(out_channels), 
                          nn.LeakyReLU(0.2, inplace = True)
                          )

class PairClassifier(nn.Module):
    def __init__(self, in_channels = 1, conv_channels = 16, depth = 5):
        """
        conv_channels (int): number of channels created by every Conv2d
        n_blocks (int): Number of convolution blocks before regressing to predicting scalar for each image
        """
        super().__init__()
        
        channels = [conv_channels * (2**i) for i in range(depth)]
        blocks = []
        previous_channels = 2*in_channels

        for c in channels:
            blocks.append(downsample_block(previous_channels, c))
            previous_channels = c
            self.downsample = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(previous_channels, 1)

    def forward(self, img_a, img_b):
        x = torch.cat([img_a, img_b], dim = 1)
        x = self.downsample(x)
        x = self.pool(x).flatten(1)
        
        return self.classifier(x)
    
class Gan():
    def __init__(self, in_channels=1, discriminator_conv_channels = 32, 
                 discriminator_depth = 5, lambda_recon = 100.0, lr = 2e-4, betas = (0.5, 0.999), device = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = UNET(in_channels, in_channels).to(self.device)
        self.discriminator = PairClassifier(in_channels, discriminator_conv_channels, discriminator_depth).to(self.device)

        self.lambda_recon = lambda_recon

        self.recon_loss_fn = nn.L1Loss()
        self.adv_loss_fn = nn.BCEWithLogitsLoss()

        self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

    def training_step(self, noisy_imgs, groundtruth_imgs):
        real_label = torch.ones(noisy_imgs.size(0), 1, device=self.device)
        fake_label = torch.zeros(noisy_imgs.size(0), 1, device=self.device)

        predictions = self.generator(noisy_imgs)

        self.opt_D.zero_grad()

        pred_real = self.discriminator(noisy_imgs, groundtruth_imgs)
        loss_D_real = self.adv_loss_fn(pred_real, real_label)

        pred_fake = self.discriminator(noisy_imgs, predictions.detach())
        loss_D_fake = self.adv_loss_fn(pred_fake, fake_label)

        loss_D = 0.5 * (loss_D_real + loss_D_fake)
        loss_D.backward()
        self.opt_D.step()

        self.opt_G.zero_grad()

        loss_recon = self.recon_loss_fn(predictions, groundtruth_imgs)

        pred_fake_G = self.discriminator(noisy_imgs, predictions)
        loss_adv = self.adv_loss_fn(pred_fake_G, real_label)

        loss_G = self.lambda_recon * loss_recon + loss_adv
        loss_G.backward()
        self.opt_G.step()

        return{
            "loss_D": loss_D.item(),
            "loss_G": loss_G.item(),
            "loss_recon": loss_recon.item(),
            "loss_adv": loss_adv.item()
        }
    
    def train(self):
        self.generator.train()
        self.discriminator.train()
        return self

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
        return self
    
    def state_dict(self):
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
        }

    def load_state_dict(self, state):
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.opt_G.load_state_dict(state["opt_G"])
        self.opt_D.load_state_dict(state["opt_D"])

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        self.device = device
        return self

