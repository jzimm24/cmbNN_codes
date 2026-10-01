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
    Double convolutional block. The output of the
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



# GAN1 -------------------------------------------------------------------------------------------------------------------------------------
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

# GAN2 -------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------

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

