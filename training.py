#training architeture broadly taken from https://github.com/Obasho10/cmbnn/blob/main/train/train.py - Obasho M.


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import functions as functions
import models as models

# ---------------------------------------------------------------------------------------------------------------------------------

class DataFileFormatError(ValueError):
    pass

class ModelArchitectureError(ValueError):
    pass

# ---------------------------------------------------------------------------------------------------------------------------------

model = "UNET"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3
BATCH_SIZE = 16
LR = 1e-4
IMG_DIM = 128
DATA_FILE = "/home/user/Physik_Bonn/master_thesis/codes/julius_master_thesis/data/errorDetection.npz"
TRAIN_SPLIT = 1-0.05

# no .pth
trained_model_name = "../models/easyData2"

#crit = nn.BCEWithLogitsLoss() 
crit = models.L1L2Loss

# General ---------------------------------------------------------------------------------------------------------------------------------

def log_memory_usage():
    #!!! Taken from https://github.com/Obasho10/cmbnn/blob/main/train/train.py
    # Log memory usage (only if CUDA is available)
    if DEVICE == "cuda":
        print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
        print(f"Max allocated memory: {torch.cuda.max_memory_allocated()} bytes")
    elif DEVICE == "cpu":
        print(f"Memry used: CPU")
    return None

def data_load_and_prep(data_file = DATA_FILE, epsilon = 1e-8):
    if data_file[-3:] == "npz":
        data, sol, paras = functions.load_npz(data_file)
        print("Data loading from .npz file")
    elif data_file[-2:] == "h5":
        data, sol = functions.load_h5py(data_file)
        print("Data loading from .h5 file")
    else:
        raise DataFileFormatError(data_file[-5])

    n_total = data.shape[0]
    print("total images: ", n_total)
    n_train = int(TRAIN_SPLIT*n_total)
    print("training images: ", n_train)

    data_img_min = data.min(axis = (1, 2), keepdims = True)
    data_img_max = data.max(axis = (1, 2), keepdims = True)
    data = (data - data_img_min) / (data_img_max - data_img_min + epsilon)

    sol_img_min = sol.min(axis = (1, 2), keepdims = True)
    sol_img_max = sol.max(axis = (1, 2), keepdims = True)
    sol = (sol - sol_img_min) / (sol_img_max - sol_img_min + epsilon)

    data_train = data[:n_train]
    sol_train = sol[:n_train]

    data_tensor = torch.from_numpy(data_train).float().unsqueeze(1)
    sol_tensor = torch.from_numpy(sol_train).float().unsqueeze(1)
    dataset = TensorDataset(data_tensor, sol_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

# UNET -------------------------------------------------------------------------------------------------------------------------------------

def init_UNET_model(in_chanels: int = 1,
                out_chanels: int = 1,
                device = DEVICE,
                img_dim = IMG_DIM,
                lr = LR):
    unet = models.UNET(in_chanels, out_chanels).to(device)
    unet_optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    return unet, unet_optimizer

def training_step_UNET(images, ground_truths, unet, unet_optimizer):
    # training step (forward and backwards pass of both generator and discriminator as well as parameter optimazation through gradient computation) for
    # one batch of image pairs in the dataloader
    images = images.to(DEVICE)  # move input to GPU
    ground_truths = ground_truths.to(DEVICE)  # move target to GPU

    # generator Forward
    preds = unet(images)

    # losses
    loss = crit(preds, ground_truths)

    # backwards
    unet_optimizer.zero_grad()
    loss.backward()
    unet_optimizer.step()

    return loss, unet_optimizer

def training_loop_UNET(unet, unet_optimizer, dataloader, num_epochs = NUM_EPOCHS):
    for epoch in range(num_epochs):
        print("##############")
        print("Epoch: ", epoch)
        print("##############")
        running_loss = 0
        for x, y in dataloader:
            current_loss, current_optimizer = training_step_UNET(x, y, unet, unet_optimizer)
            running_loss += current_loss.item()
        print("Loss: ", running_loss/len(dataloader))


    return unet, num_epochs, current_loss, current_optimizer

def training_and_saving_UNET_model(unet, unet_optimizer, dataloader, num_epochs = NUM_EPOCHS, path = trained_model_name, prev_epochs = 0):
    print("############################")
    print("Training model:")
    print("############################")
    unet, num_epochs, current_loss, current_optimizer = training_loop_UNET(unet, unet_optimizer, dataloader, num_epochs)
    checkpoint = {
    "epoch": num_epochs,
    "generator_state_dict": unet.state_dict(),
    "optimizer_state_dict": current_optimizer.state_dict(),
    "loss": current_loss
}
    print("Saving model at: ", path)
    torch.save(checkpoint, f"{path}_epoch_{num_epochs+prev_epochs}.pth")
    print("Saved")
    return None



# GAN --------------------------------------------------------------------------------------------------------------------------------------

def init_GAN_models(in_chanels_gen: int = 1,
                out_chanels_gen: int = 1,
                in_chanels_disc: int = 1,
                out_chanels_disc: int = 1,
                device = DEVICE,
                img_dim = IMG_DIM,
                lr_gen = LR,
                lr_disc = LR):
    generator = models.UNET(in_chanels_gen, out_chanels_gen).to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_gen)

    #discriminator = models.Discriminator(img_dim)
    discriminator = models.Discriminator(img_dim, in_chanels_disc, out_chanels_disc).to(device)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr_disc)

    return generator, generator_optimizer, discriminator, discriminator_optimizer

def training_step_GAN(images, ground_truths, generator, gen_optimizer, discriminator, disc_optimizer):
    # training step (forward and backwards pass of both generator and discriminator as well as parameter optimazation through gradient computation) for
    # one batch of image pairs in the dataloader
    images = images.to(DEVICE)  # move input to GPU
    ground_truths = ground_truths.to(DEVICE)  # move target to GPU

    # generator Forward
    preds = generator(images)
    # discriminator Forward
    img_groundTruths_discrimination = discriminator(images, ground_truths)
    img_preds_discrimination = discriminator(images, preds.detach())

    # losses
    gen_loss = models.generator_loss(img_preds_discrimination, preds, ground_truths)
    disc_loss = models.discriminator_loss(img_groundTruths_discrimination, img_preds_discrimination)

    # backwards
    gen_optimizer.zero_grad()
    gen_loss.backward(retain_graph=True)
    gen_optimizer.step()
    disc_optimizer.zero_grad()
    disc_loss.backward()
    disc_optimizer.step()

    return gen_loss, disc_loss, gen_optimizer, disc_optimizer

def training_loop_GAN(generator, gen_optimizer, discriminator, disc_optimizer, dataloader, num_epochs = NUM_EPOCHS):
    for epoch in range(num_epochs):
        for x, y in dataloader:
            current_gen_loss, current_disc_loss, current_gen_optimizer, current_disc_optimizer = training_step_GAN(x, y, generator, gen_optimizer, discriminator, disc_optimizer)
        print("Epoch: " + str(epoch))
        print("Loss Generator: " + str(current_gen_loss))
        print("Loss Discriminator: " + str(current_disc_loss))

    return generator, discriminator, num_epochs, current_gen_loss, current_disc_loss, current_gen_optimizer, current_disc_optimizer

def training_and_saving_GAN_model(generator, gen_optimizer, discriminator, disc_optimizer, dataloader, num_epochs = NUM_EPOCHS, path = "../models/0307_real_noise_10k_model_3_32_1e-4", prev_epochs = 0):
    print("############################")
    print("Training model:")
    print("############################")
    generator, discriminator, num_epochs, current_gen_loss, current_disc_loss, current_gen_optimizer, current_disc_optimizer = training_loop_GAN(generator, gen_optimizer, discriminator, disc_optimizer, dataloader, num_epochs)
    checkpoint = {
    "epoch": num_epochs,
    "generator_state_dict": generator.state_dict(),
    "discriminator_state_dict": discriminator.state_dict(),
    "optimizer_G_state_dict": current_gen_optimizer.state_dict(),
    "optimizer_D_state_dict": current_disc_optimizer.state_dict(),
    "loss_G": current_gen_loss,
    "loss_D": current_disc_loss,
}
    print("Saving model at: ", path)
    torch.save(checkpoint, f"{path}_epoch_{num_epochs+prev_epochs}.pth")
    print("Saved")
    return None

# ------------------------------------------------------------------------------------------------------------------------------------------

def main():
    dataloader = data_load_and_prep()

    if model == "UNET":
        unet, unet_optimizer = init_UNET_model()
        unet.train()
        training_and_saving_UNET_model(unet, unet_optimizer, dataloader)
    elif model == "GAN":
        generator, generator_optimizer, discriminator, discriminator_optimizer = init_GAN_models()
        generator.train()
        discriminator.train()
        training_and_saving_GAN_model(generator, generator_optimizer, discriminator, discriminator_optimizer, dataloader, path=trained_model_name)
    else:
        raise ModelArchitectureError(model)
    print("Done!")

    return None

if __name__ == "__main__":
    print("Executing main() in training.py")
    main()