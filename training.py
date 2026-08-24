#training architeture broadly taken from https://github.com/Obasho10/cmbnn/blob/main/train/train.py - Obasho M.


import numpy as np
import matplotlib.pyplot as plt

import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import functions as functions
import models as models

import config as config

# ---------------------------------------------------------------------------------------------------------------------------------

class DataFileFormatError(ValueError):
    pass

class ModelArchitectureError(ValueError):
    pass

# ---------------------------------------------------------------------------------------------------------------------------------

model = config.training_configs.model

DEVICE = config.training_configs.DEVICE
NUM_EPOCHS = config.training_configs.NUM_EPOCHS
BATCH_SIZE = config.training_configs.BATCH_SIZE
LR = config.training_configs.LR
IMG_DIM = config.training_configs.IMG_DIM
DATA_FILE = config.training_configs.DATA_FILE
TRAIN_SPLIT = config.training_configs.TRAIN_SPLIT

# no .pth
trained_model_name = config.training_configs.trained_model_name

#crit = nn.BCEWithLogitsLoss() 
crit = config.training_configs.crit

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

# def data_load_and_prep(data_file = DATA_FILE, epsilon = 1e-8):
#     if data_file[-3:] == "npz":
#         data, sol, paras = functions.load_npz(data_file)
#         print("Data loading from .npz file")
#     elif data_file[-2:] == "h5":
#         data, sol = functions.load_h5py(data_file)
#         print("Data loading from .h5 file")
#     else:
#         raise DataFileFormatError(data_file[-5])

#     n_total = data.shape[0]
#     print("total images: ", n_total)
#     n_train = int(TRAIN_SPLIT*n_total)
#     print("training images: ", n_train)

#     data_img_min = data.min(axis = (1, 2), keepdims = True)
#     data_img_max = data.max(axis = (1, 2), keepdims = True)
#     data = (data - data_img_min) / (data_img_max - data_img_min + epsilon)

#     sol_img_min = sol.min(axis = (1, 2), keepdims = True)
#     sol_img_max = sol.max(axis = (1, 2), keepdims = True)
#     sol = (sol - sol_img_min) / (sol_img_max - sol_img_min + epsilon)

#     data_train = data[:n_train]
#     sol_train = sol[:n_train]

#     data_tensor = torch.from_numpy(data_train).float().unsqueeze(1)
#     sol_tensor = torch.from_numpy(sol_train).float().unsqueeze(1)
#     dataset = TensorDataset(data_tensor, sol_tensor)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
#     return dataloader

def data_load_and_prep(data_file = DATA_FILE, epsilon = 1e-8):
    if data_file[-3:] == "npz":
        data, sol, paras = functions.load_npz(data_file)
        print("Data loading from .npz file")
    elif data_file[-2:] == "h5":
        data, sol = functions.load_h5py(data_file)
        print("Data loading from .h5 file")
    else:
        raise DataFileFormatError(data_file[-5])
    print("Full path: ", data_file)


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

def visualize_data_distr(data, outputfile, title, bin_width = 0.02, only_first_batch = True):
    if only_first_batch:
        batch = next(iter(data))

        noisy, clean = batch
        arr_noisy = noisy.squeeze(1).detach().cpu().numpy().flatten()
        arr_clean = clean.squeeze(1).detach().cpu().numpy().flatten()

    else:
        arr_noisy = []
        arr_clean = []
        for batch in data:
            noisy, clean = batch
            arr_noisy_current = noisy.squeeze(1).detach().cpu().numpy().flatten()
            arr_clean_current = clean.squeeze(1).detach().cpu().numpy().flatten()
            arr_noisy.append(arr_noisy_current)
            arr_clean.append(arr_clean_current)

    
    min_val_noisy = np.floor(arr_noisy.min() / bin_width) * bin_width
    max_val_noisy = np.floor(arr_noisy.max() / bin_width) * bin_width
    bins_noisy = np.arange(min_val_noisy, max_val_noisy+bin_width, bin_width)
    min_val_clean = np.floor(arr_clean.min() / bin_width) * bin_width
    max_val_clean = np.floor(arr_clean.max() / bin_width) * bin_width
    bins_clean = np.arange(min_val_clean, max_val_clean+bin_width, bin_width)

    counts_noisy, bin_edges_noisy = np.histogram(arr_noisy, bins = bins_noisy)
    counts_clean, bin_edges_clean = np.histogram(arr_clean, bins = bins_clean)

    title_noisy = title + " #noisy_values"
    title_clean = title + " #clean_values"

    noisy = noisy.squeeze(1).detach().cpu().numpy()
    clean = clean.squeeze(1).detach().cpu().numpy()

    fig, axs = plt.subplots(4, 2, figsize=(16, 16))

    axs[0][0].bar(bin_edges_noisy[:-1], counts_noisy, width=bin_width, align="edge", edgecolor="black")
    axs[0][0].text(0.8, 8000, "min: " + str(arr_noisy.min()), fontsize = 10)
    axs[0][0].text(0.8, 7000, "max: " + str(arr_noisy.max()), fontsize = 10)
    axs[0][0].set_xlabel("Values")
    axs[0][0].set_ylabel("Counts")
    axs[0][0].set_title(title_noisy)

    axs[0][1].bar(bin_edges_clean[:-1], counts_clean, width=bin_width, align="edge", edgecolor="black")
    axs[0][1].text(0.8, 80000, "min: " + str(arr_clean.min()), fontsize = 10)
    axs[0][1].text(0.8, 70000, "max: " + str(arr_clean.max()), fontsize = 10)
    axs[0][1].set_xlabel("Values")
    axs[0][1].set_ylabel("Counts")
    axs[0][1].set_title(title_clean)

    for i in range(3):
        im_noisy = axs[i+1][0].imshow(noisy[i])
        axs[i+1][0].set_xlabel("x")
        axs[i+1][0].set_ylabel("y")
        axs[i+1][0].set_title(f"title {['first','second','third'][i]} map noisy")
        fig.colorbar(im_noisy, ax=axs[i+1][0])

        im_clean = axs[i+1][1].imshow(clean[i])
        axs[i+1][1].set_xlabel("x")
        axs[i+1][1].set_ylabel("y")
        axs[i+1][1].set_title(f"title {['first','second','third'][i]} map clean")
        fig.colorbar(im_clean, ax=axs[i+1][1])

    plt.tight_layout()
    plt.savefig(outputfile)

    print("Figure saved at: ", outputfile)
    return None

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

    start_time = time.perf_counter()

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

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    functions.write_doc("../outputs/docs/first_doc_test2.txt", "training", runtime=elapsed_time, output_file=trained_model_name)

    return None

if __name__ == "__main__":
    print("Executing main() in training.py")
    main()
