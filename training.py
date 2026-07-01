#training architeture broadly taken from https://github.com/Obasho10/cmbnn/blob/main/train/train.py - Obasho M.


import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import cmbNN_codes.functions as functions
import cmbNN_codes.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 1
BATCH_SIZE = 32
LR = 1e-4
IMG_DIM = 64
DATA_FILE = "../data/1k_64_multicircle_10xs20xy20.npz"

def init_models(in_chanels_gen: int = 1,
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

def log_memory_usage():
    #!!! Taken from https://github.com/Obasho10/cmbnn/blob/main/train/train.py
    # Log memory usage (only if CUDA is available)
    if DEVICE == "cuda":
        print(f"Allocated memory: {torch.cuda.memory_allocated()} bytes")
        print(f"Max allocated memory: {torch.cuda.max_memory_allocated()} bytes")

def data_load_and_prep(data_file = DATA_FILE):
    data, sol, paras = functions.load_npz(data_file)

    data_tensor = torch.from_numpy(data).float().unsqueeze(1)
    sol_tensor = torch.from_numpy(sol).float().unsqueeze(1)
    dataset = TensorDataset(data_tensor, sol_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader

def training_step(images, ground_truths, generator, gen_optimizer, discriminator, disc_optimizer):
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

def training_loop(generator, gen_optimizer, discriminator, disc_optimizer, dataloader, num_epochs = NUM_EPOCHS):
    for epoch in range(num_epochs):
        for x, y in dataloader:
            current_gen_loss, current_disc_loss, current_gen_optimizer, current_disc_optimizer = training_step(x, y, generator, gen_optimizer, discriminator, disc_optimizer)
            print("Epoch: " + str(epoch))
            print("Loss Generator: " + str(current_gen_loss))
            print("Loss Discriminator: " + str(current_disc_loss))

    return generator, discriminator, num_epochs, current_gen_loss, current_disc_loss, current_gen_optimizer, current_disc_optimizer

def training_and_saving_model(generator, gen_optimizer, discriminator, disc_optimizer, dataloader, num_epochs = NUM_EPOCHS, path = "../models/model", prev_epochs = 0):
    print("############################")
    print("Training model:")
    print("############################")
    generator, discriminator, num_epochs, current_gen_loss, current_disc_loss, current_gen_optimizer, current_disc_optimizer = training_loop(generator, gen_optimizer, discriminator, disc_optimizer, dataloader, num_epochs)
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

def main():
    dataloader = data_load_and_prep()
    generator, generator_optimizer, discriminator, discriminator_optimizer = init_models()
    generator.train()
    discriminator.train()
    training_and_saving_model(generator, generator_optimizer, discriminator, discriminator_optimizer, dataloader)
    print("Done!")
    return None

if __name__ == "__main__":
    print("Executing main() in training.py")
    main()