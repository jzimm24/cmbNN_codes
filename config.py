# Configuration of Hyperparameters

from typing import NamedTuple
import torch
import torch.nn as nn

import models as models

# ---------------------------------------------------------------------------------------------------------------------------------------

class maps_config(NamedTuple):
    #class setting all necessary parameters for running maps.py
    save_mode = "hdf5"      # responsible for data type (choose 'hdf5' or 'csv')
    flux_mode = "gauss"     # 'lin', 'log', or 'gauss' fluxes

    num_images = 10000      # number of maps created
    image_size = 128        # size of square map
    max_source_number = 4   # maximum number of sources (uniform random distr. for all number of sources below max)

    seed = 42                   #seed for random value selection


    rc_mean  = 30.0                       # r_c in the β‐model (pixels)   --> DEFAULT 7 PIXELS
    rc_sigma = 15.0                       # std dev in r_c (in pixels)   --> DEFAULT sigma=2 PIXELS


    I0_range                 = (0.0, 4.0)    # if I0_fixed is None, draw uniformly/log-uniformly from this
                                            # (used when flux_mode='lin' or 'log')
    I0_fixed                 = None          # set to a float to force same I0 each time
    I0_mean                  = 1.0           # mean of Gaussian flux prior   (used when flux_mode='gauss')
    I0_sigma                 = 0.3           # std-dev of Gaussian flux prior (used when flux_mode='gauss')


    white_noise_amplitude    = 1.0        # σ for white Gaussian noise (~2 when with 1/f, ~7 when WN only)
    one_over_f_slope         = 3.0        # power‐law slope (1.5=pink, 3.0=red)
    one_over_f_amplitude     = 1.0        # scaling for 1/f noise (~1 when slope 3.0, ~2 when slope 1.5)

    gaussian_smoothing_fwhm  = 1.0        # if >0, smooth final image with this FWHM   --> DEFAULT 5 PIXELS


    output_h5               = '../data/realNoise128_10k_1WN_1f_3s_size30_2409.h5'

    doc_path = "../outputs/docs/realNoise128_10k_1WN_1f_3s_size30_2409.txt"

    visualization_path = "../outputs/realNoise128_10k_1WN_1f_3s_size30_2409.png"

maps_configs = maps_config()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Training

class training_config(NamedTuple):
    #class setting all necessary parameters for running training.py
    model: str = "UNET"                                     # model architecture (choose gan or unet) 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # checks available devices automatically (prefered gpu)
    NUM_EPOCHS = 4                                       # number of epochs in training
    BATCH_SIZE = 16                                         # batch size
    LR = 5e-4                                               # learning rate
    IMG_DIM = 64                                          # size of maps in data
    DATA_FILE = "../data/1k_64_multicircle_10xs20xy20.npz"   # data file storing images
    TRAIN_SPLIT = 1-0.05                                    # relative amount of data used for training
    NEW_MODEL = False
    predecessor_model = "../models/easyData2_epoch_3.pth"
    trained_model_name = "../models/testModelcontinuation_easyData2_epoch_3.pth"   # name of the trained model
    crit = nn.BCEWithLogitsLoss()                           # criterium used for calculating model loss
    doc_path = "../outputs/docs/testModelcontinuation_easyData2_epoch_3.txt"   # path for documentation file of the run
    visualization = False                                   # should the value distribution of the data + some example maps of the dataset be visualized
    visualization_file = "../outputs/testModelcontinuation_easyData2_epoch_3.png" # path for visualisation diagrams

training_configs = training_config()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Evaluation

class eval_config(NamedTuple):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    modelArchitecture = "GAN"
    MODEL_FILE = "../models/gan_epoch_16.pth"       # has to be manually inserted because of added epoch info in model save state
    DATA_FILE = "../data/100_128_single_centered_circle_varyingSize_maps.npz"
    BASE_NAME = "gan_100_128_single_centered_circle_varyingSize_maps.npz"
    OUTPUT_FILE = f"../outputs/{BASE_NAME}.png"
    EVAL_FILE = f"../outputs/{BASE_NAME}.txt"

    EVAL_SPLIT = 1 - training_configs.TRAIN_SPLIT
    BATCH_SIZE = 32


    IMG_SIZE = training_configs.IMG_DIM
    MAX_VAL = 1 # for psnr
    DATA_RANGE = [0, MAX_VAL]

    TITLE = OUTPUT_FILE

    doc_path = "../outputs/docs/gan_100_128_single_centered_circle_varyingSize_maps.txt"

evaluation_configs = eval_config()