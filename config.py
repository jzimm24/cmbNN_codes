# Configuration of Hyperparameters

from typing import NamedTuple
import torch
import torch.nn as nn

import models as models

# ---------------------------------------------------------------------------------------------------------------------------------------

# Cosmologies


PLANCK2018 = dict(
    H0=67.36, ombh2=0.02237, omch2=0.1200,
    tau=0.0544, ln1e10As=3.044, ns=0.9649, mnu=0.06,
)

# ---------------------------------------------------------------------------------------------------------------------------------------


class maps_config(NamedTuple):
    #class setting all necessary parameters for running maps.py
    save_mode = "hdf5"      # responsible for data type (choose 'hdf5' or 'csv')
    flux_mode = "gauss"     # 'lin', 'log', or 'gauss' fluxes

    num_images = 30000      # number of maps created
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


    output_h5               = '../data/cmbMaps_w1000.h5'

    doc_path = "../outputs/docs/realNoise128_30k_1WN_1f_3s_size30_0910.txt"

    visualization_path = "../outputs/realNoise128_30k_1WN_1f_3s_size30_0910.png"

maps_configs = maps_config()

class cmbMaps_config(NamedTuple):
    #class setting all necessary parameters for running maps.py for cmb map construction (different from retired real noise pipeline)
    number_maps = 1000
    map_size_arc = 256*2.0
    pixel_size_arc = 2.0

    cosmology = PLANCK2018
    powerSpectrum = None
    white_Noise_amplitude = 900
    l_max = 3000
    beam_width = 0.0

    h5_filename = "../data/cmb/cmbMaps_WN900_size512arcmin_pixelSize2arcmin_lmax3000_bw0_p18"

    seed = 42

    doc_path = "../outputs/docs/cmb/cmb_map_creation_doc_cmbMaps_WN900_size512arcmin_pixelSize2arcmin_lmax3000_bw0_p18.txt"

cmbMaps_configs = cmbMaps_config()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Training

class training_config(NamedTuple):
    #class setting all necessary parameters for running training.py
    model: str = "GAN"                                     # model architecture (choose gan or unet) 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # checks available devices automatically (prefered gpu)
    NUM_EPOCHS = 8                                       # number of epochs in training
    BATCH_SIZE = 8                                         # batch size
    LR = 5e-4                                               # learning rate
    LR_disc = 5e-5                                          # learning rate in the case of an existing discriminator
    IMG_DIM = 256                                          # size of maps in data
    DATA_FILE = "../data/cmb/cmbMaps_WN900_size512arcmin_pixelSize2arcmin_lmax3000_bw0_p18.h5"   # data file storing images
    TRAIN_SPLIT = 1-0.05                                    # relative amount of data used for training
    EPOCH_TRAINING_FRAC = 1                               # relative amount of data in training split used in each epoch
    NEW_MODEL = False
    predecessor_model = "../models/cmb/GAN_lrGen0005_lrDisc00005_cmbMaps_WN900_size512arcmin_pixelSize2arcmin_lmax3000_bw0_p18.pth_epoch_4.pth"
    trained_model_name = "../models/cmb/GAN_lrGen0005_lrDisc00005_cmbMaps_WN900_size512arcmin_pixelSize2arcmin_lmax3000_bw0_p18"   # name of the trained model
    crit = nn.BCEWithLogitsLoss()                           # criterium used for calculating model loss
    doc_path = "../outputs/docs/training_GAN_lrGen0005_lrDisc00005_cmbMaps_WN900_size512arcmin_pixelSize2arcmin_lmax3000_bw0_p18_epoch4to12.txt"   # path for documentation file of the run
    visualization = False                                        # should the value distribution of the data + some example maps of the dataset be visualized
    visualization_file = "../outputs/GAN_lrGen0005_lrDisc00005_cmbMaps_WN900_size512arcmin_pixelSize2arcmin_lmax3000_bw0_p18.png" # path for visualisation diagrams

training_configs = training_config()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Evaluation

class eval_config(NamedTuple):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    modelArchitecture = "GAN"
    MODEL_FILE = "../models/UNET_realNoise128_10k_1WN_1f_3s_size30_2409.pth_epoch_16.pth"       # has to be manually inserted because of added epoch info in model save state
    DATA_FILE = "../data/realNoise128_10k_1WN_1f_3s_size30_2409.h5"
    BASE_NAME = "UNET_realNoise128_10k_1WN_1f_3s_size30_2409.pth_epoch_16"
    OUTPUT_FILE = f"../outputs/{BASE_NAME}.png"
    EVAL_FILE = f"../outputs/{BASE_NAME}.txt"

    EVAL_SPLIT = 1 - training_configs.TRAIN_SPLIT
    BATCH_SIZE = 8


    IMG_SIZE = 256
    MAX_VAL = 1 # for psnr
    DATA_RANGE = [0, MAX_VAL]

    TITLE = OUTPUT_FILE

    doc_path = "../outputs/docs/evaluation_UNET_realNoise128_10k_1WN_1f_3s_size30_2409_epoch_16t.xt"

evaluation_configs = eval_config()