# Configuration of Hyperparameters

from typing import NamedTuple
import torch
import torch.nn as nn

import models as models

# ---------------------------------------------------------------------------------------------------------------------------------------

# Maps

maps_save_mode = "hdf5"
maps_flux_mode = "lin"

maps_num_images = 10000
maps_image_size = 128
maps_max_source_number = 3

maps_seed = 42                   #seed for random value selection


maps_rc_mean  = 7.0                       # r_c in the β‐model (pixels)   --> DEFAULT 7 PIXELS
maps_rc_sigma = 0.0                       # std dev in r_c (in pixels)   --> DEFAULT sigma=2 PIXELS


maps_I0_range                 = (0.0, 4.0)    # if I0_fixed is None, draw uniformly/log-uniformly from this
                                         # (used when flux_mode='lin' or 'log')
maps_I0_fixed                 = None          # set to a float to force same I0 each time
maps_I0_mean                  = 1.0           # mean of Gaussian flux prior   (used when flux_mode='gauss')
maps_I0_sigma                 = 0.3           # std-dev of Gaussian flux prior (used when flux_mode='gauss')


maps_white_noise_amplitude    = 1.0        # σ for white Gaussian noise (~2 when with 1/f, ~7 when WN only)
maps_one_over_f_slope         = 3.0        # power‐law slope (1.5=pink, 3.0=red)
maps_one_over_f_amplitude     = 1.0        # scaling for 1/f noise (~1 when slope 3.0, ~2 when slope 1.5)

maps_gaussian_smoothing_fwhm  = 1.0        # if >0, smooth final image with this FWHM   --> DEFAULT 5 PIXELS


maps_output_h5               = '../data/1WN_1A_3S-smooth1_10k128pix_lin04.h5'

class maps_config(NamedTuple):
    save_mode = "hdf5"
    flux_mode = "gauss"

    num_images = 10000
    image_size = 128
    max_source_number = 4

    seed = 42                   #seed for random value selection


    rc_mean  = 7.0                       # r_c in the β‐model (pixels)   --> DEFAULT 7 PIXELS
    rc_sigma = 2.0                       # std dev in r_c (in pixels)   --> DEFAULT sigma=2 PIXELS


    I0_range                 = (0.0, 4.0)    # if I0_fixed is None, draw uniformly/log-uniformly from this
                                            # (used when flux_mode='lin' or 'log')
    I0_fixed                 = None          # set to a float to force same I0 each time
    I0_mean                  = 1.0           # mean of Gaussian flux prior   (used when flux_mode='gauss')
    I0_sigma                 = 0.3           # std-dev of Gaussian flux prior (used when flux_mode='gauss')


    white_noise_amplitude    = 1.0        # σ for white Gaussian noise (~2 when with 1/f, ~7 when WN only)
    one_over_f_slope         = 3.0        # power‐law slope (1.5=pink, 3.0=red)
    one_over_f_amplitude     = 1.0        # scaling for 1/f noise (~1 when slope 3.0, ~2 when slope 1.5)

    gaussian_smoothing_fwhm  = 1.0        # if >0, smooth final image with this FWHM   --> DEFAULT 5 PIXELS


    output_h5               = '../data/realNoise128_10k_2WN_3f_2808.h5'

    doc_path = "../outputs/docs/realNoise128_10k_7rc_1WN_3f_2808_doc.txt"

    visualization_path = "../outputs/realNoise128_10k_7rc_1WN_3f_2808_preNorm.png"

maps_configs = maps_config()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Training

class training_config(NamedTuple):
    model: str = "GAN"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_EPOCHS = 16
    BATCH_SIZE = 32
    LR = 1e-4
    IMG_DIM = 128
    DATA_FILE = "../data/realNoise128_10k_7rc_1WN_3f_2808.h5"
    TRAIN_SPLIT = 1-0.05
    trained_model_name = "../models/gan_realNoise128_10k_7rc_1WN_3f_2808"
    crit = nn.BCEWithLogitsLoss() 
    doc_path = "../outputs/docs/gan_realNoise128_10k_7rc_1WN_3f_2808.txt"
    visualization = False
    visualization_file = "../outputs/realNoise128_10k_7rc_1WN_3f_2808_postNorm.png"

training_configs = training_config()

# ---------------------------------------------------------------------------------------------------------------------------------------

# Evaluation

class eval_config(NamedTuple):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    modelArchitecture = "UNET"
    MODEL_FILE = "/home/user/Physik_Bonn/master_thesis/codes/cmbNN/models/unet_simpleData_2508_epoch_3.pth"       # has to be manually inserted because of added epoch info in model save state
    DATA_FILE = "/home/user/Physik_Bonn/master_thesis/codes/cmbNN/data/realNoise128_10k_2WN_3f_2808.h5"
    OUTPUT_FILE = "unet_simple_Data_2508_eval_normalized.png"

    EVAL_SPLIT = 1 - training_configs.TRAIN_SPLIT
    BATCH_SIZE = 32


    IMG_SIZE = training_configs.IMG_DIM
    MAX_VAL = 1 # for psnr
    DATA_RANGE = [0, MAX_VAL]

    TITLE = OUTPUT_FILE

    doc_path = "../outputs/docs/0109_unet_simpleData_eval_normalized_doc.txt"

evaluation_configs = eval_config()