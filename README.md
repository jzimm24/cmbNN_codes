Cosmic Microwave Background Lensing through Machine Learning

This project builds neural networks capable of image restauration for weakly lensed CMB maps.


## Installation
### Requirements
Python 3.12.3
Hardware:   Training for networks (exclluding short runs in development) requieres GPU access.
            Epochs for GAN Training with 10k images may take up to 1hour with:
            CPU: 2 × AMD EPYC "Milan" 64-core/128-thread 2.00GHz
            RAM: 512GB DDR4 3200MHz 
            GPU: 8 × Nvidia A40 48GB
            Total Amount of Cores: 3,072
            Total RAM: 12.3TB

### Setup
    git clone https://github.com/jzimm24/cmbNN_codes
    cd cmbNN_codes
    pip install -r packages.txt

## Usage 
In general the parameters for every run are controlled via the config.py file. All parts of the code are therefor executable by simple calling the respective file
in a python environment without any other parameters being passed inside the terminal (i.e. python3 maps.py)

-> set parameters in config.py\
python maps.py\
-> set parameters in config.py\
python training.py\
-> set parameters in config.py\
python eval.py\

###Maps
CMB maps and related source maps can be created with code inside maps.py to be used as Mock Data for NN training. The code currently offers the possibility of creating maps with random or fixed number of sources and random or fixed position and sizes. The sources can have a uniform profile or beta-profile. As noise white noise and/or gaussian noise can be added. The filetype of the data can be changed.

### Training
Different models can be trained via training.py. The general model architecture can be found in moodels.py. Currently there is a working UNET, ResUNET and GAN model.
The ResUNET model can be initialized with varying depth. Before training the data will be normalized to range from [0, 1]. In order to truly recreate the original groundtruth images the data needs to be denormalized afterwrds. All normalization parameters are stored throughout training to make this possible. It is advantages to set the "visualization" parameter in the training part of config.py to "True" in order to check the in going data during training.

### Evaluation
In order to judge the networks performances a evaluation code was added. The different
