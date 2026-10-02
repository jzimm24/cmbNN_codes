Cosmic Microwave Background Lensing through Machine Learning

This project builds neural networks capable of image restauration for weakly lensed CMB maps.
In general the parameters for every run are controlled via the config.py file. All parts of the code are therefor executable by simple calling the respective file
in a python environment without any other parameters being passed inside the terminal (i.e. python3 maps.py)

CMB maps and related source maps can be created with code inside maps.py to be used as Mock Data for NN training. The code currently offers the possibility of creating maps with random or fixed number of sources and random or fixed position and sizes. The sources can have a uniform profile or beta-profile. As noise white noise and/or gaussian noise can be added. The filetype of the data can be changed.

Different models can be trained via training.py. The general model architecture can be found in moodels.py. Currently there is a working UNET, ResUNET and GAN model.
The ResUNET model can be initialized with varying depth. Before training the data will be normalized to range from [0, 1]. In order to truly recreate the original groundtruth images the data needs to be denormalized afterwrds. All normalization parameters are stored throughout training to make this possible. It is advantages to set the "visualization" parameter in the training part of config.py to "True" in order to check the in going data during training. 

In order to judge the networks performances a evaluation code was added. The different
