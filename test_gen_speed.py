
from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Add
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import load_model
import numpy as np
import pickle as pkl
import copy
import random
import stats_utils
import tensorflow as tf
import matplotlib as mpl

generator = load_model(r'/home/hk2315/MSci2/models/21256generator_1552056453.h5')
start_time = time.time()
z_list = [-1.,-0.5,0,0.5,1.]
gen_imgs_list = []
for z in z_list:
    sample = np.array([[z]]*15600)
    sample = np.reshape(sample,(15600,1))
    noise = np.random.normal(0, 1, (len(sample), 100))
    gen_imgs = generator.predict([noise, sample])
    gen_imgs_list.append(gen_imgs)
elapsed_time = time.time() - start_time
print elapsed_time
