
from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Add
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import load_model
#%%

generator = load_model('models/generator_1547468085.7927237h5')




sampled_labels = numpy.arange(-1, 1, 0.1).reshape(-1, 1)
sampled_labels_scaled = (sampled_labels.astype(np.float32)-4.5) / 4.5
print(sampled_labels_scaled)

#%%


noise = np.random.normal(0, 1, (1, 100)) #sample noise once

for i in range(len(sampled_labels)):


    imgs = generator.predict([noise, sampled_labels[i:i+1]])

    gen_imgs = 0.5 * gen_imgs + 0.5
    plt.imshow(gen_imgs[0,:,:,0])
    plt.show)()
