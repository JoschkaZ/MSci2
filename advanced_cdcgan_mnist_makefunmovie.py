
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
print('keras: %s' % keras.__version__)
#%%

generator = load_model(r'C:\\Users\\Joschka\\github\\MSci2\\models\\21256generator_1547722536.h5')



# 7, 8, 9,  10, 11
#-1, -0.5, 0, 0.5, 1
#sampled_labels = np.arange(-1.5, 1.5, 0.01).reshape(-1, 1)
sampled_labels = np.arange(-1., 1.5, 0.5).reshape(-1, 1)
print(sampled_labels)
#%%
sampled_labels_scaled = (sampled_labels.astype(np.float32)-0) / 1
print(sampled_labels_scaled)

#%%


noise = np.random.normal(0, 1, (1, 100)) #sample noise once
print(noise.shape)

#%%

for i in range(len(sampled_labels)):

    gen_imgs = generator.predict([noise, sampled_labels[i:i+1]])

    gen_imgs = 0.5 * gen_imgs + 0.5
    plt.imshow(gen_imgs[0,:,:,0])
    plt.savefig('movie/' + str(i) + '_' + str(sampled_labels[i]) + '.png')
    plt.close()
