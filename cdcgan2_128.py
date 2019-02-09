
from __future__ import print_function, division
#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Add
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import load_model
import pickle as pkl
from sys import platform
import utils
import sys
import stats_utils #TODO need to make stats_utils file
from os import listdir
from os.path import isfile, join
import copy
import random


class binary_CGAN():


    def __init__(self, use_old_model):

        self.imgs = []
        self.labels = []
        self.start_time = str(time.time()).split('.')[0]
        self.label_dim = -1
        self.start_epoch = None

        self.read_data()

        optimizer = Adam(0.0001, 0.5)
        ph_img = Input(shape=(128,128,1))
        ph_label = Input(shape=(self.label_dim,))

        self.discriminator = self.build_discriminator(ph_img, ph_label)
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])

        noise = Input(shape=(100,))
        label = Input(shape=(self.label_dim,))
        self.generator = self.build_generator(noise, label)
        img = self.generator([noise, label])

        self.discriminator.trainable = False
        valid = self.discriminator([img, label])
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)


    def read_data(self):

        #data = pkl.load(open("/home/jz8415/slices2_128_all.pkl", "rb"))
        data = pkl.load(open(r"C:\\Outputs\\slices2_128.pkl", "rb"))

        self.imgs = []
        self.labels = []
        for entry in data:
            img = entry[0]
            l_str = entry[1]
            l_z = float(l_str.split('_z')[1].split('_')[0])
            self.imgs.append(img)
            self.labels.append([l_z])

        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        self.label_dim = len(self.labels[0])

        print('dimension of label: ', self.label_dim)
        print('Shapes:')
        print(self.imgs.shape)
        print(self.labels.shape)

        self.real_imgs_index = {}
        for i, z in enumerate(self.labels):
            z = z[0]
            if not z in self.real_imgs_index:
                self.real_imgs_index[z] = []
            self.real_imgs_index[z].append(i)

        return True

    def build_generator(self, noise, con):

        con1 = Dense(12, activation='tanh')(con)
        con1 = Dense(25, activation='tanh')(con1)
        con1 = Dense(50, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)

        merged_input = Concatenate()([con1, noise])

        merged_input = Dense(200)(merged_input)
        merged_input = Dense(200)(merged_input)

        cfrom = 64
        cto = 256
        imfrom = 9
        twopot = 4

        hid = Dense(cfrom * imfrom**2)(merged_input)
        hid = Reshape((imfrom, imfrom, cfrom))(hid)

        im = imfrom
        for i in range(twopot-1):

            hid = Conv2DTranspose(int(np.round((cto/cfrom)**((i+1)/(twopot-1))*cfrom)), 5, strides=2, padding='same')(hid)
            hid = BatchNormalization(momentum=0.9)(hid)
            hid = Activation("relu")(hid)


        hid = Conv2DTranspose(1, kernel_size=5, strides=2, padding="same")(hid)
        hid = Activation("tanh")(hid)

        out = Cropping2D(cropping=((8,8),(8,8)))(hid)

        model =  Model([noise, con], out)
        model.summary()
        return model
