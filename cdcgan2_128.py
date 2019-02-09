
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

    def build_discriminator(self, img, con):

        cfrom = 64
        cto = 256
        imfrom = 128
        twopot = 4

        hid = Conv2D(cfrom, kernel_size=5, strides=2, padding='same')(img)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)

        for i in range(twopot-1):

            hid = Conv2D(int(np.round((cto/cfrom)**((i+1)/(twopot-1))*cfrom)), kernel_size=5, strides=2, padding='same')(hid)
            hid = BatchNormalization(momentum=0.9)(hid)
            hid = LeakyReLU(alpha=0.2)(hid)

        hid = Flatten()(hid)

        hid = Dense(100, activation='tanh')(hid)

        con1 = Dense(12, activation='tanh')(con)
        con1 = Dense(25, activation='tanh')(con1)
        con1 = Dense(50, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)

        merged_layer = Concatenate()([hid, con1])
        merged_layer = Dropout(0.2)(merged_layer) ####NEW
        # -> 200

        merged_layer = Dense(100, activation='tanh')(merged_layer)
        merged_layer = Dense(50, activation='tanh')(merged_layer)
        merged_layer = Dense(25, activation='tanh')(merged_layer)

        out = Dense(1, activation='sigmoid')(merged_layer)
        # -> 1

        model = Model(inputs=[img, con], outputs=out)
        model.summary()
        return model

    def f(self, x, mmin):
        return int(not x==mmin)

    def binarize_images(self):

        vf = np.vectorize(f)
        mmin = np.min(self.images)
        tolerance = 0.0001
        for i in range(len(self.images)):

            if np.abs(np.min(self.images[i]) / mmin -1.) > tolerance:
                print('WARNING - ', mmin, np.min(self.images[i]))

            self.images[i] = vf(self.images[i], mmin)

    return True



    def scale_labels(self, l, verbose=0):
        length = len(l)
        step = int(length / 5.)
        if verbose == 1:
            print('scaling labels...')
            print('initial labels')
            print(l[0:length:step])
            print(l.max(axis=0))
            print(l.min(axis=0))
        mal = l.max(axis=0)
        mil = l.min(axis=0)

        for i in range(len(l)):
            for j in range(len(l[i])):
                l[i][j] = (l[i][j] - (mal+mil)/2.) / ((mal-mil)/2.)

        if verbose == 1:
            print('scaled labels')
            print(l[0:length:step])

        self.labels = l
        return True


    def train(self, epochs, batch_size=64, sample_interval=50, save_model_interval=500):


        self.binarize_images(self.images)
        self.scale_labels(self.labels)

        print(self.images[14])

        print('FINAL SHAPES')
        print(self.imgs.shape)
        print(self.labels.shape)

        input('...')

        #Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        last_acc = .75

        # if the model was loaded, start at start_epoch
        if self.start_epoch != None:
            efrom = self.start_epoch
        else:
            efrom = 0
        for epoch in range(efrom,epochs): #these are not proper epochs, it just selects one batch randomly each time

            # Select a random half batch of images
            idx = np.random.randint(0, self.imgs.shape[0], batch_size)
            imgs, labels = self.imgs[idx], self.labels[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels])

            #Train the discriminator
            #if last_acc > 0.8:
            if (epoch < 200) and (epoch%5!=0):
                print('Only testing discriminator')
                d_loss_real = self.discriminator.test_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.test_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            idx = np.random.randint(0, self.imgs.shape[0], batch_size)
            sampled_labels = self.labels[idx] #sample random labels from the data for training the generator

            # Train the generator
            if last_acc < 0.7 and 1==2:
                print('Only testing generator')
                g_loss = self.combined.test_on_batch([noise, sampled_labels], valid)
            else:
                g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)


            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100.*d_loss[1], g_loss))
            last_acc = d_loss[1]

            if epoch % 2000 == 0:
                print('calculating ps...')
                self.calc_ps(epoch)
                print('calculating brihgtness peak count...')
                self.calc_peak_count_brightness(epoch)


            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch % save_model_interval == 0:
                print('saving models...')
                self.discriminator.save('models/21256discriminator_' + str(self.start_time) + '.h5')
                self.discriminator.save_weights('models/21256discriminatorweights_' + str(self.start_time) + '.h5')
                self.combined.save('models/21256combined_' + str(self.start_time) + '.h5')
                self.discriminator.save_weights('models/21256combinedweights_' + str(self.start_time) + '.h5')
                self.generator.save('models/21256generator_' + str(self.start_time) + '.h5')
                self.discriminator.save_weights('models/21256generatorweights_' + str(self.start_time) + '.h5')
