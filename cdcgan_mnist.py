#%% IMPORTS

from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Concatenate
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import cifar10
import keras.backend as K
import matplotlib.pyplot as plt
import sys
import numpy as np
from keras.preprocessing import image

#%% DEFINE GENERATOR

def get_generator(input_layer, condition_layer):
    #input layer should be a random vector yo

    merged_input = Concatenate()([input_layer, condition_layer])
    #100+10

    hid = Dense(128 * 7 * 7, activation='relu')(merged_input)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    hid = Reshape((7, 7, 128))(hid)
    #7x7x128

    hid = Conv2D(128, kernel_size=4, strides=1,padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #7x7x128

    hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #14x14x128

    hid = Conv2D(128, kernel_size=5, strides=1,padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #14x14x128

    hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #28x28x128

    hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #28x28x128

    hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #28x28x128

    hid = Conv2D(1, kernel_size=5, strides=1, padding="same")(hid)
    out = Activation("relu")(hid)
    #28x28x1

    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    model.summary()

    #model takes [input_layer, condition_layer] as input and returns the generated image
    # why return out as well ?
    return model, out



#%% DEFINE DISCRIMINATOR

def get_discriminator(input_layer, condition_layer):
    #input layer should be a crazy image yo (32x32x3)

    hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(input_layer)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #28x28x128

    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #14x14x128

    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #7x7x128


    hid = Flatten()(hid)
    #7*7*128

    merged_layer = Concatenate()([hid, condition_layer])
    #4*4*128+1
    hid = Dense(512, activation='relu')(merged_layer)
    #hid = Dropout(0.4)(hid)
    #512

    out = Dense(1, activation='sigmoid')(hid)
    #1

    model = Model(inputs=[input_layer, condition_layer], outputs=out)

    model.summary()

    return model, out



def train(epochs, batch_size=128, sample_interval=50):

    # Load the dataset
    (X_train, y_train), (_, _) = mnist.load_data()

    # Configure input
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    y_train = y_train.reshape(-1, 1)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generate a half batch of new images
        gen_imgs = generator.predict([noise, labels])

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
        d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
