
from __future__ import print_function, division
#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, add
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
import tensorflow.keras.backend as tfback
import tensorflow as tf

def build_binary_generator(noise, con, ones, half):

    con1 = Dense(12, activation='tanh')(con)
    con1 = Dense(25, activation='tanh')(con1)
    con1 = Dense(50, activation='tanh')(con1)
    con1 = Dense(100, activation='tanh')(con1)

    merged_input = Concatenate()([con1, noise])

    merged_input = Dense(200)(merged_input)
    merged_input = Dense(200)(merged_input)

    hid = Dense(128*9*9, activation='relu')(merged_input)
    # -> 128*9*9

    hid = Reshape((9,9,128))(hid)
    # -> 128x9x9

    #hid = UpSampling2D()(hid)
    hid = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(hid)
    hid = BatchNormalization(momentum=0.8)(hid)
    hid = Activation("relu")(hid)
    # -> 128x18x18

    #hid = UpSampling2D()(hid)
    hid = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(hid)
    hid = BatchNormalization(momentum=0.8)(hid)
    hid = Activation("relu")(hid)
    # -> 128x36x36

    #hid = UpSampling2D()(hid)
    hid = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(hid)
    hid = BatchNormalization(momentum=0.8)(hid)
    hid = Activation("relu")(hid)
    # -> 128x72x72

    hid = Conv2DTranspose(1, kernel_size=5, strides=2, padding="same")(hid)
    hid = Activation("tanh")(hid) # images are binary, sigmoid should work welle

    out = Cropping2D(cropping=((8,8),(8,8)))(hid)

    out = add([out, ones]) # ones are negative

    out = multiply([out,half])





    model =  Model([noise, con, ones, half], out)
    model.summary()
    return model

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

        ones = Input(shape=(128,128,1))
        half = Input(shape=(128,128,1))

        self.discriminator = self.build_discriminator(ph_img, ph_label)
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])

        noise = Input(shape=(100,))
        label = Input(shape=(self.label_dim,))
        self.generator = build_binary_generator(noise, label, ones, half)
        img = self.generator([noise, label, ones, half])

        self.discriminator.trainable = False
        valid = self.discriminator([img, label])
        self.combined = Model([noise, label, ones, half], valid)
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

    def build_discriminator(self, img, con):

        hid = Conv2D(32, kernel_size=5, strides=2, padding="same")(img)
        hid = BatchNormalization(momentum=0.8)(hid) ####NEW
        hid = LeakyReLU(alpha=0.2)(hid)
        #hid = Dropout(0.25)(hid)
        # -> 32x64x64

        hid = Conv2D(64, kernel_size=5, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)
        #hid = Dropout(0.25)(hid)
        # -> 64x32x32

        hid = Conv2D(128, kernel_size=4, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)
        #hid = Dropout(0.25)(hid)
        # -> 128x16x16

        hid = Conv2D(128, kernel_size=4, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)
        #hid = Dropout(0.25)(hid)
        # -> 128x8x8

        hid = Conv2D(128, kernel_size=3, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)
        #hid = Dropout(0.25)(hid)
        # -> 128x4x4

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

    def f(self, x, mmin, mmax, tolerance):
        return float(not (np.abs((x - mmin) / (mmax-mmin)) < tolerance))

    def binarize_images(self):

        print('binarising images')
        vf = np.vectorize(self.f)
        mmin = np.min(self.imgs)
        mmax = np.max(self.imgs)
        tolerance = 0.00001

        for i in range(len(self.imgs)):

            if i % (int(len(self.imgs)/100.)) == 0:
                print(np.round(i/len(self.imgs)*100,2))

            #if i == 0:
            #    f, axarr = plt.subplots(2)
            #    axarr[0].imshow(self.imgs[i])

            if np.abs((np.min(self.imgs[i]) - mmin) / (mmax-mmin)) > tolerance:
                print('WARNING - ', mmin, np.min(self.imgs[i]))

            mmin = copy.deepcopy(np.min(self.imgs[i]))
            mmax = copy.deepcopy(np.max(self.imgs[i]))
            self.imgs[i] = vf(self.imgs[i], mmin, mmax, tolerance)


            #if i == 0:
            #    axarr[1].imshow(self.imgs[i])
            #    plt.show()
            #    plt.close()

        self.imgs = np.expand_dims(self.imgs, axis=3)

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

        return l

    def train(self, epochs, batch_size=4, sample_interval=50, save_model_interval=500):

        self.binarize_images()
        self.labels = self.scale_labels(self.labels)

        print('FINAL SHAPES')
        print(self.imgs.shape)
        print(self.labels.shape)

        #Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        last_acc = .75

        ones = 1*np.ones(shape=(batch_size,128,128,1))
        half = 0.5*np.ones(shape=(batch_size,128,128,1))

        # if the model was loaded, start at start_epoch
        if self.start_epoch != None:
            efrom = self.start_epoch
        else:
            efrom = 0
        for epoch in range(efrom,epochs):

            # Select a random half batch of images
            idx = np.random.randint(0, self.imgs.shape[0], batch_size)
            imgs, labels = self.imgs[idx], self.labels[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels, ones, half])

            # NOTE GET NOISE LABEL VECTORS
            #p_flip = 0.01
            #noise_range = 0.1
            #valid_noisy  = np.array([random.uniform(1.-noise_range,1.) if (random.uniform(0,1)<1.-p_flip) else random.uniform(0.,noise_range) for _ in range(batch_size)])
            #fake_noisy  = np.array([random.uniform(0.,noise_range) if (random.uniform(0,1)<1.-p_flip) else random.uniform(1.-noise_range,1.) for _ in range(batch_size)])

            #imgs = imgs + np.random.normal(0, 0.01, size=imgs.shape)
            #gen_imgs = gen_imgs + np.random.normal(0, 0.01, size=imgs.shape)


            #Train the discriminator
            #if (epoch < 200) and (epoch%5!=0):
            if last_acc > 0.8:
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
                g_loss = self.combined.test_on_batch([noise, sampled_labels, ones, half], valid)
            else:
                g_loss = self.combined.train_on_batch([noise, sampled_labels, ones, half], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100.*d_loss[1], g_loss))
            last_acc = d_loss[1]

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch % save_model_interval == 0:
                print('saving models...')
                self.discriminator.save('models/b_128_discriminator_' + str(self.start_time) + '.h5')
                self.discriminator.save_weights('models/b_128_discriminator_weights_' + str(self.start_time) + '.h5')
                self.combined.save('models/b_128_combined_' + str(self.start_time) + '.h5')
                self.combined.save_weights('models/b_128_combined_weights_' + str(self.start_time) + '.h5')
                self.generator.save('models/b_128_generator_' + str(self.start_time) + '.h5')
                self.generator.save_weights('models/b_128_generator_weights_' + str(self.start_time) + '.h5')

    def sample_images(self, epoch):

        sample_at0 = [
        [7.],
        [7.5],
        [8.],
        [8.5],
        [9.],
        [9.5],
        [10.],
        [10.5],
        [11.]
        ]
        r = len(sample_at0)
        c = 3

        temp_copy = copy.deepcopy(sample_at0)

        sample_at0 = np.array(sample_at0)
        sample_at = self.scale_labels(sample_at0)
        print('sampling images at labels:', sample_at)

        noise = np.random.normal(0, 1, (len(sample_at), 100))

        ones = 1*np.ones(shape=(batch_size,128,128,1))
        half = 0.5*np.ones(shape=(batch_size,128,128,1))

        gen_imgs = self.generator.predict([noise, sample_at, ones, half])

        # Rescale images 0 - 1
        ######gen_imgs = 0.5 * gen_imgs + 0.5
        """
        print('fake_min',np.min(gen_imgs))
        print('fake_max',np.max(gen_imgs))

        print('fake_min',np.min(self.imgs))
        print('fake_max',np.max(self.imgs))
        """

        fig, axs = plt.subplots(r, c, figsize=(4,18), dpi=250)
        cnt = 0
        for i in range(r):
            for j in range(c): # c=0: fake, c=1, real
                if cnt >= len(gen_imgs) and j == 0:
                    axs[i,j].axis('off')
                    break
                else:
                    if j == 0:
                        axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='hot', clim=(-1,1))
                        axs[i,j].set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in temp_copy[cnt]))
                        axs[i,j].axis('off')
                        print('fake_min',np.min(gen_imgs[cnt,:,:,0]))
                        print('fake_max',np.max(gen_imgs[cnt,:,:,0]))
                        cnt += 1

                    if j ==1: #show a real image
                        #print(temp_copy[i][0])
                        #print(self.real_imgs_index.keys())
                        if temp_copy[i][0] in self.real_imgs_index: #real images for that z are available

                            sample_i = random.randint(0,len(self.real_imgs_index[temp_copy[i][0]])-1)
                            #print(self.real_imgs_index[temp_copy[i][0]])
                            sample_i = self.real_imgs_index[temp_copy[i][0]][sample_i]
                            #print(sample_i)
                            #print(self.imgs.shape)
                            axs[i,j].imshow(self.imgs[sample_i,:,:,0], cmap='hot', clim=(-1,1))
                            axs[i,j].set_title("")
                            print('real_min',np.min(self.imgs[sample_i,:,:,0]))
                            print('real_max',np.max(self.imgs[sample_i,:,:,0]))
                        axs[i,j].axis('off')

                    if j ==2:
                        if temp_copy[i][0] in self.real_imgs_index: #real images for that z are available
                            fake_pix_val = stats_utils.get_pixel_val(gen_imgs[[cnt-1],:,:,:])
                            real_pix_val = stats_utils.get_pixel_val(self.imgs[[sample_i],:,:,:])
                            axs[i,j].hist(real_pix_val, bins=100, color="blue", label="real", alpha=0.7)
                            axs[i,j].hist(fake_pix_val, bins=100, color="orange", label="fake", alpha=0.7)
                            axs[i,j].legend()

        if platform == 'linux':
            user = utils.get_user()
            print(user)
            fig.savefig(r"/home/" + user + r"/MSci2/images/%d.png" % epoch)
        else: #windows
            fig.savefig("images/%d.png" % epoch)
        plt.close()

class combined_CGAN():

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

        self.find_ps = False
        self.find_peak_brightness = False

        self.discriminator = self.build_discriminator(ph_img, ph_label)
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])

        # IMPORT BINARY GENERATOR
        binary_noise = Input(shape=(100,))
        label = Input(shape=(self.label_dim,))
        self.binary_generator = build_binary_generator(binary_noise, label) # uses the same label
        self.binary_generator.trainable = False
        binary_img = self.binary_generator([binary_noise, label])

        noise = Input(shape=(100,))
        ph_binary_img = Input(shape=(128,128,1))
        ones = Input(shape=(128,128,1))
        twos = Input(shape=(128,128,1))
        self.generator = self.build_generator(noise, label, ph_binary_img, ones, twos)
        img = self.generator([noise, label, binary_img, ones, twos])

        self.discriminator.trainable = False
        valid = self.discriminator([img, label])
        self.combined = Model([noise, binary_noise, label, ones, twos], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

        # LOAD BINARY GENERATOR MODEL
        # identify model to load
        if platform == 'linux':
            user = utils.get_user()
            #mypath = r'/home/' + user + r'/MSci2/models' # NOTE LINUX OR WINDOWS
        else:
            mypath = r"C:\\Users\\Joschka\\github\\MSci2\\models"
        files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        print('Models found:')
        print(files)
        time_to_load = 0
        for file in files:
            if 'b_' in file and 'generator_weights' in file:
                model_time = int(file.split('weights_')[-1].split('.')[0])
                time_to_load = max(time_to_load, model_time)
        print('Time to load: ', time_to_load)


        print('Reading weights from: ')
        print(mypath + r'\\b_128_generator_weights_' + str(time_to_load) + '.h5') # NOTE LINUX OR WINDOWS
        self.binary_generator.load_weights(mypath + r'\\b_128_generator_weights_' + str(time_to_load) + '.h5')

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

    def build_generator(self, noise, con, binary_img, ones, twos):


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
        hid = Activation("sigmoid")(hid)

        hid = Cropping2D(cropping=((8,8),(8,8)))(hid)
        #- > [0, 1]

        out = multiply([hid, binary_img])
        #- > [0, 1]

        #ones = Input(np.ones(shape=tfback.shape(out)))

        out = multiply([out,twos])

        out = add([out, ones]) # ones are negative
        #out = out * 2.
        #out = out - 1.


        model =  Model([noise, con, binary_img, ones, twos], out)
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

        """
        #EXPAND LABEL
        con1 = Dense(100, activation='tanh')(con)
        con1 = Dense(100, activation='tanh')(con1)
        """

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

    def min_max_scale_images(self):
        print('minmax scaling images...')
        mmax = np.max(self.imgs)
        mmin = np.min(self.imgs)
        self.imgs = (self.imgs - (mmax+mmin)/2.) / ((mmax-mmin) / 2.)
        print('expanding dimension of images...')
        self.imgs = np.expand_dims(self.imgs, axis=3)

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

        return l

    def train(self, epochs, batch_size=64, sample_interval=50, save_model_interval=500):

        #SCALE IMAGES
        self.min_max_scale_images()
        self.labels = self.scale_labels(self.labels)

        print('FINAL SHAPES')
        print(self.imgs.shape)
        print(self.labels.shape)

        #Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        last_acc = .75

        # if the model was loaded, start at start_epoch
        if self.start_epoch != None:
            efrom = self.start_epoch
        else:
            efrom = 0

        ones = -1*np.ones(shape=(batch_size,128,128,1))
        twos = 2.*np.ones(shape=(batch_size,128,128,1))

        for epoch in range(efrom,epochs): #these are not proper epochs, it just selects one batch randomly each time

            # Select a random half batch of images
            idx = np.random.randint(0, self.imgs.shape[0], batch_size)
            imgs, labels = self.imgs[idx], self.labels[idx]

            # obtain binary images
            binary_noise = np.random.normal(0,1, (batch_size, 100))
            binary_imgs = self.binary_generator.predict([binary_noise, labels])

            binary_imgs = np.round(binary_imgs)

            if epoch == 0:
                plt.imshow(binary_imgs[0][:,:,0])
                plt.show()
                plt.imshow(binary_imgs[1][:,:,0])
                plt.show()
                plt.imshow(binary_imgs[2][:,:,0])
                plt.show()

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels, binary_imgs, ones, twos])

            if epoch == 0:
                plt.imshow(gen_imgs[0][:,:,0])
                plt.show()
                plt.imshow(gen_imgs[1][:,:,0])
                plt.show()
                plt.imshow(gen_imgs[2][:,:,0])
                plt.show()

            # NOTE GET NOISE LABEL VECTORS
            p_flip = 0.05
            noise_range = 0.1
            valid_noisy  = np.array([random.uniform(1.-noise_range,1.) if (random.uniform(0,1)<1.-p_flip) else random.uniform(0.,noise_range) for _ in range(batch_size)])
            fake_noisy  = np.array([random.uniform(0.,noise_range) if (random.uniform(0,1)<1.-p_flip) else random.uniform(1.-noise_range,1.) for _ in range(batch_size)])

            imgs = imgs + np.random.normal(0, 0.01, size=imgs.shape)
            gen_imgs = gen_imgs + np.random.normal(0, 0.01, size=imgs.shape)

            #Train the discriminator
            #if last_acc > 0.8:
            if (epoch < 200) and (epoch%5!=0):
                print('Only testing discriminator')
                d_loss_real = self.discriminator.test_on_batch([imgs, labels], valid_noisy)
                d_loss_fake = self.discriminator.test_on_batch([gen_imgs, labels], fake_noisy)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid_noisy)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake_noisy)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            idx = np.random.randint(0, self.imgs.shape[0], batch_size)
            sampled_labels = self.labels[idx] #sample random labels from the data for training the generator

            # Train the generator
            if last_acc < 0.7 and 1==2:
                print('Only testing generator')
                g_loss = self.combined.test_on_batch([noise, binary_noise, sampled_labels, ones, twos], valid)
            else:
                g_loss = self.combined.train_on_batch([noise, binary_noise, sampled_labels, ones, twos], valid)


            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100.*d_loss[1], g_loss))
            last_acc = d_loss[1]

            if epoch % 2000 == 0:
                1
                #print('calculating ps...')
                #self.calc_ps(epoch)
                #print('calculating brihgtness peak count...')
                #self.calc_peak_count_brightness(epoch)


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

    def calc_ps(self, epoch):
        if self.find_ps == True:
            self.real_imgs_dict_ps = {}
            for i in self.real_imgs_index:
                intv = int(len(self.real_imgs_index[i])/99)
                index_list = self.real_imgs_index[i][::intv]
                real_imgs = self.imgs[index_list]
                real_imgs = np.squeeze(real_imgs)
                #print(real_imgs)
                #print(np.shape(real_imgs))
                real_ave_ps,real_ps_std,k_list_real = stats_utils.produce_average_ps(real_imgs)
                self.real_imgs_dict_ps[i] = [real_ave_ps[1:],k_list_real[1:]]
            self.find_ps = False

        else:
            noise = np.random.normal(0, 1, (100, 100))

            mal = max(self.real_imgs_index.keys())
            mil = min(self.real_imgs_index.keys())
            for z in self.real_imgs_index:
                z_vec = np.array([[z]]*100)
                z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
                gen_imgs = self.generator.predict([noise, np.reshape(z_vec,(100,1))])
                gen_imgs = self.generator.predict([noise, np.reshape(z_vec,(100,1)), binary_imgs, ones, twos])
                gen_imgs = np.squeeze(gen_imgs)
                #fake_ave_ps,fake_ps_std,k_list_fake = stats_utils.produce_average_ps(gen_imgs)
                plt.errorbar(x=k_list_fake[1:], y=fake_ave_ps[1:], yerr=fake_ps_std[1:], alpha=0.5, label="fake z="+str(z))
                plt.yscale('log')
                plt.plot(self.real_imgs_dict_ps[z][1],self.real_imgs_dict_ps[z][0], label="real z="+str(z))
                plt.yscale('log')
                plt.legend()
                plt.savefig("images/ps_%d.png" % epoch)
            plt.close()

    def calc_peak_count_brightness(self, epoch):
        z_list_pk = range(7,12)
        if self.find_peak_brightness == True:
            self.real_imgs_dict_pk = {}
            for i in self.real_imgs_index:
                intv = int(len(self.real_imgs_index[i])/10)
                index_list = self.real_imgs_index[i][::intv]
                real_imgs = self.imgs[index_list]
                #real_imgs = np.squeeze(real_imgs)
                rl_brightness_list = stats_utils.get_peak_vs_brightness(real_imgs)
                self.real_imgs_dict_pk[i] = [rl_brightness_list]
            self.find_peak_brightness = False

        else:
            noise = np.random.normal(0, 1, (10, 100))

            mal = max(self.real_imgs_index.keys())
            mil = min(self.real_imgs_index.keys())
            self.fake_imgs_dict_pk = {}
            for z in self.real_imgs_index:
                z_vec = np.array([[z]]*100)
                z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
                gen_imgs = self.generator.predict([noise, np.reshape(z_vec,(100,1))])
                #gen_imgs = np.squeeze(gen_imgs)
                fk_brightness_list = stats_utils.get_peak_vs_brightness(gen_imgs)
                self.fake_imgs_dict_pk[z] = [fk_brightness_list]

            r = 5
            c = 1
            fig, axs = plt.subplots(r, c, figsize=(4,18), dpi=250)
            cnt = 0
            for z in z_list_pk:
                axs[cnt].hist(self.real_imgs_dict_pk[z], bins=100, color="blue", label="real", alpha=0.7)
                axs[cnt].hist(self.fake_imgs_dict_pk[z], bins=100, color="orange", label="fake", alpha=0.7)
                axs[cnt].set_title("Labels: %d" % z)
                axs[cnt].legend()
                cnt += 1
            fig.savefig("images/peak_brightness_%d.png" % epoch)
            plt.close()

    def sample_images(self, epoch):

        sample_at0 = [
        [7.],
        [7.5],
        [8.],
        [8.5],
        [9.],
        [9.5],
        [10.],
        [10.5],
        [11.]
        ]
        r = len(sample_at0)
        c = 3

        temp_copy = copy.deepcopy(sample_at0)

        sample_at0 = np.array(sample_at0)
        sample_at = self.scale_labels(sample_at0)
        print('sampling images at labels:', sample_at)

        #noise = np.random.normal(0, 1, (len(sample_at), 100))
        #gen_imgs = self.generator.predict([noise, sample_at])
        ones = -1*np.ones(shape=(len(sample_at),128,128,1))
        twos = 2.*np.ones(shape=(len(sample_at),128,128,1))

        binary_noise = np.random.normal(0,1, (len(sample_at), 100))
        binary_imgs = self.binary_generator.predict([binary_noise, sample_at])
        binary_imgs = np.round(binary_imgs)
        noise = np.random.normal(0, 1, (len(sample_at), 100))
        gen_imgs = self.generator.predict([noise, sample_at, binary_imgs, ones, twos])

        # Rescale images 0 - 1
        ######gen_imgs = 0.5 * gen_imgs + 0.5
        """
        print('fake_min',np.min(gen_imgs))
        print('fake_max',np.max(gen_imgs))

        print('fake_min',np.min(self.imgs))
        print('fake_max',np.max(self.imgs))
        """

        fig, axs = plt.subplots(r, c, figsize=(4,18), dpi=250)
        cnt = 0
        for i in range(r):
            for j in range(c): # c=0: fake, c=1, real
                if cnt >= len(gen_imgs) and j == 0:
                    axs[i,j].axis('off')
                    break
                else:
                    if j == 0:
                        axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='hot', clim=(-1,1))
                        axs[i,j].set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in temp_copy[cnt]))
                        axs[i,j].axis('off')
                        print('fake_min',np.min(gen_imgs[cnt,:,:,0]))
                        print('fake_max',np.max(gen_imgs[cnt,:,:,0]))
                        cnt += 1

                    if j ==1: #show a real image
                        #print(temp_copy[i][0])
                        #print(self.real_imgs_index.keys())
                        if temp_copy[i][0] in self.real_imgs_index: #real images for that z are available

                            sample_i = random.randint(0,len(self.real_imgs_index[temp_copy[i][0]])-1)
                            #print(self.real_imgs_index[temp_copy[i][0]])
                            sample_i = self.real_imgs_index[temp_copy[i][0]][sample_i]
                            #print(sample_i)
                            #print(self.imgs.shape)
                            axs[i,j].imshow(self.imgs[sample_i,:,:,0], cmap='hot', clim=(-1,1))
                            axs[i,j].set_title("")
                            print('real_min',np.min(self.imgs[sample_i,:,:,0]))
                            print('real_max',np.max(self.imgs[sample_i,:,:,0]))
                        axs[i,j].axis('off')

                    if j ==2:
                        if temp_copy[i][0] in self.real_imgs_index: #real images for that z are available
                            fake_pix_val = stats_utils.get_pixel_val(gen_imgs[[cnt-1],:,:,:])
                            real_pix_val = stats_utils.get_pixel_val(self.imgs[[sample_i],:,:,:])
                            axs[i,j].hist(real_pix_val, bins=100, color="blue", label="real", alpha=0.7)
                            axs[i,j].hist(fake_pix_val, bins=100, color="orange", label="fake", alpha=0.7)
                            axs[i,j].legend()

        if platform == 'linux':
            user = utils.get_user()
            print(user)
            fig.savefig(r"/home/" + user + r"/MSci2/images/%d.png" % epoch)
        else: #windows
            fig.savefig("images/%d.png" % epoch)
        plt.close()



if __name__ == '__main__':

    args = sys.argv
    if len(args) == 3:
        if args[2] == 'binary':
            if args[1] == 'new':
                bcgan = binary_CGAN(use_old_model=False)
                bcgan.train(epochs=400000, batch_size=4, sample_interval=100, save_model_interval = 500)
            elif args[1] == 'continue':
                bcgan = binary_CGAN(use_old_model=True)
                bcgan.train(epochs=400000, batch_size=4, sample_interval=50, save_model_interval = 500)
            else:
                print('argument not recognised')

        elif args[2] == 'combined':
            if args[1] == 'new':
                ccgan = combined_CGAN(use_old_model=False)
                ccgan.train(epochs=400000, batch_size=4, sample_interval=50, save_model_interval=500)
            else:
                print('WHUUUUUUU HAHAHAHA')
        else:
            print('argument not recognised')
    else:
        print('2 arguments required: (new,continue) (binary, combined)')
