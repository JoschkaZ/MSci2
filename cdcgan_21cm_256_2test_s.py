
from __future__ import print_function, division
#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Add, ReLU
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu, tanh
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

class CGAN():

    def __init__(self, use_old_model):
        self.imgs = []
        self.labels = []
        self.start_time = str(time.time()).split('.')[0]
        self.label_dim = -1
        self.start_epoch = None

        self.read_data()

        optimizer = Adam(0.0002, 0.5)
        ph_img = Input(shape=(64,64,1))
        ph_label = Input(shape=(self.label_dim,))

        self.find_ps = True

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

        if use_old_model == True:
            # identify model to load
            #user = utils.get_user()
            #mypath = r'/home/' + user + r'/MSci2/models'
            mypath = r'C:\\Users\\Joschka\\github\\MSci2\\models'
            files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            print('Models found:')
            print(files)
            time_to_load = 0
            for file in files:
                if 'weights' in file:
                    model_time = int(file.split('weights_')[-1].split('.')[0])
                    time_to_load = max(time_to_load, model_time)
            print('Time to load: ', time_to_load)

            # read epoch from images NOTE this assumes that the image with the heighest epoch was created by the newest model
            #imgpath = r'/home/' + user + r'/MSci2/images'
            imgpath = r'C:\\Users\\Joschka\\github\\MSci2\\Images'
            files = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
            start_epoch = 0
            for file in files:
                if '_' not in file:
                    img_epoch = int(file.split('.')[0])
                    start_epoch = max(start_epoch, img_epoch)

            print('Epoch to start at: ', start_epoch)
            self.start_epoch = start_epoch +1

            print('Reading weights from: ')
            print(mypath +'/21256discriminator_' + str(time_to_load) + '.h5')
            self.discriminator.load_weights(mypath +'/21256discriminator_' + str(time_to_load) + '.h5')
            print(mypath +'/21256generator_' + str(time_to_load) + '.h5')
            self.generator.load_weights(mypath +'/21256generator_' + str(time_to_load) + '.h5')
            print(mypath +'/21256combined_' + str(time_to_load) + '.h5')
            self.combined.load_weights(mypath +'/21256combined_' + str(time_to_load) + '.h5')





    '''

    def __init__(self, use_old_model):
        self.imgs = []
        self.labels = []
        self.start_time = str(time.time()).split('.')[0]
        self.label_dim = -1
        self.start_epoch = None

        self.read_data()

        if use_old_model == False:
            optimizer = Adam(0.00005, 0.5)
            ph_img = Input(shape=(256,256,1))
            ph_label = Input(shape=(self.label_dim,))

        self.find_ps = True

        if use_old_model == False:
            self.discriminator = self.build_discriminator(ph_img, ph_label)
            self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])
        else:

            # identify model to load
            user = utils.get_user()
            mypath = r'/home/' + user + r'/MSci2/models'
            files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            print('Models found:')
            print(files)
            time_to_load = 0
            for file in files:
                model_time = int(file.split('_')[-1].split('.')[0])
                time_to_load = max(time_to_load, model_time)
            print('Time to load: ', time_to_load)

            # read epoch from images
            imgpath = r'/home/' + user + r'/MSci2/images'
            files = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
            start_epoch = 0
            for file in files:
                if '_' not in file:
                    img_epoch = int(file.split('.')[0])
                    start_epoch = max(start_epoch, img_epoch)

            print('Epoch to start at: ', start_epoch)
            self.start_epoch = start_epoch +1

            print('Reading discriminator from: ', mypath +'/21256discriminator_' + str(time_to_load) + '.h5')
            #self.discriminator = load_model(mypath +'/21256discriminator_' + str(time_to_load) + '.h5')



        if use_old_model == False:
            noise = Input(shape=(100,))
            label = Input(shape=(self.label_dim,))
            self.generator = self.build_generator(noise, label, use_old_model)
            img = self.generator([noise, label])
        else:
            print('Reading generator from: ', mypath +'/21256generator_' + str(time_to_load) + '.h5')
            #self.generator = load_model(mypath +'/21256generator_' + str(time_to_load) + '.h5')


        if use_old_model == False:
            self.discriminator.trainable = False
            valid = self.discriminator([img, label])
            self.combined = Model([noise, label], valid)
            self.combined.compile(loss=['binary_crossentropy'],
                optimizer=optimizer)
        else:
            print('Reading combined from: ', mypath +'/21256combined_' + str(time_to_load) + '.h5')
            self.combined = load_model(mypath +'/21256combined_' + str(time_to_load) + '.h5')
    '''



    def read_data(self):

        print('importing data...')
        #data = pkl.load( open( "C:\Outputs\slices2_32.pkl", "rb" ) )
        #data = pkl.load(open("/home/jz8415/slices2.pkl", "rb"))
        data = pkl.load(open(r"C:\\Users\\Joschka\\github\\MSci2\\faketest_images.pkl", "rb"))
        print('data imported!')

        self.imgs = []
        self.labels = []
        for entry in data:
            img = entry[0]
            l_str = entry[1]

            #DEFINE LABELS HERE
            l_z = float(l_str.split('_z')[1].split('_')[0])

            #APPEND LABELS HERE
            self.imgs.append(img)
            self.labels.append([l_z])

        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        self.label_dim = len(self.labels[0])
        print('dimension of label: ', self.label_dim)

        print('Shapes:')
        print(self.imgs.shape)
        print(self.labels.shape)
        #(60000, 28, 28)
        #(60000,1) #after reshaping

        self.real_imgs_index = {}
        for i, z in enumerate(self.labels):
            z = z[0]
            if not z in self.real_imgs_index:
                self.real_imgs_index[z] = []
            self.real_imgs_index[z].append(i)

        return 'Done'

    def build_generator(self, noise, con):

        #con2 = Dense(5, activation='tanh')(con)
        #con2 = Dense(22, activation='tanh')(con2)
        con1 = Dense(2)(con) #TODO this is likely bad because it squases the ends too much
        con1 = Dense(4)(con1)
        con1 = Dense(8)(con1)
        con1 = Dense(16, activation='tanh')(con1)
        con1 = Dense(32, activation='tanh')(con1)
        con1 = Dense(64, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)
        #100

        noise1 = Dense(100,activation='tanh')(noise)

        #merged_input = Concatenate()([con1, noise1])
        #100+100

        hid = Dense(560)(noise1)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #560

        hid = Dense(128 * 8 * 8)(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #128*7*7


        hid = Reshape((8, 8, 128))(hid)
        #8

        hid = Conv2D(128, kernel_size=4, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #8

        hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #16

        hid = Conv2D(128, kernel_size=5, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #116

        hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #32

        hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #32

        hid = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #64

        hid = Conv2D(1, kernel_size=5, strides=1, padding="same")(hid)
        out = Activation("tanh")(hid)
        #64

        model =  Model([noise, con], out)
        model.summary()
        return model

        '''
        noise1 = BatchNormalization(momentum=0.9)(noise)

        #con1 = Dense(100, activation='tanh')(con)
        #con1 = BatchNormalization(momentum=0.9)(con1)

        #merged_input = Concatenate()([con1, noise1])
        # -> 200

        hid = Dense(256*4*4, activation='relu')(noise1) #TODO removed conditional data
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        # -> 256*4*4

        hid = Reshape((4,4,256))(hid)
        # -> 256x4x4

        hid = Conv2DTranspose(128, 5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        # -> 128x8x8

        hid = Conv2DTranspose(64, 5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        # -> 64x16x16

        hid = Conv2DTranspose(32, 5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        # -> 32x32x32

        hid = Conv2DTranspose(1, 5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        out = Activation('tanh')(hid)
        # -> 1x64x64

        model =  Model([noise, con], out)
        model.summary()
        return model
        '''

    def build_discriminator(self, img, con):

        '''
        hid = Conv2D(32, kernel_size=5, strides=2, padding="same")(img)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)
        # -> 32x32x32

        hid = Conv2D(64, kernel_size=5, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)
        # -> 64x16x16

        hid = Conv2D(128, kernel_size=5, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)
        # -> 128x8x8

        hid = Conv2D(256, kernel_size=5, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)
        # -> 256x4x4

        hid = Flatten()(hid)

        hid = Dense(100, activation='tanh')(hid)
        hid = BatchNormalization(momentum=0.8)(hid)

        #EXPAND LABEL
        #con1 = Dense(100, activation='tanh')(con)
        #con1 = BatchNormalization(momentum=0.8)(con1)

        #merged_layer = Concatenate()([hid, con1]) #TODO removed conditional
        # -> 200

        out = Dense(1, activation='sigmoid')(hid)
        # -> 1

        model = Model(inputs=[img, con], outputs=out)
        model.summary()
        return model
        '''


        hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(img)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #64

        hid = Conv2D(128, kernel_size=4, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #64

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #32

        hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #32

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #16

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #8

        hid = Flatten()(hid)
        #8

        hid = Dense(560, activation='tanh')(hid)
        #640

        hid = Dense(100, activation='tanh')(hid)
        #64

        con1 = Dense(2)(con) #TODO same issue here....
        con1 = Dense(4)(con1)
        con1 = Dense(8)(con1)
        con1 = Dense(16, activation='tanh')(con1)
        con1 = Dense(32, activation='tanh')(con1)
        con1 = Dense(64, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)
        #con1 = Dense(64, activation='tanh')(con1)

        #merged_layer = Concatenate()([hid, con1])
        #100+100

        hid = Dense(34, activation='tanh')(hid)
        #34
        hid = Dense(6, activation='tanh')(hid)
        #6
        out = Dense(1, activation='sigmoid')(hid)
        #1

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

    def train(self, epochs, batch_size=128, sample_interval=50, save_model_interval=500):

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
        for epoch in range(efrom,epochs): #these are not proper epochs, it just selects one batch randomly each time

            # Select a random half batch of images
            idx = np.random.randint(0, self.imgs.shape[0], batch_size)
            imgs, labels = self.imgs[idx], self.labels[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels])

            #print(imgs.shape)
            #print(gen_imgs.shape)
            #plt.imshow(np.squeeze(imgs[0]))
            #plt.show()
            #plt.imshow(np.squeeze(gen_imgs[0]))
            #plt.show()

            #Train the discriminator
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
            if last_acc < 0.7: #and 1==2:
                print('Only testing generator')
                g_loss = self.combined.test_on_batch([noise, sampled_labels], valid)
            else:
                g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)


            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100.*d_loss[1], g_loss))
            last_acc = d_loss[1]

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

                #if epoch % 2000 == 0:
                print('calculating stats...')
                #self.calc_stats(epoch)

    def calc_stats(self, epoch):
        if self.find_ps == True:
            self.real_imgs_dict = {}
            for i in self.real_imgs_index:
                intv = int(len(self.real_imgs_index[i])/99)
                index_list = self.real_imgs_index[i][::intv]
                real_imgs = self.imgs[index_list]
                real_imgs = np.squeeze(real_imgs)
                #print(real_imgs)
                #print(np.shape(real_imgs))
                real_ave_ps,real_ps_std,k_list_real = stats_utils.produce_average_ps(real_imgs)
                self.real_imgs_dict[i] = [real_ave_ps[1:],k_list_real[1:]]
            self.find_ps = False

        else:
            noise = np.random.normal(0, 1, (100, 100))

            mal = max(self.real_imgs_index.keys())
            mil = min(self.real_imgs_index.keys())
            for z in self.real_imgs_index:
                z_vec = np.array([[z]]*100)
                z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
                gen_imgs = self.generator.predict([noise, np.reshape(z_vec,(100,1))])
                gen_imgs = np.squeeze(gen_imgs)
                fake_ave_ps,fake_ps_std,k_list_fake = stats_utils.produce_average_ps(gen_imgs)
                plt.errorbar(x=k_list_fake[1:], y=fake_ave_ps[1:], yerr=fake_ps_std[1:], alpha=0.5, label="fake z="+str(z))
                plt.yscale('log')
                plt.plot(self.real_imgs_dict[z][1],self.real_imgs_dict[z][0], label="real z="+str(z))
                plt.yscale('log')
                plt.legend()
            if platform == 'linux':
                user = utils.get_user()
                print(user)
                plt.savefig(r"/home/" + user + r"/MSci2/images/ps_%d.png" % epoch)
            else: #windows
                plt.savefig("images/ps_%d.png" % epoch)
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
        c = 2

        temp_copy = copy.deepcopy(sample_at0)

        sample_at0 = np.array(sample_at0)
        sample_at = self.scale_labels(sample_at0)
        print('sampling images at labels:', sample_at)

        noise = np.random.normal(0, 1, (len(sample_at), 100))

        gen_imgs = self.generator.predict([noise, sample_at])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5


        fig, axs = plt.subplots(r, c, figsize=(4,18), dpi=250)
        cnt = 0
        for i in range(r):
            for j in range(c): # c=0: fake, c=1, real
                if cnt >= len(gen_imgs) and j == 0:
                    axs[i,j].axis('off')
                    break
                else:
                    if j == 0:
                        axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='hot')
                        axs[i,j].set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in temp_copy[cnt]))
                        axs[i,j].axis('off')
                        cnt += 1
                    else: #show a real image
                        #print(temp_copy[i][0])
                        #print(self.real_imgs_index.keys())
                        if temp_copy[i][0] in self.real_imgs_index: #real images for that z are available
                            sample_i = random.randint(0,len(self.real_imgs_index[temp_copy[i][0]])-1) #TODO
                            #print(self.real_imgs_index[temp_copy[i][0]])
                            sample_i = self.real_imgs_index[temp_copy[i][0]][sample_i]
                            #print(sample_i)
                            #print(self.imgs.shape)
                            axs[i,j].imshow(self.imgs[sample_i,:,:,0], cmap='hot')
                            axs[i,j].set_title("")
                        axs[i,j].axis('off')
        if platform == 'linux':
            user = utils.get_user()
            print(user)
            fig.savefig(r"/home/" + user + r"/MSci2/images/%d.png" % epoch)
        else: #windows
            fig.savefig("images/%d.png" % epoch)
        plt.close()

if __name__ == '__main__':

    args = sys.argv
    if args[1] == 'new':
        cgan = CGAN(use_old_model=False)
        #cgan.train(epochs=20000, batch_size=128, sample_interval=10, save_model_interval = 100)
        cgan.train(epochs=400000, batch_size=128, sample_interval=50, save_model_interval = 2000)
    elif args[1] == 'continue':
        cgan = CGAN(use_old_model=True)
        #cgan.train(epochs=20000, batch_size=128, sample_interval=10, save_model_interval = 100)
        cgan.train(epochs=400000, batch_size=128, sample_interval=50, save_model_interval = 2000)
    else:
        print('Argument required.')
        print('write: "python cdcgan_21cm_256.py new" to use a new model.')
        print('write: "python cdcgan_21cm_256.py continue" to continue training the last model.')