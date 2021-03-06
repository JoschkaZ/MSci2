
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
import pickle as pkl


class CGAN():

    def __init__(self):
        self.imgs = []
        self.labels = []
        self.start_time = str(time.time()).split('.')[0]
        self.label_dim = -1

        self.read_data()

        optimizer = Adam(0.0002, 0.5)

        ph_img = Input(shape=(32,32,1))
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

        print('importing data...')
        data = pkl.load( open( "C:\Outputs\slices2_32.pkl", "rb" ) )

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

        return 'Done'

    def build_generator(self, noise, con):

        #EXPAND LABELS
        con1 = Dense(32, activation='tanh')(con)
        con1 = Dense(64, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)

        #COMBINE LABELS AND NOISE
        merged_input = Concatenate()([con1, noise])
        # -> 200

        hid = Dense(500)(merged_input)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 1280

        hid = Dense(128 * 4 * 4)(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 128*8*8 = 8192

        hid = Reshape((4, 4, 128))(hid)
        # -> 4x4x512

        hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 8x8x256

        hid = Conv2D(128, kernel_size=4, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 8x8x128

        hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 16x16x128

        hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 16x16x128

        hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 32x32x128

        hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 32x32x128

        hid = Conv2D(1, kernel_size=5, strides=1, padding="same")(hid)
        out = Activation("tanh")(hid)
        # -> 32x32x1

        model =  Model([noise, con], out)
        model.summary()
        return model

    def build_discriminator(self, img, con):

        hid = Conv2D(128, kernel_size=4, strides=1, padding='same')(img)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 32x32x128

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 16x16x128

        hid = Conv2D(128, kernel_size=4, strides=1, padding='same')(img)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 16x16x128

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 8x8x256

        hid = Conv2D(128, kernel_size=4, strides=1, padding='same')(img)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 8x8x256

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        # -> 4x4x512

        hid = Flatten()(hid)
        # -> 4*4*512 = 8192

        hid = Dense(500, activation='tanh')(hid)
        # -> 560

        hid = Dense(100, activation='tanh')(hid)
        # -> 100

        #EXPAND LABEL
        con1 = Dense(8)(con)
        con1 = Dense(16, activation='tanh')(con1)
        con1 = Dense(32, activation='tanh')(con1)
        con1 = Dense(64, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)

        merged_layer = Concatenate()([hid, con1])
        # -> 200

        hid = Dense(34, activation='tanh')(merged_layer)
        # -> 34
        hid = Dense(6, activation='tanh')(hid)
        # -> 6
        out = Dense(1, activation='sigmoid')(hid)
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

        for epoch in range(epochs): #these are not proper epochs, it just selects one batch randomly each time

            # Select a random half batch of images
            idx = np.random.randint(0, self.imgs.shape[0], batch_size)
            imgs, labels = self.imgs[idx], self.labels[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            gen_imgs = self.generator.predict([noise, labels])

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
            if last_acc < 0.7 and 1==2:
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
                self.discriminator.save('models/2132discriminator_' + str(self.start_time) + '.h5')
                self.combined.save('models/2132combined_' + str(self.start_time) + '.h5')
                self.generator.save('models/2132generator_' + str(self.start_time) + '.h5')

    def sample_images(self, epoch):

        sample_at = [
        [6.],
        [7.],
        [8.],
        [9.],
        [10.],
        [11.],
        [12.]
        ]
        r = 4
        c = 2

        sample_at = np.array(sample_at)
        sample_at = self.scale_labels(sample_at)
        print('sampling images at labels:', sample_at)

        noise = np.random.normal(0, 1, (len(sample_at), 100))

        gen_imgs = self.generator.predict([noise, sample_at])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5


        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if cnt >= len(gen_imgs):
                    axs[i,j].axis('off')
                    break
                else:
                    axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='hot')
                    axs[i,j].set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in sample_at[cnt]))
                    #axs[i,j].set_title("Digit: %d" % '_'.join(sample_at[cnt]))
                    axs[i,j].axis('off')
                    cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()



if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=128, sample_interval=10, save_model_interval = 100)
