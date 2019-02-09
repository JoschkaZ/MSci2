from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Add, ReLU
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle as pkl

class CGAN():
    def __init__(self):

        self.imgs = []
        self.labels = []
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_dim = 100
        self.con_dim = 1
        self.start_time = str(time.time()).split('.')[0]

        self.read_data()

        optimizer = Adam(0.0002, 0.5)

        ph_img = Input(shape=self.img_shape) # no , ?
        ph_con = Input(shape=(self.con_dim,))
        self.discriminator = self.build_discriminator(ph_img,ph_con)
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])

        noise = Input(shape=(self.noise_dim,))
        con = Input(shape=(self.con_dim,))
        self.generator = self.build_generator(noise, con)

        img = self.generator([noise, con])

        self.discriminator.trainable = False #combined model

        valid = self.discriminator([img, con])

        self.combined = Model([noise, con], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)


    def build_generator(self, noise, con):

        #con2 = Dense(5, activation='tanh')(con)
        #con2 = Dense(22, activation='tanh')(con2)
        con1 = Dense(2, activation='tanh')(con) #TODO this is likely bad because it squases the ends too much
        con1 = Dense(4, activation='tanh')(con1)
        con1 = Dense(8, activation='tanh')(con1)
        con1 = Dense(16, activation='tanh')(con1)
        con1 = Dense(32, activation='tanh')(con1)
        con1 = Dense(64, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)
        #100

        noise1 = noise

        merged_input = Concatenate()([con1, noise1])
        #100+100

        #mnist version
        hid = Dense(560)(merged_input)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #560

        hid = Dense(128 * 7 * 7)(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #128*7*7


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
        out = Activation("tanh")(hid)
        #28x28x1
        '''

        hid = Dense(128 * 7 * 7)(merged_input)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        #128*7*7

        hid = Reshape((7, 7, 128))(hid)
        #7x7x128

        hid = Conv2D(128, kernel_size=5, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        #7x7x128

        hid = Conv2DTranspose(128, 5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        #14x14x128

        hid = Conv2D(128, kernel_size=5, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        #14x14x128

        hid = Conv2DTranspose(128, 5, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        #28x28x128

        hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        #28x28x128

        hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)
        #28x28x128

        hid = Conv2D(1, kernel_size=5, strides=1, padding="same")(hid)
        out = Activation("tanh")(hid)
        #28x28x1
        '''

        model =  Model([noise, con], out)
        model.summary()
        return model


    def build_discriminator(self, img, con):

        hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(img)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #28x28x128

        hid = Conv2D(128, kernel_size=4, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #28x28x128

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #14x14x128

        hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #14x14x128

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #7x7x128

        hid = Flatten()(hid)
        #7*7*128

        hid = Dense(560, activation='tanh')(hid)
        #640

        hid = Dense(100, activation='tanh')(hid)
        #64

        con1 = Dense(2, activation='tanh')(con) #TODO same issue here....
        con1 = Dense(4, activation='tanh')(con1)
        con1 = Dense(8, activation='tanh')(con1)
        con1 = Dense(16, activation='tanh')(con1)
        con1 = Dense(32, activation='tanh')(con1)
        con1 = Dense(64, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)
        #con1 = Dense(64, activation='tanh')(con1)

        merged_layer = Concatenate()([hid, con1])
        #100+100

        hid = Dense(34, activation='tanh')(merged_layer)
        #34
        hid = Dense(6, activation='tanh')(hid)
        #6
        out = Dense(1, activation='sigmoid')(hid)
        #1

        model = Model(inputs=[img, con], outputs=out)

        model.summary()

        return model

    def read_data(self):

        print('importing data...')
        #data = pkl.load( open( "C:\Outputs\slices2_32.pkl", "rb" ) )
        data = pkl.load(open("/home/hk2315/MSci2/faketest_images.pkl", "rb"))
        #data = pkl.load(open(r"C:\\Users\\Joschka\\github\\MSci2\\faketest_images.pkl", "rb"))
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

    def min_max_scale_images(self):
        print('minmax scaling images...')
        print(self.imgs.shape)
        mmax = np.max(self.imgs) #shouldnt the scaling be done per image rather than looking at min/max out of all images???
        mmin = np.min(self.imgs)
        #print('mmax', mmax)
        #print('mmin', mmin)
        #print(self.imgs[0][14])
        self.imgs = (self.imgs.astype(np.float32) - (mmax+mmin)/2.) / ((mmax-mmin) / 2.)
        #print(self.imgs[0][14])
        print('expanding dimension of images...')
        self.imgs = np.expand_dims(self.imgs, axis=3)

    def scale_labels(self, l, verbose=0):
        length = len(l)
        step = int(length / 5.)
        if verbose == 0:
            print('scaling labels...')
            print('initial labels')
            print(l[0:length:step])
            print(l.max(axis=0))
            print(l.min(axis=0))
        mal = l.max(axis=0)
        mil = l.min(axis=0)

        for i in range(len(l)):
            for j in range(len(l[i])):
                l[i][j] = (l[i][j].astype(np.float32) - (mal+mil)/2.) / ((mal-mil)/2.) #wouldnt j always be 0? (thats not a problem but just a question)

        if verbose == 0:
            print('scaled labels')
            print(l[0:length:step])

        return l

    def train(self, epochs, batch_size=128, sample_interval=50, save_multiple=10):

        #SCALE IMAGES
        self.min_max_scale_images()
        self.labels = self.scale_labels(self.labels)

        print('FINAL SHAPES')
        print(self.imgs.shape)
        print(self.labels.shape)

        """
        (X_train, y_train), (_, _) = mnist.load_data()
        print(X_train.shape)
        print(y_train.shape)

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)
        y_train = (y_train.astype(np.float32)-4.5) / 4.5
        print(y_train)
        print(len(y_train))

        print('FINAL SHAPES')
        print(X_train.shape)
        print(y_train.shape)
        #print(X_train[0])
        """
        print('FINAL SHAPES')
        print(self.imgs.shape)
        print(self.labels.shape)
        #print(self.imgs[0])

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        last_acc = .75

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            """
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]
            """
            idx = np.random.randint(0, self.imgs.shape[0], batch_size)
            imgs, labels = self.imgs[idx], self.labels[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            #print(noise)
            #print(labels)
            #input('...')
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            if last_acc > 0.8:
                print('Only testing discriminator')
                d_loss_real = self.discriminator.test_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.test_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            sampled_labels = (sampled_labels.astype(np.float32)-4.5) / 4.5

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
                if epoch % (sample_interval*save_multiple) == 0:
                    self.discriminator.save('models/discriminator_' + str(self.start_time) + '.h5')
                    self.combined.save('models/combined_' + str(self.start_time) + '.h5')
                    self.generator.save('models/generator_' + str(self.start_time) + '.h5')

                self.sample_images(epoch)


    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        sampled_labels = (sampled_labels.astype(np.float32)-4.5) / 4.5

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0])
                axs[i,j].set_title("Digit: %f" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=256, sample_interval=10, save_multiple = 10)
