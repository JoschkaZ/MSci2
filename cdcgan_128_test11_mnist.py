from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Add, Multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time

class CGAN():
    def __init__(self):

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_dim = 100
        self.con_dim = 1
        self.start_time = str(time.time()).split('.')[0]

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

        n_channel = 64
        kernel_size = 3

        con1 = Dense(n_channel, activation='tanh')(con) #model settings
        con1 = Reshape((1,1,n_channel))(con1)
        con1 = UpSampling2D((28,28))(con1)

        hid = Dense(n_channel*7*7, activation='relu')(noise)
        hid = Reshape((7,7,n_channel))(hid)

        hid = Conv2DTranspose(n_channel, kernel_size=kernel_size, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = Activation("relu")(hid)

        hid = Conv2DTranspose(n_channel, kernel_size=kernel_size, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = Activation("relu")(hid) # -> 128x144x144
        hid = Multiply()([hid, con1])

        hid = Conv2D(n_channel, kernel_size=kernel_size, strides=1, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = Activation("relu")(hid) # -> 128x144x144
        hid = Multiply()([hid, con1])

        hid = Conv2D(n_channel, kernel_size=kernel_size, strides=1, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = Activation("relu")(hid) # -> 128x144x144
        hid = Multiply()([hid, con1])

        hid = Conv2D(1, kernel_size=kernel_size, strides=1, padding="same")(hid)
        out = Activation("tanh")(hid)

        model =  Model([noise, con], out)
        model.summary()
        return model


    def build_discriminator(self, img, con):

        n_channel = 64
        kernel_size = 3

        con1 = Dense(n_channel, activation='tanh')(con) #model settings
        con1 = Reshape((1,1,n_channel))(con1)
        con1 = UpSampling2D((28,28))(con1)


        hid = Conv2D(n_channel, kernel_size=kernel_size, strides=1, padding="same")(img)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid) # -> 32
        hid = Multiply()([hid, con1]) # -> 128x128xn_channel

        hid = Conv2D(n_channel, kernel_size=kernel_size, strides=1, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid) # -> 32
        hid = Multiply()([hid, con1])

        hid = Conv2D(n_channel, kernel_size=kernel_size, strides=1, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid) # -> 32
        hid = Multiply()([hid, con1])


        hid = Conv2D(n_channel, kernel_size=kernel_size, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid) # -> 64

        hid = Conv2D(n_channel, kernel_size=kernel_size, strides=2, padding="same")(hid)
        hid = BatchNormalization(momentum=0.8)(hid)
        hid = LeakyReLU(alpha=0.2)(hid) # -> 32

        hid = Flatten()(hid)

        hid = Dropout(0.1)(hid)

        out = Dense(1, activation='sigmoid')(hid)

        model = Model(inputs=[img, con], outputs=out)
        model.summary()
        return model


    def train(self, epochs, batch_size=128, sample_interval=50, save_multiple=10):

        # Load the dataset
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

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        last_acc = .75

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            #print(noise)
            #print(labels)
            #input('...')
            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            if last_acc > 0.8 and False:
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
            if last_acc < 0.7 and False:
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
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=256, sample_interval=10, save_multiple = 10)
