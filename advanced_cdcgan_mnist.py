from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

class CGAN():
    def __init__(self):

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_dim = 100
        self.con_dim = 1

        optimizer = Adam(0.0002, 0.5)

        ph_img = Input(shape=self.img_shape) # no , ?
        ph_con = Input(shape=(self.con_dim,))
        self.discriminator = self.build_discriminator(ph_img,ph_con)
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=optimizer,metrics=['accuracy'])

        noise = Input(shape=(self.latent_dim,))
        con = Input(shape=(con_dim,))
        self.generator = self.build_generator(noise, con)

        img = self.generator([noise, con])

        self.discriminator.trainable = False #combined model

        valid = self.discriminator([img, con])

        # stacked
        self.combined = Model([noise, con], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)


    def build_generator(self, noise, con):

        hid = Dense(16, activation='relu')(noise)
        merged_input = Concatenate()([hid, con])
        #?+10

        hid = Dense(128 * 7 * 7, activation='relu')(merged_input)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        hid = Reshape((7, 7, 128))(hid)
        #8x8x128

        hid = Conv2D(128, kernel_size=4, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #8x8x128

        hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #16x16x128

        hid = Conv2D(128, kernel_size=5, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #16x16x128

        hid = Conv2DTranspose(128, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #32x32x128

        hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #32x32x128

        hid = Conv2D(128, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #32x32x128
        hid = Conv2D(1, kernel_size=5, strides=1, padding="same")(hid)
        out = Activation("tanh")(hid)

        model =  Model([input_layer, condition_layer], out)
        model.summary()
        return model


    def build_discriminator(self, img, con):

        #input layer should be a crazy image yo (32x32x3)
        hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(img)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #32x32x128

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #16x16x128

        hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        #7x7x128

        hid = Flatten()(hid)
        #4*4*128

        hid = Dense(16, activation='relu')(hid)

        merged_layer = Concatenate()([hid, con])
        #4*4*128+?
        hid = Dense(512, activation='relu')(merged_layer)
        #hid = Dropout(0.4)(hid)
        #512

        out = Dense(1, activation='sigmoid')(hid)
        #1

        model = Model(inputs=[input_layer, condition_layer], outputs=out)

        model.summary()

        return model


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
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
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

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
cgan.train(epochs=3000, batch_size=32, sample_interval=200)
