from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import matplotlib.pyplot as plt
import sys
from sys import platform
import numpy as np
import scipy
from scipy import stats
from scipy.misc import imresize
import pickle
import tensorflow as tf

try:
    %run utils.ipynb
    user = get_user()
except:
    import utils
    user = utils.get_user()


def crossraPsd2d(img1,img2,show=False):
    s1 = len(img1)
    s2 = len(img2)

    img1 = np.array(img1) - np.mean(img1)
    img2 = np.array(img2) - np.mean(img2)
    #a = np.random.randint(2, size=(10,10))
    #k = [[1,1,1],[1,1,1],[1,1,1]]

    tensor_a = tf.constant(img1, tf.float32)
    tensor_k = tf.constant(img2, tf.float32)
    conv = tf.nn.convolution(
    tf.reshape(tensor_a, [1, s1, s1, 1]),
    tf.reshape(tensor_k, [s2, s2, 1, 1]),
    #use_cudnn_on_gpu=True,
    padding='SAME')
    conv = tf.Session().run(conv)
    conv = np.reshape(conv,(s1,s1))


    k = np.ones((s1,s1))
    #print(k)
    tensor_a = tf.constant(k, tf.float32)
    tensor_k = tf.constant(k, tf.float32)
    convc = tf.nn.convolution(
    tf.reshape(tensor_a, [1, s1, s1, 1]),
    tf.reshape(tensor_k, [s1, s1, 1, 1]),
    #use_cudnn_on_gpu=True,
    padding='SAME')
    convc = tf.Session().run(convc)
    convc = np.reshape(convc,(s1,s1))

    conv = conv / convc

    imgf = np.fft.fft2(conv)
    imgfs = np.fft.fftshift(imgf)
    S = np.zeros(128)
    Sconv = np.zeros(128)
    C = np.zeros(128)
    k_list = []
    for i in range(256):
        for j in range(256):

            i2 = i - (256.-1)/2.
            j2 = j - (256.-1)/2.

            r = int(np.round(np.sqrt(i2**2+j2**2)))


            if r <= 127:
                S[r] += imgfs[i][j]
                Sconv[r] += conv[i][j]
                C[r] += 1

    for i in range(128):
        k = i*(1.*2*np.pi/300)
        if C[i] == 0:
            S[i] = 0
            Sconv[i] = 0
        else:
            #print(k**2 * S[i] / C[i])
            S[i] = np.real(k**2 * S[i] / C[i])
            Sconv[i] = np.real(k**0 * Sconv[i] / C[i])

        k_list.append(k)

    if show == True:
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
        plt.imshow(conv)
        plt.show()
        plt.imshow(convc)
        plt.show()

    #S,k_list = raPsd2d(conv,s1,show=show)



    return S,Sconv,k_list









def raPsd2d(img, res, show=False):
    # get averaged power spectral density of image with resolution res


    #compute power spectrum
    imgf = np.fft.fft2(img)
    imgfs = np.fft.fftshift(imgf)
    imgfsp = (np.abs(imgfs)) **2.
    #print(np.shape(imgfsp))

    S = np.zeros(128)
    C = np.zeros(128)
    k_list = []
    for i in range(256):
        for j in range(256):

            i2 = i - (256.-1)/2.
            j2 = j - (256.-1)/2.

            r = int(np.round(np.sqrt(i2**2+j2**2)))


            if r <= 127:
                S[r] += imgfsp[i][j]
                C[r] += 1

    for i in range(128):
        k = i*(1.*2*np.pi/300)
        if C[i] == 0:
            S[i] = 0
        else:
            S[i] = k**2 * S[i] / C[i]

        k_list.append(k)

    if show == True:
        print('Original')
        plt.imshow(np.log(np.abs(img)), cmap='hot', interpolation='nearest')
        plt.show()
        print('Fourier')
        plt.imshow(np.log(np.abs(imgf)), cmap='hot', interpolation='nearest')
        plt.show()
        print('Fourier + Shift')
        plt.imshow(np.log(np.abs(imgfs)), cmap='hot', interpolation='nearest')
        plt.show()
        print('Fourier + Shift + Squared')
        plt.imshow(np.log(np.abs(imgfsp)), cmap='hot', interpolation='nearest')
        plt.show()

    return S,k_list



def produce_average_ps(slices):
    PS = np.zeros(127) #the first value of the PS is always zero so ignore that
    N = len(slices)
    values_list = [ [] for i in range(127) ] #list of 127 empty lists
    std_list = []
    S0 = []
    S1 = []

    for i in range(N):
        slc = slices[i]
        S,k_list = raPsd2d(slc,(256,256))
        S = S[1:] #ignore the first element in PS becuase its always zero
        #k_list = k_list[1:]
        #S0.append(S[0])
        #S1.append(S[1])
        PS = np.add(PS,S)
        for j in range(len(S)):
            values_list[j].append(S[j])
    PS = PS / N
    #print(len(k_list))
    k_list = k_list[1:]
    #print(len(k_list))
    #print(S0)
    #print(S1)

    for k in range(len(values_list)):
        std = np.std(values_list[k])
        std_list.append(std)
    return PS,std_list,k_list



def compare_ps(real_PS,fake_PS,ps_std):
    diff = np.subtract(real_PS,fake_PS)
    for i in range(len(diff)):
        diff[i] = (1.*diff[i])/ps_std[i]
    mod_diff2 = diff**2
    tot_diff = np.sum(mod_diff2)
    return tot_diff



def get_pk_hist(slices):
    N = len(slices)
    count_list = []

    for i in range(N):
        counts = []
        slc = slices[i]
        for j in range(2,np.shape(slc)[0]-2): #dont consider edge rows
            for k in range(2,np.shape(slc)[1]-2): #dont consider edge columns
                middle = slc[j,k] #middle cell that we are considering
                largest = True #set middle cell to be the largest out of neighbours
                done = False
                for p in range(-2,3):
                    if done == True: break
                    for q in range(-2,3):
                        if done == True: break
                        if p != 0 or q != 0: #dont compare with middle cell
                            neighbour = slc[j+p,k+q]
                            if neighbour >= middle:##################what if the peak is more than a pixek big
                                largest = False
                                done = True
                if largest == True:
                    counts.append(middle)
        count_list.append(len(counts))
    return count_list


def get_peak_vs_brightness(slices):
    N = len(slices)
    brightness_list = []

    for i in range(N):
        slc = slices[i]
        for j in range(2,np.shape(slc)[0]-2): #dont consider edge rows
            for k in range(2,np.shape(slc)[1]-2): #dont consider edge columns
                middle = slc[j,k] #middle cell that we are considering
                largest = True #set middle cell to be the largest out of neighbours
                done = False
                for p in range(-2,3):
                    if done == True: break
                    for q in range(-2,3):
                        if done == True: break
                        if p != 0 or q != 0: #dont compare with middle cell
                            neighbour = slc[j+p,k+q]
                            if neighbour >= middle:
                                largest = False
                                done = True
                if largest == True:
                    brightness_list.append(middle[0])
    return brightness_list


def get_pixel_val(slices):
    N = len(slices)
    elem = [] #list of elements to use for histogram

    for i in range(N):
        slc = slices[i]
        for j in range(np.shape(slc)[0]):
            for k in range(np.shape(slc)[1]):
                elem.append(slc[j,k][0])

    return elem

"""
def kolmogorov(reallist,fakelist):
    n_bins = 100
    fig, ax = plt.subplots(figsize=(8,4))

    #plot the real cumulative histogram
    nreal, realbins, realpatches = ax.hist(reallist, n_bins, density=True, histtype='step', cumulative=True, label='Real')
    #plot the fake cumulative histogram
    nfake, fakebins, fakepatches = ax.hist(fakelist, n_bins, density=True, histtype='step', cumulative=True, label='Fake')
    plt.legend()

    p_val = scipy.stats.ks_2samp(reallist,fakelist)

    n = len(reallist)
    m = len(fakelist)
    D = 0.
    c_alpha = np.sqrt(-0.5 * np.log(alpha))
    rjct = c_alpha * np.sqrt((1.*(n+m))/(n*m))
    for i in range(len(nreal)):
        d = abs(nreal[i]-nfake[i]) ###can i take the magnitude???
        if d > D:
            D = d

    return p_val#,fig
"""

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.00001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 9 * 9, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((9, 9, 128)))
        #8

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        #18

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        #36

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        #72

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        #144

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        #288

        model.add(Conv2D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation("tanh"))
        #288

        model.add(Cropping2D(cropping=((16, 16), (16, 16))))
        #256


        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=5, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # 128

        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #64

        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #32

        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #16

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #8

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #4 x 4 x 128


        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        #new = []
        #for x in X_train[:1000]:
        #    newx = imresize(x, (256,256))
        #    new.append(newx)
        #print('DONE RESHAPING')
        #X_train = np.array(new)
        #print(np.shape(X_train))

        if platform == "linux":
            #user = get_user()
            slices = pickle.load(open(r"/home/" + user + r"/Outputs/slices.pkl","rb"))
        else:
            slices = pickle.load(open(r"C:\\Outputs\\slices.pkl", "rb"))

        slices = np.array(slices)
        X_train = np.interp(slices, (slices.min(), slices.max()), (-1, +1))

        X_train_no_dim = X_train ##############################################################
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        d_loss = [1,0]
        i = 0

        #diff_list = []
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            if d_loss[1] > 0.95:
                d_loss_real = self.discriminator.test_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.test_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                print('Training Disc')
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            i+=1
            if d_loss[1] > 0.6 or i < 3:
            # Train the generator (wants discriminator to mistake images as real)
                print('Training Gen')
                g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            """
            # power spectrum stuff
            if epoch == 0:
                hist_PSD_real = [[0]]*100
                hist_PSD_fake = [[0]]*100
                hist_PSD_i = 0
            PSD_real,k_list_rl= raPsd2d(imgs[0], (256,256))
            PSD_fake,k_list_fk = raPsd2d(gen_imgs[0], (256,256))
            hist_PSD_real[hist_PSD_i] = PSD_real
            hist_PSD_fake[hist_PSD_i] = PSD_fake
            hist_PSD_i = (hist_PSD_i + 1) % (100)
            if epoch % (100) == 0:
                for i in range(100):
                    if len(hist_PSD_real[i]) != 1:
                        if i == 0:
                            plt.plot(hist_PSD_real[i][5::], color="blue", label="real", alpha=0.1)
                            plt.plot(hist_PSD_fake[i][5::], color="orange", label="fake", alpha=0.1)
                        else:
                            plt.plot(hist_PSD_real[i][5::], color="blue", alpha=0.1)
                            plt.plot(hist_PSD_fake[i][5::], color="orange", alpha=0.1)
                plt.legend()
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/ps_%d.png" % epoch)
                else:
                    plt.savefig("images/ps_%d.png" % epoch)
                plt.close()

            """

            look_for_cnvrg_at = 5000 ######5000
            if epoch == 0:
                idx = np.random.randint(0, X_train_no_dim.shape[0], 100)#100
                real_imgs = X_train_no_dim[idx]
                #print(real_imgs)
                real_ave_ps,real_ps_std,k_list_real = produce_average_ps(real_imgs)

                mod_sq_fake_std = []
                mod_sq_real_std = []
                mod_sq_comb_std = []
                #epoch_list = []
                epoch_list2 = []

            if epoch % look_for_cnvrg_at == 0 and epoch != 0:
                noise = np.random.normal(0, 1, (100, self.latent_dim))#100
                gen_imgs = self.generator.predict(noise)
                gen_imgs = np.squeeze(gen_imgs)
                #print(gen_imgs)
                fake_ave_ps,fake_ps_std,k_list_fake = produce_average_ps(gen_imgs)
                plt.plot(k_list_real[1:],real_ave_ps[1:], color="blue", label="real")
                plt.yscale('log')
                plt.errorbar(x=k_list_fake[1:], y=fake_ave_ps[1:], yerr=fake_ps_std[1:], color="orange", alpha=0.5, label="fake")
                plt.yscale('log')
                plt.legend()
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/ave_ps_%d.png" % epoch)
                else:
                    plt.savefig("images/ave_ps_%d.png" % epoch)
                plt.close()

                epoch_list2.append(epoch)
                ##convergence test using fake std
                diff_sq_fake_std = compare_ps(real_ave_ps,fake_ave_ps,fake_ps_std)
                mod_sq_fake_std.append(diff_sq_fake_std)

                plt.plot(epoch_list2,mod_sq_fake_std)
                plt.yscale('log')
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/convergence_test2_with_fakestd.png")
                else:
                    plt.savefig("images/convergence_test2_with_fakestd")
                plt.close()

                ##convergence test using real std
                diff_sq_real_std = compare_ps(real_ave_ps,fake_ave_ps,real_ps_std)
                mod_sq_real_std.append(diff_sq_real_std)

                plt.plot(epoch_list2,mod_sq_real_std)
                plt.yscale('log')
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/convergence_test2_with_realstd.png")
                else:
                    plt.savefig("images/convergence_test2_with_realstd")
                plt.close()

                ##convergence test using real + fake std
                comb_ps_std = []
                for l in range(len(real_ps_std)):
                    comb = np.sqrt(real_ps_std[l]**2 + fake_ps_std[l]**2)
                    comb_ps_std.append(comb)
                diff_sq_comb_std = compare_ps(real_ave_ps,fake_ave_ps,comb_ps_std)
                mod_sq_comb_std.append(diff_sq_comb_std)

                plt.plot(epoch_list2,mod_sq_comb_std)
                plt.yscale('log')
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/convergence_test2_with_combstd.png")
                else:
                    plt.savefig("images/convergence_test2_with_combstd")
                plt.close()


                #if epoch > look_for_cnvrg_at:
                    #diff_sq_aft = compare_ps(real_ave_ps,fake_ave_ps,fake_ps_std)
                    #print(diff_sq_aft)
                    #diff = (diff_sq_bfr-diff_sq_aft)/diff_sq_aft
                    #mod_sq.append(diff_sq_aft)
                    #epoch_list2.append(epoch)
                    #diff_list.append(diff)
                    #epoch_list.append(epoch)
                    #diff_sq_bfr = diff_sq_aft
                    #plt.plot(epoch_list,diff_list)

                    #if platform == "linux":
                    #    plt.savefig(r"/home/" + user + r"/Important/Images/convergence_test.png")
                    #else:
                    #    plt.savefig("images/convergence_test")
                    #plt.close()


            #normal peak counts
            if epoch == 0:
                idx = np.random.randint(0, X_train.shape[0], 500)#500
                real_imgs = X_train[idx]
                real_count_list = get_pk_hist(real_imgs)
            if epoch % 5000 == 0 and epoch != 0:
                noise = np.random.normal(0, 1, (500, self.latent_dim))#500
                gen_imgs = self.generator.predict(noise)
                fake_count_list = get_pk_hist(gen_imgs)
                plt.hist(real_count_list, bins=100, color="blue", label="real", alpha=0.7)
                plt.hist(fake_count_list, bins=100, color="orange", label="fake", alpha=0.7)
                plt.legend()
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/pk_%d.png" % epoch)
                else:
                    plt.savefig("images/pk_%d.png" % epoch)
                plt.close()


            #peak count vs brightness
            if epoch == 0:
                idx = np.random.randint(0, X_train.shape[0], 10)
                rl_imgs = X_train[idx]
                rl_brightness_list = get_peak_vs_brightness(rl_imgs)

                p_val_pk_bright_list = []
                p_val_pk_bright_epoch = []
            if epoch % 5000 == 0 and epoch != 0:
                noise = np.random.normal(0, 1, (10, self.latent_dim))
                gn_imgs = self.generator.predict(noise)
                fake_brightness_list = get_peak_vs_brightness(gn_imgs)
                plt.hist(rl_brightness_list, bins=100, color="blue", label="real", alpha=0.7)
                plt.hist(fake_brightness_list, bins=100, color="orange", label="fake", alpha=0.7)
                plt.legend()
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/pk_vs_brightness_%d.png" % epoch)
                else:
                    plt.savefig("images/pk_vs_brightness_%d.png" % epoch)
                plt.close()

                #kolmogorov
                p_val_pk_bright = scipy.stats.ks_2samp(rl_brightness_list,fake_brightness_list)[1]
                p_val_pk_bright_list.append(p_val_pk_bright)
                p_val_pk_bright_epoch.append(epoch)
                plt.plot(p_val_pk_bright_epoch,p_val_pk_bright_list)
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/pk_vs_brightness_kolmogorov.png")
                else:
                    plt.savefig("images/pk_vs_brightness_kolmogorov.png")
                plt.close()


            #pixel value histogram
            find_pixel_val_at = 5000
            if epoch == 0:
                idx = np.random.randint(0, X_train.shape[0], 1)
                real_imgs = X_train[idx]
                real_pix_val = get_pixel_val(real_imgs)

                p_val_pix_val_list = []
                p_val_pix_val_epoch = []
            if epoch % find_pixel_val_at == 0 and epoch != 0:
                noise = np.random.normal(0, 1, (1, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                fake_pix_val = get_pixel_val(gen_imgs)

                plt.hist(real_pix_val, bins=100, color="blue", label="real", alpha=0.7)
                plt.hist(fake_pix_val, bins=100, color="orange", label="fake", alpha=0.7)
                plt.legend()

                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/pixel_val_%d.png" % epoch)
                else:
                    plt.savefig("images/pixel_val_%d.png" % epoch)
                plt.close()

                #kolmogorov
                p_val_pix_val = scipy.stats.ks_2samp(real_pix_val,fake_pix_val)[1]
                p_val_pix_val_list.append(p_val_pk_bright)
                p_val_pix_val_epoch.append(epoch)
                plt.plot(p_val_pix_val_epoch,p_val_pix_val_list)
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/pixel_val_kolmogorov.png")
                else:
                    plt.savefig("images/pixel_val_kolmogorov.png")
                plt.close()


            #cross ps
            if epoch % 5000 == 0:
                idx = np.random.randint(0, X_train_no_dim.shape[0], 1)
                real_im = X_train_no_dim[idx][0]
                noise = np.random.normal(0, 1, (1, self.latent_dim))
                fake_im = self.generator.predict(noise)[0]
                fake_im = np.squeeze(fake_im)

                #cross ps with FT
                CPS,CPSconv,k_lst = crossraPsd2d(real_im,fake_im)
                #CPS,CPSconv,k_lst = crossraPsd2d(real_im,real_im)
                plt.plot(k_lst[1:],CPS[1:])
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/cross_ps_FT_%d.png" % epoch)
                else:
                    plt.savefig("images/cross_ps_FT_%d.png" % epoch)
                plt.close()

                #cross ps with conv
                CPS,CPSconv,k_lst = crossraPsd2d(real_im,fake_im)
                #CPS,CPSconv,k_lst = crossraPsd2d(real_im,real_im)
                plt.plot(k_lst[1:],CPSconv[1:])
                if platform == "linux":
                    plt.savefig(r"/home/" + user + r"/Important/Images/cross_ps_conv_%d.png" % epoch)
                else:
                    plt.savefig("images/cross_ps_conv_%d.png" % epoch)
                plt.close()


            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch,X_train)

    def save_imgs(self, epoch, X_train):
        r, c = 4, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # Select a random half of images
        idx = np.random.randint(0, X_train.shape[0], 10)
        imgs2 = X_train[idx]

        fig, axs = plt.subplots(r, c)
        cnt = 0
        dnt = 0
        for i in range(r):
            for j in range(c):
                if j == 0:
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='hot')
                    axs[i,j].axis('off')
                    cnt += 1
                else:
                    axs[i,j].imshow(imgs2[dnt, :,:,0], cmap='hot')
                    axs[i,j].axis('off')
                    dnt +=1

            if platform == "linux":
                    #user = get_user()
                    plt.savefig(r"/home/" + user + r"/Important/Images/map_%d.png" % epoch)

            else:
                fig.savefig("images/map_%d.png" % epoch)
        plt.close()

    def save_models(self):

        model_json = self.generator.to_json()
        with open("model_generator.json", "w") as json_file:
            json_file.write(model_json)
        self.generator.save_weights("model_generator.h5")

        model_json = self.combined.to_json()
        with open("model_combined.json", "w") as json_file:
            json_file.write(model_json)
        self.combined.save_weights("model_combined.h5")

        model_json = self.discriminator.to_json()
        with open("model_combined.json", "w") as json_file:
            json_file.write(model_json)
        self.discriminator.save_weights("model_combined.h5")


        return 1


if __name__ == '__main__':

    dcgan = DCGAN()
    dcgan.train(epochs=400000, batch_size=16, save_interval=500)
    dcgan.save_models()



# run same code on density field rather then ionized field to see if there are still two peaks
# in peak count
# primarily look at power spectrum
