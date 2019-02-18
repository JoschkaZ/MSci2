
from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Add
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import load_model
import numpy as np
import pickle as pkl
import copy
import random
import stats_utils



generator = load_model(r'/home/hk2315/MSci2/models/21256generator_1550323874.h5')

def read_data():

    print('importing data...')
    #data = pkl.load(open(r"C:\\Outputs\\slices2_128.pkl", "rb"))
    data = pkl.load(open("/home/jz8415/slices2_128_all.pkl", "rb"))
    print('data imported!')

    imgs = []
    labels = []
    for entry in data:
        img = entry[0]
        l_str = entry[1]

        #DEFINE LABELS HERE
        l_z = float(l_str.split('_z')[1].split('_')[0])

        #APPEND LABELS HERE
        imgs.append(img)
        labels.append([l_z])

    imgs = np.array(imgs)
    labels = np.array(labels)
    label_dim = len(labels[0])
    print('dimension of label: ', label_dim)

    print('Shapes:')
    print(imgs.shape)
    print(labels.shape)
    #(60000, 28, 28)
    #(60000,1) #after reshaping

    real_imgs_index = {}
    for i, z in enumerate(labels):
        z = z[0]
        if not z in real_imgs_index:
            real_imgs_index[z] = []
        real_imgs_index[z].append(i)

    return imgs, labels, real_imgs_index

def scale_labels(l, verbose=0):
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

def min_max_scale_images(imgs):
    print('minmax scaling images...')
    mmax = np.max(imgs)
    mmin = np.min(imgs)
    imgs = (imgs - (mmax+mmin)/2.) / ((mmax-mmin) / 2.)
    print('expanding dimension of images...')
    #print('_______________',imgs.shape)
    imgs = np.expand_dims(imgs, axis=3)
    #print('_______________',imgs.shape)

    return imgs


imgs, labels, real_imgs_index = read_data()

imgs = min_max_scale_images(imgs)

sample_at0 = [
[7.],
[8.],
[9.],
[10.],
[11.]
]
r = len(sample_at0)
c = 5

temp_copy = copy.deepcopy(sample_at0)

sample_at0 = np.array(sample_at0)
sample_at = scale_labels(sample_at0)
print('sampling images at labels:', sample_at)

noise = np.random.normal(0, 1, (len(sample_at), 100))

gen_imgs = generator.predict([noise, sample_at])

fig1 = plt.figure(figsize=(25,20), constrained_layout=False)
widths = [1, 1, 1, 1, 1]
heights = [1, 1, 1, 1, 1]
spec1 = fig1.add_gridspec(ncols=c, nrows=r, width_ratios=widths, height_ratios=heights)
plt.subplots_adjust(wspace=0.3, hspace=0.3)

for i in range(r):
    for j in range(c):

        if j == 0: #show a real image
            #print(temp_copy[i][0])
            #print(self.real_imgs_index.keys())
            ax = fig1.add_subplot(spec1[i, j])
            if temp_copy[i][0] in real_imgs_index: #real images for that z are available

                sample_i = random.randint(0,len(real_imgs_index[temp_copy[i][0]])-1)
                #print(self.real_imgs_index[temp_copy[i][0]])
                sample_i = real_imgs_index[temp_copy[i][0]][sample_i]
                #print(sample_i)
                #print(self.imgs.shape)
                #print(imgs.shape)
                ax.imshow(imgs[sample_i,:,:,0], cmap='hot', clim=(-1,1))
                #ax.set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in temp_copy[i]))
                ax.axis('off')
                print('real_min',np.min(imgs[sample_i,:,:,0]))
                print('real_max',np.max(imgs[sample_i,:,:,0]))
            ax.axis('off')

        if j == 1:
            ax = fig1.add_subplot(spec1[i, j])
            ax.imshow(gen_imgs[i,:,:,0], cmap='hot', clim=(-1,1))
            #axs[i,j].set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in temp_copy[i]))
            ax.axis('off')
            print('fake_min',np.min(gen_imgs[i,:,:,0]))
            print('fake_max',np.max(gen_imgs[i,:,:,0]))
            #cnt += 1

        if j == 2:
            if temp_copy[i][0] in real_imgs_index: #real images for that z are available
                ax = fig1.add_subplot(spec1[i, j])
                print(imgs.shape)
                #intv = int(len(real_imgs_index[temp_copy[i][0]])/5)###100
                idx = np.random.randint(0, len(real_imgs_index[temp_copy[i][0]])-1, 10)
                index_list = np.array(real_imgs_index[temp_copy[i][0]])[idx]
                real_imgs = imgs[index_list]
                real_pix_val = stats_utils.get_pixel_val(real_imgs)


                noise = np.random.normal(0, 1, (10, 100))###5
                mal = max(real_imgs_index.keys())
                mil = min(real_imgs_index.keys())
                z_vec = np.array([[temp_copy[i][0]]]*10)###100
                z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
                gen_imgs2 = generator.predict([noise, np.reshape(z_vec,(10,1))])
                fake_pix_val = stats_utils.get_pixel_val(gen_imgs2)


                ax.hist(real_pix_val, range=(-1,1), bins=100, color="blue", label="real", alpha=0.7)
                ax.hist(fake_pix_val, range=(-1,1), bins=100, color="orange", label="fake", alpha=0.7)
                plt.xlabel('Pixel value', fontsize=10)
                plt.ylabel('Pixel Count', fontsize=10)
                ax.legend()

        if j == 3:
            ax = fig1.add_subplot(spec1[i, j])

            idx = np.random.randint(0, len(real_imgs_index[temp_copy[i][0]])-1, 100)
            index_list = np.array(real_imgs_index[temp_copy[i][0]])[idx]
            real_imgs = imgs[index_list]
            real_imgs = np.squeeze(real_imgs)
            real_ave_ps, real_ps_std, k_list_real = stats_utils.produce_average_ps(real_imgs)

            noise = np.random.normal(0, 1, (100, 100))###100
            mal = max(real_imgs_index.keys())
            mil = min(real_imgs_index.keys())
            z_vec = np.array([[temp_copy[i][0]]]*100)###100
            z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
            gen_imgs2 = generator.predict([noise, np.reshape(z_vec,(100,1))])###100
            gen_imgs2 = np.squeeze(gen_imgs2)
            fake_ave_ps, fake_ps_std, k_list_fake = stats_utils.produce_average_ps(gen_imgs2)

            ax.errorbar(x=k_list_fake[10:], y=fake_ave_ps[10:], yerr=fake_ps_std[10:], alpha=0.5, label="fake")
            ax.set_yscale('log')
            ax.plot(k_list_real[10:], real_ave_ps[10:], label="real")
            ax.set_yscale('log')
            plt.xlabel(r'Frequency[Mpc' + str(r'$^{-1}$') + r']', fontsize=10)
            plt.ylabel(r'Power[Mpc' + str(r'$^{-2}$') + r']', fontsize=10)
            ax.legend()

        if j == 4:
            ax = fig1.add_subplot(spec1[i, j])

            idx = np.random.randint(0, len(real_imgs_index[temp_copy[i][0]])-1, 10)
            index_list = np.array(real_imgs_index[temp_copy[i][0]])[idx]
            real_imgs = imgs[index_list]
            rl_brightness_list = stats_utils.get_peak_vs_brightness(real_imgs)

            noise = np.random.normal(0, 1, (10, 100))
            mal = max(real_imgs_index.keys())
            mil = min(real_imgs_index.keys())
            z_vec = np.array([[temp_copy[i][0]]]*10)
            z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
            gen_imgs2 = generator.predict([noise, np.reshape(z_vec,(10,1))])
            fk_brightness_list = stats_utils.get_peak_vs_brightness(gen_imgs2)

            ax.hist(rl_brightness_list, bins=100, color="blue", label="real", alpha=0.7)
            ax.hist(fk_brightness_list, bins=100, color="orange", label="fake", alpha=0.7)
            plt.xlabel('Pixel value', fontsize=10)
            plt.ylabel('Pixel count', fontsize=10)
            ax.legend()

        #ax = fig1.add_subplot(spec1[row, col])
        #label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
        #ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')



"""
fig, axs = plt.subplots(r, c, figsize=(4,10), dpi=250)
#cnt = 0
for i in range(r):
    for j in range(c): # c=0: fake, c=1, real

        if j == 0: #show a real image
            #print(temp_copy[i][0])
            #print(self.real_imgs_index.keys())
            if temp_copy[i][0] in real_imgs_index: #real images for that z are available

                sample_i = random.randint(0,len(real_imgs_index[temp_copy[i][0]])-1)
                #print(self.real_imgs_index[temp_copy[i][0]])
                sample_i = real_imgs_index[temp_copy[i][0]][sample_i]
                #print(sample_i)
                #print(self.imgs.shape)
                #print(imgs.shape)
                axs[i,j].imshow(imgs[sample_i,:,:,0], cmap='hot', clim=(-1,1))
                axs[i,j].set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in temp_copy[i]))
                axs[i,j].axis('off')
                print('real_min',np.min(imgs[sample_i,:,:,0]))
                print('real_max',np.max(imgs[sample_i,:,:,0]))
            axs[i,j].axis('off')

        if j == 1:
            axs[i,j].imshow(gen_imgs[i,:,:,0], cmap='hot', clim=(-1,1))
            #axs[i,j].set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in temp_copy[i]))
            axs[i,j].axis('off')
            print('fake_min',np.min(gen_imgs[i,:,:,0]))
            print('fake_max',np.max(gen_imgs[i,:,:,0]))
            #cnt += 1

        if j ==2:
            if temp_copy[i][0] in real_imgs_index: #real images for that z are available
                print(imgs.shape)
                fake_pix_val = stats_utils.get_pixel_val(gen_imgs[[i],:,:,:])
                real_pix_val = stats_utils.get_pixel_val(imgs[[sample_i],:,:,:])
                axs[i,j].hist(real_pix_val, range=(-1,1), bins=100, color="blue", label="real", alpha=0.7)
                axs[i,j].hist(fake_pix_val, range=(-1,1), bins=100, color="orange", label="fake", alpha=0.7)
                axs[i,j].legend()
"""

fig1.savefig("images/all_stats.png")
plt.close()
