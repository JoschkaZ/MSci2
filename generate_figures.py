
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
import tensorflow as tf
import matplotlib as mpl

choose = 711
#choose = 59

if choose == 711:
    generator = load_model(r'/home/hk2315/cdcgan_bu/models_test11_no9/21256generator_1550767593.h5')


def read_data():

    print('importing data...')
    #data = pkl.load(open(r"C:\\Outputs\\slices2_128.pkl", "rb"))
    if choose == 711:
        data = pkl.load(open("/home/jz8415/slices2_128_all.pkl", "rb"))
    if choose == 59:
        data = pkl.load(open("/home/jz8415/128_all.pkl", "rb"))
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

    return imgs, mmax, mmin


def make_table():
    imgs, labels, real_imgs_index = read_data()
    imgs, mmax, mmin = min_max_scale_images(imgs)

    if choose == 711:
        sample_at0 = [
        [7.],
        [8.],
        [9.],
        [10.],
        [11.]
        ]
    if choose == 59:
        sample_at0 = [
        [5.],
        [6.],
        [7.],
        [8.],
        [9.]
        ]

    r = len(sample_at0)
    c = 5

    temp_copy = copy.deepcopy(sample_at0)

    sample_at0 = np.array(sample_at0)
    sample_at = scale_labels(sample_at0)
    print('sampling images at labels:', sample_at)

    noise = np.random.normal(0, 1, (len(sample_at), 100))

    gen_imgs = generator.predict([noise, sample_at])

    fig1 = plt.figure(figsize=(25,20), constrained_layout=False, dpi=300)
    widths = [1, 1, 1, 1, 1]
    heights = [1, 1, 1, 1, 1]
    spec1 = fig1.add_gridspec(ncols=c, nrows=r, width_ratios=widths, height_ratios=heights)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

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
                    print('real min before scaling', mmin)
                    print('real_max',np.max(imgs[sample_i,:,:,0]))
                    print('real max before scaling', mmax)
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
                #pixel value
                ave_pv_from = 10 ###10

                ax = fig1.add_subplot(spec1[i, j])
                print(imgs.shape)
                #intv = int(len(real_imgs_index[temp_copy[i][0]])/5)###100
                idx = np.random.randint(0, len(real_imgs_index[temp_copy[i][0]])-1, ave_pv_from)
                index_list = np.array(real_imgs_index[temp_copy[i][0]])[idx]
                real_imgs = imgs[index_list]
                real_pix_val = stats_utils.get_pixel_val(real_imgs)


                noise = np.random.normal(0, 1, (ave_pv_from, 100))###5
                mal = max(real_imgs_index.keys())
                mil = min(real_imgs_index.keys())
                z_vec = np.array([[temp_copy[i][0]]]*ave_pv_from)###100
                z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
                gen_imgs2 = generator.predict([noise, np.reshape(z_vec,(ave_pv_from,1))])
                fake_pix_val = stats_utils.get_pixel_val(gen_imgs2)

                x2 = np.arange(-1., 1., 0.399)
                print(x2)
                my_ticks = [str(np.round((x+1)*mmax/2,1)) for x in x2]
                ax.set_xticks(x2, my_ticks)
                plt.xticks(x2, my_ticks)
                ax.hist(real_pix_val, range=(-1,1), bins=100, color="blue", label="real", alpha=0.7)
                ax.hist(fake_pix_val, range=(-1,1), bins=100, color="orange", label="fake", alpha=0.7)
                plt.xlabel('Pixel value', fontsize=20)
                plt.ylabel('Pixel Count', fontsize=20)
                ax.legend(prop={'size': 15})

            if j == 3:
                #power spectrum
                ave_ps_from = 100 ###100

                ax = fig1.add_subplot(spec1[i, j])

                idx = np.random.randint(0, len(real_imgs_index[temp_copy[i][0]])-1, ave_ps_from)
                index_list = np.array(real_imgs_index[temp_copy[i][0]])[idx]
                real_imgs = imgs[index_list]
                real_imgs = np.squeeze(real_imgs)
                real_ave_ps, real_ps_std, k_list_real = stats_utils.produce_average_ps(real_imgs)

                noise = np.random.normal(0, 1, (ave_ps_from, 100))###100
                mal = max(real_imgs_index.keys())
                mil = min(real_imgs_index.keys())
                z_vec = np.array([[temp_copy[i][0]]]*ave_ps_from)###100
                z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
                gen_imgs2 = generator.predict([noise, np.reshape(z_vec,(ave_ps_from,1))])###100
                gen_imgs2 = np.squeeze(gen_imgs2)
                fake_ave_ps, fake_ps_std, k_list_fake = stats_utils.produce_average_ps(gen_imgs2)
                fake_ps_std = fake_ps_std/(np.sqrt(ave_ps_from))

                ax.errorbar(x=k_list_fake[10:], y=fake_ave_ps[10:], yerr=fake_ps_std[10:], color='orange', alpha=0.5, label="fake")
                ax.set_yscale('log')
                ax.plot(k_list_real[10:], real_ave_ps[10:], color='blue', label="real")
                ax.set_yscale('log')
                plt.xlabel(r'Frequency[Mpc' + str(r'$^{-1}$') + r']', fontsize=20)
                plt.ylabel(r'Power[Mpc' + str(r'$^{-2}$') + r']', fontsize=20)
                ax.legend(prop={'size': 15})

            if j == 4:
                #peak brightness
                ave_pb_from = 10 ###10

                ax = fig1.add_subplot(spec1[i, j])

                idx = np.random.randint(0, len(real_imgs_index[temp_copy[i][0]])-1, ave_pb_from)
                index_list = np.array(real_imgs_index[temp_copy[i][0]])[idx]
                real_imgs = imgs[index_list]
                rl_brightness_list = stats_utils.get_peak_vs_brightness(real_imgs)

                noise = np.random.normal(0, 1, (ave_pb_from, 100))
                mal = max(real_imgs_index.keys())
                mil = min(real_imgs_index.keys())
                z_vec = np.array([[temp_copy[i][0]]]*ave_pb_from)
                z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
                gen_imgs2 = generator.predict([noise, np.reshape(z_vec,(ave_pb_from,1))])
                fk_brightness_list = stats_utils.get_peak_vs_brightness(gen_imgs2)

                x2 = np.arange(-1., 1., 0.399)
                print(x2)
                my_ticks = [str(np.round((x+1)*mmax/2,1)) for x in x2]
                ax.set_xticks(x2, my_ticks)
                plt.xticks(x2, my_ticks)
                ax.hist(rl_brightness_list, range=(-1,1), bins=100, color="blue", label="real", alpha=0.7)
                ax.hist(fk_brightness_list, range=(-1,1), bins=100, color="orange", label="fake", alpha=0.7)
                plt.xlabel('Pixel value', fontsize=20)
                plt.ylabel('Pixel count', fontsize=20)
                ax.legend(prop={'size': 15})

    fig1.savefig("images/all_stats_%d.png" % choose)
    plt.close()


def make_cb():
    imgs, labels, real_imgs_index = read_data()
    imgs, mmax, mmin = min_max_scale_images(imgs)

    fig, ax = plt.subplots(figsize=(6, 1), dpi=300)
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.cm.hot
    norm = mpl.colors.Normalize(vmin=mmin, vmax=mmax)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Brightness temperature[mK]', fontsize=20)
    fig.savefig("images/colourbar_%d.png" % choose, bbox_inches = "tight")
    plt.close()



def make_cor_fig():
    imgs, labels, real_imgs_index = read_data()
    imgs, mmax, mmin = min_max_scale_images(imgs)

    if choose == 711:
        z_list = [7,8,9,10,11]
    if choose == 59:
        z_list = [5,6,7,8,9]
    for z in z_list:
        #make = True
        #count = 0
        noise = np.random.normal(0, 1, (100, 100))###100
        mal = max(real_imgs_index.keys())
        mil = min(real_imgs_index.keys())
        z_vec = np.array([[z]]*100)###100
        z_vec = (z_vec - (mal+mil)/2.) / ((mal-mil)/2.)
        gen_imgs2 = generator.predict([noise, np.reshape(z_vec,(100,1))])###100
        gen_imgs2 = np.squeeze(gen_imgs2)

        y_list = []
        cor = True
        for iii,s42 in enumerate(gen_imgs2):
                    for jjj,s43 in enumerate(gen_imgs2):
                        if cor == True:
                            if iii == 10 and jjj == 10:
                                Sconv, k_list_cross = stats_utils.xps(s42,s43)
                                #if make == True:
                                plt.plot(k_list_cross[2:], Sconv[2:], 'r--', label='Correlated')
                                cor = False

                        if iii != jjj and iii % 10 == 0 and jjj % 5 == 0 and jjj>iii:
                            print(iii)
                            Sconv, k_list_cross = stats_utils.xps(s42,s43)
                            #if make == True:
                            #plt.plot(k_list_cross[2:], Sconv[2:], alpha=0.2)
                            y_list.append(Sconv)
        y_list = np.array(y_list)
        y = np.mean(y_list, axis=0)
        error = np.std(y_list, axis=0)
        error = error / (np.sqrt(len(y_list)))
        plt.errorbar(x=k_list_cross[2:], y=Sconv[2:], yerr=error[2:])
        plt.savefig("images/cross_%d_%d.png" % (z,choose))
        plt.close()


def make_ps_ave_models():
    imgs, labels, real_imgs_index = read_data()
    imgs, mmax, mmin = min_max_scale_images(imgs)
    if choose == 711:
        ps_ave_from_models = pkl.load(open("/home/hk2315/cdcgan_bu/models_test11_no9/ps_ave_from_models.pkl", "rb"))
        z_list = [7,8,9,10,11]
    if choose == 59:
        ps_ave_from_models = pkl.load(open("/home/hk2315/MSci2/models/ps_ave_from_models.pkl", "rb"))
        z_list = [5,6,7,8,9]

    k_list = ps_ave_from_models['k_list']
    z_dict = {}

    r = 5
    c = 1
    fig, axs = plt.subplots(r, c, figsize=(4,18))
    cnt = 0


    for z in z_list:
        z_lists = ps_ave_from_models[z]
        z_lists = z_lists[-10:]
        values_list = [ [] for i in range(len(z_lists[0])) ] #list of empty lists
        std_list = []
        ps = np.zeros(len(z_lists[0]))
        print(len(z_lists))

        for l in range(len(z_lists)):
            s = z_lists[l]
            ps = np.add(ps,s)
            for j in range(len(s)):
                values_list[j].append(s[j])
        ps = ps/(len(z_lists))

        for k in range(len(values_list)):
            std = np.std(values_list[k])
            std = std/np.sqrt(len(values_list[k]))
            std_list.append(std)

        #z_dict[z] = [ps]
        #z_dict[z].append(std_list)

        idx = np.random.randint(0, len(real_imgs_index[z])-1, 10)
        index_list = np.array(real_imgs_index[z])[idx]
        real_imgs = imgs[index_list]
        real_imgs = np.squeeze(real_imgs)
        real_ave_ps, real_ps_std, k_list_real = stats_utils.produce_average_ps(real_imgs)

        print(len(k_list))
        print(len(ps))
        #print(len(std))

        axs[cnt].errorbar(x=k_list[10:], y=ps[10:], yerr=std_list[10:], color='orange', label='Fake') ###fix
        axs[cnt].set_yscale('log')
        axs[cnt].plot(k_list_real[10:], real_ave_ps[10:], color='blue', label="real")
        axs[cnt].set_yscale('log')
        axs[cnt].set_title("Labels: %d" % z)
        axs[cnt].legend()
        cnt += 1
    fig.savefig("images/ps_ave_from_models_%d.png" % choose, bbox_inches = "tight")
    plt.close()



make_table()
#make_cor_fig()
#make_ps_ave_models()
