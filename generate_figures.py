
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

def min_max_scale_images():
    print('minmax scaling images...')
    mmax = np.max(imgs)
    mmin = np.min(imgs)
    imgs = (imgs - (mmax+mmin)/2.) / ((mmax-mmin) / 2.)
    print('expanding dimension of images...')
    imgs = np.expand_dims(imgs, axis=3)


imgs, labels, real_imgs_index = read_data()

min_max_scale_images()

sample_at0 = [
[7.],
[8.],
[9.],
[10.],
[11.]
]
r = len(sample_at0)
c = 3

temp_copy = copy.deepcopy(sample_at0)

sample_at0 = np.array(sample_at0)
sample_at = scale_labels(sample_at0)
print('sampling images at labels:', sample_at)

noise = np.random.normal(0, 1, (len(sample_at), 100))

gen_imgs = generator.predict([noise, sample_at])

fig, axs = plt.subplots(r, c, figsize=(4,18), dpi=250)
#cnt = 0
for i in range(r):
    for j in range(c): # c=0: fake, c=1, real

        if j == 1:
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='hot', clim=(-1,1))
            axs[i,j].set_title("Labels: %s" % '_'.join(str(np.round(e,3)) for e in temp_copy[cnt]))
            axs[i,j].axis('off')
            print('fake_min',np.min(gen_imgs[cnt,:,:,0]))
            print('fake_max',np.max(gen_imgs[cnt,:,:,0]))
            #cnt += 1

        if j == 0: #show a real image
            #print(temp_copy[i][0])
            #print(self.real_imgs_index.keys())
            if temp_copy[i][0] in real_imgs_index: #real images for that z are available

                sample_i = random.randint(0,len(real_imgs_index[temp_copy[i][0]])-1)
                #print(self.real_imgs_index[temp_copy[i][0]])
                sample_i = real_imgs_index[temp_copy[i][0]][sample_i]
                #print(sample_i)
                #print(self.imgs.shape)
                axs[i,j].imshow(imgs_all[sample_i,:,:,0], cmap='hot', clim=(-1,1))
                axs[i,j].set_title("")
                print('real_min',np.min(imgs[sample_i,:,:,0]))
                print('real_max',np.max(imgs[sample_i,:,:,0]))
            axs[i,j].axis('off')

        if j ==2:
            if temp_copy[i][0] in real_imgs_index: #real images for that z are available
                fake_pix_val = stats_utils.get_pixel_val(gen_imgs[[cnt-1],:,:,:])
                real_pix_val = stats_utils.get_pixel_val(imgs[[sample_i],:,:,:])
                axs[i,j].hist(real_pix_val, range=(-1,1), bins=100, color="blue", label="real", alpha=0.7)
                axs[i,j].hist(fake_pix_val, range=(-1,1), bins=100, color="orange", label="fake", alpha=0.7)
                axs[i,j].legend()

fig.savefig("all_stats.png")
plt.close()
