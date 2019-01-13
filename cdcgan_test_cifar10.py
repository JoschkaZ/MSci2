

#%% IMPORTS

from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Concatenate
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import cifar10
import keras.backend as K
import matplotlib.pyplot as plt
import sys
import numpy as np
from keras.preprocessing import image



#%% DEFINE GENERATOR

def get_generator(input_layer, condition_layer):
    #input layer should be a random vector yo

    merged_input = Concatenate()([input_layer, condition_layer])
    #?+10

    hid = Dense(128 * 8 * 8, activation='relu')(merged_input)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    hid = Reshape((8, 8, 128))(hid)
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

    hid = Conv2D(3, kernel_size=5, strides=1, padding="same")(hid)
    out = Activation("tanh")(hid)
    #32x32x3

    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    model.summary()

    #model takes [input_layer, condition_layer] as input and returns the generated image
    # why return out as well ?
    return model, out



#%% DEFINE DISCRIMINATOR

def get_discriminator(input_layer, condition_layer):
    #input layer should be a crazy image yo (32x32x3)

    hid = Conv2D(128, kernel_size=3, strides=1, padding='same')(input_layer)
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
    #8x8x128

    hid = Conv2D(128, kernel_size=4, strides=2, padding='same')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = LeakyReLU(alpha=0.1)(hid)
    #4x4x128

    hid = Flatten()(hid)
    #4*4*128

    merged_layer = Concatenate()([hid, condition_layer])
    #4*4*128+?
    hid = Dense(512, activation='relu')(merged_layer)
    #hid = Dropout(0.4)(hid)
    #512

    out = Dense(1, activation='sigmoid')(hid)
    #1

    model = Model(inputs=[input_layer, condition_layer], outputs=out)

    model.summary()

    return model, out



#%% DEFINE HELPER FUNCTIONS

def one_hot_encode(y): # onehot encodes a label vector y
  z = np.zeros((len(y), 10))
  idx = np.arange(len(y))
  z[idx, y] = 1
  return z

def generate_noise(n_samples, noise_dim): # returns a normal random matrix (n_samples, noise_dim)
  X = np.random.normal(0, 1, size=(n_samples, noise_dim))
  return X

def generate_random_labels(n): #returns n onehot encoded random labels
  y = np.random.choice(10, n)
  y = one_hot_encode(y)
  return y

tags = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def show_samples(batchidx): #generate 10*3 images and plot them
  fig, axs = plt.subplots(5, 6, figsize=(10,6))
  plt.subplots_adjust(hspace=0.3, wspace=0.1)
  #fig, axs = plt.subplots(5, 6)
  #fig.tight_layout()
  for classlabel in range(10): #loo over classes
    row = int(classlabel / 2)  #row to plot in
    coloffset = (classlabel % 2) * 3
    lbls = one_hot_encode([classlabel] * 3) #onehot encode the label 3 times
    noise = generate_noise(3, 100) #generate noise three times
    gen_imgs = generator.predict([noise, lbls]) #generate three images

    for i in range(3):
        # Dont scale the images back, let keras handle it
        img = image.array_to_img(gen_imgs[i], scale=True) #this turns a 3d tensor into an image?
        axs[row,i+coloffset].imshow(img)
        axs[row,i+coloffset].axis('off')
        if i ==1:
          axs[row,i+coloffset].set_title(tags[classlabel])
  plt.savefig(str(batchidx) + '.png')
  plt.close()



#%% BUILD DA GAN

img_input = Input(shape=(32,32,3))
disc_condition_input = Input(shape=(10,))

discriminator, disc_out = get_discriminator(img_input, disc_condition_input)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False

noise_input = Input(shape=(100,))
gen_condition_input = Input(shape=(10,))
generator, gen_out = get_generator(noise_input, gen_condition_input)

gan_input = Input(shape=(100,))
x = generator([gan_input, gen_condition_input])
gan_out = discriminator([x, disc_condition_input])
gan = Model(inputs=[gan_input, gen_condition_input, disc_condition_input], output=gan_out)
gan.summary()

gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')



#%% LOAD AND SCALE TRAINING DATA

BATCH_SIZE = 16
(X_train, y_train), (X_test, _) = cifar10.load_data()
X_train = (X_train - 127.5) / 127.5
y_train = one_hot_encode(y_train[:,0])
print ("Training shape: {}".format(X_train.shape))
num_batches = int(X_train.shape[0]/BATCH_SIZE)

#%%
img = image.array_to_img(X_train[1], scale=True) #this turns a 3d tensor into an image?
plt.imshow(img)
plt.axis('off')
plt.show()
#%%


exp_replay = [] # Array to store samples for experience replay

N_EPOCHS = 200
for epoch in range(N_EPOCHS):

    cum_d_loss = 0.
    cum_g_loss = 0.

    for batch_idx in range(num_batches): # how many batches fit in one epoch

        # Get the next set of real images to be used in this iteration
        images = X_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
        labels = y_train[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]

        noise_data = generate_noise(BATCH_SIZE, 100)
        random_labels = generate_random_labels(BATCH_SIZE)
        # We use same labels for generated images as in the real training batch
        generated_images = generator.predict([noise_data, labels])

        # Train on soft targets (add noise to targets as well)
        noise_prop = 0.05 # Randomly flip 5% of targets

        # Prepare labels for real data
        true_labels = np.zeros((BATCH_SIZE, 1)) + np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(true_labels)), size=int(noise_prop*len(true_labels)))
        true_labels[flipped_idx] = 1 - true_labels[flipped_idx]

        # Train discriminator on real data
        d_loss_true = discriminator.train_on_batch([images, labels], true_labels)

        # Prepare labels for generated data
        gene_labels = np.ones((BATCH_SIZE, 1)) - np.random.uniform(low=0.0, high=0.1, size=(BATCH_SIZE, 1))
        flipped_idx = np.random.choice(np.arange(len(gene_labels)), size=int(noise_prop*len(gene_labels)))
        gene_labels[flipped_idx] = 1 - gene_labels[flipped_idx]

        # Train discriminator on generated data
        d_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)

        # Store a random point for experience replay
        r_idx = np.random.randint(BATCH_SIZE)
        exp_replay.append([generated_images[r_idx], labels[r_idx], gene_labels[r_idx]])

        #If we have enough points, do experience replay
        if len(exp_replay) == BATCH_SIZE:
            generated_images = np.array([p[0] for p in exp_replay])
            labels = np.array([p[1] for p in exp_replay])
            gene_labels = np.array([p[2] for p in exp_replay])
            expprep_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)
            exp_replay = []
            break

        d_loss = 0.5 * np.add(d_loss_true, d_loss_gene)
        cum_d_loss += d_loss

        # Train generator
        noise_data = generate_noise(BATCH_SIZE, 100)
        random_labels = generate_random_labels(BATCH_SIZE)
        g_loss = gan.train_on_batch([noise_data, random_labels, random_labels], np.zeros((BATCH_SIZE, 1)))
        cum_g_loss += g_loss

    print('\tEpoch: {}, Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, cum_g_loss/num_batches, cum_d_loss/num_batches))
    show_samples("epoch" + str(epoch))
