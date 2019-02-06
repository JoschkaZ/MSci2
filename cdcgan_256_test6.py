from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Concatenate, Add, ReLU
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose, Cropping2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import pickle as pkl
from skimage.transform import resize
import random
import stats_utils
import utils
import sys

#zeta parameter


class CGAN():
    def __init__(self, use_old_model):

        self.imgs = []
        self.labels = []
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.noise_dim = 100
        self.con_dim = 1
        self.start_time = str(time.time()).split('.')[0]

        self.read_data()

        optimizer = Adam(0.0002, 0.5)

        ph_img = Input(shape=self.img_shape)
        ph_con = Input(shape=(self.con_dim,))

        self.find_ps = True

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


    def build_generator(self, noise, con):

        # 64 Mpc box isze is minimum tolerable. otherwise abprubt
        # 128 should be fune. #
        #ionisation boxes around 30 Mpc
        # if go down to 64 should also reduce physical size of box. 128 should be fine#
        noise1 = noise

        con1 = Dense(12, activation='tanh')(con)
        con1 = Dense(25, activation='tanh')(con1)
        con1 = Dense(50, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)

        merged_input = Concatenate()([con1, noise1])

        merged_input = Dense(200)(merged_input)
        merged_input = Dense(200)(merged_input)

        cfrom = 512
        cto = 128
        imfrom = 9
        twopot = 5

        hid = Dense(cfrom * imfrom**2)(merged_input)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = ReLU()(hid)

        hid = Reshape((imfrom, imfrom, cfrom))(hid)

        im = imfrom
        for i in range(twopot-1):

            hid = Conv2DTranspose(int(np.round((cto/cfrom)**((i+1)/(twopot-1))*cfrom)), 5, strides=2, padding='same')(hid)
            hid = BatchNormalization(momentum=0.9)(hid)
            hid = ReLU()(hid)


        hid = Conv2DTranspose(1, kernel_size=5, strides=2, padding="same")(hid)
        hid = Activation("tanh")(hid)

        out = Cropping2D(cropping=((16, 16), (16, 16)))(hid)

        model =  Model([noise, con], out)
        model.summary()
        return model

    def build_discriminator(self, img, con):

        cfrom = 128
        cto = 512
        imfrom = 256
        twopot = 5

        hid = Conv2D(cfrom, kernel_size=5, strides=2, padding='same')(img)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.2)(hid)

        for i in range(twopot-1):

            hid = Conv2D(int(np.round((cto/cfrom)**((i+1)/(twopot-1))*cfrom)), kernel_size=5, strides=2, padding='same')(hid)
            hid = BatchNormalization(momentum=0.9)(hid)
            hid = LeakyReLU(alpha=0.2)(hid)
            #14x14x128

        hid = Flatten()(hid)
        hid = Dropout(0.4)(hid) # NOTE LARGE DROPOUT

        hid = Dense(100, activation='tanh')(hid)

        con1 = Dense(12, activation='tanh')(con)
        con1 = Dense(25, activation='tanh')(con1)
        con1 = Dense(50, activation='tanh')(con1)
        con1 = Dense(100, activation='tanh')(con1)

        merged_layer = Concatenate()([hid, con1])
        merged_layer = Dropout(0.1)(merged_layer) # NOTE SMALLER DROPOUT


        merged_layer = Dense(100, activation='tanh')(merged_layer)
        merged_layer = Dense(50, activation='tanh')(merged_layer)
        merged_layer = Dense(25, activation='tanh')(merged_layer)

        out = Dense(1, activation='sigmoid')(merged_layer)

        model = Model(inputs=[img, con], outputs=out)
        model.summary()

        return model

    def read_data(self):


        print('importing data...')


        data = pkl.load(open("/home/jz8415/slices2.pkl", "rb"))
        #data = pkl.load(open("/home/hk2315/MSci2/faketest_images.pkl", "rb"))
        #data = pkl.load(open(r"C:\\Users\\Joschka\\github\\MSci2\\faketest_images_256.pkl", "rb"))
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
        print(self.imgs[5])
        print(self.labels[5])

        '''
        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        print(X_train.shape)
        print(y_train.shape)

        y_train = y_train.reshape(-1, 1)

        self.imgs = X_train.astype(np.float32)
        self.labels = y_train.astype(np.float32)
        '''

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
        mmax = np.max(self.imgs)
        mmin = np.min(self.imgs)
        print('mmax', mmax)
        print('mmin', mmin)
        print(self.imgs[0][14])
        self.imgs = (self.imgs.astype(np.float32) - (mmax+mmin)/2.) / ((mmax-mmin) / 2.)
        print(self.imgs[0][14])
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
                # for now its always 0, but if there are multiple labels later on they will need to be scaled separately

        if verbose == 0:
            print('scaled labels')
            print(l[0:length:step])

        return l

    def train(self, epochs, batch_size=128, sample_interval=50, save_model_interval=500, stats_interval = 500):


        #SCALE IMAGES
        self.min_max_scale_images()
        self.labels = self.scale_labels(self.labels)

        print('FINAL SHAPES')
        print(self.imgs.shape)
        print(self.labels.shape)


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        last_acc = .75

        for epoch in range(epochs):
            print('EPOCH', epoch)

            '''
            if epoch % 50 == 0: # TODO NEED TO CHANGE THIS
                print('aaaa')
                idx = np.random.randint(0, self.imgs.shape[0], 500)
                sub_imgs = copy.deepcopy(self.imgs[idx])
                sub_labels = self.labels[idx]

                new = []
                print(sub_imgs.shape)
                for i in range(len(sub_imgs)):
                    s = copy.deepcopy(sub_imgs[i][:,:,0])
                    #print(s.shape)
                    #plt.imshow(s)
                    #plt.show()
                    s = resize(s, (256, 256))
                    #plt.imshow(s)
                    #plt.show()
                    new.append(s)
                sub_imgs= np.array(new)
                sub_imgs = np.expand_dims(sub_imgs, axis=3)

                print('After selecting subset: ', sub_imgs.shape)

            '''

            # NOTE GET NOISE LABEL VECTORS
            p_flip = 0.05
            noise_range = 0.1
            valid_noisy  = np.array([random.uniform(1.-noise_range,1.) if (random.uniform(0,1)<1.-p_flip) else random.uniform(0.,noise_range) for _ in range(batch_size)])
            fake_noisy  = np.array([random.uniform(0.,noise_range) if (random.uniform(0,1)<1.-p_flip) else random.uniform(1.-noise_range,1.) for _ in range(batch_size)])
            #05022019 changed this to arrays


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            """
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]
            """
            #idx = np.random.randint(0, sub_imgs.shape[0], batch_size)
            #imgs, labels = sub_imgs[idx], sub_labels[idx]
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


            imgs = imgs + np.random.normal(0, 0.05, size=imgs.shape)
            gen_imgs = gen_imgs + np.random.normal(0, 0.05, size=imgs.shape)

            if (epoch < 200) and (epoch%5!=0): # only slowed down at the start. 1:1 training later
                print('Only testing discriminator')
                d_loss_real = self.discriminator.test_on_batch([imgs, labels], valid_noisy)
                d_loss_fake = self.discriminator.test_on_batch([gen_imgs, labels], fake_noisy)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid_noisy)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake_noisy)
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
                # TODO try testing right afterwards and check if the losses actuallry agree   . . . .


            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100.*d_loss[1], g_loss))
            last_acc = d_loss[1]

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch % save_model_interval == 0:
                print('saving model...')
                self.discriminator.save('models/21256discriminator_' + str(self.start_time) + '.h5')
                self.discriminator.save_weights('models/21256discriminatorweights_' + str(self.start_time) + '.h5')
                self.combined.save('models/21256combined_' + str(self.start_time) + '.h5')
                self.discriminator.save_weights('models/21256combinedweights_' + str(self.start_time) + '.h5')
                self.generator.save('models/21256generator_' + str(self.start_time) + '.h5')
                self.discriminator.save_weights('models/21256generatorweights_' + str(self.start_time) + '.h5')

            #if epoch % stats_interval == 0:
                #print('calculating ps...')
                #self.calc_ps(epoch)


    def calc_ps(self, epoch):
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
                            sample_i = random.randint(0,len(self.real_imgs_index[temp_copy[i][0]])-1)
                            #print(self.real_imgs_index[temp_copy[i][0]])
                            sample_i = self.real_imgs_index[temp_copy[i][0]][sample_i]
                            #print(sample_i)
                            #print(self.imgs.shape)
                            axs[i,j].imshow(self.imgs[sample_i,:,:,0], cmap='hot')
                            axs[i,j].set_title("")
                        axs[i,j].axis('off')
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    args = sys.argv
    if args[1] == 'new':
        cgan = CGAN(use_old_model=False)
        #cgan.train(epochs=20000, batch_size=128, sample_interval=10, save_model_interval = 100)
        cgan.train(epochs=400000, batch_size=64, sample_interval=10, save_model_interval = 10, stats_interval = 10)
    elif args[1] == 'continue':
        cgan = CGAN(use_old_model=True)
        #cgan.train(epochs=20000, batch_size=128, sample_interval=10, save_model_interval = 100)
        cgan.train(epochs=400000, batch_size=64, sample_interval=10, save_model_interval = 10, stats_interval = 10)
    else:
        print('Argument required.')
        print('write: "python cdcgan_21cm_256.py new" to use a new model.')
        print('write: "python cdcgan_21cm_256.py continue" to continue training the last model.')
