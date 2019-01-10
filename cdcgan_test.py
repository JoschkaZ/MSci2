
from keras.layers import Input, Dense, Dropout, Reshape, BatchNormalization, Conv2DTranspose
from keras.layers.convolutional import Conv2D
from keras.layers.advanced_activations import LeakyReLU

# onehot representation
disc_condition_input = Input(shape=(10,))
gen_condition_input = Input(shape=(10,))



def get_generator(input_layer, condition_layer):

  # using concatenate to combine input and condition
  # the dense to 128*8*8
  merged_input = Concatenate()([input_layer, condition_layer])

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
  mdl = Model(inputs=[input_layer, condition_layer], outputs=out)
  mdl.summary()
   #32x32x3

  return mdl, out


def get_discriminator(input_layer, condition_layer):

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
  #2048

  merged_layer = Concatenate()([hid, condition_layer])
  #2058

  hid = Dense(512, activation='relu')(merged_layer)
  #512
  #hid = Dropout(0.4)(hid)
  out = Dense(1, activation='sigmoid')(hid)
  1
  mdl = Model(inputs=[input_layer, condition_layer], outputs=out)
  mdl.summary()

  return mdl, out


# DEFINE GAN MODEL

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



# TRAIN
