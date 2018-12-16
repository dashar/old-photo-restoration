from keras.layers import Flatten, Dense, Input, Reshape, Concatenate, Lambda
from keras.layers.convolutional import Convolution2D, Conv2D, Conv3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf

def PatchGanDiscriminator(img_dim, patch_dim, nb_patches, nb_conv):
    """
    Creates the generator according to the specs in the paper below.
    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]

    PatchGAN only penalizes structure at the scale of patches. This
    discriminator tries to classify if each N x N patch in an
    image is real or fake. We run this discriminator convolutationally
    across the image, averaging all responses to provide
    the ultimate output of D.

    The discriminator has two parts. First part is the actual discriminator
    seconds part we make it a PatchGAN by running each image patch through the model
    and then we average the responses

    Discriminator does the following:
    1. Runs many pieces of the image through the network
    2. Calculates the cost for each patch
    3. Returns the costs for each patch as the output of the network

    :param patch_dim: (channels, width, height) T
    :param nb_patches:
    :return:
    """
    # -------------------------------
    # DISCRIMINATOR
    # C64-C128-C256-C512-C512-C512 (for 256x256)
    # otherwise, it scales from 64
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    bn_mode = 2
    axis = 1
    patch_dim = patch_dim[:2]
    #input_img = Input(shape=patch_dim)
    #input_cond = Input(shape=patch_dim)
    #input_to_disc = [input_img, input_cond]
    #input_layer = Concatenate()([input_img, input_cond])
    input_img = Input(shape=img_dim)
    input_cond = Input(shape=img_dim)

    input_layer = Concatenate()([input_img, input_cond])

    #generate patches
    patches = Lambda(lambda z: tf.extract_image_patches(z, ksizes=(1,) + patch_dim + (1,), 
                rates=(1, 1, 1,1), strides = (1,) + patch_dim + (1,), padding = 'VALID'))(input_layer)
    patches = Reshape((nb_patches, *patch_dim, 2 * img_dim[-1]))(patches)

    # We have to build the discriminator dynamically because
    # the size of the disc patches is dynamic
    num_filters_start = 64
    #nb_conv = 4#int(np.floor(np.log(img_dim[1]) / np.log(2)))
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    k_size = [1, 4, 4]
    strides = [1, stride, stride]
    # CONV 1
    # Do first conv bc it is different from the rest
    # paper skips batch norm for first layer
    disc_out = Conv3D(filters=64, kernel_size = k_size, padding='same', 
                      strides=strides, name='disc_conv_1')(patches)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # CONV 2 - CONV N
    # do the rest of the convs based on the sizes from the filters
    for i, filter_size in enumerate(filters_list[1:]):
        name = 'disc_conv_{}'.format(i+2)

        disc_out = Conv3D(filters=filter_size, kernel_size = k_size, padding='same', 
                          strides=strides, name=name)(disc_out)
        disc_out = BatchNormalization(name=name + '_bn')(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    #per patch - dense layer analog
    disc_out = Conv3D(filters=1, kernel_size = [1, int(disc_out.shape[2]), int(disc_out.shape[3])], 
           padding='valid', strides=strides, activation = 'sigmoid')(disc_out)
    disc_out = Reshape((nb_patches,))(disc_out)
    patch_gan_discriminator = Model(inputs = [input_img, input_cond], outputs = disc_out, name = 'discriminator_nn')
    return patch_gan_discriminator