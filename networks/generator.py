from keras.layers import Activation, Input, Dropout, Concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

"""
There are two models available for the generator:
1. AE Generator
2. UNet with skip connections
"""


def make_generator_ae(input_layer, num_output_filters):
    """
    Creates the generator according to the specs in the paper below.
    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
    :param model:
    :return:
    """
    # -------------------------------
    # ENCODER
    # C64-C128-C256-C512-C512-C512-C512-C512
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    filter_sizes = [64, 128, 256, 512, 512, 512, 512, 512]

    encoder = input_layer
    for filter_size in filter_sizes:
        encoder = Conv2D(filters=filter_size, kernel_size = 4, padding='same', strides=stride)(encoder)
        # paper skips batch norm for first layer
        if filter_size != 64:
            encoder = BatchNormalization()(encoder)
        encoder = Activation(LeakyReLU(alpha=0.2))(encoder)

    # -------------------------------
    # DECODER
    # CD512-CD512-CD512-C512-C512-C256-C128-C64
    # 1 layer block = Conv - Upsample - BN - DO - Relu
    # -------------------------------
    stride = 2
    filter_sizes = [512, 512, 512, 512, 512, 256, 128, 64]#[512, 512, 512, 512, 512, 256, 128, 64]

    decoder = encoder
    for filter_size in filter_sizes:
        decoder = UpSampling2D(size=(2, 2))(decoder)
        decoder = Conv2D(filters=filter_size, kernel_size = 4, padding='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(rate=0.5)(decoder)
        decoder = Activation('relu')(decoder)

    # After the last layer in the decoder, a convolution is applied
    # to map to the number of output channels (3 in general,
    # except in colorization, where it is 2), followed by a Tanh
    # function.
    decoder = Conv2D(filters=filter_size, kernel_size = 4, padding='same')(decoder)
    generator = Activation('tanh')(decoder)
    return generator


def UNETGenerator(input_img_dim, num_output_channels):
    """
    Creates the generator according to the specs in the paper below.
    It's basically a skip layer AutoEncoder

    Generator does the following:
    1. Takes in an image
    2. Generates an image from this image

    Differs from a standard GAN because the image isn't random.
    This model tries to learn a mapping from a suboptimal image to an optimal image.

    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
    :param input_img_dim: (channel, height, width)
    :param output_img_dim: (channel, height, width)
    :return:
    """
    # -------------------------------
    # ENCODER
    # C64-C128-C256-C512-C512-C512-C512-C512
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    
    input_layer = Input(shape=input_img_dim, name="unet_input")
    filters_en = [64, 128, 256, 512, 512, 512]#, 512, 512]
    layers = []
    en_i = input_layer
    
    for (i, nb_filters) in enumerate(filters_en):
        en_i = Conv2D(filters=nb_filters, kernel_size = 4, padding='same', strides=stride)(en_i)
        # skip batchnorm on first layer on purpose (from paper)
        if i > 0:
            en_i = BatchNormalization()(en_i)
        en_i = LeakyReLU(alpha=0.2)(en_i)
        layers += [en_i]
    
    # 1 encoder C64
    # skip batchnorm on this layer on purpose (from paper)
    #en_1 = Conv2D(filters=64, kernel_size = 4, padding='same', strides=stride)(input_layer)
    #en_1 = LeakyReLU(alpha=0.2)(en_1)

    # 2 encoder C128
    #en_2 = Conv2D(filters=128, kernel_size = 4, padding='same', strides=stride)(en_1)
    #en_2 = BatchNormalization(name='gen_en_bn_2')(en_2)
    #en_2 = LeakyReLU(alpha=0.2)(en_2)

    # 3 encoder C256
    #en_3 = Conv2D(filters=256, kernel_size = 4, padding='same', strides=stride)(en_2)
    #en_3 = BatchNormalization(name='gen_en_bn_3')(en_3)
    #en_3 = LeakyReLU(alpha=0.2)(en_3)

    # 4 encoder C512
    #en_4 = Conv2D(filters=512, kernel_size = 4, padding='same', strides=stride)(en_3)
    #en_4 = BatchNormalization(name='gen_en_bn_4')(en_4)
    #en_4 = LeakyReLU(alpha=0.2)(en_4)

    # 5 encoder C512
    #en_5 = Conv2D(filters=512, kernel_size = 4, padding='same', strides=stride)(en_4)
    #en_5 = BatchNormalization(name='gen_en_bn_5')(en_5)
    #en_5 = LeakyReLU(alpha=0.2)(en_5)

    # 6 encoder C512
    #en_6 = Conv2D(filters=512, kernel_size = 4, padding='same', strides=stride)(en_5)
    #en_6 = BatchNormalization(name='gen_en_bn_6')(en_6)
    #en_6 = LeakyReLU(alpha=0.2)(en_6)

    # 7 encoder C512
    #en_7 = Conv2D(filters=512, kernel_size = 4, padding='same', strides=stride)(en_6)
    #en_7 = BatchNormalization(name='gen_en_bn_7')(en_7)
    #en_7 = LeakyReLU(alpha=0.2)(en_7)

    # 8 encoder C512
    #en_8 = Conv2D(filters=512, kernel_size = 4, padding='same', strides=stride)(en_7)
    #en_8 = BatchNormalization(name='gen_en_bn_8')(en_8)
    #en_8 = LeakyReLU(alpha=0.2)(en_8)

    # -------------------------------
    # DECODER
    # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    # 1 layer block = Conv - Upsample - BN - DO - Relu
    # also adds skip connections (merge). Takes input from previous layer matching encoder layer
    # -------------------------------
    
    de_i = en_i
    filters_dec = [512, 1024, 1024, #1024, 1024, 
                   512, 256]
    for (i, nb_filters) in enumerate(filters_dec):
        de_i = UpSampling2D(size=(2, 2))(de_i)
        de_i = Conv2D(filters=nb_filters, kernel_size = 4, padding='same')(de_i)
        de_i = BatchNormalization()(de_i)
        de_i = Dropout(rate=0.5)(de_i)
        de_i = Concatenate(axis = -1)([de_i, layers[-(i + 2)]])
        de_i = Activation('relu')(de_i)
        
    # 1 decoder CD512 (decodes en_8)
    #de_1 = UpSampling2D(size=(2, 2))(en_8)
    #de_1 = Conv2D(filters=512, kernel_size = 4, padding='same')(de_1)
    #de_1 = BatchNormalization(name='gen_de_bn_1')(de_1)
    #de_1 = Dropout(rate=0.5)(de_1)
    #de_1 = Concatenate(axis = -1)([de_1, en_7])
    #de_1 = Activation('relu')(de_1)

    # 2 decoder CD1024 (decodes en_7)
    #de_2 = UpSampling2D(size=(2, 2))(de_1)
    #de_2 = Conv2D(filters=1024, kernel_size = 4, padding='same')(de_2)
    #de_2 = BatchNormalization(name='gen_de_bn_2')(de_2)
    #de_2 = Dropout(rate=0.5)(de_2)
    #de_2 = Concatenate(axis = -1)([de_2, en_6])
    #de_2 = Activation('relu')(de_2)

    # 3 decoder CD1024 (decodes en_6)
    #de_3 = UpSampling2D(size=(2, 2))(de_2)
    #de_3 = Conv2D(filters=1024, kernel_size = 4, padding='same')(de_3)
    #de_3 = BatchNormalization(name='gen_de_bn_3')(de_3)
    #de_3 = Dropout(rate=0.5)(de_3)
    #de_3 = Concatenate(axis = -1)([de_3, en_5])
    #de_3 = Activation('relu')(de_3)

    # 4 decoder CD1024 (decodes en_5)
    #de_4 = UpSampling2D(size=(2, 2))(de_3)
    #de_4 = Conv2D(filters=1024, kernel_size = 4, padding='same')(de_4)
    #de_4 = BatchNormalization(name='gen_de_bn_4')(de_4)
    #de_4 = Dropout(rate=0.5)(de_4)
    #de_4 = Concatenate(axis = -1)([de_4, en_4])
    #de_4 = Activation('relu')(de_4)

    # 5 decoder CD1024 (decodes en_4)
    #de_5 = UpSampling2D(size=(2, 2))(de_4)
    #de_5 = Conv2D(filters=1024, kernel_size = 4, padding='same')(de_5)
    #de_5 = BatchNormalization(name='gen_de_bn_5')(de_5)
    #de_5 = Dropout(rate=0.5)(de_5)
    #de_5 = Concatenate(axis = -1)([de_5, en_3])
    #de_5 = Activation('relu')(de_5)

    # 6 decoder C512 (decodes en_3)
    #de_6 = UpSampling2D(size=(2, 2))(de_5)
    #de_6 = Conv2D(filters=512, kernel_size = 4, padding='same')(de_6)
    #de_6 = BatchNormalization(name='gen_de_bn_6')(de_6)
    #de_6 = Dropout(rate=0.5)(de_6)
    #de_6 = Concatenate(axis = -1)([de_6, en_2])
    #de_6 = Activation('relu')(de_6)

    # 7 decoder CD256 (decodes en_2)
    #de_7 = UpSampling2D(size=(2, 2))(de_6)
    #de_7 = Conv2D(filters=256, kernel_size = 4, padding='same')(de_7)
    #de_7 = BatchNormalization(name='gen_de_bn_7')(de_7)
    #de_7 = Dropout(rate=0.5)(de_7)
    #de_7 = Concatenate(axis = -1)([de_7, en_1])
    #de_7 = Activation('relu')(de_7)

    # After the last layer in the decoder, a convolution is applied
    # to map to the number of output channels (3 in general,
    # except in colorization, where it is 2), followed by a Tanh
    # function.
    de_i = UpSampling2D(size=(2, 2))(de_i)#(de_7)
    de_i = Conv2D(filters=num_output_channels, kernel_size = 4, padding='same')(de_i)
    de_i = Activation('tanh')(de_i)

    unet_generator = Model(inputs=input_layer, outputs=de_i, name='unet_generator')
    return unet_generator