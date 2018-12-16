from keras.layers import Input, Lambda, Concatenate
from keras.models import Model


def DCGAN(generator_model, discriminator_model, input_img_dim, patch_dim):
    """
    Here we do the following:
    1. Generate an image with the generator
    2. break up the generated image into patches
    3. feed the patches to a discriminator to get the avg loss across all patches
        (i.e is it fake or not)
    4. the DCGAN outputs the generated image and the loss

    This differs from standard GAN training in that we use patches of the image
    instead of the full image (although a patch size = img_size is basically the whole image)

    :param generator_model:
    :param discriminator_model:
    :param img_dim:
    :param patch_dim:
    :return: DCGAN model
    """

    generator_input = Input(shape=input_img_dim, name="DCGAN_input")

    # generated image model from the generator
    generated_image = generator_model(generator_input)
    
    #input the 2 images into the discriminator
    dcgan_output = discriminator_model([generated_image, generator_input])

    # actually turn into keras model
    dc_gan = Model(inputs=generator_input, outputs=[generated_image, dcgan_output], name="DCGAN")
    return dc_gan