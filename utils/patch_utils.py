import numpy as np


def num_patches(output_img_dim=(256, 256, 1), sub_patch_dim=(64, 64)):
    """
    Creates non-overlaping patches to feed to the PATCH GAN
    (Section 2.2.2 in paper)
    The paper provides 3 options.
    Pixel GAN = 1x1 patches (aka each pixel)
    PatchGAN = nxn patches (non-overlaping blocks of the image)
    ImageGAN = im_size x im_size (full image)

    Ex: 4x4 image with patch_size of 2 means 4 non-overlaping patches

    :param output_img_dim:
    :param sub_patch_dim:
    :return:
    """
    # num of non-overlaping patches
    nb_non_overlaping_patches = (output_img_dim[0] / sub_patch_dim[0]) * (output_img_dim[1] / sub_patch_dim[1])

    # dimensions for the patch discriminator
    patch_disc_img_dim = (sub_patch_dim[0], sub_patch_dim[1], output_img_dim[-1])

    return int(nb_non_overlaping_patches), patch_disc_img_dim


def extract_patches(images, sub_patch_dim):
    """
    Cuts images into k subpatches
    Each kth cut as the kth patches for all images
    ex: input 3 images [im1, im2, im3]
    output [[im_1_patch_1, im_2_patch_1], ... , [im_n-1_patch_k, im_n_patch_k]]

    :param images: array of Images (num_images, im_height, im_width, num_channels)
    :param sub_patch_dim: (height, width) ex: (30, 30) Subpatch dimensions
    :return:
    """
    im_height, im_width = images.shape[1:3]
    patch_height, patch_width = sub_patch_dim

    # list out all xs  ex: 0, 29, 58, ...
    x_spots = range(0, im_width, patch_width)

    # list out all ys ex: 0, 29, 58
    y_spots = range(0, im_height, patch_height)
    all_patches = []

    for y in y_spots:
        for x in x_spots:
            # indexing here is rac
            # images[num_images, width, height, num_channels]
            # this says, cut a patch across all images at the same time with this width, height
            image_patches = images[:, y: y+patch_height, x: x+patch_width, :]
            all_patches.append(np.asarray(image_patches, dtype=np.float32))
    return all_patches

def get_disc_batch(X_original_batch, X_conditional_batch, generator_model, patch_dim, nb_patches = 256,
                   label_smoothing=False, label_flipping=0):
    
    #randomly select images to be fake:
    idx = np.random.choice(X_conditional_batch.shape[0], int(X_conditional_batch.shape[0] / 2))
    X_disc = np.zeros(X_conditional_batch.shape)
    # Create X_disc: alternatively generated or real images
    
    # generate fake image
    X_disc[idx, :, :, :] = generator_model.predict(X_conditional_batch[idx, :, :, :])

    # each image will produce a nb_patches vector for the results (aka is fake or not)
    y_disc = np.ones((X_disc.shape[0], nb_patches), dtype=np.uint8)

    # these are fake iamges
    y_disc[idx, :] = 0

    #    if label_flipping > 0:
    #        p = np.random.binomial(1, label_flipping)
    #        if p > 0:
    #            y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    #else:
    # generate real images
    idx_r = np.setdiff1d(np.arange(X_conditional_batch.shape[0]), idx)
    X_disc[idx_r, :, :, :] = X_original_batch[idx_r, :, :, :]

    # each image will produce a 1x2 vector for the results (aka is fake or not)
    #    if label_smoothing:
    #        y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
    #    else:
    # these are real images

    #    if label_flipping > 0:
    #        p = np.random.binomial(1, label_flipping)
    #        if p > 0:
    #           y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc

    return X_disc, y_disc


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], batch_size, replace=False)
        x1 = X1[idx]
        x2 = X2[idx]
        yield x1, x2