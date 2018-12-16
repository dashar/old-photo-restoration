import os
import numpy as np
import h5py
from keras.preprocessing.image import ImageDataGenerator

def normalize(X):
    return X / 255.0

def data_batch_generator(X, Y, aug_dict = None, batch_size = 16, color_mode = "grayscale", seed = 1, \
                         X_test = None, Y_test = None):
    '''
    Creates an ImageDataGenerator object with aug_dict as parameters and fits X and Y to it (i.e. centering and
    normalization if the are listed in aug_dict). The output is a train batches generator if X_test is not provived
    and train and test data generators if it is provided. (test data is not augmented, but is centered and normalized
    as the train data)
    '''
    if aug_dict is None:
        aug_dict = dict(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=2,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.01,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.01,
        shear_range=0.01,  # set range for random shear
        zoom_range=0.01,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=1,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)


    x_datagen = ImageDataGenerator(**aug_dict)
    y_datagen = ImageDataGenerator(**aug_dict)
    x_datagen.fit(X, augment=True, seed=seed)
    y_datagen.fit(Y, augment=True, seed=seed)
    x_generator = x_datagen.flow(
        X,
        batch_size=batch_size,
        seed = seed)
    y_generator = y_datagen.flow(
        Y,
        batch_size=batch_size,
        seed = seed)
    
    train_generator = zip(x_generator, y_generator)
    
    if X_test is not  None:
        keys = ['featurewise_center', 'samplewise_center', 'featurewise_std_normalization', 
        'samplewise_std_normalization', 'zca_whitening', 'zca_epsilon']
        test_dict = {}
        for key in keys:
            test_dict[key] = aug_dict[key]
        
        
        x_datagen_test = ImageDataGenerator(**test_dict)
        y_datagen_test = ImageDataGenerator(**test_dict)
        x_datagen_test.fit(X, augment=True, seed=seed)
        y_datagen_test.fit(Y, augment=True, seed=seed)
        x_test_generator = x_datagen_test.flow(
            X_test,
            batch_size=X_test.shape[0],
            seed = seed)
        y_test_generator = y_datagen_test.flow(
            Y_test,
            batch_size=X_test.shape[0],
            seed = seed)
    
        test_generator = zip(x_test_generator, y_test_generator)
    
        return train_generator, test_generator
    else:
        return train_generator, None