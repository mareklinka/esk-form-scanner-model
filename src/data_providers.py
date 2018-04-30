from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import re
import random
import itertools

def training_data(folder_path):
    """
    Returns an endless generator of training data from the specified folder.
    Examples are permuted randomly between passes.

    Parameters
    ----------

    folder_path : string
        The path to a folder to read training examples from
    """

    batch_size = 5
    images = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) & f.endswith('.jpg') ]
    labels = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) & f.endswith('.txt') ]

    i = 0
    while True:
        X_train = np.empty((5, 847, 1200, 3), dtype="float")
        Y_train = np.empty((5, 8))

        for b in range(batch_size):
            if i == len(images):
                i = 0
                __shuffle_in_unison(images, labels)
            image = load_img(images[i], target_size=(847,1200))
            img_array = img_to_array(image)
            img_array = img_array.reshape((1,) + img_array.shape)

            X_train[b, :, :, :] = img_array / 255

            with open(labels[i], "r") as content:
                points = re.split('[ ,]', content.read().strip())
                Y_train[b, :] = [float(p) for p in points]

            i += 1

        yield X_train, Y_train

def validation_data(folder_path):
    """
    Returns a generator of validation data from the specified folder.
    The generator will return all examples in the folder once, then exit.

    Parameters
    ----------

    folder_path : string
        The path to a folder to read training examples from
    """

    batch_size = 5
    images = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) & f.endswith('.jpg') ]
    labels = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) & f.endswith('.txt') ]

    tuples = list(zip(images, labels))

    for batch in __split_list(tuples, batch_size):
        X = np.empty((5, 847, 1200, 3), dtype="float")
        Y = np.empty((5, 8))

        b = 0
        for x, y in batch:
            image = load_img(x, target_size=(847,1200))
            img_array = img_to_array(image)
            img_array = img_array.reshape((1,) + img_array.shape)

            X[b, :, :, :] = img_array / 255

            with open(y, "r") as content:
                points = re.split('[ ,]', content.read().strip())
                Y[b, :] = [float(p) for p in points]
            b += 1

        yield X, Y

def __split_list(list, chunk_size):
    """
    Splits the provided list into chunks of a specific size.

    Parameters
    ----------

    list : list
        The list to split into chunks
    chunk_size : int
        The size of a chunk
    """
    
    return [list[offs:offs+chunk_size] for offs in range(0, len(list), chunk_size)]

def __shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
