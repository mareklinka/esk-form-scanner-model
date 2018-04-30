from keras.models import load_model as load_keras_model
from os.path import join

def save_model(model, filename):
    """
    Saves the specified Keras model into a file.

    Parameters
    ----------

    model : Keras model
        The model to store
    filename : string
        The file name to give to the resulting file
    """

    model.save(__construct_path(filename))

def load_model(filename):
    """
    Loads the specified Keras model from a file.

    Parameters
    ----------
    filename : string
        The name of the file to read from

    Returns
    -------
    Keras model
        The Keras model loaded from a file
    """

    return load_keras_model(__construct_path(filename))

def __construct_path(filename):
    """
    Creates a model path from filname.

    Parameters
    ----------

    filename : string
        The filename to construct path from
    """
    
    return join("models", filename + ".h5")