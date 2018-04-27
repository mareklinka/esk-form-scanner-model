from keras.models import load_model as load_keras_model
from os.path import join

def save_model(model, filename):
    model.save(__construct_path(filename))

def load_model(filename):
    return load_keras_model(__construct_path(filename))

def __construct_path(filename):
    return join("models", filename + ".h5")