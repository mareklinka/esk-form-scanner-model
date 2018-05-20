from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
import data_providers as gen
import model_storage
import constants as c

def train(trainig_data_path):
    """
    Trains a new model.

    Parameters
    ----------

    trainig_data_path : string
        The path to a folder with training data
    """

    model = Sequential()
    model.add(Conv2D(256, (3, 3), strides=(2,2), input_shape=(c.image_height, c.image_width, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(384, (3, 3),strides=(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3),strides=(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(768, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(1024, (3, 3)))
    model.add(Activation('relu'))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(c.prediction_size))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'accuracy'])

    model.summary()

    tbCallback = TensorBoard(log_dir='./TB', histogram_freq=0, write_graph=True, write_images=True)

    model.fit_generator(gen.training_data(trainig_data_path), epochs=30, steps_per_epoch=60, callbacks=[tbCallback])

    model_storage.save_model(model, "current_model")