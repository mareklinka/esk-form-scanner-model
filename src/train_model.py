from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import regularizers
from keras.callbacks import TensorBoard, ModelCheckpoint
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
    model.add(Conv2D(64, (5, 5), strides=(2,2), input_shape=(c.image_height, c.image_width, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5),strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(192, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(192))
    
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(c.prediction_size))

    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    model.summary()

    # tbCallback = TensorBoard(log_dir='./TB', histogram_freq=0, write_graph=True, write_images=True)
    cpCallback = ModelCheckpoint("models\\current_model_best.h5", save_best_only=True, monitor="val_loss", mode="min", save_weights_only=False)
    model.fit_generator(gen.training_data(trainig_data_path), epochs=100, steps_per_epoch=160, validation_data=gen.training_data("data\\validation"),validation_steps=12, callbacks=[cpCallback])

    model_storage.save_model(model, "current_model")