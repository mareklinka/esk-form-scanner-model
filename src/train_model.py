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
    
    model.add(Conv2D(16, (5, 5), strides=(2,2), input_shape=(c.image_height, c.image_width, 3),kernel_regularizer=regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (5, 5),strides=(2,2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(c.prediction_size))

    model.compile(loss='mae', optimizer='adam', metrics=['mae'])

    model.summary()

    tbCallback = TensorBoard(log_dir='./TB', histogram_freq=0, write_graph=True, write_images=True)
    cpCallback = ModelCheckpoint("models\\current_model_best.h5", save_best_only=True, monitor="val_loss", mode="min", save_weights_only=False)
    model.fit_generator(gen.infinite_generator(trainig_data_path), epochs=100, steps_per_epoch=240, validation_data=gen.infinite_generator("data\\validation"),validation_steps=30, callbacks=[cpCallback, tbCallback])

    model_storage.save_model(model, "current_model")