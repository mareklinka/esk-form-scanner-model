from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import data_providers as gen
import model_storage

model = Sequential()
model.add(Conv2D(256, (3, 3), strides=(2,2), input_shape=(847, 1200, 3)))
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

model.add(Dense(70))
model.add(Activation('relu'))

model.add(Dense(35))
model.add(Activation('relu'))

model.add(Dense(8))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit_generator(gen.training_data("data\\training"), epochs=40, steps_per_epoch=100)

model_storage.save_model(model, "current_model")