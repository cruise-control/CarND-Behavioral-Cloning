from idlelib.idle_test.test_io import S

import os.path
import pandas as pd
import pickle
import numpy as np
from scipy.misc import imread, imresize
# import matplotlib as plt
from sklearn.model_selection import train_test_split
# import math
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import random


class RawDataHandler:
    # csv_file = ""
    csv_headers = {"center": 0, 'left': 1, 'right': 2, 'steering_angle': 3}
    pickle_file = './train.p'

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.csv_data = pd.read_csv(csv_file)

    def get_pickle_data(self, location='center', regenerate='false'):

        self.pickle_raw_data(location=location, regenerate=regenerate)

        return pickle.load(open(self.pickle_file, "rb"))

    def pickle_raw_data(self, location='center', regenerate='false'):

        generate_pickled_file = 'false'

        # Does the file exist? Yes? Are we forcing a regemeration? Np? Generate it
        if os.path.isfile(self.pickle_file):
            if regenerate == 'true':
                generate_pickled_file = 'true'
        else:
            generate_pickled_file = 'true'

        if generate_pickled_file == 'true':
            x = self.get_images(location)
            y = self.get_steering_angles()
            y = y.tolist()
            y = np.asarray(y)
            # Scale result to be between 0 - 50
            y += 25
            # Convert to a float between 0 and 1
            y /= 50
            pickle.dump((x, y), open(self.pickle_file, "wb"))

    def get_image_locations(self, location="center"):

        index = self.csv_headers[location]
        # print(self.csv_data.info())
        image_paths = self.csv_data.icol(index)
        return image_paths

    def pre_process_images(self, images):
        result = list()
        for img in images:
            result.append(imresize(img, 50))
        return result

    def get_images(self, location="center"):

        image_locations = self.get_image_locations(location)
        images_from_car = list()
        for img in image_locations:
            images_from_car.append(imread(img).astype(np.float32))
        # Scale and order between -0.5 and 0.5
        images_from_car = np.asarray(images_from_car)
        images_from_car /= 255
        images_from_car -= np.mean(images_from_car)
        images_from_car = self.pre_process_images(images_from_car)
        return np.asarray(images_from_car)

    def get_steering_angles(self):
        return self.csv_data.icol(self.csv_headers['steering_angle'])


class CloningModel:
    model_name = "model.json"
    weights_name = "model.h5"

    @staticmethod
    def BehaviorModel(input_shape):
        mdl = Sequential()

        mdl.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=input_shape))
        mdl.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
        mdl.add(Dropout(0.5))
        mdl.add(Activation('relu'))
        mdl.add(Convolution2D(128, 3, 3, border_mode='valid', input_shape=input_shape))
        mdl.add(MaxPooling2D(pool_size=(4, 4), strides=None, border_mode='valid', dim_ordering='default'))
        mdl.add(Activation('relu'))
        mdl.add(Flatten())
        mdl.add(Dense(256))
        mdl.add(Dense(128))
        mdl.add(Dense(1))
        # mdl.add(Activation('relu'))
        mdl.summary()
        mdl.compile(loss='mean_absolute_error', optimizer='rmsprop')

        return mdl

    def SaveMoodel(self, mdl):
        json_string = mdl.to_json()
        with open(self.model_name, 'w') as json:
            json.write(json_string)
        mdl.save_weights(self.weights_name)
        print("Saved the model and weights")

    def LoadModel(self):
        with open(self.model_name, 'r') as jfile:
            model = model_from_json(jfile.read())
        model.compile("adam", "mse")
        weights_file = self.model_name.replace('json', 'h5')
        model.load_weights(weights_file)
        return model


def train_flow(X_train, y_train, X_val, y_val):
    batch_size = 50
    nb_epoch = 20

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=False)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    cl = CloningModel()

    # Figure out the input shape of the images
    input_shape = X_data[1].shape

    # Get the model to train on
    clone_model = cl.BehaviorModel(input_shape=input_shape)
    print(X_train.shape)

    history = clone_model.fit_generator(train_generator, samples_per_epoch=2000,
                                        nb_epoch=nb_epoch, validation_data=(X_val, y_val)
                                        )
    scr = clone_model.evaluate(X_val, y_val, batch_size=batch_size)
    # cl.SaveMoodel(clone_model)
    print("Final Score (on validation data is: ", scr)


def train(X_train, y_train, X_val, y_val):
    batch_size = 50
    nb_epoch = 20

    cl = CloningModel()

    # Figure out the input shape of the images
    input_shape = X_data[1].shape

    # Get the model to train on
    clone_model = cl.BehaviorModel(input_shape=input_shape)
    print(X_train.shape)

    history = clone_model.fit(X_train, y_train,
                              batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val),
                              shuffle='true')
    scr = clone_model.evaluate(X_val, y_val, batch_size=batch_size)
    # cl.SaveMoodel(clone_model)
    print("Final Score (on validation data is: ", scr)


def test_on_images(X_val, y_val):
    cln = CloningModel()
    mdl = cln.LoadModel()
    print(len(y_val))
    print(X_val.shape)
    for index in range(0, len(y_val)):
        test_image = X_val[index]
        test_result = y_val[index]
        transformed_image_array = test_image[None, :, :, :]
        result = float(mdl.predict(transformed_image_array, batch_size=1))
        print("Result: ", result, " Expected: ", test_result)


if __name__ == '__main__':

    train_flag = 1

    # Load the data ( place it into a pickle file if it is not already for future runs )
    raw_access = RawDataHandler("./simulator/driving_log.csv")
    X_data, y_data = raw_access.get_pickle_data(location='center', regenerate='false')

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.33, random_state=50)

    if train_flag:
        train_flow(X_train, y_train, X_val, y_val)

    # batch_size = 32
    # nb_epoch = 20
    #
    #
    # # Load the data ( place it into a pickle file if it is not already for future runs )
    # raw_access = RawDataHandler("./simulator/driving_log.csv")
    # X_data, y_data = raw_access.get_pickle_data(location='center',regenerate='false')
    #
    # # Split into train and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.33, random_state=50)
    #
    # clone = CloningModel()
    #
    # if train_flag:
    #     # Figure out the input shape of the images
    #     input_shape = X_data[1].shape
    #
    #     # Get the model to train on
    #     clone_model = clone.BehaviorModel(input_shape=input_shape)
    #     print(X_train.shape)
    #
    #     history = clone_model.fit(X_train, y_train,
    #                               batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_val, y_val),
    #                               shuffle='true')
    #     score = clone_model.evaluate(X_val, y_val, batch_size=batch_size)
    #     clone.SaveMoodel(clone_model)
    #     print("Final Score (on validation data is: ", score)

    test_on_images(X_val, y_val)

    # Train the model
    print('Finished')
