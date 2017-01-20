from idlelib.idle_test.test_io import S

import os.path, getopt
import sys
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
from keras.layers import Convolution2D, MaxPooling2D, ELU, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import random
import argparse
import gc


class Parameters:

    def __init__(self):
        self.epochs = 10
        self.batch_size = 32
        self.train = True
        self.test = False
        self.regenerate = False
        self.samples_per_epoch = 1000
        self.raw_samples = 5000

    def dump(self):
        print("epochs ", self.epochs)
        print("batch_size ", self.batch_size)
        print("train ", self.train)
        print("test ", self.test)
        print("regenerate ", self.regenerate)
        print("samples_per_epoch ", self.samples_per_epoch)

    def parse_parameters(self):
        parser = argparse.ArgumentParser(description='Pass arguemnts to the program')
        parser.add_argument('-t', '--test', help='Run the final tests', required=False, default=False)
        parser.add_argument('-T', '--train', help='Train the model', required=False, default=False)
        parser.add_argument('-e', '--epochs', type=int, help='number of epochs to run for', required=False, default=10)
        parser.add_argument('-r', '--regen', help='regenerate the test data', required=False, default=False)
        parser.add_argument('-s', '--samples_per_epoch', type=int, help='samples to generate per epoch', required=False, default=1000)
        parser.add_argument('-b', '--batch_size', type=int, help='size of batches to run every epoch', required=False, default=50)
        parser.add_argument('-R', '--raw_samples', type=int, help='number of raw samples to load', required=False, default=5000)
        args = parser.parse_args()
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.raw_samples = args.raw_samples
        self.regenerate = args.regen
        self.train = args.train
        self.test = args.test
        self.samples_per_epoch = args.samples_per_epoch


class RawDataHandler:
    # csv_file = ""
    csv_headers = {"center": 0, 'left': 1, 'right': 2, 'steering_angle': 3, 'random':4}
    pickle_file = './train.p'
    smooth_steering = False
    side_steering_modifier = 1.1

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.csv_data = pd.read_csv(csv_file)

    def get_test_size(self):
        return len(self.csv_data)

    def get_data_set(self, location="random", nb_samples=1000):
        
        assert(nb_samples < self.get_test_size())
        
        # Choose a sample of random indices
        random_sample_locations = random.sample(range(1,self.get_test_size()),nb_samples)

        # Load either the left, right or center image and corresponding steering angle at this point
        image_locations = list()
        y_values = list()
        for index in random_sample_locations:
            # Get the center image initially
            lcr_image = 0
            
            # Left, Right Image, randomly choose
            if location == 'random':
                lcr_image = random.randint(1,2)
            else:
                lcr_image = self.csv_headers['center']
                
            image_locations.append(self.csv_data.iget_value(index, lcr_image))
            steer_angle = self.csv_data.iget_value(index, self.csv_headers['steering_angle'])
            if lcr_image == 1:
                steer_angle += 0.3
            if lcr_image == 2:
                steer_angle -= 0.3
            y_values.append(steer_angle)
        # Strip any white space in the image locations
        image_locations = [x.strip() for x in image_locations]
        l = list()
        for path in image_locations:
            if "simulator/" in path:
                left, right = path.split("simulator/",1)
                l.append(right)
            else:
                l.append(path)
        image_locations = ["./simulator/" + i for i in l]
        images_from_car = list()
        # Load an image and reduce in size before adding to array
        # This is to reduce the required memory
        for img in image_locations:
            image = imread(img, mode='RGB').astype(np.float32)
            image = imresize(image, 50).astype(np.float32)
            images_from_car.append(image)
        # Scale and order between -0.5 and 0.5
        images_from_car = np.asarray(images_from_car)
        images_from_car /= 255
        images_from_car -= np.mean(images_from_car)
        return images_from_car, np.asarray(y_values)

    def get_pickle_data(self, location='center', regenerate='False'):

        self.pickle_raw_data(location=location, regenerate=regenerate)

        return pickle.load(open(self.pickle_file, "rb"))

    def pickle_raw_data(self, location='center', regenerate='False'):

        generate_pickled_file = 'False'

        # Does the file exist? Yes? Are we forcing a regemeration? Np? Generate it
        if os.path.isfile(self.pickle_file):
            if regenerate == 'True':
                generate_pickled_file = 'True'
        else:
            generate_pickled_file = 'True'

        if generate_pickled_file == 'True':
            x = self.get_images('center')
            y = self.get_steering_angles()
            pickle.dump((x, y), open(self.pickle_file, "wb"))
            gc.collect()

            xl = self.get_images('left')
            yl = self.get_steering_angles()
            yl *= self.side_steering_modifier
            pickle.dump((xl, yl), open(self.pickle_file, "ab"))
            gc.collect()

            xr = self.get_images('right')
            yr = self.get_steering_angles()
            yr *= self.side_steering_modifier
            pickle.dump((xr, yr), open(self.pickle_file, "ab"))
            gc.collect()

    def get_image_locations(self, location="center"):

        index = self.csv_headers[location]
        image_paths = self.csv_data.icol(index)
        image_paths = [x.strip() for x in image_paths]

        return image_paths

    def pre_process_images(self, images):
        result = list()
        for img in images:
            result.append(imresize(img, 50))
        return result

    def get_images(self, location="center"):

        image_locations = self.get_image_locations(location)
        # set the offset for the images here
        image_locations = ["./simulator/" + i for i in image_locations]
        images_from_car = list()
        # Load an image and reduce in size before adding to array
        # This is to reduce the required memory
        for img in image_locations:
            image = imread(img).astype(np.float32)
            image = imresize(image, 50).astype(np.float32)
            images_from_car.append(image)
        # Scale and order between -0.5 and 0.5
        images_from_car = np.asarray(images_from_car)
        images_from_car /= 255
        images_from_car -= np.mean(images_from_car)
        return images_from_car
        # images_from_car = self.pre_process_images(images_from_car)
        # return np.asarray(images_from_car)

    def pre_process_steering_angles(self, angles):
        """
        Ref: http://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
        :param angles:
        :return:
        """
        # # Scale result to be between 0 - 50
        # angles += 1
        # # Convert to a float between 0 and 1
        # angles /= 50

        N = 5
        # Perform an averaging pass over the data
        angles = np.asarray(angles.tolist())

        window = np.ones(int(N)) / float(N)

        ret = np.convolve(a=angles, v=window, mode='SAME')

        return ret

    def get_steering_angles(self):
        """

        :return:
        """
        angles = self.csv_data.icol(self.csv_headers['steering_angle'])

        if self.smooth_steering:
            return self.pre_process_steering_angles(angles)

        return np.asarray(angles)


class CloningModel:
    model_name = "model.json"
    weights_name = "model.h5"


    @staticmethod
    def BehaviorModel(input_shape):
        """
        Ref: https://github.com/commaai/research/blob/master/train_steering_model.py
        :param input_shape:
        :return:
        """
        mdl = Sequential()

        mdl.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=input_shape))
        mdl.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='default'))
        mdl.add(ELU())
        #mdl.add(Activation('tanh'))
        mdl.add(Convolution2D(36, 5, 5, border_mode='valid'))
        mdl.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='default'))
        mdl.add(ELU())
        #mdl.add(Activation('tanh'))
        mdl.add(Convolution2D(48, 5, 5, border_mode='valid'))
        mdl.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='default'))
        mdl.add(ELU())
        # mdl.add(Activation('tanh'))
        mdl.add(Convolution2D(64, 3, 3, border_mode='valid'))
        # mdl.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
        mdl.add(ELU())
        # mdl.add(Activation('tanh'))
        mdl.add(Convolution2D(96, 3, 3, border_mode='valid'))
        # mdl.add(Dropout(.2))
        # mdl.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
        mdl.add(ELU())
        # mdl.add(Activation('tanh'))
        mdl.add(Flatten())
        mdl.add(Dropout(.5))
        mdl.add(Dense(100))
        #mdl.add(ELU())
        mdl.add(Activation('tanh'))
        mdl.add(Dense(50))
        mdl.add(Dropout(.5))
        #mdl.add(ELU())
        mdl.add(Activation('tanh'))
        mdl.add(Dense(10))
        mdl.add(Dense(1))
        mdl.add(Activation('tanh')) # Push the output between an allowed range

        mdl.summary()
        mdl.compile(optimizer="adam", loss="mse")

        return mdl


    def BehaviorModelV2(self,input_shape):
        ch, row, col = input_shape  # camera format

        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=(ch, row, col),
                         output_shape=(ch, row, col)))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")

        return model

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


def train_flow_manual(input_shape, samples_to_load=1000, samples_per_epoch=5000, nb_epoch=20, batch_size=50):

    image_access = RawDataHandler("./simulator/driving_log.csv")

    train_datagen = ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.0,
        zoom_range=0.0,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False)

    cl = CloningModel()

    # Get the model to train on
    clone_model = cl.BehaviorModel(input_shape=input_shape)

    for e in range(nb_epoch):
        print('Epoch', e)

        # Load a fresh set of center images and 1/10th of random data from the total dataset here
        X_data, y_data = image_access.get_data_set(location='center',nb_samples=samples_to_load)
        X_data_r, y_data_r = image_access.get_data_set(location='random',nb_samples=int(samples_to_load/40))
        X_data = np.concatenate([X_data,X_data_r])
        y_data = np.concatenate([y_data,y_data_r])
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.33, random_state=50)

        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

        history = clone_model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
                                            nb_epoch=1, validation_data=(X_val, y_val)
                                            )

    X_test, y_test = image_access.get_data_set(location='center', nb_samples=200)
    scr = clone_model.evaluate(X_test, y_test, batch_size=batch_size)
    print("Final Score (on random test data) is: ", scr)
    cl.SaveMoodel(clone_model)


def train_flow(X_train, y_train, X_val, y_val,samples_per_epoch=5000, nb_epoch=20):
    batch_size = 50

    train_datagen = ImageDataGenerator(
        rotation_range=3,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False)

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    cl = CloningModel()

    # Figure out the input shape of the images
    input_shape = X_data[1].shape

    # Get the model to train on
    clone_model = cl.BehaviorModel(input_shape=input_shape)
    print(X_train.shape)

    history = clone_model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
                                        nb_epoch=nb_epoch, validation_data=(X_val, y_val)
                                        )
    scr = clone_model.evaluate(X_val, y_val, batch_size=batch_size)

    cl.SaveMoodel(clone_model)

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


def display_images(X_train, y_train):
    # How many unique classes/labels there are in the dataset.
    y_value = set()
    for y in y_train:
        y_value.add(y)
    n_classes = len(y_value)
    import random
    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(1):
        # From LeNet lab
        index = random.randint(0, len(X_train))
        image = X_train[index].squeeze()
        plt.figure(figsize=(5, 5))
        plt.title('Class ' + str(y_train[index]))
        plt.imshow(image)
    y_data = [None] * n_classes

    # Generate and plot a histogram of the dataset
    for y in range(0, n_classes):
        y_data[y] = 0

    unique, unique_counts = np.unique(y_train,return_counts=True)
    for index in range(0, len(unique)):
        print( unique[index], " :: ", unique_counts[index])


    idx = np.arange(n_classes)
    plt.figure(figsize=(20, 20))
    plt.title('Chart of classes in the dataset')
    plt.bar(unique, unique_counts, linewidth=0.1)
    plt.show()


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


params = Parameters()

if __name__ == '__main__':

    params.parse_parameters()
    params.dump()

    # Load the data ( place it into a pickle file if it is not already for future runs )
    raw_access = RawDataHandler("./simulator/driving_log.csv")

    # X_data, y_data = raw_access.get_pickle_data(location='center', regenerate=params.regenerate)
    #
    # print("Total train / val dataset contains ", len(X_data))
    # # Split into train and validation sets
    # X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.33, random_state=50)
    #
    # display_images(X_train,y_train)

    if params.train == 'True':
        x, y = raw_access.get_data_set(2)
        train_flow_manual(x[0].shape, samples_to_load=5000, nb_epoch=params.epochs,
                          samples_per_epoch=params.samples_per_epoch, batch_size=params.batch_size)

    if params.test == 'True':
        X_val, y_val = raw_access.get_data_set(200)
        test_on_images(X_val, y_val)

    # Train the model
    print('Finished')
