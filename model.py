
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Convolution2D, MaxPooling2D, ELU, Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread, imresize
from sklearn.model_selection import train_test_split
# import BatchNormalization
from keras.layers.normalization import BatchNormalization


class Parameters:
    """
    Handle arguments to the program
    """

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
        print("raw_samples ", self.raw_samples)

    def parse_parameters(self):
        parser = argparse.ArgumentParser(description='Pass arguments to the program')
        parser.add_argument('-t', '--test', help='Run the final tests', required=False, default=False)
        parser.add_argument('-T', '--train', help='Train the model', required=False, default=False)
        parser.add_argument('-e', '--epochs', type=int, help='number of epochs to run for', required=False, default=10)
        parser.add_argument('-r', '--regen', help='regenerate the test data', required=False, default=False)
        parser.add_argument('-s', '--samples_per_epoch', type=int, help='samples to generate per epoch', required=False,
                            default=1000)
        parser.add_argument('-b', '--batch_size', type=int, help='size of batches to run every epoch', required=False,
                            default=50)
        parser.add_argument('-R', '--raw_samples', type=int, help='number of raw samples to load', required=False,
                            default=5000)
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
    csv_headers = {"center": 0, 'left': 1, 'right': 2, 'steering_angle': 3, 'random': 4}
    pickle_file = './train.p'
    smooth_steering = False
    side_steering_modifier = 1.1
    test_sample_size = 300

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.csv_data = pd.read_csv(csv_file)

    def get_test_set(self):

        # Choose a sample of random indices
        test_locations = range(1, self.test_sample_size)

        # Load either the left, right or center image and corresponding steering angle at this point
        image_locations = list()
        y_values = list()
        for index in test_locations:
            lcr_image = self.csv_headers['center']
            image_locations.append(self.csv_data.iget_value(index, lcr_image))
            steer_angle = self.csv_data.iget_value(index, self.csv_headers['steering_angle'])
            y_values.append(steer_angle)
        # Strip any white space in the image locations
        image_locations = [x.strip() for x in image_locations]
        l = list()
        for path in image_locations:
            if "simulator/" in path:
                left, right = path.split("simulator/", 1)
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

    def get_train_size(self):
        return len(self.csv_data) - self.test_sample_size

    def get_data_set(self, location="random", nb_samples=1000):

        assert (nb_samples < self.get_train_size())

        # Choose a sample of random indices
        random_sample_locations = random.sample(range(self.test_sample_size, self.get_train_size()), nb_samples)

        # Load either the left, right or center image and corresponding steering angle at this point
        image_locations = list()
        y_values = list()
        for index in random_sample_locations:
            lcr_image = 0

            if location == 'random':
                # Left, Right Image, randomly choose
                lcr_image = random.randint(1, 2)
            else:
                lcr_image = self.csv_headers['center']

            image_locations.append(self.csv_data.iget_value(index, lcr_image))
            steer_angle = self.csv_data.iget_value(index, self.csv_headers['steering_angle'])
            if lcr_image == self.csv_headers['left']:
                steer_angle += 0.3
            if lcr_image == self.csv_headers['right']:
                steer_angle -= 0.3
            y_values.append(steer_angle)
        # Strip any white space in the image locations
        image_locations = [x.strip() for x in image_locations]
        l = list()
        for path in image_locations:
            if "simulator/" in path:
                left, right = path.split("simulator/", 1)
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


class CloningModel:
    model_name = "model.json"
    weights_name = "model.h5"

    @staticmethod
    def get_model(input_shape):
        """
        Get the keras model for steering detection
        Ref: https://github.com/commaai/research/blob/master/train_steering_model.py
        Ref: The NVIDIA paper "End to End Learning for Self-Driving Cars"
        :param input_shape:
        :return:
        """
        mdl = Sequential()

        mdl.add(Convolution2D(24, 5, 5, border_mode='valid', input_shape=input_shape))
        mdl.add(BatchNormalization())
        mdl.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='default'))
        mdl.add(ELU())
        mdl.add(Convolution2D(36, 5, 5, border_mode='valid'))
        mdl.add(BatchNormalization())
        mdl.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='default'))
        mdl.add(ELU())
        mdl.add(Convolution2D(48, 5, 5, border_mode='valid'))
        mdl.add(BatchNormalization())
        mdl.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid', dim_ordering='default'))
        mdl.add(ELU())
        mdl.add(Convolution2D(64, 3, 3, border_mode='valid'))
        mdl.add(BatchNormalization())
        mdl.add(ELU())
        mdl.add(Convolution2D(96, 3, 3, border_mode='valid'))
        mdl.add(BatchNormalization())
        mdl.add(ELU())
        mdl.add(Flatten())
        mdl.add(Dropout(.5))
        mdl.add(Dense(100))
        mdl.add(Activation('tanh'))
        mdl.add(Dense(50))
        mdl.add(Dropout(.5))
        mdl.add(Activation('tanh'))
        mdl.add(Dense(10))
        mdl.add(Dense(1))

        # Push the output between an allowed range
        mdl.add(Activation('tanh'))

        mdl.summary()
        mdl.compile(optimizer="adam", loss="mse")

        return mdl

    def save_model(self, mdl):
        json_string = mdl.to_json()
        with open(self.model_name, 'w') as json:
            json.write(json_string)
        mdl.save_weights(self.weights_name)
        print("Saved the model and weights")

    def load_model(self):
        with open(self.model_name, 'r') as jfile:
            model = model_from_json(jfile.read())
        model.compile("adam", "mse")
        weights_file = self.model_name.replace('json', 'h5')
        model.load_weights(weights_file)
        return model


def train_flow_manual(input_shape, samples_to_load=1000, samples_per_epoch=5000, nb_epoch=20, batch_size=50):
    """
    Datagenerator implementation referenced from 
    https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    """
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

    # Get the model to train on
    cl = CloningModel()
    clone_model = cl.get_model(input_shape=input_shape)

    for e in range(nb_epoch):
        print('Epoch', e)

        # Load a fresh set of center images and 1/40th of random Left or Right data from the total data-set
        x_data, y_data = image_access.get_data_set(location='center', nb_samples=samples_to_load)
        x_data_r, y_data_r = image_access.get_data_set(location='random', nb_samples=int(samples_to_load / 40))
        x_data = np.concatenate([x_data, x_data_r])
        y_data = np.concatenate([y_data, y_data_r])

        # Split into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.33, random_state=50)

        train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

        # Train for 1 epoch with this set of data
        clone_model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch,
                                  nb_epoch=1, validation_data=(x_val, y_val)
                                  )

    x_test, y_test = image_access.get_test_set()
    # get_data_set(location='center', nb_samples=200)
    scr = clone_model.evaluate(x_test, y_test, batch_size=batch_size)
    print("Final Score (on random test data) is: ", scr)
    cl.save_model(clone_model)


def display_images(x_data, y_data):
    # How many unique classes/labels there are in the data-set.
    y_value = set()
    for cls in y_data:
        y_value.add(cls)
    n_classes = len(y_value)

    for i in range(1):
        # From LeNet lab
        index = random.randint(0, len(x_data))
        image = x_data[index].squeeze()
        plt.figure(figsize=(5, 5))
        plt.title('Class ' + str(y_data[index]))
        plt.imshow(image)
    y_data = [None] * n_classes

    # Generate and plot a histogram of the data-set
    for cls in range(0, n_classes):
        y_data[cls] = 0

    unique, unique_counts = np.unique(y_data, return_counts=True)
    for index in range(0, len(unique)):
        print(unique[index], " :: ", unique_counts[index])

    plt.figure(figsize=(20, 20))
    plt.title('Chart of classes in the dataset')
    plt.bar(unique, unique_counts, linewidth=0.1)
    plt.show()


def test_on_images(x_test, y_test):
    cln = CloningModel()
    mdl = cln.load_model()
    print(len(y_test))
    print(x_test.shape)
    for index in range(0, len(y_test)):
        test_image = x_test[index]
        test_result = y_test[index]
        transformed_image_array = test_image[None, :, :, :]
        result = float(mdl.predict(transformed_image_array, batch_size=1))
        print("Result: ", result, " Expected: ", test_result)


params = Parameters()

if __name__ == '__main__':

    # Parse the program parameters
    params.parse_parameters()

    # Get access to the Pandas data
    raw_access = RawDataHandler("./simulator/driving_log.csv")

    if params.train == 'True':
        x, y = raw_access.get_data_set(nb_samples=2)
        train_flow_manual(x[0].shape, samples_to_load=5000, nb_epoch=params.epochs,
                          samples_per_epoch=params.samples_per_epoch, batch_size=params.batch_size)

    if params.test == 'True':
        X_val, y_val = raw_access.get_data_set(nb_samples=200)
        test_on_images(X_val, y_val)

    # Train the model
    print('Exiting')
