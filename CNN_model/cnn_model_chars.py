"""
62-character classifier:
0-9
a-z
A-Z
"""
import time
import os
import warnings
import numpy as np
import keras
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

class CharacterClassifier:
    def __init__(self, img_height, img_width):
        self.dropout = 0.5
        self.strides = 2
        self.height = img_height
        self.width = img_width
        self.kernel_size = [5,5]
        self.kernel_size_2 = [3,3]
        self.pool_size = (2,2)
        self.conv1_filters = 32
        self.conv2_filters = self.conv1_filters*2
        self.conv3_filters = self.conv2_filters*2
        self.conv4_filters = self.conv3_filters*2
        self.conv5_filters = self.conv3_filters*2
        self.num_units = 128
        self.num_units_2 = int(self.num_units/2)
        self.num_labels = 62
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()

        # CONVOLUTION LAYER 1
        model.add(Conv2D(
            input_shape=(self.height, self.width, 1),
            filters=self.conv1_filters,
            kernel_size=self.kernel_size,
            padding="same"))
        BatchNormalization(axis=-1)
        model.add(Activation('relu'))

        model.add(MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides))

        # CONVOLUTION LAYER 2
        model.add(Conv2D(
            filters=self.conv2_filters,
            kernel_size=self.kernel_size,
            padding="same"))
        BatchNormalization(axis=-1)
        model.add(Activation('relu'))

        model.add(MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides))
        #"""
        # CONVOLUTION LAYER 3
        model.add(Conv2D(
            filters=self.conv3_filters,
            kernel_size=self.kernel_size_2,
            padding="same"))
        BatchNormalization(axis=-1)
        model.add(Activation('relu'))

        model.add(MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides))
        #"""
        """
        # CONVOLUTION LAYER 4
        model.add(Conv2D(
            filters=self.conv4_filters,
            kernel_size=self.kernel_size_2,
            padding="same"))
        BatchNormalization(axis=-1)
        model.add(Activation('relu'))

        model.add(MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides))
        """
        """
        # CONVOLUTION LAYER 5
        model.add(Conv2D(
            filters=self.conv5_filters,
            kernel_size=self.kernel_size_2,
            padding="same"))
        BatchNormalization(axis=-1)
        model.add(Activation('relu'))

        model.add(MaxPooling2D(
            pool_size=self.pool_size,
            strides=self.strides))
        """
        model.add(Flatten())

        # DENSE LAYER 1
        model.add(Dense(
            units=self.num_units,
            activation='relu'))

        model.add(Dropout(
            rate=self.dropout/2))

        # DENSE LAYER 2
        model.add(Dense(
            units=self.num_units_2,
            activation='relu'))

        model.add(Dropout(
            rate=self.dropout))

        # CLASSIFICATION LAYER
        model.add(Dense(
            units=self.num_labels,
            activation='softmax'))

        start = time.time()
        model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
        print('> Compilation Time: {}'.format(time.time() - start))
        return model
