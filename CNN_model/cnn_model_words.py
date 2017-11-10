"""
Keras implementation of word classifier
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

# Hyperparameters
DROPOUT = 0.5 # Fifty percent chance to keep neuron activated
STRIDES = 2
HEIGHT = 32
WIDTH = 100
KERNEL_SIZE = [5,5]
KERNEL_SIZE_2 = [3,3]
POOL_SIZE = (2,2)
CONV1_FILTERS = 64
CONV2_FILTERS = 128
CONV3_FILTERS = 256
CONV4_FILTERS = 512
CONV5_FILTERS = 512
num_units = 100
num_words = 5000

def build_model():
	model = Sequential()

	# CONVOLUTION LAYER 1
	model.add(Conv2D(
		input_shape=(HEIGHT, WIDTH, 1),
		filters=CONV1_FILTERS,
		kernel_size=KERNEL_SIZE,
		padding="same"))
	BatchNormalization(axis=-1)
	model.add(Activation('relu'))

	model.add(MaxPooling2D(
		pool_size=POOL_SIZE,
		strides=STRIDES))

	# CONVOLUTION LAYER 2
	model.add(Conv2D(
		filters=CONV2_FILTERS,
		kernel_size=KERNEL_SIZE,
		padding="same"))
	BatchNormalization(axis=-1)
	model.add(Activation('relu'))

	model.add(MaxPooling2D(
		pool_size=POOL_SIZE,
		strides=STRIDES))

	# CONVOLUTION LAYER 3
	model.add(Conv2D(
		filters=CONV3_FILTERS,
		kernel_size=KERNEL_SIZE_2,
		padding="same"))
	BatchNormalization(axis=-1)
	model.add(Activation('relu'))

	model.add(MaxPooling2D(
		pool_size=POOL_SIZE,
		strides=STRIDES))

	# CONVOLUTION LAYER 4
	model.add(Conv2D(
		filters=CONV4_FILTERS,
		kernel_size=KERNEL_SIZE_2,
		padding="same"))
	BatchNormalization(axis=-1)
	model.add(Activation('relu'))

	model.add(MaxPooling2D(
		pool_size=POOL_SIZE,
		strides=STRIDES))

	# CONVOLUTION LAYER 5
	model.add(Conv2D(
		filters=CONV5_FILTERS,
		kernel_size=KERNEL_SIZE_2,
		padding="same"))
	BatchNormalization(axis=-1)
	model.add(Activation('relu'))

	model.add(MaxPooling2D(
		pool_size=POOL_SIZE,
		strides=STRIDES))

	model.add(Flatten())

	# DENSE LAYER 1
	model.add(Dense(
		units=num_units,
		activation='relu'))

	model.add(Dropout(
		rate=DROPOUT/2))

	# DENSE LAYER 1
	model.add(Dense(
		units=num_units,
		activation='relu'))

	model.add(Dropout(
		rate=DROPOUT))

	# CLASSIFICATION LAYER
	model.add(Dense(
		units=num_words,
		activation='softmax'))

	start = time.time()
	model.compile(loss=keras.losses.categorical_crossentropy,
			optimizer=keras.optimizers.Adadelta(),
			metrics=['accuracy'])
	print('> Compilation Time: {}'.format(time.time() - start))
	return model
