#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN Model for the Handwriting OCR

Created on Thu Sep  14 15:02:43 2017

@author: Alexander Chiu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os
import fnmatch
import re
from PIL import Image
import _pickle as pickle
from sets import Set

tf.logging.set_verbosity(tf.logging.INFO)

# Hyperparameters
LEARNING_RATE = 0.01
NUM_INPUT = 3000 # Our data input (img size: 100 * 30)
NUM_CLASSES = 90000 # 90000 classes in our dataset
NUM_UNITS = 3000
DROPOUT = 0.5 # Fifty percent chance to keep neuron activated
STRIDES = 2
HEIGHT = 32
WIDTH = 100
KERNEL_SIZE = [5,5]
KERNEL_SIZE_2 = [3,3]
POOL_SIZE = [2,2]
CONV1_FILTERS = 64
CONV2_FILTERS = 128
CONV3_FILTERS = 256
CONV4_FILTERS = 512
CONV5_FILTERS = 512

def load_data(image_directory, num_folders, num_words):
	images = []
	labels = []
	regexp = '[a-zA-Z]+'
	count = 0
	words = Set()

	try:
		# Attempts to load data from pickle
		data = pickle.load(open("dataset.p", "rb"))
		images, labels, words = zip(*data)
		print('data loaded from dataset.p')
	except:
		for root, dirnames, filenames in os.walk(image_directory):
			for filename in fnmatch.filter(filenames, '*.jpg'):
				fname = os.path.splitext(filename)[0]
				m = re.search(regexp, fname)
				if(m):
					word = m.group(0).lower()
					if(len(words) < num_words or word in words):
						image_path = os.path.join(root, filename)
						try:
							img = convert_to_pixel_array(image_path)
							labels.append(word)
							images.append(img)
							words.add(word)
						except:
							print('image not valid: ', filename)
			count = count + 1
			if (count > num_folders):
				break
		images = np.array(images).astype(np.float32)
		words = list(words)
		words = {k: v for v, k in enumerate(words)}
		labels = [words[l] for l in labels]
		# Pickle data so this process doesn't need to be repeated
		data = zip(images, labels, words)
		pickle.dump(data, open("dataset.p", "wb"))
		print('data saved to dataset.p')

	global NUM_CLASSES
	NUM_CLASSES = len(words)
	print(NUM_CLASSES)
	return images, labels, words

def convert_to_pixel_array(image_path):
	pixels = []

	im = Image.open(image_path, 'r').resize((WIDTH, HEIGHT), Image.BICUBIC).convert('L')
	pixels = list(im.getdata())

	# Normalize and zero center pixel data
	std_dev = np.std(pixels)
	img_mean = np.mean(pixels)

	pixels = [(pixels[offset:offset+WIDTH]-img_mean)/std_dev for offset in range(0, WIDTH*HEIGHT, WIDTH)]
	
	return pixels

# Define the CNN model
def conv_net(features, labels, mode, reuse, is_training):
	with tf.variable_scope('ConvNet', reuse=reuse):
		# INPUT LAYER
		input_layer = tf.reshape(features["images"], [-1, HEIGHT, WIDTH, 1])

		# CONVOLUTION LAYER #1
		conv1 = tf.layers.conv2d(inputs=input_layer,
					 filters=CONV1_FILTERS,
					 kernel_size=KERNEL_SIZE,
					 padding="same",
					 activation=tf.nn.relu)
		# POOLING LAYER #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1,
						pool_size=POOL_SIZE,
						strides=STRIDES)

		# CONVOLUTION LAYER #2
		conv2 = tf.layers.conv2d(inputs=pool1,
					 filters=CONV2_FILTERS,
					 kernel_size=KERNEL_SIZE,
					 padding="same",
					 activation=tf.nn.relu)

		# POLLING LAYER #2
		pool2 = tf.layers.max_pooling2d(inputs=conv2,
						pool_size=POOL_SIZE,
						strides=STRIDES)

		# CONVOLUTION LAYER #3
		conv3 = tf.layers.conv2d(inputs=pool2,
					 filters=CONV3_FILTERS,
					 kernel_size=KERNEL_SIZE_2,
					 padding="same",
					 activation=tf.nn.relu)

		# POOLING LAYER #3
		pool3 = tf.layers.max_pooling2d(inputs=conv3,
						pool_size=POOL_SIZE,
						strides=STRIDES)

		# CONVOLUTION LAYER #4
		conv4 = tf.layers.conv2d(inputs=pool3,
					 filters=CONV4_FILTERS,
					 kernel_size=KERNEL_SIZE_2,
					 padding="same",
					 activation=tf.nn.relu)

		# POOLING LAYER #4
		#pool4 = tf.layers.max_pooling2d(inputs=conv4,
						#pool_size=POOL_SIZE,
						#strides=STRIDES)

		# CONVOLUTION LAYER #5
		conv5 = tf.layers.conv2d(inputs=conv4, #pool4,
					filters=CONV5_FILTERS,
					kernel_size=KERNEL_SIZE_2,
					padding="same",
					activation=tf.nn.relu)

		# POOLING LAYER #5
		pool5 = tf.layers.max_pooling2d(inputs=conv5,
					pool_size=POOL_SIZE,
					strides=STRIDES)

		# DENSE LAYER #1
		#pool4_flat = tf.reshape(pool4, [-1, 4 * 13 * 512])
		#pool4_flat = tf.contrib.layers.flatten(pool4)
		pool4_flat = tf.contrib.layers.flatten(pool5)
		fc1 = tf.layers.dense(inputs=pool4_flat,
				      units=NUM_UNITS,
				      activation=tf.nn.relu)
		dropout1 = tf.layers.dropout(inputs=fc1,
					     rate=DROPOUT,
					     training=is_training)

		# DENSE LAYER #2
		fc2 = tf.layers.dense(inputs=dropout1,
				      units=NUM_UNITS,
				      activation=tf.nn.relu)
		dropout2 = tf.layers.dropout(inputs=fc2,
					     rate=DROPOUT,
					     training=is_training)

		# LOGITS/CLASSIFIER LAYER
		output = tf.layers.dense(inputs=dropout2,
					 units=NUM_CLASSES)
		
	return output

def model_fn(features, labels, mode):
	logits_train = conv_net(features, labels, mode, reuse=False, is_training=True)
	logits_test = conv_net(features, labels, mode, reuse=True, is_training=False)

	# Predictions
	pred_classes = tf.argmax(logits_test, axis=1)
	pred_probas = tf.nn.softmax(logits_test)

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
	
	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
	train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

	# Evaluate the accuracy of the model
	acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

	# TF Estimators requires to return a EstimatorSpec, that specify
	# the different ops for training, evaluating, ...
	estim_specs = tf.estimator.EstimatorSpec(
	mode=mode,
	predictions=pred_classes,
	loss=loss_op,
	train_op=train_op,
	eval_metric_ops={'accuracy': acc_op})

	return estim_specs
