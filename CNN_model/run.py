# Trains the model
import cnn_model
import time
import os
import fnmatch
import re
from PIL import Image
import numpy as np
import tensorflow as tf
import random

if __name__=='__main__':
	global_start_time = time.time()
	
	batch_size = 128
	num_steps = 2000
	image_directory = '../../../Dataset/synth-words/mnt/ramdisk/max/90kDICT32px/'
	training_images_path = '1'
	test_image_path = '68/1/235_minion_48634.jpg'

	# Load data
	images, labels, words = cnn_model.load_data(image_directory + training_images_path)

	# Shuffle data
	data = list(zip(images, labels))
	random.shuffle(data)
	images, labels = zip(*data)

	# Split into train and test sets
	split_idx = int(len(images)*0.8)
	train_images = np.array(images[0:split_idx])
	train_labels = np.array(labels[0:split_idx])
	test_images = np.array(images[split_idx+1:])
	test_labels = np.array(labels[split_idx+1:])

	# Build model
	model = tf.estimator.Estimator(cnn_model.model_fn)

	# Define the input function for training
	input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'images':train_images}, y=train_labels,
		batch_size=batch_size, num_epochs=None, shuffle=True)

	# Train the Model
	model.train(input_fn, steps=num_steps)
	print('Training duration (s): ', time.time() - global_start_time)

	# Evaluate the Model
	# Define the input function for evaluating
	input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'images':train_images}, y=train_labels,
		batch_size=batch_size, shuffle=False)
	# Use the Estimator 'evaluate' method
	e = model.evaluate(input_fn)

	print("Training Accuracy:", e['accuracy'])

	input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'images':test_images}, y=test_labels,
		batch_size=batch_size, shuffle=False)
	# Use the Estimator 'evaluate' method
	e = model.evaluate(input_fn)

	print("Testing Accuracy:", e['accuracy'])

	# Test classifier
	#cnn_model.classify_image(image_directory + test_image_path)

