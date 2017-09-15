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

tf.logging.set_verbosity(tf.logging.INFO)

# Hyperparameters
LEARNING_RATE = 0.001
NUM_STEPS = 2000
BATCH_SIZE = 5000
NUM_INPUT = 3000 # Our data input (img size: 100 * 30)
NUM_CLASSES = 90000 # 90000 classes in our dataset
DROPOUT = 0.5 # Fifty percent chance to keep neuron activated

# Define the CNN model
def conv_Net(features, labels, mode):
    with tf.variable_scope('ConvNet', reuse=reuse):
        # INPUT LAYER
        input_layer = tf.reshape(features["image"], [-1, 32, 100, 1])

        # CONVOLUTION LAYER #1
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)

        # POOLING LAYER #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2, 2],
                                        strides=2)

        # CONVOLUTION LAYER #2
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=128,
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)

        # POLLING LAYER #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2, 2],
                                        strides=2)

        # CONVOLUTION LAYER #3
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.relu)

        # POOLING LAYER #3
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                        pool_size=[2, 2],
                                        strides=2)

        # CONVOLUTION LAYER #4
        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=512,
                                 kernel_size=[3, 3],
                                 padding="same",
                                 activation=tf.nn.relu)

        # POOLING LAYER #4
        pool4 = tf.layers.max_pooling2d(inputs=conv4,
                                        pool_size=[2, 2],
                                        strides=2)
        # DENSE LAYER #1
        pool4_flat = tf.reshape(pool4, [-1, 4 * 13 * 512])
        fc1 = tf.layers.dense(inputs=pool4_flat,
                              units=4096,
                              activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(inputs=fc1,
                                     rate=DROPOUT,
                                     training=mode == tf.estimator.ModeKeys.TRAIN)

        # DENSE LAYER #2
        fc2 = tf.layers.dense(inputs=dropout1,
                              units=4096,
                              activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(inputs=fc2,
                                     rate=DROPOUT,
                                     training=mode == tf.estimator.ModeKeys.TRAIN)

        # LOGITS/CLASSIFIER LAYER
        output = tf.layers.dense(inputs=dropout2,
                                 units=NUM_CLASSES)

# Define the main function to train
# and export the train model

# Main logic
if __name__ == "__main__":
    tf.app.run()
