# Neural Network
This repository contains the files necessary to describe our convolutional neural network that will allow our application to detect handwritten words.

# Explanation of the code
(run.py and cnn\_model.py are old versions of the code, will be removed)

## CNN\_model
Contains the files for running the word classifier

### cnn\_model\_keras.py
Defines the model to be trained. Uses 5 CNN layers, each with batch normalization before activation, and a max pooling layer after activation, which feed into 2 dense layers, which feed into a classification layer. The classification layer must contain units equal to the number of labels. The dense layers should have roughly 5-10% of the units in the classification layer.

### dataload.py
Contains functions for loading the dataset, converting images to a pixel array, and running a word classifier from a Keras model. 

load\_data first attempts to load data from a pickle file; if it is not found, it generates a datafile by walking through the image directory looking for images matching the words given in 1-1000.txt.

convert\_to\_pixel\_array takes an image and resizes it to 100x32, converts it to grayscale, then converts it to a 2D array of pixel intensities, normalized and zero-centered by subtracting each pixel value by the image mean and dividing it by the standard deviation.

WordClassifier loads the given model and, given an image, will give the top 5 words and their probabilities. It will be changed to load a light version of the dataset containing only words and their int ID, rather than the entire dataset.

### run\_keras.py
Loads the word data, separates it into train and test sets, trains a model, and saves it to a file.

### run\_classifier.py
Loads the model and runs it against a set of test images.
