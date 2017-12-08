import os
import time
import cnn_model_words
import dataload
import keras
import random
import numpy as np
from keras import backend as K

if __name__ == '__main__':
    BATCH_SIZE = 100
    EPOCHS = 100
    NUM_WORDS = 5000
    HEIGHT = 32
    WIDTH = 100

    # Load data
    word_data = dataload.load_data(NUM_WORDS)
    print('Loaded data')

    # Split into train and test sets
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    random.seed(100)
    for k, v in word_data.items():
        # Shuffle data
        pixels = [p['pixel_array'] for p in v['points']]
        random.shuffle(pixels)
        del pixels[int(len(pixels)*0.5):] # Trims list to save memory

        split_idx = int(len(pixels)*0.8)
        train_images.extend(pixels[0:split_idx])
        train_labels += [v['id']] * split_idx
        test_images.extend(pixels[split_idx:])
        test_labels += [v['id']] * (len(pixels)-split_idx)

        # Save memory by deleting unused lists
        del v['points'][:]
        del pixels

    print('{} training images, {} testing images'.format(len(train_images), len(test_images)))
    train_images = np.array(train_images)
    train_labels = keras.utils.to_categorical(train_labels, NUM_WORDS)
    test_images = np.array(test_images)
    test_labels = keras.utils.to_categorical(test_labels, NUM_WORDS)

    # Reshape data to fit image channel
    train_images = train_images.reshape(train_images.shape[0], HEIGHT, WIDTH, 1)
    test_images = test_images.reshape(test_images.shape[0], HEIGHT, WIDTH, 1)

    # Build model
    print('building')
    model = cnn_model_words.build_model()

    print('fitting')
    model.fit(
        train_images,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(test_images, test_labels))

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    # Save model
    modelname = 'model-{}word.h5'.format(NUM_WORDS)
    model.save(modelname)
