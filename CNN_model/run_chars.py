import os
import time
import load_chars
import cnn_model_chars
import keras
import random
import numpy as np
import string
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    HEIGHT = 32
    WIDTH = 32
    BATCH_SIZE = 200
    EPOCHS = 100

    # Load data
    loader = load_chars.Dataloader(HEIGHT, WIDTH)
    #char_data = loader.load_all()
    char_data = loader.load_char74k()
    nist_data = loader.load_nist()

    # Split into train and test sets
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for k, v in char_data.items():
        if k in string.ascii_lowercase:
            random.seed(100)
            # Shuffle data
            pixels = [p['pixel_array'] for p in char_data[k]['points']]
            random.shuffle(pixels)

            # Trim to use only a fraction of the images
            #pixels = pixels[:int(len(pixels)*0.01)]
            #pixels = pixels[:200]

            split_idx = int(len(pixels)*0.8)
            train_images.extend(pixels[0:split_idx])
            train_labels += [char_data[k]['id']] * split_idx
            test_images.extend(pixels[split_idx:])
            test_labels += [char_data[k]['id']] * (len(pixels)-split_idx)

    print('{} training images, {} testing images'.format(len(train_images), len(test_images)))
    train_images = np.array(train_images)
    uniq = list(set(train_labels))
    # Must convert ordinal id of char to int
    # TODO: move this conversion to load_chars
    train_labels = [uniq.index(char) for char in train_labels]
    train_labels = keras.utils.to_categorical(train_labels, 62)
    test_images = np.array(test_images)
    test_labels = [uniq.index(char) for char in test_labels]
    test_labels = keras.utils.to_categorical(test_labels, 62)

    # Reshape data to fit image channel
    train_images = train_images.reshape(train_images.shape[0], HEIGHT, WIDTH, 1)
    test_images = test_images.reshape(test_images.shape[0], HEIGHT, WIDTH, 1)

    # Build model
    classifier = cnn_model_chars.CharacterClassifier(HEIGHT, WIDTH)

    # Fit model
    classifier.model.fit(
            train_images,
            train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(test_images, test_labels))

    score = classifier.model.evaluate(train_images, train_labels, verbose=0)
    print('Train loss: ', score[0])
    print('Train accuracy: ', score[1])

    score = classifier.model.evaluate(test_images, test_labels, verbose=0)
    print('\nTest loss: ', score[0])
    print('Test accuracy: ', score[1])

    # Per character accuracy
    print('\nPer character (NIST):')
    for k, v in sorted(nist_data.items()):
        if v['id'] in uniq:
            print(k)
            pixels = [p['pixel_array'] for p in v['points']]
            pixels = np.array(pixels)
            pixels = pixels.reshape(pixels.shape[0], HEIGHT, WIDTH, 1)
            labels = [uniq.index(v['id'])] * len(pixels)
            labels = keras.utils.to_categorical(labels, 62)

            score = classifier.model.evaluate(pixels, labels, verbose=0)
            print('Accuracy: ', score[1])

    # Save model
    modelname = 'model-char74k-{}lowercase.h5'.format(len(uniq))
    classifier.model.save(modelname)
