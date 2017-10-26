import os
import time
import load_chars
import keras
import random
import numpy as np

if __name__ == '__main__':
    HEIGHT = 32
    WIDTH = 32

    # Load data
    loader = load_chars.Dataloader(HEIGHT, WIDTH)
    char_data = loader.load_all()

    # Split into train and test sets
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    random.seed(100)
    for k, v in char_data.items():
        # Shuffle data
        pixels = [p['pixel_array'] for p in char_data[k]['points']]
        random.shuffle(pixels)

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
