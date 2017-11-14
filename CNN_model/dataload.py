import numpy as np
import os
import fnmatch
import re
from PIL import Image
import pickle
import configparser
import keras

HEIGHT = 32
WIDTH = 100

config = configparser.ConfigParser()
config.read('settings.ini')
# Path to image dataset
image_directory = config['paths']['IMAGE_DIRECTORY']

# Name of converted dataset file
word_id_file = 'word_ids.p'


def load_data(num_words=10000):
    # Path to text file containing words to learn, one word per line
    # Should be sorted in descending order of word priority
    words = set(open('google-10000-english-usa.txt').read().split()[:num_words])

    datafile = 'dataset-{}-words.p'.format(num_words)

    regexp = '[a-zA-Z]+'
    word_data = {}

    try:
        # Attempts to load data from pickle
        word_data = pickle.load(open(datafile, "rb"))
        print('data loaded from {}'.format(datafile))
    except:
        for root, dirnames, filenames in os.walk(image_directory):
            for filename in fnmatch.filter(filenames, '*.jpg'):
                fname = os.path.splitext(filename)[0]
                m = re.search(regexp, fname)
                if(m):
                    word = m.group(0).lower()
                    if(word in words):
                        image_path = os.path.join(root, filename)
                        try:
                            img = convert_to_pixel_array(image_path, WIDTH, HEIGHT)
                            if (word not in word_data):
                                word_data[word] = {}
                                word_data[word]['id'] = len(word_data) - 1
                                word_data[word]['points'] = []
                            point = {}
                            point['filename'] = filename
                            point['image_path'] = image_path
                            point['pixel_array'] = img
                            word_data[word]['points'].append(point)
                        except:
                            print('image not valid: ', filename)
        # Pickle data so this process doesn't need to be repeated
        pickle.dump(word_data, open(datafile, "wb"))
        print('data saved to {}'.format(datafile))
        word_ids = word_data.copy()
        for word, data in words_ids:
            data.pop('points')
        pickle.dump(word_ids, open(word_id_file, 'wb'))

    global NUM_CLASSES
    NUM_CLASSES = len(word_data)
    return word_data

def convert_to_pixel_array(image_path, width, height, trim_whitespace=False):
    pixels = []

    im = Image.open(image_path, 'r').convert('L')
    if (trim_whitespace):
        im_width, im_height = im.size
        pixels = list(im.getdata())
        pixels = [pixels[offset:offset+im_width] for offset in range(0, im_width*im_height, im_width)]
        top = find_topmost_nonwhite_pixel(pixels, im_height, im_width)
        left = find_leftmost_nonwhite_pixel(pixels, im_height, im_width)
        bottom = find_bottommost_nonwhite_pixel(pixels, im_height, im_width)
        right = find_rightmost_nonwhite_pixel(pixels, im_height, im_width)
        trim = min([top,left,im_height-bottom,im_width-right])
        top = trim
        left = trim
        bottom = im_height - trim
        right = im_width - trim
        im = im.crop((top,left,bottom,right))

    im = im.resize((width, height), Image.BICUBIC)
    pixels = list(im.getdata())

    # Normalize and zero center pixel data
    std_dev = np.std(pixels)
    img_mean = np.mean(pixels)

    pixels = [(pixels[offset:offset+width]-img_mean)/std_dev for offset in range(0, width*height, width)]
    pixels = np.array(pixels).astype(np.float32)

    return pixels

def find_topmost_nonwhite_pixel(pixels, height, width):
    for i in range(height):
        for j in range(width):
            if (pixels[i][j] != 0):
                return i
def find_leftmost_nonwhite_pixel(pixels, height, width):
    for j in range(width):
        for i in range(height):
            if (pixels[i][j] != 0):
                return j
def find_bottommost_nonwhite_pixel(pixels, height, width):
    for i in range(height-1,-1,-1):
        for j in range(width):
            if (pixels[i][j] != 0):
                return i
def find_rightmost_nonwhite_pixel(pixels, height, width):
    for j in range(width-1,-1,-1):
        for i in range(height):
            if (pixels[i][j] != 0):
                return j

class WordClassifier:
    def __init__(self, modelPath=None, model=None):
        if (model is not None):
            self.model = model
        elif (modelPath is not None):
            self.model = keras.models.load_model(modelPath)
        else:
            raise ValueError('either model or modelPath must be given')
        self.word_ids = pickle.load(open(word_id_file, 'rb'), encoding='latin1')

        def classify_image(self, image_path):
            try:
                image_pixels = convert_to_pixel_array(image_path, WIDTH, HEIGHT)
                image_pixels = np.array(image_pixels)
                inp = np.array([image_pixels])
                inp = inp.reshape(inp.shape[0], 32, 100, 1)

                outp = self.model.predict(inp)[0]
                outp = np.array(outp)

                top5_idx = outp.argsort()[-5:]

                top5_words = [(k, outp[v['id']]) for k, v in self.word_ids.items() if v['id'] in top5_idx]
                top5_words = sorted(top5_words, key=lambda x: x[1], reverse=True)

                return top5_words
            except FileNotFoundError:
                print('Image not found at path {}'.format(image_path))
