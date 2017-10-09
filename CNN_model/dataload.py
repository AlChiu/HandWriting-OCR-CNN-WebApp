import numpy as np
import os
import fnmatch
import re
from PIL import Image
import pickle

HEIGHT = 32
WIDTH = 100

def load_data(image_directory, words):
	regexp = '[a-zA-Z]+'
	word_data = {}
	datafile = 'dataset.p'

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
							img = convert_to_pixel_array(image_path)
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

	global NUM_CLASSES
	NUM_CLASSES = len(word_data)
	return word_data

def convert_to_pixel_array(image_path):
	pixels = []

	im = Image.open(image_path, 'r').resize((WIDTH, HEIGHT), Image.BICUBIC).convert('L')
	pixels = list(im.getdata())

	# Normalize and zero center pixel data
	std_dev = np.std(pixels)
	img_mean = np.mean(pixels)

	pixels = [(pixels[offset:offset+WIDTH]-img_mean)/std_dev for offset in range(0, WIDTH*HEIGHT, WIDTH)]
	pixels = np.array(pixels).astype(np.float32)
	
	return pixels
