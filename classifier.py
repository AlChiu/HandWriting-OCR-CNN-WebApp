import glob
import os
import re
from dataload import WordClassifier

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def classify():
  classifier = WordClassifier(modelPath='NeuralNet/model-1000word.h5')

  words = []
  string = []

  for image in sorted(glob.iglob('word_*.png'), key=numericalSort):
    words.append(classifier.classify_image(image))
    os.remove(image)

  return words
