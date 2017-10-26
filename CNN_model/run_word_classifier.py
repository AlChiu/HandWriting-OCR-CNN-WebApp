from dataload import WordClassifier

classifier = WordClassifier(modelPath='model-1000word.h5')

images = [('earth1.jpg', 'earth'),
		('wordpics/word_36.png', 'clear'),
		('wordpics/word_37.png', 'contents'), # word not a part of the 1000 word set
		('wordpics/word_38.png', 'we'), # may fail, possibly because image is too tightly cropped
		('wordpics/word_39.png', 'have'),
		('wordpics/word_56.png', 'shall'),
		('wordpics/word_52.png', 'the')]

for image in images:
	print('Actual: {}'.format(image[1]))
	print(classifier.classify_image(image[0]))
