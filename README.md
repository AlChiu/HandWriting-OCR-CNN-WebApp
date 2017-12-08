# Handwriting Detection and Recognition with Neural Networks
### Alexander Chiu, Andrew Quintanilla, Milan Ghori, Akshay Pawar, Chinmayee Tapaskar
 
## Summary
Our project consists of a model trained to convert the handwriting in images to text. The model will be trained on a desktop GPU to speed up the training. The model will then be saved and migrated to a server for use in a Flask web application. The application will be hosted on one of our laptops, but will be capable of being hosted on a service such as Heroku. Client-side, the page will upload images to the server, then output the results of the server’s analysis of the image. By using Tesseract and OpenCV to detect distinct blocks of text, we can output the text for each block separately. Client-side functionality could potentially be expanded by capturing photos from the user’s webcam, or even by saving handwriting drawn on the page. 

## How to Use
To use the this end-to-end optical character recognition system, you must install the prerequisite libraries and packages for Python 3.6:

* OpenCV 

* Numpy

* Pillow

* Tesserocr

* Tesseract

* Tensorflow

* Keras

* Flask

Once those requirements are installed, you only need to go to the project directory and run the command,

`python app.py`

Running the above command will start the Flask web server, and you can now upload your images for translation.

## Detection and Segmentation
The detector is built with Tesseract and OpenCV. When the user uploads an image, the detector will preprocess the image to create a binarized image. The detector performs binarization using OpenCV’s implementation of the Otsu method. This algorithm will search the entire image for a threshold value that minimizes the intra-variance of the image. The image will be a black and white image that Tesseract will use to find words. After finding the words, the detector crops out the detected word and saves it locally. After the detected words are cropped out, the detector removes the binarized image since it is not needed anymore. During the entire detection process, the original input image is preserved. 

## Neural Network Classification
Our model consists of multiple neural network layers, with a classifier layer for output. It includes five CNN layers to extract features from the images and reduce dimensionality, which then feed into two dense layers. A final dense layer, with as many neurons as there are distinct words to classify from our word dataset, is used to output the final classification. It is written in Python, using libraries such as numpy, Tensorflow, OpenCV, Tesseract and Keras.

We used multiple strategies for training the model. The paper we had based our model on described a 10,000 word classifier trained on 9 million images. Due to computational limitations, we could only train a 5,000 word classifier using a fraction of the images. We also had to train in stages, doing initial training on half of the whole dataset, then training the model again using the other half. To prioritize which words of the 10,000 to use, we obtained a list of the 10,000 most common words according to Google, which came sorted, and used the first n, where n is how many words we wished our model to classify.

## Web Server
We used the Flask framework, allowing us to use Python for both the model and the web server, making it easier to import our pre-trained model into the application. We chose Flask over the Django framework due to Flask’s simplicity. As our application is currently just a simple demonstration of our model’s capabilities, it does not need Django’s numerous features for templating, routing, authentication, and database administration. 

## Client side
We have a basic web page whose function is to simply ask the user for images, upload those images to the server, and then display the results of the server’s analysis of those images back to the user. At the most basic level, we only need a control to upload an image, and then a text element to display the text found in the images. We can add more functionality with jQuery; however, we don’t want to spend too much time learning a full Javascript framework such as Backbone or Angular since we were under a time limit.

## Languages used
For the majority of the project, we used Python 3 as our primary language. Python 3 was used because it made the coding easy and it was also compatible with Tensorflow and Keras. For the frontend, we built an UI by using HTML, CSS for styling and Javascript for handling communication with the detector and the backend system.

## Python libraries
We used Numpy to access basic statistical functions and to use numpy arrays, and Pillow to convert the images to pixel intensity arrays. For the neural network, we used Tensorflow, with Keras on top to streamline writing the code. The detector uses OpenCV and Tesseract to process images. Flask was used to write the frontend.

## Classifier Pipeline
When a user uploads the image into the website, the backend will save that image into its own local storage. With that image, the detector will grayscale, binarize it, and create a temporary copy of the image for word detection. For every word detected, the detector will crop the word out and save it for classification. After the word detection is complete, the server will remove the temporary image. 

With the cropped word images, the classifier will create a list of word results. For each word image, the classifier will reshape the image and send it through its network for prediction. Each word classification will have a list of words and their resulting confidences. The classifier will extract the highest confidence and its respective word and append it to the word list. If the confidence level is below 50%, the system will return “(N/A)” as a the word since we know the word is not in the dictionary. When all of words have been classified, the system will return the list of words to the web front to be displayed as a sentence of classified words. 

## Future Work
Currently, this end-to-end system is a prototype that is only able to detect a subset of the English language. Below is a list of improvements that we can implement when given more time:

* Design and implement a custom detector that does not rely on Tesseract. This allows us to customize how we detect words and characters for the classifier.

* Train the neural network so that it does not classify by words but by characters. This allows the system to recognize and translate entire documents without relying on a dictionary of words. This reduces the complexity and training time for the network since the total size of the English alphabet is much smaller than the dictionary of ENglish words.

* Implement a neural network ruleset that predicts words when given combinations of characters. This may increase the complexity of the network, and conflict with the second above point. 

## References
* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* [A Beginner's Guide to Understanding Convolutional Neural Networks](https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
* [Nvidia Digits](https://developer.nvidia.com/digits)
* [Synckey Tensorflow OCR](https://github.com/synckey/tensorflow_lstm_ctc_ocr)
* [VGG Synthetic Word Dataset](http://www.robots.ox.ac.uk/~vgg/data/text)
* [Tensorflow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [Keras](https://keras.io/)
* [Tesseract](https://github.com/tesseract-ocr/tesseract)
