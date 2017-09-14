# HandWriting-OCR-CNN-WebApp
This website will allow users to upload images of handwritten text, and the server will return a string of text of what it thinks it says through a handwriting, optical character recognition, convolutional neural network.

## Components
This repository is split into three sections:

The frontend will house the website that users will interact with by uploading an image.

The backend will house the components for the server.

The model will house the files that describe our neural network model.

## Requirements
1) User shall be able to upload an image.
2) Image shall be sent to the server.
3) Image shall be preprocessed by grayscaling the image.
4) Image shall be sent through a detector to find regions of text.
5) Text filled regions shall be sent into a convolutional neural network (CNN).
6) The CNN shall return what it thinks the region says.
7) Webiste shall return the output of the CNN and display it for the user. 

## Tools
**_Python_** will be the main programming language we will use. We will use it for configuring our neural network and, through the **_Flask_** framework, be our main backend language.

