# import the necessary packages
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
import numpy
import argparse
import csv
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
  help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh", help="type of preprocessing to be done")
args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image
if args["preprocess"] == "thresh":
  gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# Blur if preprocess has bluring
elif args["preprocess"] == "blur":
  gray = cv2.medianBlur(gray, 3)

# Write the grayscale to temp file
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

with PyTessBaseAPI() as api:
  # Send the new grayscale image into Tesseract
  api.SetImageFile(filename)

  # OCR the grayscale image for comparison
  translation = api.GetUTF8Text()

  # We will segment the grayscale by words
  boxes = api.GetComponentImages(RIL.WORD, True)

  # With each bouding box for the words
  for i, (im, box, _, _) in enumerate(boxes):
    # Grab the coordinates of the bounding box
    api.SetRectangle(box['x'], box['y'], box['w'], box['h'])

    # What is the word in the bounding box
    ocrResult = api.GetUTF8Text()

    # Get the confidence of its translation
    conf = api.MeanTextConf()

    # Turn the returned bounding box coordinates into an array of coordinates
    coord = list(box.values())

    # Load grayscale for cropping
    cropper = Image.open(filename)

    # Cropped image is saved into new variable
    crop_image = cropper.crop(
      (coord[0],
      coord[1],
      coord[0]+coord[2],
      coord[1]+coord[3])
    )

    # Convert the new image into a numpy array
    cropped = numpy.array(crop_image) 

    # Create the new file name for the word
    word_file = "word_" + str(i) + ".png"

    # Have OpenCV save the cropped image into the new file
    cv2.imwrite(word_file, cropped) 

# Remove the grayscale image
os.remove(filename)

# Print out Tessearct's translation for comparison
print(translation)
