# import the necessary packages
from PIL import Image
from pytesseract import pytesseract
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
  gray = cv2.GaussianBlur(gray, (5,5), 0)

# Blur if preprocess has bluring
elif args["preprocess"] == "blur":
  gray = cv2.medianBlur(gray, 3)

# Write the grayscale to temp file
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

pytesseract.run_tesseract(filename, 'output', lang=None, boxes=True, config="hocr")

# Load the image as a PIL image, OCR, then delete
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
