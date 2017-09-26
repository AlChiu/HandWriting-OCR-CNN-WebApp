import cv2
import numpy as np
from matplotlib import pyplot as plt

# Parameters
MAX_AREA = 150
MIN_AREA = 10

# Read in the image
IMAGE = cv2.imread('gray.jpg')
cv2.imshow('image', IMAGE)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the Image into a grayscale image
IMAGE_GRAY = cv2.cvtColor(IMAGE, cv2.COLOR_RGB2GRAY)
cv2.imshow('image', IMAGE_GRAY)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Thresholds
ret, IMAGE_THRESH = cv2.threshold(IMAGE_GRAY, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('image', IMAGE_THRESH)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Keep only small componenets but not too small
COMPONENT = cv2.connectedComponentsWithStats(IMAGE_THRESH)

labels = COMPONENT[1]
labelStats = COMPONENT[2]
labelAreas = labelStats[:,4]

for compLabel in range(1,COMPONENT[0],1):
    if labelAreas[compLabel] >  MAX_AREA or labelAreas[compLabel] < MIN_AREA:
        labels[labels==compLabel] = 0

labels[labels>0] = 1

# Do dilation
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
IdilateText = cv2.morphologyEx(labels.astype(np.uint8),cv2.MORPH_DILATE,se)

# Find connected component again
COMPONENT = cv2.connectedComponentsWithStats(IdilateText)

# Draw a rectangle around the text
labels = COMPONENT[1]
labelStats = COMPONENT[2]

for compLabel in range(1,COMPONENT[0],1):
    # Draw the actual rectangle
    cv2.rectangle(IMAGE,(labelStats[compLabel,0],labelStats[compLabel,1]),(labelStats[compLabel,0]+labelStats[compLabel,2],labelStats[compLabel,1]+labelStats[compLabel,3]),(0,0,255),2)

    # Print the coordinates
    print("Word #%d" % compLabel)
    print("First Coordinate: (%d,%d)" % (labelStats[compLabel,0], labelStats[compLabel,1]))
    print("Second Coordinate: (%d,%d)" % (labelStats[compLabel,0]+labelStats[compLabel,2],labelStats[compLabel,1]+labelStats[compLabel,3]))

    # Create the crop image for the word
    crop_image = IMAGE[labelStats[compLabel,1]:labelStats[compLabel,1]+labelStats[compLabel,3], labelStats[compLabel,0]:labelStats[compLabel,0]+labelStats[compLabel,2]]
    cv2.imshow('cropped', crop_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the cropped image
    filename = "word%d.jpg" % compLabel
    cv2.imwrite(filename, crop_image)

cv2.imshow('image', IMAGE)
cv2.waitKey(0)
cv2.destroyAllWindows()   