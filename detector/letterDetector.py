# import packages
# import csv
import os
import argparse
import cv2
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
# from matplotlib import pyplot as plt

# Construct the image argument parser for testing purposes
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to imput image to be processed")
args = vars(ap.parse_args())

# load the example image and convert it to grascale
inputImage = cv2.imread(args["image"])
inputResize = cv2.resize(inputImage, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
inputImageBlur = cv2.bilateralFilter(inputResize, 13, 75, 75)
grayInputImage = cv2.cvtColor(inputImageBlur, cv2.COLOR_BGR2GRAY)

# apply thresholds to the image
ret, otsu = cv2.threshold(grayInputImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# thresheld_adaptive_mean = cv2.adaptiveThreshold(grayInputImage, 255,
#  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 2)
# thresheld_adaptive_gauss = cv2.adaptiveThreshold(grayInputImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

# write thresheld image to a new, temporary file
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, otsu)

# display new, temporary file
# titles = ['Original Image','Otsu', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [grayInputImage, otsu, thresheld_adaptive_mean, thresheld_adaptive_gauss]
# for i in range(4):
#    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
# plt.show()

# perform OCR and word detection using Tesseract engine
with PyTessBaseAPI() as api:
    api.SetImageFile(filename)
    WORD_BOXES = api.GetComponentImages(RIL.WORD, True)
    print('Found {} word images'.format(len(WORD_BOXES)))
    for i, (im, box, _, _) in enumerate(WORD_BOXES):
        api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
        # ocrResult = api.GetUTF8Text()
        # conf = api.MeanTextConf()
        # print("Box[{0}]: x={x}, y={y}, w={w}, h={h}, confidence: {1}, text:{2}".format(i, conf, ocrResult, **box))
        coord = list(box.values())
        cropper = Image.open(filename)
        crop_image = cropper.crop((coord[0], coord[1], coord[0]+coord[2], coord[1]+coord[3]))
        cropped = np.array(crop_image)
        constant = cv2.copyMakeBorder(cropped, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
        word_file = "word_" + str(i) + ".png"
        cv2.imwrite(word_file, constant)

# Use openCV findContours capability to extract individual characters
KERNEL_1 = np.ones((3, 3), np.uint8)
KERNEL_2 = np.ones((5, 5), np.uint8)
KERNEL_3 = np.ones((7, 7), np.uint8)
KERNEL_4 = np.ones((9, 9), np.uint8)

for j, val in enumerate(WORD_BOXES):
    word_file2 = "word_" + str(j) + ".png"
    word = cv2.imread(word_file2)
    wordGray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
    # wordGrayBlur = cv2.bilateralFilter(wordGray, 9, 35, 35)
    # dilate = cv2.dilate(wordGrayBlur, kernel1, iterations=2)
    # erode = cv2.erode(dilate, kernel3, iterations=1)
    im2, contours, hierarchy = cv2.findContours(wordGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    new_box = []

    for k, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        CENTER_X = x + w / 2
        CENTER_Y = y + h / 2
        center_coord = (CENTER_X, CENTER_Y, x, y, w, h)
        centers.append(center_coord)

    cent = np.asarray(centers)
    # NEED TO SORT CONTOURS LEFT TO RIGHT
    sorted_cent = cent[cent[:, 2].argsort()]
    print(sorted_cent)

    for i in range(len(sorted_cent) - 1):
        # print(cent[0, 0])
        diff_X = abs(sorted_cent[i, 0] - sorted_cent[i+1, 0])
        diff_Y = abs(sorted_cent[i, 1] - sorted_cent[i+1, 1])
        if diff_X < 10:
            if diff_Y < 50:
                print("Merge")
            else:
                print("diff_Y= ", diff_Y)
        else:
            print("diff_X= ", diff_X)

    #os.remove(word_file2)
        #     if abs(cent[l, 1] - cent[m, 1]) < 30:
        #         # FIND THE BOTTOM LEFT CORNER OF LOWEST BOUNDING BOX
        #         # MINIMUM OF THE TWO X's and MINIMUM OF THE TWO Y's
        #         MIN_X = min(cent[l, 2], cent[m, 2])
        #         MIN_Y = min(cent[l, 3], cent[m, 3])
        #         # FIND THE LARGEST HEIGHT AND WIDTH OF NEW BOX
        #         MAX_HEIGHT = max(cent[l, 5], cent[m, 5])
        #         MAX_WIDTH = max(cent[l, 4], cent[m, 4])
        #         # CREATE NEW BOUNDING RECTANGLE BASED ON THESE NUMBERS
        #         # cv2.rectangle(word, (MIN_X, MIN_Y), (MIN_X+LARGEST_WIDTH, MIN_Y+LARGEST_HEIGHT), (0, 255, 0), 2)
        #         new_rect = (MIN_X, MIN_Y, MAX_WIDTH, MAX_HEIGHT)
        #         new_box.append(new_rect)
        #     else:
        #         continue
        # else:
        #     new_rect = (cent[l, 2], cent[l, 3], cent[l, 4], cent[l, 5])
        #     new_box.append(new_rect)
    # print(new_rect)



        # if h < 5 and w < 5:
        #    continue

        # cv2.rectangle(word, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.drawContours(word, contours, -1, (0,255,0), 3)
        # cv2.imshow("contours", word)
        # cv2.waitKey(0)

    # SIMPLE BLOB DETECTION CODE
    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(word)

    # im_with_keypoints = cv2.drawKeypoints(word, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv2.imshow("keypoints", im_with_keypoints)
    # cv2.waitKey(0)

    # EROSION AND DILATION OF WORD IMAGES
    # kernel1 = np.ones((5,5), np.uint8)
    # kernel2 = np.ones((3,3), np.uint8)
    # Erode the word images so that letters are more distinct
#    word_file2 = "word_" + str(j) + ".png"
#    word = cv2.imread(word_file2)
#    res = cv2.resize(word,None,fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
#    closing = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel1)
#    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
#    erosion = cv2.erode(opening, kernel1, iterations=1)
#    laplacian = cv2.Laplacian(erosion, cv2.CV_64F)
#    cv2.imwrite(word_file2, erosion)

    # MSER DETECTION CODE
#    word_file3 = cv2.imread(word_file2)
#    vis = word_file3.copy()
#    mser = cv2.MSER_create()
#    regions = mser.detectRegions(word_file3)
#    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
#    cv2.polylines(vis, hulls, 1, (0,255,0))

#    cv2.namedWindow('img', 0)
#    cv2.imshow('img', vis)
#    while(cv2.waitKey()!=ord('q')):
#      continue
#    cv2.destroyAllWindows()

    # path = "words_" + str(i)
    # word_path = path+"/"+word_file
    # os.mkdir(path)
    # os.rename(word_file, word_path)

# delete the new, temporary file
os.remove(filename)
