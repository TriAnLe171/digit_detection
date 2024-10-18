# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from contrast import increase_contrast
# load the image
image = cv2.imread("example1.png")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
# ratio = image.size / 500

image = imutils.resize(image, height=500)

image=increase_contrast(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

cv2.imshow('og',image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


## contours of the display
displayCnt = None
# # loop over the contours
for c in cnts:
	# approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
        displayCnt = approx
    # break
    
# extract the display, apply a perspective transform
# to it
if displayCnt is not None:
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    thresh = cv2.threshold(warped, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh=thresh[15:thresh.shape[0]-15,15:thresh.shape[1]-15]
    cv2.imshow('thresh',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
     print("No display contours found")

     
# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []

# loop over the digit area candidates
xy=[]
for dgit in cnts:
	# compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(dgit)
	# if the contour is sufficiently large, it must be a digit
    if w >= 10 and h >= 10:
        xy.append((x,y,w,h))
        digitCnts.append(dgit)

print(len(cnts))
print(len(digitCnts))
print(len(xy))
        

#Draw bounding boxes around the digits
if len(digitCnts)>0:
    
    for (x, y, w, h) in xy:
        cv2.rectangle(output, (x+15, y+15), (x + w+15, y + h+15), (0, 255, 0), 2)

else:
    print('NO')

cv2.imshow('Input',image)
cv2.imshow('Output',output)
cv2.waitKey(0)
cv2.destroyAllWindows