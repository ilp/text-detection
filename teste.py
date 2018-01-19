import cv2
import sys
import numpy as np


def detecting_edges(filepath):
    gray_img = cv2.imread(filepath, 0)
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 175, 320, apertureSize=3)
    sobel_x = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=-1)
    sobel_y = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=-1)
    theta = np.arctan2(sobel_x, sobel_y)

    return (edges, sobel_x, sobel_y, theta)

filepath = sys.argv[1]

canny, sobelx, sobely, theta = detecting_edges(filepath)

image = cv2.imread("img1.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
cv2.imwrite("bordas.jpg", canny)
_, contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

# for each contour found, draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h>300 and w>300:
        continue

    # discard areas that are too small
    if h<40 or w<40:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)

# write original image with added contours to disk  
cv2.imwrite("contoured.jpg", image) 