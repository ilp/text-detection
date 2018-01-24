#!/usr/bin/python

import cv2
import sys
import math
import os
import time
import numpy as np
from matplotlib import pyplot as plt
from StrokeWidthTransform import swt

t0 = time.clock()

diagnostics = True

def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    
    return detected_edges

def detecting_edges(filepath):
    gray_img = cv2.imread(filepath, 0)
    #gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 175, 320, apertureSize=3)
    sobel_x = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=-1)
    sobel_y = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=-1)
    theta = np.arctan2(sobel_x, sobel_y)

    return (edges, sobel_x, sobel_y, theta)

def _swt(theta, edges, sobelx64f, sobely64f):
    # create empty image, initialized to infinity
    swt = np.empty(theta.shape)
    swt[:] = np.Infinity
    rays = []

    print time.clock() - t0

    # now iterate over pixels in image, checking Canny to see if we're on an edge.
    # if we are, follow a normal a ray to either the next edge or image border
    # edgesSparse = scipy.sparse.coo_matrix(edges)
    step_x_g = -1 * sobelx64f
    step_y_g = -1 * sobely64f
    mag_g = np.sqrt( step_x_g * step_x_g + step_y_g * step_y_g )
    grad_x_g = step_x_g / mag_g
    grad_y_g = step_y_g / mag_g

    for x in xrange(edges.shape[1]):
        for y in xrange(edges.shape[0]):
            if edges[y, x] > 0:
                step_x = step_x_g[y, x]
                step_y = step_y_g[y, x]
                mag = mag_g[y, x]
                grad_x = grad_x_g[y, x]
                grad_y = grad_y_g[y, x]
                ray = []
                ray.append((x, y))
                prev_x, prev_y, i = x, y, 0
                while True:
                    i += 1
                    cur_x = math.floor(x + grad_x * i)
                    cur_y = math.floor(y + grad_y * i)

                    if cur_x != prev_x or cur_y != prev_y:
                        # we have moved to the next pixel!
                        try:
                            if edges[cur_y, cur_x] > 0:
                                # found edge,
                                ray.append((cur_x, cur_y))
                                theta_point = theta[y, x]
                                alpha = theta[cur_y, cur_x]
                                if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                    thickness = math.sqrt( (cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y) )
                                    for (rp_x, rp_y) in ray:
                                        swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                    rays.append(ray)
                                break
                            # this is positioned at end to ensure we don't add a point beyond image boundary
                            ray.append((cur_x, cur_y))
                        except IndexError:
                            # reached image boundary
                            break
                        prev_x = cur_x
                        prev_y = cur_y

    for ray in rays:
        median = np.median([swt[y, x] for (x, y) in ray])
        for (x, y) in ray:
            swt[y, x] = min(median, swt[y, x])
    if diagnostics:
        cv2.imwrite('swt.jpg', swt * 100)

    return rays

filepath = sys.argv[1]

canny, sobelx, sobely, theta = detecting_edges(filepath)
#swt = _swt(theta, canny, sobelx, sobely)
cv2.imwrite("canny.jpg", canny)
cv2.imwrite("sobelx.jpg", sobelx)
cv2.imwrite("sobely.jpg", sobely)
cv2.imwrite("theta.jpg", theta)
#cv2.imwrite("swt.jpg", swt)


lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread(filepath)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite("canny2.jpg", CannyThreshold(0))
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()