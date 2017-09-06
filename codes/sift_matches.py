__author__ = 'liuzhen'

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('/Users/liuzhen-mac/Desktop/image1.tif',0) #queryImage
img2 = cv2.imread('/Users/liuzhen-mac/Desktop/font_test_2.png',0) #trainImage

# Initiate SIFT detector
#sift = cv2.xfeatures2d.SURF_create()

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
figure = plt.figure()
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, img2, flags=2)
print('the matched features: ', len(good))
plt.title('the number of matches:{0}'.format(len(good)))
plt.imshow(img3),plt.show()

figure.savefig('/Users/liuzhen-mac/Desktop/matches.jpg')