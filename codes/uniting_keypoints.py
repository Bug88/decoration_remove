__author__ = 'liuzhen'

# uniting keypoints for duplicate searching

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

img1 = cv2.imread('/Users/liuzhen-mac/Desktop/image1.tif',0) # queryImage
img2 = cv2.imread('/Users/liuzhen-mac/Desktop/image5.tif',0) # trainImage

# Initiate SIFT detector
sift =  cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

points_1 = np.array([kp1[idx].pt for idx in range(len(kp1))]) #.reshape(-1, 1, 2)
pints_2 = np.array([kp2[idx].pt for idx in range(len(kp2))])

km1 = KMeans(n_clusters=5, random_state=0).fit(points_1)
km2 = KMeans(n_clusters=5, random_state=0).fit(pints_2)

print(km1.labels_[:10])
print(km2.labels_[:10])

for q in range(10):
    keypoints = points_1[km1.labels_ == q]

    print(keypoints.shape[0])

    img1 = cv2.imread('/Users/liuzhen-mac/Desktop/image1.tif', 0)  # queryImage

    c = 0
    for k in range(keypoints.shape[0]):
        tmp = keypoints[k]
        cv2.circle(img1,(int(tmp[0]),int(tmp[1])), 3,(255, 0, 0),-1)
        c = c + 1

    print('the number of local features are: ', c)

    cv2.imshow('SIFT_features',img1)
    cv2.waitKey()
    cv2.destroyAllWindows()

print('c')

