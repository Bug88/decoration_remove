__author__ = 'liuzhen'

from PIL import Image
import cv2
import numpy as np

im = cv2.imread('/Users/liuzhen-mac/Desktop/image1.tif')
#gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(gray, None)
#img = cv2.drawKeypoints(gray, kp, im)
#Image._show(img)

cv2.imshow('original',im)
#cv2.waitKey()

#im_lowers = cv2.pyrDown(im)
#cv2.imshow('im_lowers',im_lowers)

s = cv2.SIFT()
#s = cv2.SURF()
keypoints = s.detect(im)

c = 0
for k in keypoints:
    cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),1,(0,255,0),-1)
    #cv2.circle(im,(int(k.pt[0]),int(k.pt[1])),int(k.size),(0,255,0),2)
    c = c + 1

print('the number of local features are: ', c)

cv2.imshow('SIFT_features',im)
cv2.waitKey()
cv2.destroyAllWindows()