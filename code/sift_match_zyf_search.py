#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pickle

i = 0

for filename in [ ls for ls in os.listdir(r"../chinese_sub") if ls.split('.')[1] == 'png' ]:

    MIN_MATCH_COUNT = 0

    img1 = cv2.imread('../chinese_sub/' + filename, 0)          # queryImage
    img2 = cv2.imread('../examples_image/test.jpg', 0) # trainImage

# Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1,des2, k=2)

# store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    if len(good) / float(len(kp1)) > MIN_MATCH_COUNT:

        print(len(good) / float(len(kp1)))

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if ( mask == None):
            print "not inner pot"
        else:
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            figure = plt.figure()
            i = i + 1
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

            plt.imshow(img3, 'gray'), plt.show()
            figure.savefig('../save_match_image/' + filename)

            fp = open('../data.pkl', 'w')
            pickle.dump([src_pts, dst_pts], fp)
            fp.close()

            src = src_pts
            dst = dst_pts
            pt = np.zeros(shape=(src.shape[0], 4))

            for i in range(src.shape[0]):
                pt[i][0] = src[i][0][0]
                pt[i][1] = src[i][0][1]
                pt[i][2] = dst[i][0][0]
                pt[i][3] = dst[i][0][1]

            src_pts = np.float32([kp1[m].pt for m in range(len(kp1))]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m].pt for m in range(len(kp2))]).reshape(-1, 1, 2)

            fp = open('../data_all.pkl', 'w')
            pickle.dump([img2.shape, src_pts, dst_pts, pt, img2], fp)
            fp.close()

            pt2 = np.float32([kp2[m].pt for m in range(len(kp2))]).reshape(-1, 1, 2)

            fp = open('../query.pkl', 'w')

            pickle.dump([des2.tolist(), pt2], fp)

            fp.close()

    else:
        print "Not enough matches are found - %d/%f" % (len(good), MIN_MATCH_COUNT)
        matchesMask = None