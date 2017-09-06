__author__ = 'liuzhen'
# this is a script
# -*- coding: utf-8 -*-

import sys
import const_params
sys.path.append(const_params.__faiss_lib_path__)
import os
import cv2
import pickle

# collecting sift features in the database
data_dir = const_params.__data_dir_path__
filenames = [file for file in os.listdir(data_dir)
             if file.split('.')[1] in const_params.__post__]

feats = []
pos = []
imgID = []

c = 0

featNum = []

for file in filenames:
    img = cv2.imread(data_dir+file, 0)

    sift = cv2.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(img, None)

    for i in range(len(kp)):
        feats.append(des[i])
        pos.append(kp[i].pt)
        imgID.append(c)

    featNum.append(len(kp))

    print('{0} image is processed!'.format(file))

    c = c + 1

fp = open('../database.pkl', 'w')
pickle.dump([feats, pos, imgID, filenames, featNum], fp)
fp.close()

