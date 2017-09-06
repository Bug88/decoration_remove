__author__ = 'liuzhen'

# -*- coding: utf-8 -*-

# the main pipeline for test filtering
#import extract_sift
import pickle
import index
import matching
import rejection
import const_params
import sys
sys.path.append(const_params.__faiss_lib_path__)
import faiss
import numpy as np
import cv2
import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

print('=============extract features===============')

#if not const_params.__DEBUG__:
#    import extract_sift

print('=================indexing====================')

# load the database
fp = open('../database.pkl', 'r')
feats, pos, imgID, filenames, featNum = pickle.load(fp)
fp.close()

pos_database = pos
imgID_database = imgID

#index_, pos_np, imgID_np = index.indexing(feats, pos, imgID)

#faiss.write_index(index_, '../index_.faiss')
#fp = open('../index.pkl', 'w')
#pickle.dump([pos_np, imgID_np], fp)
#fp.close()

print('===================query======================')

img2 = cv2.imread('../examples_image/test_1.jpg', 0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp2, des2 = sift.detectAndCompute(img2, None)

# store all the good matches as per Lowe's ratio test.
pt = np.float32([kp2[m].pt for m in range(len(kp2))]).reshape(-1, 1, 2)
des = des2.tolist()

print('==================matching====================')

# load the database
fp = open('../database.pkl', 'r')
feats, pos, imgID, filenames, featNum = pickle.load(fp)
fp.close()

index_ = faiss.read_index('../index_.faiss')
fp = open('../index.pkl', 'r')
pos_np, imgID_np = pickle.load(fp)
fp.close()

imgNum = 500000

final_id, final_sim, final_mat = matching.query(des, pt, index_, pos_np, imgID_np, imgNum)

final_id_d = []
for i in range(len(final_id)):
    final_id_d.append((final_id[i], final_sim[i]))

final_id_1 = sorted(final_id_d, key=lambda x: x[1], reverse=True)

matched_imgs = []
for i in range(len(final_id)):
    matched_imgs.append(filenames[final_id_1[i][0]])
    print matched_imgs[i], final_id_1[i][1], featNum[final_id_1[i][0]]

# reverse the final matches
r_final_mat = []
for i in range(len(final_id)):
    t = np.zeros(shape=final_mat[i].shape)
    for j in range(final_mat[i].shape[0]):
        t[j, [2, 3]] = final_mat[i][j, [0, 1]]
        t[j, [0, 1]] = final_mat[i][j, [2, 3]]
    #r_final_mat.append(final_mat)
    r_final_mat.append(t)

print('final match img list: {0}'.format(final_id))

print('=====================rejection==========================')

# src database; dst query
fp = open('../data_all.pkl', 'r')
imgShape, src_pts, dst_pts, matches, img2 = pickle.load(fp)
fp.close()

imgShape = img2.shape
dst_pts = pt
imgID_database = np.asarray(imgID_database)
pos_database = np.asarray(pos_database)

label_map_list = []

for i in range(len(final_id)):
    c_id = final_id[i]
    matches = r_final_mat[i]
    #src_pts = matches[:, [0, 1]]

    src_pts = pos_database[imgID_database == c_id]

    src_pt_np = np.zeros(shape=(src_pts.shape[0], 1, 2))
    dst_pt_np = np.zeros(shape=(dst_pts.shape[0], 2))

    for i in range(src_pts.shape[0]):
        src_pt_np[i][0][0] = src_pts[i][0]
        src_pt_np[i][0][1] = src_pts[i][1]

        dst_pt_np[i][0] = dst_pts[i][0][0]
        dst_pt_np[i][1] = dst_pts[i][0][1]

    label, label_map = rejection.points_rejection(imgShape, src_pt_np, dst_pts, matches)

    label_map_list.append(label_map)

from PIL import Image

im = Image.open("../examples_image/test_1.jpg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

im = np.array(im, dtype=np.uint8)

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(im)

for j in range(len(label_map_list)):
    label_map = label_map_list[j]
    if label_map is None:
        continue
    for i in range(len(label_map)):
        x = label_map[i][0]
        y = label_map[i][1]

        rect = patches.Rectangle((x*const_params.__grid_width__, y*const_params.__grid_width__),
                         const_params.__grid_width__, const_params.__grid_width__, linewidth=1,
                         edgecolor='b', facecolor='none')

        ax.add_patch(rect)

plt.show()

if label is not None:
    print('==================')
    print('the deleted points number is {0}'.format(sum(label[:, 0].tolist())))
    print('==================')

