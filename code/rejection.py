__author__ = 'liuzhen'

# filter out some useless keypoints
# by an estimated homography
# based on the density of the mapped features

import cv2
import numpy as np
import const_params

__grid_width__ = const_params.__grid_width__
__thred__ = const_params.__thred__

def points_rejection(dst_img_shape, src_img_points, dst_img_points, matched_points):
    # grid filtering
    src_pts = matched_points[:, [0, 1]]
    dst_pts = matched_points[:, [2, 3]]

    # learn the transformation mask and perform the points projection
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        print('no inner points')
        return None, None
    else:
        projected_points = cv2.perspectiveTransform(src_img_points, M)

    # reject points with grid projection
    h, w = dst_img_shape
    w_n = int(w / __grid_width__)
    h_n = int(h / __grid_width__)

    grid_map = np.zeros(shape=(w_n, h_n))

    #src_img_points = projected_points
    if const_params.__NotMat__:
        src_img_points = np.zeros(shape=(matched_points.shape[0], 1, 2))

        for i in range(matched_points.shape[0]):
            src_img_points[i][0][0] = src_pts[i][0]
            src_img_points[i][0][1] = src_pts[i][1]
    else:
        src_img_points = projected_points

    for i in range(src_img_points.shape[0]):
        c_x = int(src_img_points[i][0][0] / __grid_width__)
        c_y = int(src_img_points[i][0][1] / __grid_width__)

        if c_x >= w_n:
            continue
        if c_y >= h_n:
            continue

        if c_x < 0:
            continue
        if c_y < 0:
            continue

        grid_map[c_x][c_y] = grid_map[c_x][c_y] + 1

    label_map = []
    label = np.zeros(shape=(dst_img_points.shape[0], 1))
    for i in range(dst_img_points.shape[0]):
        c_x = int(dst_img_points[i][0][0] / __grid_width__)
        c_y = int(dst_img_points[i][0][1] / __grid_width__)

        #print(c_x)
        #print(c_y)

        if c_x >= w_n:
            continue
        if c_y >= h_n:
            continue
        if c_x < 0:
            continue
        if c_y < 0:
            continue

        if grid_map[c_x][c_y] > __thred__:
            label[i][0] = 1
            label_map.append([c_x, c_y])

    return label, label_map

if __name__ == '__main__':
    import pickle

    print('testing........')

    fp = open('../data_all.pkl', 'r')
    imgShape, src_pts, dst_pts, matches, img2 = pickle.load(fp)
    fp.close()

    src_pt_np = np.zeros(shape=(src_pts.shape[0], 2))
    dst_pt_np = np.zeros(shape=(dst_pts.shape[0], 2))

    for i in range(src_pts.shape[0]):
        src_pt_np[i][0] = src_pts[i][0][0]
        src_pt_np[i][1] = src_pts[i][0][1]

        dst_pt_np[i][0] = dst_pts[i][0][0]
        dst_pt_np[i][1] = dst_pts[i][0][1]

    #src_pts = src_pt_np
    #dst_pts = dst_pt_np
    label, label_map = points_rejection(imgShape, src_pts, dst_pts, matches)

    from PIL import Image, ImageDraw


    im = Image.open("../examples_image/test_image5.tiff")

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np

    im = np.array(im, dtype=np.uint8)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    for i in range(len(label_map)):
        x = label_map[i][0]
        y = label_map[i][1]

        #print x, y
        # Create a Rectangle patch
        rect = patches.Rectangle((x*__grid_width__, y*__grid_width__), __grid_width__, __grid_width__, linewidth=1,
                                 edgecolor='b', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

    print('==================')
    print('the deleted points number is {0}'.format(sum(label[:, 0].tolist())))
    print('==================')