__author__ = 'liuzhen'

# -*- coding: utf-8 -*-

import numpy as np
import const_params
import verification
import sys
sys.path.append(const_params.__faiss_lib_path__)

# match the templates in the lib by inverted file

K = const_params.__nearest_NN__
#M = const_params.__max_match__

def query(qDes, qPts, index, pos_np, imgID_np, imgNum):

    # query the index file to get matched points
    score = [0] * imgNum   #np.zeros(shape=(imgNum, 1))
    index.nprobe = 5
    des = np.asarray(qDes).astype('float32')
    D, I = index.search(des, K)

    # compute the score for each image
    for i in range(qPts.shape[0]):
        for j in range(K):
            c_img = int(imgID_np[I[i, j], 0])
            score[c_img] = score[c_img] + 1

    st_score = sorted(score, reverse=True)
    # filter out the matched points
    selected_imgID = []
    matched_pts = []
    for i in range(imgNum):
        if score[i] > 0: # all matched images are selected
            selected_imgID.append(i)
            matched_pts.append([])

#    for i in range(M):
#        selected_imgID.append(score.index[st_score[i]])
#        matched_pts.append([])

    for i in range(qPts.shape[0]):
        for j in range(K):
            c_idx = I[i, j]
            c_imgID = int(imgID_np[c_idx, 0])

            if c_imgID not in selected_imgID:
                continue

            c_p = selected_imgID.index(c_imgID)
            matched_pts[c_p].append([qPts[i, 0, 0], qPts[i, 0, 1], pos_np[c_idx, 0], pos_np[c_idx, 1]])



    # global verification for the matches
    final_id = []
    final_sim = []
    final_matches = []
    for i in range(len(selected_imgID)):
        if len(matched_pts[i]) <= 1:
            continue

        if const_params.__verification__:
            c_m = verification.verification(np.asarray(matched_pts[i]))
        else:
            c_m =  np.asarray(matched_pts[i]) #verification.verification(np.asarray(matched_pts[i]))

        if c_m.shape[0] > const_params.__match_thred__:
            final_id.append(selected_imgID[i])
            final_sim.append(c_m.shape[0])
            final_matches.append(np.asarray(matched_pts[i]))

    return final_id, final_sim, final_matches

if __name__ == '__main__':
    import pickle
    import faiss
    import index as idx

    # load the database
    fp = open('../database.pkl', 'r')
    feats, pos, imgID, filenames, featNum = pickle.load(fp)
    fp.close()

    index_ = faiss.read_index('../index.faiss', True)

    #index_, pos_np, imgID_np = idx.index(feats, pos, imgID)

    fp = open('../query.pkl', 'r')
    des, pt = pickle.load(fp)
    fp.close()

    fp = open('../index.pkl', 'r')
    pos_np, imgID_np = pickle.load(fp)
    fp.close()

    imgNum = 500000

    final_id, final_sim = query(des, pt, index_, pos_np, imgID_np, imgNum)

    final_id_d = []
    for i in range(len(final_id)):
        final_id_d.append((final_id[i], final_sim[i]))

    final_id_1 = sorted(final_id_d, key=lambda x:x[1], reverse=True)

    matched_imgs = []
    for i in range(len(final_id)):
        matched_imgs.append(filenames[final_id_1[i][0]])
        print matched_imgs[i], final_id_1[i][1], featNum[final_id_1[i][0]]

    print('finanl match img list: {0}'.format(final_id))
