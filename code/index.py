__author__ = 'liuzhen'

# construct the inverted file
import const_params
import sys
sys.path.append(const_params.__faiss_lib_path__)
import faiss
import numpy as np
import pickle


def indexing(feats, pos, imgID):

    feats_np = np.zeros(shape=(len(feats), feats[0].shape[0]))
    pos_np = np.zeros(shape=(len(feats), 2))
    imgID_np = np.zeros(shape=(len(feats), 1))

    for i in range(len(feats)):
        feats_np[i, :] = feats[i]
        pos_np[i, :] = pos[i]
        imgID_np[i, :] = imgID[i]

    # construct the visual vocabulary
    voc_size = const_params.__voc_size__

    niter = 20
    verbose = False
    d = feats[0].shape[0]

    code_size = 8
    quantizer = faiss.IndexFlatL2(d)  # this remains the same
    index_ = faiss.IndexIVFPQ(quantizer, d, voc_size, code_size, 8)
                                  # 8 specifies that each sub-vector is encoded as 8 bits
    index_.train(feats_np.astype('float32'))
    index_.add(feats_np.astype('float32'))
    index_.nprobe = 5

    #faiss.write_index(faiss.clone_index(index_), '../index.faiss')

    #fp = open('../query.pkl', 'r')
    #des, pt = pickle.load(fp)
    #fp.close()

    #q = np.asarray(des).astype('float32')
    #D, I = index_.search(q, 10)

    return [faiss.clone_index(index_), pos_np, imgID_np]


if __name__ == '__main__':
    import pickle

    # load the database
    fp = open('../database.pkl', 'r')
    feats, pos, imgID, filenames, featNum = pickle.load(fp)
    fp.close()

    index_, pos_np, imgID_np = indexing(feats, pos, imgID)

    #fp = open('../query.pkl', 'r')
    #des, pt = pickle.load(fp)
    #fp.close()

    #imgNum = 500000

    #q = np.asarray(des).astype('float32')
    #D, I = index_.search(q, 10)


    fp = open('../index.pkl', 'w')
    pickle.dump([pos_np, imgID_np], fp)
    fp.close()

    faiss.write_index(faiss.clone_index(index_), '../index.faiss')

    print('testing...............')

    print('=======================')

    print('the number of points {0}'.format(imgID_np.shape[0]))
    print('the pos is {0}'.format(pos_np[0, :]))

    print('=======================')
