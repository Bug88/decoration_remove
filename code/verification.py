__author__ = 'liuzhen'

# geometric verification

__dim__ = 2
import numpy as np

def verification(matched_points):
    # global verification by geometric clues for image pairs
    # until all matched points are consistent

    while 1:
        match_num = matched_points.shape[0]
        map = np.zeros(shape=(__dim__, __dim__, match_num, match_num))

        # anchor image
        for i in range(match_num):
            anc_x = matched_points[i][0]
            anc_y = matched_points[i][1]
            for j in range(match_num):
                cur_x = matched_points[j][0]
                cur_y = matched_points[j][1]

                if cur_x - anc_x > 0:
                    map[0][0][i][j] = 1
                    map[0][0][j][i] = 0
                else:
                    map[0][0][i][j] = 0
                    map[0][0][j][i] = 1

                if cur_y - anc_y > 0:
                    map[0][1][i][j] = 1
                    map[0][1][j][i] = 0
                else:
                    map[0][1][i][j] = 0
                    map[0][1][j][i] = 1

        # target image
        for i in range(match_num):
            anc_x = matched_points[i][2]
            anc_y = matched_points[i][3]
            for j in range(match_num):
                cur_x = matched_points[j][2]
                cur_y = matched_points[j][3]

                if cur_x - anc_x > 0:
                    map[1][0][i][j] = 1
                    map[1][0][j][i] = 0
                else:
                    map[1][0][i][j] = 0
                    map[1][0][j][i] = 1

                if cur_y - anc_y > 0:
                    map[1][1][i][j] = 1
                    map[1][1][j][i] = 0
                else:
                    map[1][1][i][j] = 0
                    map[1][1][j][i] = 1

        # check the consistency
        bit_map = np.zeros(shape=(__dim__, match_num, match_num))
        bit_map[0] = np.logical_xor(map[0][0], map[1][0])
        bit_map[1] = np.logical_xor(map[0][1], map[1][1])

        # filter out false matches
        error_num = []
        for i in range(match_num):
            terr = sum(bit_map[0][i].tolist())
            terr = terr + sum(bit_map[1][i].tolist())
            error_num.append(terr)

        if max(error_num) == 0:
            break;

        idx = error_num.index(max(error_num))

        # update matched points
        new_matched_points = np.zeros(shape=(matched_points.shape[0]-1, 4))
        c = 0
        for i in range(matched_points.shape[0]):
            if i == idx:
                continue
            new_matched_points[c] = matched_points[i]
            c = c + 1
        matched_points = new_matched_points

    return matched_points

if __name__ == '__main__':

    print('testing....')

    import pickle
    import numpy as np

    fp = open('../data.pkl', 'r')

    src, dst = pickle.load(fp)
    pt = np.zeros(shape=(src.shape[0], 4))

    for i in range(src.shape[0]):
        pt[i][0] = src[i][0][0]
        pt[i][1] = src[i][0][1]
        pt[i][2] = dst[i][0][0]
        pt[i][3] = dst[i][0][1]

    fp.close()

    q = verification(pt)

    print('==========================================================')
    print('the original number of matches is {0}'.format(src.shape[0]))
    print('the verified number of matches is {0}'.format(q.shape[0]))
    print('==========================================================')