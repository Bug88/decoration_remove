//
//  index.h
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/26.
//  Copyright © 2017年 willard. All rights reserved.
//

#ifndef index_h
#define index_h

#pragama once

#include <cstdio>
#include <cstdlib>

//#include <faiss/IndexFlat.h>
//#include <faiss/IndexIVFPQ.h>

#endif /* index_h */

class index_details  // the index details
{
public:
    index_details();
    ~index_details();
public:
    int d; // dimension
    int nb; // number of points
    int nq; // number of query points
    //// product quantization with inverted file indexing structure
    int nlist; // the vocabulary size
    int k; // bytes per vector
    int m; // the other index
    int nprob; // the number of word to be visited
public:
    void set_index_params(int, int, int, int, int, int, int);
}
