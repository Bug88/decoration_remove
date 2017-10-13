//
//  matches.hpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/27.
//  Copyright © 2017年 willard. All rights reserved.
//

#ifndef matches_hpp
#define matches_hpp

#include <stdio.h>
//#include <../faiss-master/IndexFlat.h>
//#include <../faiss-master/IndexIVFPQ.h>
#include "build_index.hpp"

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>

// OpenCV3
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/flann/flann.hpp>


// the class for performing the sift matching

using namespace std;
using namespace cv;
class matches
{
public:
    const float match_dis = 15000; // the distance threshold for the matched sift features
public:
    matches();
    ~matches();
public:
    index_details* pIndexDet;
    cvflann::Index<cvflann::L2<float>> * pIndex;  // the flann index files
    float * pDes; // the descriptors for the query image, change to float type
    float *pPos;  // the positions for geometric verification
    int featNum;
    int NN;  // the number of nearest neighbors
public:
    float *pMatchedPos;  // the record for the matched positions
public:
    void performQuery();
    void setParams(int n, index_details* idxDt, cvflann::Index<cvflann::L2<float>> *idx);
    cv::Mat extractQueryFeats(cv::Mat image);
    int stepIdx(float * Data, int length);
public:
    void printMatchedImgs();
public:
    std::vector<std::vector<float>> vMatchedPos;
};
#endif /* matches_hpp */
