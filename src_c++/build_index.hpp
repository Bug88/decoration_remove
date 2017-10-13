//
//  build_index.hpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/26.
//  Copyright © 2017年 willard. All rights reserved.
//
#ifndef build_index_hpp
#define build_index_hpp

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <iostream>
#include <fstream>

#include <vector>
#include <list>
#include <memory>
#include <algorithm>
#include <numeric>

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

using namespace std;

class index_details  // the index details
{
public:
    index_details();
    ~index_details();
public:
    int d; // dimension
    //// product quantization with inverted file indexing structure
    std::string dataDir;
public:
    int *pImgIDs;   // the image ids stored by seriel order
    int *pFeatNum;  // the feature number of each image stored by the seriel order
    float *pFeatPos;  // the feature positions of each feature stored by the seriel order
    std::string *pFilename; // the filenames for the dataset images
    float *pFeatsD;
public:
    const int maxImgNum = 50000;   // the max image number in the database
    const int maxFeatNum = 200;    // the max feature number for each image
public:
    int imgNum;   // the total image number in the database
    int featNum;  // the total feature number in the database
    std::string database_dir;  // the database dir path
    std::string sub_dir;  // the sub data dir
    std::string filenames;  // the file path containing database images
public:
    void set_database_info(std::string database_dir, std::string sub_dir, std::string filenames);
    void set_index_params(int);
    void build_index();
public:
    void save_to_file(std::string filepath);
    void load_from_file(std::string filepath);
};
#endif /* build_index_hpp */
