//
//  rejection.hpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/27.
//  Copyright © 2017年 willard. All rights reserved.
//

#ifndef rejection_hpp
#define rejection_hpp
#include <string>
#include <vector>
#include <set>
#include <stdio.h>
#include "build_index.hpp"
#include "matches.hpp"

#include <iostream>
#include <fstream>
#include "pairwise_validation.hpp"
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>

#include <opencv2/features2d/features2d.hpp>

// perform the rejection procedure for filtering out the outlier data points
using namespace std;
class rejection
{
public:
    rejection();
    ~rejection();
public:
    const int grid_width = 15; // the partition width for the cell
    const int thred = 2; // the threshold for the rejection cell
    const float ratio = 0.005;//0.005; // the threshold ration for the whole font
    const int font_size = 400; // the max size for the font
    const int mat_num_thred = 1; // the threshold for the number of matched features
    const int error = 2; // the allowed error
public:
    std::vector<int> label;  // the label for each grid cell is rejected or not
    std::vector<int> numPt;   // the number of projected points for each grid cell
    int queryImgWidth;   // the width for the query image
    int queryImgHeight;    // the height for the query image
    int grid_w;   // equal to queryImgWidth / grid_width
    int grid_h;   // equal to queryImgHeight / grid_height
public:
    std::set<std::vector<float>> map;
public:
    cv::Mat image;
    matches *pMatches;
public:
    bool font_flag;
public:
    void setQueryImgParams(cv::Mat image, matches *pMatches);
    std::vector<int> singleRejection_v2(std::vector<float> pos, std::vector<float> all_points, cv::Mat image);
    void performRejection_font();
    void performRejection_dec();
};

#endif /* rejection_hpp */
