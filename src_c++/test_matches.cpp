//
//  test_matches.cpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/10/9.
//  Copyright © 2017年 willard. All rights reserved.
//

#include "test_matches.hpp"
#include "matches.hpp"

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

// the function to test the matches class
using namespace cv;
using namespace std;
int main()
{
    int nn = 3;
    cvflann::Matrix<float> dataset;
    cvflann::Matrix<float> query;
    //cvflann::Index<flann::L2<float>> index;
    
    
    return 0;
}
