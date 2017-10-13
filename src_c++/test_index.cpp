//
//  test_index.cpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/26.
//  Copyright © 2017年 willard. All rights reserved.
//
#include "rejection.hpp"
#include "build_index.hpp"
#include "matches.hpp"
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

#define CPU_IMP true
#ifdef CPU_IMP

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

#endif

// this file is for testing the build index functions
// save the index files
using namespace cv;

int main()
{
    using namespace std;
    
    std::string database_dir("/Users/liuzhen-mac/Desktop/projects/decoration_remove/");
    std::string sub_dir("chinese/");
    std::string filenames("filenames.txt");
    std::string test_image("/Users/liuzhen-mac/Desktop/projects/decoration_remove/test_19.jpg");
    
    // the saved index file
    std::string idxfilepath("/Users/liuzhen-mac/Desktop/projects/decoration_remove/idx.txt");
    bool to_be_saved = false;
    
    // build the index
    index_details details;
    details.set_database_info(database_dir, sub_dir, filenames);
    if(to_be_saved)
    {
        details.build_index();
        details.save_to_file(idxfilepath);
    }
    else
    {
        details.load_from_file(idxfilepath);
    }

    // build the index file with the flann lib
    cvflann::Matrix<float> dataset(details.pFeatsD, details.featNum, details.d);
    cvflann::Index<cvflann::L2<float>> index(dataset, cvflann::KDTreeIndexParams(8));
    index.buildIndex();
    
    // find the matched fonts
    std::string queryImg = test_image;
    
    cv::Mat image;
    image = imread(queryImg, CV_LOAD_IMAGE_COLOR);
    float ratio = 1.0; //512.0/(std::max(image.rows, image.cols));
    cv::resize(image, image, Size(int(image.cols*ratio), int(image.rows*ratio)));
    
    matches mt;
    mt.setParams(30, &details, &index);
    cv::Mat qimage = mt.extractQueryFeats(image);
    
//    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
//    cv::imshow( "Display window", qimage );
//    cv::waitKey(0);
//    cv::destroyWindow("Display window");
//    image.release();
    
    mt.performQuery();
    
    // perform the rejection
    rejection rjt;
    rjt.setQueryImgParams(image, &mt);
    rjt.performRejection_font();
    //rjt.performRejection_dec();
    
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", rjt.image );
    cv::waitKey(0);
    cv::destroyWindow("Display window");
    image.release();

    return 0;
}
