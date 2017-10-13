//
//  decorator.h
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/26.
//  Copyright © 2017年 willard. All rights reserved.
//

#ifndef decorator_h
#define decorator_h

#endif /* decorator_h */

#include "pairwise_validation.hpp"

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
//OpenCV2
//#include <opencv2/nonfree/features2d.hpp>

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

#endif


