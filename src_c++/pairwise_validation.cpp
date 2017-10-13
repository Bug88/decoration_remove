
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

#ifdef GPU_IMP_2
#include "opencv2/nonfree/gpu.hpp"
#endif
#ifdef GPU_IMP_3
#include "opencv2/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#endif

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


using std::vector;
using std::list;

const int multiple_frames = 0;

PairWiseValidation::PairWiseValidation() {
}
PairWiseValidation::~PairWiseValidation() {
}

class PairWiseValidationGPU : public PairWiseValidation {
public:
    PairWiseValidationGPU();
    virtual ~PairWiseValidationGPU();
    virtual int IsMatch(const std::vector<char>& buf1, const std::vector<char>& buf2, int* is_match);
    
private:
    
    int FilterByDist(vector<vector<cv::DMatch> >& matches, \
                     list<cv::DMatch>* good_matches);
    
    int AngleDiff(float angle1, float angle2, int* idx);
    int CheckAngleDiff(vector<cv::KeyPoint>& kpts1, vector<cv::KeyPoint>& kpts2, \
                       list<cv::DMatch>* good_matches);
    
    int OctaveDiff(float octave1, float octave2, int* idx);
    int CheckOctaveDiff(vector<cv::KeyPoint>& kpts1, vector<cv::KeyPoint>& kpts2, \
                        list<cv::DMatch>* good_matches);
    
    int CheckAffine(vector<cv::KeyPoint>& queryKpts, vector<cv::KeyPoint>& trainKpts, \
                    list<cv::DMatch>& good_matches, int* match_count, cv::Mat &H);
    
    int CheckAreaWarpTransform(cv::Mat& img1, cv::Mat& img2, vector<cv::KeyPoint>& kpts1, \
                               vector<cv::KeyPoint>& kpts2, cv::Mat& H, list<cv::DMatch>& good_matches, \
                               int* is_match, float &dist);
    
    int CheckRigidTransform(vector<cv::KeyPoint>& queryKpts, vector<cv::KeyPoint>& trainKpts, \
                            list<cv::DMatch>& good_matches, int* match_count);
    
    int CheckBoundingBox(vector<cv::KeyPoint>& queryKpts, vector<cv::KeyPoint>& trainKpts, \
                         list<cv::DMatch>& good_matches, int* is_match);
    
    int ResizeImage(cv::Mat& raw_img, cv::Mat* img);
    
    float computeHOG(cv::Mat& img1, cv::Mat& img2, cv::HOGDescriptor& hog, cv::Rect& region);
    
private:
#ifdef GPU_IMP_2
    std::shared_ptr<cv::gpu::SURF_GPU> surf;
#endif
#ifdef GPU_IMP_3
    std::shared_ptr<cv::cuda::SURF_CUDA> surf;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
#endif
#ifdef CPU_IMP
    //OpenCV 2
    //std::shared_ptr<cv::SURF> surf;
    //OpenCV 3
    cv::Ptr<cv::Feature2D> surf;
    
    std::shared_ptr<cv::BFMatcher> matcher;
#endif
    std::vector<int32_t> octave_diff;
    std::vector<int32_t> angle_diff;
    
    const static uint32_t kMaxMatchPair = 4098;
    const static uint32_t kMaxAngleSlices = 36;
    const static uint32_t kMaxOctaveSlices = 8;
    const static uint32_t kShorterEdge = 300;
    const static uint32_t kMinMatchPair = 6;
    
    // 噪声描述子
    //cv::flann::Index noises_surf_KDindex;
};

int PairWiseValidationFactory::SetGPU(int gpu_id) {
#ifdef GPU_IMP_3
    cv::cuda::setDevice(gpu_id);
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
#endif
#ifdef GPU_IMP_2
    cv::gpu::setDevice(gpu_id);
#endif
    return 0;
}

PairWiseValidation* PairWiseValidationFactory::Create(const char* name) {
    return reinterpret_cast<PairWiseValidation*>(new PairWiseValidationGPU());
}

PairWiseValidationGPU::PairWiseValidationGPU() {
#ifdef GPU_IMP_2
    surf = std::make_shared<cv::gpu::SURF_GPU>();
#endif
#ifdef GPU_IMP_3
    surf = std::make_shared<cv::cuda::SURF_CUDA>();
    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf->defaultNorm());
#endif
#ifdef CPU_IMP
    //OpenCV 2
    //surf = std::make_shared<cv::SURF>();
    //OpenCV 3
    surf = cv::xfeatures2d::SIFT::create();
    matcher = std::make_shared<cv::BFMatcher>();
#endif
    this->octave_diff.resize(kMaxOctaveSlices);
    this->angle_diff.resize(kMaxAngleSlices);
}

PairWiseValidationGPU::~PairWiseValidationGPU() {
}


int PairWiseValidationGPU::ResizeImage(cv::Mat& raw_img, cv::Mat* img) {
    double ratio;
    if(raw_img.rows < raw_img.cols) {
        ratio = double(kShorterEdge / double(raw_img.rows));
    }else {
        ratio = double(kShorterEdge / double(raw_img.cols));
    }
    cv::Rect my_roi(raw_img.cols >> 3, raw_img.rows >> 3,  (raw_img.cols * 3) >> 2, (raw_img.rows *3) >> 2);
    cv::Mat croped_img = raw_img(my_roi);
    cv::resize(croped_img, *img, cv::Size(0,0), ratio, ratio);
    return 0;
}


float PairWiseValidationGPU::computeHOG(cv::Mat& img1, cv::Mat& img2, cv::HOGDescriptor& hog, cv::Rect& region){
    float dist = 0.0;
    //Center area
    cv::Mat region1_img1 = img1(region);
    cv::Mat region1_img2 = img2(region);
    cv::resize(region1_img1, region1_img1, cv::Size(hog.winSize.width, hog.winSize.height));
    cv::resize(region1_img2, region1_img2, cv::Size(hog.winSize.width, hog.winSize.height));
    std::vector<float> feat_1;
    std::vector<float> feat_2;
    hog.compute(region1_img1, feat_1, cv::Size(1, 1), cv::Size(0, 0));
    hog.compute(region1_img2, feat_2, cv::Size(1, 1), cv::Size(0, 0));
    
    //Compute cosine similarity between two vectors
    float dot = std::inner_product( feat_1.begin(), feat_1.end(), feat_2.begin(), 0.0 );
    float denom_a = std::inner_product( feat_1.begin(), feat_1.end(), feat_1.begin(), 0.0 );
    float denom_b = std::inner_product( feat_2.begin(), feat_2.end(), feat_2.begin(), 0.0 );
    if(denom_a == 0.0 || denom_b == 0.0){
        dist = 1.0;
    }else{
        dist = dot / (sqrt(denom_a) * sqrt(denom_b));
    }
    return dist;
}

int PairWiseValidationGPU::IsMatch(const std::vector<char>& buf1, \
                                   const std::vector<char>& buf2, int* is_match) {
    *is_match = 0;
    if(!surf) {
        std::cout << "Failed in new SURF descriptor" << std::endl;
        return -1;
    }
    if(buf1.size() == 0 || buf2.size() == 0) {
        std::cout << "buf1 size: " << buf1.size() << ", buf2 size: " << buf2.size() << std::endl;
        return -2;
    }
    std::cout << "buf1 size: " << buf1.size() << ", buf2 size: " << buf2.size() << std::endl;
    cv::Mat raw_img1, raw_img2, img1, img2;
    raw_img1 = cv::imdecode(buf1, 0);
    raw_img2 = cv::imdecode(buf2, 0);
    
#if 0
    cv::namedWindow("raw img1");
    cv::namedWindow("raw img2");
    cv::imshow("raw img1", raw_img1);
    cv::imshow("raw img2", raw_img2);
    cv::waitKey(0);
#endif
    
    if(raw_img1.empty() || raw_img2.empty()) {
        std::cout << "Failed in imdecoding" << std::endl;
        return -3;
    }
    ResizeImage(raw_img1, &img1);
    ResizeImage(raw_img2, &img2);
    
#if 0
    cv::namedWindow("img1");
    cv::namedWindow("img2");
    cv::imshow("img1", img1);
    cv::imshow("img2", img2);
    cv::waitKey(0);
#endif
    
    //GpuMat gpu_img1;  gpu_img1.upload(img1);
    vector<cv::KeyPoint> kpts1, kpts2;
    cv::Mat dsp1, dsp2; // CPU
    //vector<float> dsp1, dsp2;
#ifdef GPU_IMP_2
    cv::gpu::GpuMat gpu_img1(img1), gpu_img2(img2);
    cv::gpu::GpuMat gpu_dsp1, gpu_dsp2;
    (*surf)(gpu_img1, cv::gpu::GpuMat(), kpts1, gpu_dsp1);
    (*surf)(gpu_img2, cv::gpu::GpuMat(), kpts2, gpu_dsp2);
    gpu_dsp1.upload(dsp1);
    gpu_dsp2.upload(dsp2);
#endif
#ifdef GPU_IMP_3
    cv::cuda::GpuMat gpu_img1(img1), gpu_img2(img2);
    cv::cuda::GpuMat gpu_kpts1, gpu_kpts2;
    cv::cuda::GpuMat gpu_dsp1, gpu_dsp2;
    (*surf)(gpu_img1, cv::cuda::GpuMat(), gpu_kpts1, gpu_dsp1);
    (*surf)(gpu_img2, cv::cuda::GpuMat(), gpu_kpts2, gpu_dsp2);
    (*surf).downloadKeypoints(gpu_kpts1, kpts1);
    (*surf).downloadKeypoints(gpu_kpts2, kpts2);
    (*surf).downloadDescriptors(gpu_dsp1, dsp1);
    (*surf).downloadDescriptors(gpu_dsp2, dsp2);
#endif
#ifdef CPU_IMP
    //OpenCV 2
    //(*surf)(img1, cv::Mat(), kpts1, dsp1);
    //(*surf)(img2, cv::Mat(), kpts2, dsp2);
    //OpenCV 3
    surf->detectAndCompute(img1, cv::Mat(), kpts1, dsp1);
    surf->detectAndCompute(img2, cv::Mat(), kpts2, dsp2);
#endif
    if(kpts1.size() == 0 || kpts2.size() == 0) {
        std::cout << "SURF_GPU return (" << kpts1.size() << "," << kpts2.size() << ")";
        return 0;
    }
    std::cout << "img1 kpts nb:" << kpts1.size() << ", img2 kpts nb:" << kpts2.size() << std::endl;
    vector<vector<cv::DMatch> > matches;
    list<cv::DMatch> good_matches;
    //dsp1:query, dsp2:train
#ifdef CPU_IMP
    matcher->knnMatch(dsp1, dsp2, matches, 2);
#endif
#ifdef GPU_IMP_3
    matcher->knnMatch(gpu_dsp1, gpu_dsp2, matches, 2);
#endif
    
    //std::cout << "the descriptor is: " << dsp1.data << std::endl;
    
    std::cout << "the number of matches: " << matches.size() << std::endl;
    
    int ret = 0;
    if((ret = FilterByDist(matches, &good_matches))) {
        //std::cout << "the value of ret: " << ret << std::endl;
        std::cout << "Failed in FilterByDist" << std::endl;
        return 0;
    }
    std::cout << "FilterByDist match pair" << good_matches.size() << std::endl;
    if(good_matches.size() < kMinMatchPair) {
        std::cout << "Reject by FilterByDist" << std::endl;
        return 0;
    }
    
    if(multiple_frames == 1)
    {
        if((ret = CheckAngleDiff(kpts1, kpts2, &good_matches))) {
            std::cout << "Failed in CheckAngleDiff" << std::endl;
            return 0;
        }
        std::cout << "CheckAngleDiff match pair" << good_matches.size() << std::endl;
        if(good_matches.size() < kMinMatchPair) {
            std::cout << "Reject by CheckAngleDiff" << std::endl;
            return 0;
        }
        if((ret = CheckOctaveDiff(kpts1, kpts2, &good_matches))) {
            std::cout << "Failed in CheckOctaveDiff" << std::endl;
            return 0;
        }
        std::cout << "CheckOctave match pair: " << good_matches.size() << std::endl;
        if(good_matches.size() < kMinMatchPair) {
            std::cout << "Reject by CheckOctaveDiff" << std::endl;
            return 0;
        }
    }
    //Final GeoPosition Validation
    int match_count = 0;
    cv::Mat H;
    if((ret = CheckAffine(kpts1, kpts2, good_matches, &match_count, H))) {
        std::cout << "Failed in CheckAffine: " << std::endl;
        return 0;
    }
    
    
    std::cout << good_matches.size() << " match count " << match_count << std::endl;
    
    
    std::cout << "CheckAffine match pair: " << match_count << std::endl << std::endl;
    if(match_count < kMinMatchPair) {
        std::cout << "Reject by CheckAffine";
        return 0;
    }
    
    //Check BoundingBox
    if((ret = CheckBoundingBox(kpts1, kpts2, good_matches, is_match))) {
        std::cout << "Failed in CheckBoundingBox" << std::endl;
        return 0;
    }
    
    
    
//    //Check AreaWarpTransform: still under development, maybe not success
//    float dist = 0.0;
//     if (!H.empty()){
//     if(PairWiseValidationGPU::CheckAreaWarpTransform(img1, img2, kpts1, kpts2, H, good_matches, is_match, dist)){
//     std::cout << "Reject by Area Warp Transform, dist: " << dist;
//     return 0;
//     }
//     }
    
    *is_match = 1;
    std::cout << "is_match: " << *is_match << ", address: " << long(is_match) << std::endl;
    
    std::fstream outfile;
    outfile.open("/Users/liuzhen-mac/Desktop/siftDis.txt", std::ios::app);
    
    for(auto i=good_matches.begin(); i != good_matches.end(); i++)
    {
        outfile << i->distance << std::endl;
    }
    
    outfile.close();
    
    
    
    return 0;
}

int PairWiseValidationGPU::CheckBoundingBox(vector<cv::KeyPoint>& queryKpts, \
                                            vector<cv::KeyPoint>& trainKpts, \
                                            list<cv::DMatch>& good_matches, \
                                            int* is_match) {
    float query_min_x, query_min_y, query_max_x, query_max_y;
    float train_min_x, train_min_y, train_max_x, train_max_y;
    query_min_x = query_min_y = train_min_x = train_min_y = 1000000.0f;
    query_max_x = query_max_y = train_max_x = train_max_y = 0.0f;
    for(auto i = good_matches.begin(); i != good_matches.end(); i++) {
        query_min_x = (queryKpts[i->queryIdx].pt.x < query_min_x) ? queryKpts[i->queryIdx].pt.x : query_min_x;
        query_min_y = (queryKpts[i->queryIdx].pt.y < query_min_y) ? queryKpts[i->queryIdx].pt.y : query_min_y;
        query_max_x = (queryKpts[i->queryIdx].pt.x > query_max_x) ? queryKpts[i->queryIdx].pt.x : query_max_x;
        query_max_y = (queryKpts[i->queryIdx].pt.y > query_max_y) ? queryKpts[i->queryIdx].pt.y : query_max_y;
        
        train_min_x = (trainKpts[i->trainIdx].pt.x < train_min_x) ? trainKpts[i->trainIdx].pt.x : train_min_x;
        train_min_y = (trainKpts[i->trainIdx].pt.y < train_min_y) ? trainKpts[i->trainIdx].pt.y : train_min_y;
        train_max_x = (trainKpts[i->trainIdx].pt.x > train_max_x) ? trainKpts[i->trainIdx].pt.x : train_max_x;
        train_max_y = (trainKpts[i->trainIdx].pt.y > train_max_y) ? trainKpts[i->trainIdx].pt.y : train_max_y;
    }
    //LOG(INFO) << "query:(min_x, min_y, max_x, max_y): " << query_min_x << "," << query_min_y << "," \
    << query_max_x << "," << query_max_y;
    //LOG(INFO) << "train:(min_x, min_y, max_x, max_y): " << train_min_x << "," << train_min_y << "," \
    << train_max_x << "," << train_max_y;
    const static float kMinShortEdge = 150.0f;
    if(std::abs<float>(query_min_x - query_max_x) > kMinShortEdge && \
       std::abs<float>(query_min_y - query_max_y) > kMinShortEdge && \
       std::abs<float>(train_min_x - train_max_x) > kMinShortEdge && \
       std::abs<float>(train_min_y - train_max_y) > kMinShortEdge) {
        *is_match = 1;
    }else {
        std::cout << "(query_diff_x, query_diff_y, train_diff_x, train_diff_y): " <<\
        (query_min_x - query_max_x) << "," << (query_min_y - query_max_y) << "," << \
        (train_min_x - train_max_x) << "," << (train_min_y - train_max_y) << \
        ". Shoter than" << kMinShortEdge << std::endl;
    }
    return 0;
}

int PairWiseValidationGPU::CheckAffine(vector<cv::KeyPoint>& queryKpts, \
                                       vector<cv::KeyPoint>& trainKpts, \
                                       list<cv::DMatch>& good_matches, \
                                       int* match_count, cv::Mat &H) {
    vector<unsigned char> match_mask;
    *match_count = 0;
    std::vector<cv::Point2f> match_kpts1, match_kpts2;
    for(auto i = good_matches.begin(); i != good_matches.end(); i++) {
        match_kpts1.push_back(queryKpts[i->queryIdx].pt);
        match_kpts2.push_back(trainKpts[i->trainIdx].pt);
    }
    //LOG(INFO) << "estimateAffineTransform inputs: " << match_kpts1.size() << ", " << match_kpts2.size();
    //std::vector<char> inliner_mask(good_matches.size());
    std::vector<unsigned char> inliner_mask(good_matches.size());
    H = cv::findHomography(match_kpts1, match_kpts2, CV_RANSAC, 20, inliner_mask);
    //LOG(INFO) << "estimateAffineTransform ret: matrix:\n" <<
//    std::cout << H.at<double>(0, 0) << ","<< H.at<double>(0, 1) <<"," << H.at<double>(0,2) << "\n" << \
//    H.at<double>(1, 0) << ","<< H.at<double>(1, 1) <<"," << H.at<double>(1,2) << "\n" << \
//    H.at<double>(2, 0) << ","<< H.at<double>(2, 1) <<"," << H.at<double>(2,2);
    if(H.empty()) {
        good_matches.clear();
        return 0;
    }
    
    //std::cout << H.at<double>(2, 0) << H.at<double>(2, 1) << std::endl;
    //std::cout << H.at<double>(2, 2) << std::endl;
    //std::cout << inliner_mask[0] << std::endl;
    float thred_h = 0.003;
    if(multiple_frames != 2)
    {
        thred_h = 0.01;
    }
    if(std::abs<double>(H.at<double>(2,0)) < thred_h && std::abs<double>(H.at<double>(2,1)) < thred_h) {
        auto match_iter = good_matches.begin();
        for(int i = 0; i < good_matches.size(); i++) {
            if(inliner_mask[i]) {
                (*match_count)++;
                match_iter++;
            }else {
                match_iter = good_matches.erase(match_iter);
            }
        }
    }else{
        good_matches.clear();
    }
    //LOG(INFO) << "Affine input size: " << inliner_mask.size() << "match_count: " << *match_count;
    if (multiple_frames == 1)
    {
        if(inliner_mask.size() > (*match_count)*2 && (*match_count) < 15) {
        (*match_count) = 0;
        good_matches.clear();
        }
    }
    return 0;
}

int PairWiseValidationGPU::CheckRigidTransform(vector<cv::KeyPoint>& queryKpts, \
                                               vector<cv::KeyPoint>& trainKpts, \
                                               list<cv::DMatch>& good_matches, \
                                               int* match_count) {
    vector<unsigned char> match_mask;
    *match_count = 0;
    std::vector<cv::Point2f> match_kpts1, match_kpts2;
    for(auto i = good_matches.begin(); i != good_matches.end(); i++) {
        match_kpts1.push_back(queryKpts[i->queryIdx].pt);
        match_kpts2.push_back(trainKpts[i->trainIdx].pt);
    }
    //LOG(INFO) << "estimateRigidTransform inputs: " << match_kpts1.size() << ", " << match_kpts2.size();
    cv::Mat H = cv::estimateRigidTransform(match_kpts1, match_kpts2, false);
    if(!H.empty()) {
        //LOG(INFO) << "estimateRigidTransform ret: matrix:\n" << \
        H.at<double>(0, 0) << ","<< H.at<double>(0, 1) <<"," << H.at<double>(0,2) << "\n" << \
        H.at<double>(1, 0) << ","<< H.at<double>(1, 1) <<"," << H.at<double>(1,2);
        cv::Mat R = cv::Mat(3, 3, H.type());
        R.at<double>(0, 0) = H.at<double>(0, 0);
        R.at<double>(0, 1) = H.at<double>(0, 1);
        R.at<double>(0, 2) = H.at<double>(0, 2);
        
        R.at<double>(1, 0) = H.at<double>(1, 0);
        R.at<double>(1, 1) = H.at<double>(1, 1);
        R.at<double>(1, 2) = H.at<double>(1, 2);
        
        R.at<double>(2, 0) = 0.0;
        R.at<double>(2, 1) = 0.0;
        R.at<double>(2, 2) = 1.0;
        std::vector<cv::Point2f> groundtruth_kpts2;
        cv::perspectiveTransform(match_kpts1, groundtruth_kpts2, R);
        
        for(int i = 0; i < groundtruth_kpts2.size(); i++) {
            double loss = cv::norm(groundtruth_kpts2[i] - match_kpts2[i]);
            //LOG(INFO) << "x:" << match_kpts1[i].x << "y:" << match_kpts1[i].y << \
            "x:" << match_kpts2[i].x << "y:" << match_kpts2[i].y << \
            "x:" << groundtruth_kpts2[i].x << "y:" << groundtruth_kpts2[i].y;
            //LOG(INFO) << "loss: " << loss;
            if(loss < 10.0) {
                (*match_count)++;
            }
        }
    }
    return 0;
}

int PairWiseValidationGPU::AngleDiff(float angle1, float angle2, int* idx) {
    float angle_diff = angle1 - angle2;
    while(angle_diff < 0.0) angle_diff += 720.0;
    while(angle_diff >= 720.0) angle_diff -= 720.0;
    *idx = int(angle_diff + 0.5) / kMaxAngleSlices;
    *idx = (*idx < 0) ? 0 : *idx;
    *idx = (*idx >= kMaxAngleSlices) ? (kMaxAngleSlices-1) : *idx;
    return 0;
}

int PairWiseValidationGPU::CheckAngleDiff(vector<cv::KeyPoint>& kpts1, vector<cv::KeyPoint>& kpts2, \
                                          list<cv::DMatch>* good_matches) {
    std::fill(angle_diff.begin(), angle_diff.end(), 0);
    int angle_diff_idx = -1;
    for(auto i = good_matches->begin(); i != good_matches->end(); i++) {
        AngleDiff(kpts1[i->queryIdx].angle, kpts2[i->trainIdx].angle, &angle_diff_idx);
        angle_diff[angle_diff_idx] += 1;
    }
    int max_idx = (int)std::distance(angle_diff.begin(), \
                                std::max_element(angle_diff.begin(), angle_diff.end()));
    std::cout << "CheckAngleDiff max_idx: " << max_idx << ", nb: " << angle_diff[max_idx] <<std::endl;
    if(std::abs<int>(max_idx) < 3 || \
       std::abs<int>(max_idx - kMaxAngleSlices/4) < 3 || \
       std::abs<int>(max_idx - kMaxAngleSlices/2) < 3 || \
       std::abs<int>(max_idx - 3*kMaxAngleSlices/4) < 3 || \
       std::abs<int>(max_idx - kMaxAngleSlices) < 3) {
        //pass
    }else {
        std::cout << "angle diff is not 0, 90, 180, 270, 360" << std::endl;
        good_matches->clear();
        return 0;
    }
    auto i = good_matches->begin();
    do {
        AngleDiff(kpts1[i->queryIdx].angle, kpts2[i->trainIdx].angle, &angle_diff_idx);
        if(abs(angle_diff_idx - max_idx) > 2) {
            if((max_idx == 0 || max_idx == (kMaxAngleSlices-1)) && \
               ((max_idx + angle_diff_idx + 1) == kMaxAngleSlices)) {
                continue;
            }else {
                i = good_matches->erase(i);
            }
        }else {
            i++;
        }
    }while(i != good_matches->end());
    return 0;
}

int PairWiseValidationGPU::OctaveDiff(float octave1, float octave2, int* idx) {
    float octave_diff = octave1 - octave2;
    while(octave_diff < 0.0) octave_diff += 8;
    while(octave_diff >= 8) octave_diff -= 8;
    *idx = int(octave_diff + 0.5) / kMaxOctaveSlices;
    *idx = (*idx < 0) ? 0 : *idx;
    *idx = (*idx >= kMaxOctaveSlices) ? (kMaxOctaveSlices-1) : *idx;
    return 0;
}

int PairWiseValidationGPU::CheckOctaveDiff(vector<cv::KeyPoint>& kpts1, vector<cv::KeyPoint>& kpts2, \
                                           list<cv::DMatch>* good_matches) {
    std::fill(octave_diff.begin(), octave_diff.end(), 0);
    int octave_diff_idx = -1;
    for(auto i = good_matches->begin(); i != good_matches->end(); i++) {
        OctaveDiff(kpts1[i->queryIdx].octave, kpts2[i->trainIdx].octave, &octave_diff_idx);
        octave_diff[octave_diff_idx] += 1;
    }
    int max_idx = (int)std::distance(octave_diff.begin(), std::max_element(octave_diff.begin(), octave_diff.end()));
    std::cout << "CheckOctaveDiff max_idx: " << max_idx << std::endl;
    auto i = good_matches->begin();
    do {
        OctaveDiff(kpts1[i->queryIdx].octave, kpts2[i->trainIdx].octave, &octave_diff_idx);
        if(abs(octave_diff_idx - max_idx) > 1) {
            i = good_matches->erase(i);
        }else {
            i++;
        }
    }while(i != good_matches->end());
    return 0;
}

//CheckAreaWarpTransform
int PairWiseValidationGPU::CheckAreaWarpTransform(cv::Mat &img1, cv::Mat &img2, vector<cv::KeyPoint>& kpts1,
                                                  vector<cv::KeyPoint>& kpts2, cv::Mat &H, list<cv::DMatch>& good_matches, int* is_match, float &dist){
    cv::Mat im1_out;
    vector<cv::Point2f> kpts1_pt;
    vector<cv::Point2f> kpts1_warp;
    for(int i = 0; i < kpts1.size(); i++){
        kpts1_pt.push_back(kpts1.at(i).pt);
    }
    
    //Compute 4 coordinates of image
    vector<cv::Point2f> img1_4coords;
    vector<cv::Point2f> img4coords_warp;
    img1_4coords.push_back(cv::Point2f(0.0, 0.0)); // left top
    img1_4coords.push_back(cv::Point2f(img1.cols, 0.0)); // right top
    img1_4coords.push_back(cv::Point2f(0.0,img1.rows)); // left bottom
    img1_4coords.push_back(cv::Point2f(img1.cols,img1.rows)); // right bottom
    
    //Compute 4 coordinates of image
    vector<cv::Point2f> img2_4coords;
    img2_4coords.push_back(cv::Point2f(0.0, 0.0)); // left top
    img2_4coords.push_back(cv::Point2f(img2.cols, 0.0)); // right top
    img2_4coords.push_back(cv::Point2f(0.0,img2.rows)); // left bottom
    img2_4coords.push_back(cv::Point2f(img2.cols,img2.rows)); // right bottom
    
    //ALL coordinates of image
    vector<cv::Point2f> imgCoords;
    vector<cv::Point2f> imgCoords_warp;
    for(int i = 0; i < img1.cols; i++){
        for(int j = 0; j < img1.rows; j++){
            imgCoords.push_back(cv::Point2f(i, j));
        }
    }
    
    if(! H.empty()) {
        //Warp source image to destination based on homography
        cv::warpPerspective(img1, im1_out, H, img2.size());
        cv::perspectiveTransform(kpts1_pt, kpts1_warp, H);
        cv::perspectiveTransform(img1_4coords, img4coords_warp, H);
        cv::perspectiveTransform(imgCoords, imgCoords_warp, H);
    }
    
    cv::Mat img1_color;
    cv::cvtColor(im1_out, img1_color, cv::COLOR_GRAY2BGR);
    
    //Compute area after perspective transform
    int warpPTArea = 0;
    int gravity_x = 0;
    int gravity_y = 0;
    cv::Mat mask = cv::Mat::zeros(img2.rows, img2.cols, CV_8U);
    vector<cv::Point2f> imgCoords_warp_img1in;
    for(int i = 0; i < imgCoords_warp.size(); i++){
        if(imgCoords_warp.at(i).x>=0 && imgCoords_warp.at(i).x<img2.cols && imgCoords_warp.at(i).y>=0 && imgCoords_warp.at(i).y<img2.rows){
            ++warpPTArea;
            gravity_x += imgCoords_warp.at(i).x;
            gravity_y += imgCoords_warp.at(i).y;
            imgCoords_warp_img1in.push_back(cv::Point2f(imgCoords_warp.at(i).x, imgCoords_warp.at(i).y));
            //mask.at<uchar>((int)imgCoords_warp.at(i).y, imgCoords_warp.at(i).x) = 255;
            //circle(img1_color, imgCoords_warp.at(i), 1, cv::Scalar(255), CV_FILLED, 8,0);
        }
    }

    //cv::imshow("mask", mask);
    //cv::waitKey();
    cv::flann::Index tree = cv::flann::Index(cv::Mat(imgCoords_warp_img1in).reshape(1), cv::flann::LinearIndexParams(), cvflann::FLANN_DIST_L2);
    cv::Mat nearest_index_result;
    cv::Mat nearest_dist_result;
    //tree.knnSearch(img2_4coords, nearest_index_result, nearest_dist_result, num_knn, cv::flann::SearchParams(64));
    
    cv::Point point_center = cv::Point(int(gravity_x/warpPTArea), int(gravity_y/warpPTArea));
    
    
    //Draw only "good" matches
    cv::Mat img_matches;
    std::vector<cv::DMatch> matches_{ std::begin(good_matches), std::end(good_matches) };
    cv::drawMatches(img1, kpts1, img2, kpts2, matches_, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                    vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Good Matches", img_matches );
    
    //Show new matching
    vector<cv::KeyPoint> kpts1_warp_kp = kpts1;
    for(int i = 0; i < kpts1_warp.size(); i++){
        kpts1_warp_kp.at(i).pt = kpts1_warp.at(i);
    }
    
    //Show 4 corners
    //circle(img1_color, point_center, 10, cv::Scalar(255), CV_FILLED, 8,0);
    circle(img1_color, img4coords_warp.at(0), 6, cv::Scalar(0,0,255), CV_FILLED, 8,0);
    circle(img1_color, img4coords_warp.at(1), 6, cv::Scalar(0,0,255), CV_FILLED, 8,0);
    circle(img1_color, img4coords_warp.at(2), 6, cv::Scalar(0,0,255), CV_FILLED, 8,0);
    circle(img1_color, img4coords_warp.at(3), 6, cv::Scalar(0,0,255), CV_FILLED, 8,0);
    
    //rectangle
    int region_radius = 20;
    cv::Rect region_center = cv::Rect(point_center.x - region_radius, point_center.y - region_radius, 2*region_radius, 2*region_radius);
    
    //Show img1 warp
    //cv::rectangle(img1_color, region_center.tl(), region_center.br(), cv::Scalar(0, 0, 255), 3);
    imshow( "warp image",  img1_color );
    
    //Show warp images
    cv::Mat img_matches_warp;
    cv::Mat img2_color;
    cv::cvtColor(img2, img2_color, cv::COLOR_GRAY2BGR);
    circle(img2_color, point_center, 10, cv::Scalar(255), CV_FILLED, 8,0);
    cv::rectangle(img2_color, region_center.tl(), region_center.br(), cv::Scalar(0, 0, 255), 3);
    
    //Compute max boundary area
    float black_area_ratio = 1.0 - warpPTArea*1.0/(img2.cols*img2.rows);
    /*if (black_area_ratio > 0.58){
        *is_match = 0;
        return 1;
    }*/
    
    //HOG setting
    int HH = 64;
    int WW = 128;
    cv::HOGDescriptor hog(cv::Size(WW, HH), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
    
    int count = 0;
    cv::Rect region_tr;
    cv::Rect region_tl;
    cv::Rect region_bl;
    cv::Rect region_br;
    cv::Rect region_tr_nn;
    cv::Rect region_tl_nn;
    cv::Rect region_bl_nn;
    cv::Rect region_br_nn;
    if (black_area_ratio <= 0.2){
        
        //Compute region pairwise HOG distance
        float dist_center = computeHOG(im1_out, img2, hog, region_center);
        if (dist_center < 0.45){
            ++count;
        }
        
        int xtl = point_center.x - 3*region_radius;
        int ytl = point_center.y - 5*region_radius;
        int xbr = point_center.x + 3*region_radius;
        int ybr = point_center.y + 5*region_radius;
        if(xtl>=0 && xtl<img2.cols && ytl>=0 && ytl<img2.rows && xbr>=0 && xbr<img2.cols && ybr>=0 && ybr<img2.rows){
            region_tl = cv::Rect(point_center.x - 3*region_radius, point_center.y - 5*region_radius, 2*region_radius, 2*region_radius);
            region_tr = cv::Rect(point_center.x + region_radius, point_center.y - 5*region_radius, 2*region_radius, 2*region_radius);
            region_bl = cv::Rect(point_center.x - 3*region_radius, point_center.y + 3*region_radius, 2*region_radius, 2*region_radius);
            region_br = cv::Rect(point_center.x + region_radius, point_center.y + 3*region_radius, 2*region_radius, 2*region_radius);
    
            region_tl_nn = cv::Rect(point_center.x - 3*region_radius, point_center.y - 3*region_radius, 2*region_radius, 2*region_radius);
            region_tr_nn = cv::Rect(point_center.x + region_radius, point_center.y - 3*region_radius, 2*region_radius, 2*region_radius);
            region_bl_nn = cv::Rect(point_center.x - 3*region_radius, point_center.y + 1*region_radius, 2*region_radius, 2*region_radius);
            region_br_nn = cv::Rect(point_center.x + region_radius, point_center.y + 1*region_radius, 2*region_radius, 2*region_radius);
            
            //ff
            float dist_tr = computeHOG(im1_out, img2, hog, region_tr);
            if (dist_tr < 0.43){
                ++count;
            }
            float dist_tl = computeHOG(im1_out, img2, hog, region_tl);
            if (dist_tl < 0.43){
                ++count;
            }
            float dist_bl = computeHOG(im1_out, img2, hog, region_bl);
            if (dist_bl < 0.43){
                ++count;
            }
            float dist_br = computeHOG(im1_out, img2, hog, region_br);
            if (dist_br < 0.43){
                ++count;
            }
            
            //nn
            float dist_tr_nn = computeHOG(im1_out, img2, hog, region_tr_nn);
            if (dist_tr_nn < 0.43){
                ++count;
            }
            float dist_tl_nn = computeHOG(im1_out, img2, hog, region_tl_nn);
            if (dist_tl_nn < 0.43){
                ++count;
            }
            float dist_bl_nn = computeHOG(im1_out, img2, hog, region_bl_nn);
            if (dist_bl_nn < 0.43){
                ++count;
            }
            float dist_br_nn = computeHOG(im1_out, img2, hog, region_br_nn);
            if (dist_br_nn < 0.43){
                ++count;
            }
            circle(img1_color, cv::Point2f(xtl, ytl), 8, cv::Scalar(0, 0, 255), CV_FILLED, 8,0);
            circle(img1_color, cv::Point2f(xbr, ybr), 8, cv::Scalar(0, 0, 255), CV_FILLED, 8,0);
        
            cv::rectangle(img1_color, region_tr.tl(), region_tr.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img1_color, region_tl.tl(), region_tl.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img1_color, region_bl.tl(), region_bl.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img1_color, region_br.tl(), region_br.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img2_color, region_tr.tl(), region_tr.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img2_color, region_tl.tl(), region_tl.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img2_color, region_bl.tl(), region_bl.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img2_color, region_br.tl(), region_br.br(), cv::Scalar(0, 255, 255), 3);
            
            cv::rectangle(img2_color, region_tr_nn.tl(), region_tr_nn.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img2_color, region_tl_nn.tl(), region_tl_nn.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img2_color, region_bl_nn.tl(), region_bl_nn.br(), cv::Scalar(0, 255, 255), 3);
            cv::rectangle(img2_color, region_br_nn.tl(), region_br_nn.br(), cv::Scalar(0, 255, 255), 3);
        }
    }
    
    cv::drawMatches(img1_color, kpts1_warp_kp, img2_color, kpts2, matches_, img_matches_warp, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow( "Good Matches with Perspective Transform",  img_matches_warp );
    cv::waitKey();
    
    if(count >= 5){
        *is_match = 0;
        return 1;
    }
    
    return 0;
}

int PairWiseValidationGPU::FilterByDist(vector<vector<cv::DMatch> >& matches, \
                                        list<cv::DMatch>* good_matches) {
    if(good_matches == nullptr) {
        std::cout << "good_matches is null" << std::endl;
        return -1;
    }
    good_matches->clear();
    const float dist_ratio = 0.9;
    for(auto i = matches.begin(); i != matches.end(); i++) {
        //LOG(INFO) << "matches list" << ":" << i->size();
        //LOG(INFO) << "queryid:" << (*i)[0].queryIdx << ",trainidx:" << (*i)[0].trainIdx;
        //LOG(INFO) << "queryid:" << (*i)[1].queryIdx << ",trainidx:" << (*i)[1].trainIdx;
        //LOG(INFO) << "(dist0, dist1) : (" << (*i)[0].distance << "," << (*i)[1].distance << ")";
        //std::cout << (*i)[0].distance << " " << (*i)[1].distance << std::endl;
        if((*i)[0].distance > (*i)[1].distance * dist_ratio) {
            i->clear();
        }
    }
    float min, max;
    min = 1.0;
    max = -1.0;
    for(auto i = matches.begin(); i != matches.end(); i++) {
        if(i->size() == 2) {
            if((*i)[0].distance < min) {
                min = (*i)[0].distance;
            }
            if((*i)[1].distance > max) {
                max = (*i)[1].distance;
            }
        }
    }
    float dist_threshold = 0.0;
    if(min * 3 < 0.2) {
        dist_threshold = 0.2;
    }else if( min * 3 < 0.6 ){
        dist_threshold = min * 3;
    }else {
        dist_threshold = 0.6;
    }
    if (multiple_frames != 2)
    {
        dist_threshold = 0.6;
    }
    std::cout << "the dist threshold is: " << dist_threshold << std::endl;
    std::cout << "min, max, threshold: " << min << ", " << max << ", " << dist_threshold << std::endl;
    for(auto i = matches.begin(); i != matches.end(); i++) {
        if(i->size() == 2) {
            //std::cout << "the distance is: " << (*i)[0].distance << dist_threshold << std::endl;
            if((*i)[0].distance < dist_threshold * (*i)[1].distance) {
                good_matches->push_back((*i)[0]);
            }
        }
    }
    return 0;
}
