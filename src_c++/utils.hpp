

#ifndef utils_hpp
#define utils_hpp

#include <set>
#include <map>
#include <list>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


void ResizeImage(const cv::Mat& raw_img, cv::Mat &img, int kShorterEdge) {
    double ratio;
    if(raw_img.rows < raw_img.cols) {
        ratio = double(kShorterEdge / double(raw_img.rows));
    }else {
        ratio = double(kShorterEdge / double(raw_img.cols));
    }
    cv::Rect my_roi(int(raw_img.cols*1.0/8.0), int(raw_img.rows*1.0/8.0),  int(raw_img.cols*6.0/8.0), int(raw_img.rows*6.0/8.0));
    cv::Mat croped_img = raw_img(my_roi);
    cv::resize(croped_img, img, cv::Size(0,0), ratio, ratio);
}

template<typename Out>
void split(const std::string &s, char delim, Out result, std::string &frame_id) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, delim)) {
        if (i == 1) frame_id = item;
        if (i > 1) *(result++) = std::stof(item);
        ++i;
    }
}

std::vector<float> split(const std::string &s, char delim, std::string &frame_id) {
    std::vector<float> elems;
    split(s, delim, std::back_inserter(elems), frame_id);
    return elems;
}

std::vector<std::string> split(const String &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

// convert vector of vector to cv::Mat
cv::Mat convert2Mat(std::vector<std::vector<float> > &angles){
    cv::Mat matAngles((int)angles.size(), (int)angles.at(0).size(), CV_32FC1);
    for(int i=0; i<matAngles.rows; ++i)
        for(int j=0; j<matAngles.cols; ++j)
            matAngles.at<float>(i, j) = angles.at(i).at(j);
    return matAngles;
}

// read image from buffer
std::vector<char> read_to_buffer(std::string img_path){
    std::ifstream file(img_path);
    std::vector<char> data;
    file >> std::noskipws;
    std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(), std::back_inserter(data));
    return data;
}

#endif /* utils_hpp */
