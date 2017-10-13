//
//  rejection.cpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/27.
//  Copyright © 2017年 willard. All rights reserved.
//
//

#include "rejection.hpp"

rejection::rejection()
{
    this->font_flag = true;
}
rejection::~rejection()
{}
void rejection::performRejection_dec()
{
    this->font_flag = false;
    matches *mt = this->pMatches;
    int idx;
    int tmp = 0;
    for(int i=0; i < mt->vMatchedPos.size(); i ++)
    {
        if(mt->vMatchedPos[i].size() > tmp)
        {
            idx = i;
            tmp = mt->vMatchedPos[i].size();
        }
    }
    // performing the feature rejection for each dataset image
    int fn = 0;
    for(int i=0; i < mt->vMatchedPos.size(); i ++)
    {
        if(i != idx)
        {
            fn += mt->pIndexDet->pFeatNum[i];
            continue;
        }
        // perform the rejection procedure for those fonts with more than 3 pairs matches
        std::vector<int> gd_map;
        if(mt->vMatchedPos[i].size() / 4.0 > this->mat_num_thred)
        {
            // all the feature points for the specific database image
            std::vector<float> all_pts;
            for(int j=0; j < mt->pIndexDet->pFeatNum[i]; j ++)
            {
                all_pts.push_back(mt->pIndexDet->pFeatPos[(fn+j)*2+0]);
                all_pts.push_back(mt->pIndexDet->pFeatPos[(fn+j)*2+1]);
            }
            
            std::cout << mt->pIndexDet->pFilename[i] << std::endl;
            gd_map = this->singleRejection_v2(mt->vMatchedPos[i], all_pts, image);
        }
        
        fn += mt->pIndexDet->pFeatNum[i];
        
        // show in the query image
        for(int j=0; j < gd_map.size(); j ++)
        {
            int tw, th;
            tw = int(j / this->grid_h);
            th = j % this->grid_h;
            
            if(gd_map[j] > this->thred)
            {
                cv::rectangle(image, cvPoint((th)*this->grid_width, (tw)*this->grid_width), cvPoint((th+1)*this->grid_width, (tw+1)*this->grid_width), cvScalar(255), 2, 8, 0);
                
                std::vector<float> pq;
                pq.push_back(th*this->grid_width);
                pq.push_back(tw*this->grid_width);
                pq.push_back((th+1)*this->grid_width);
                pq.push_back((tw+1)*this->grid_width);
                this->map.insert(pq);
            }
        }
    }
    
    return;
}
void rejection::performRejection_font()
{
    matches *mt = this->pMatches;
    // performing the feature rejection for each dataset image
    int fn = 0;
    for(int i=0; i < mt->vMatchedPos.size(); i ++)
    {
        // perform the rejection procedure for those fonts with more than 3 pairs matches
        std::vector<int> gd_map;
        if(mt->vMatchedPos[i].size() / 4.0 > this->mat_num_thred)
        {
            // all the feature points for the specific database image
            std::vector<float> all_pts;
            for(int j=0; j < mt->pIndexDet->pFeatNum[i]; j ++)
            {
                all_pts.push_back(mt->pIndexDet->pFeatPos[(fn+j)*2+0]);
                all_pts.push_back(mt->pIndexDet->pFeatPos[(fn+j)*2+1]);
            }
            
            std::cout << mt->pIndexDet->pFilename[i] << std::endl;
            gd_map = this->singleRejection_v2(mt->vMatchedPos[i], all_pts, image);
        }
        
        fn += mt->pIndexDet->pFeatNum[i];
        
        // show in the query image
        for(int j=0; j < gd_map.size(); j ++)
        {
            int tw, th;
            tw = int(j / this->grid_h);
            th = j % this->grid_h;
            
            if(gd_map[j] > this->thred)
            {
                cv::rectangle(this->image, cvPoint((th)*this->grid_width, (tw)*this->grid_width), cvPoint((th+1)*this->grid_width, (tw+1)*this->grid_width), cvScalar(255), 2, 8, 0);
                
                std::vector<float> pq;
                pq.push_back(th*this->grid_width);
                pq.push_back(tw*this->grid_width);
                pq.push_back((th+1)*this->grid_width);
                pq.push_back((tw+1)*this->grid_width);
                this->map.insert(pq);
            }
        }
    }
    
    //cv::Mat image = cv::imread("img.jpg")；
    
    return ;
}
std::vector<int> rejection::singleRejection_v2(std::vector<float> pos, std::vector<float> all_points, cv::Mat image)
{
    std::vector<int> empty;
    // filter out the complex font
    // 
//    if( pos.size() / (2 * all_points.size()) < this->ratio)
//    {
//        return empty;
//    }
    
    // all_points are the feature positions for the dataset image
    int w, h;
    w = this->grid_w;
    h = this->grid_h;
    
    if(this->numPt.size() > 0)
    {
        this->numPt.clear();
    }
    
    for(int i=0; i < w*h; i ++)
    {
        this->numPt.push_back(0);
    }
    
    // match_kpts1  the query points
    // match_kpts2  the database points
    std::vector<cv::Point2f> match_kpts1, match_kpts2;
    for(int i=0; i < pos.size() / 4; i ++)
    {
        cv::Point2f pt;
        
        pt.x = pos[i*4+0];
        pt.y = pos[i*4+1];
        match_kpts1.push_back(pt);
        
        pt.x = pos[i*4+2];
        pt.y = pos[i*4+3];
        match_kpts2.push_back(pt);
    }
    
    std::vector<cv::Point2f> all_points_2f;
    std::vector<cv::Point2f> trans_points_2f;
    for(int i=0; i < all_points.size() / 2; i ++)
    {
        cv::Point2f pt;
        pt.x = all_points[i*2+0];
        pt.y = all_points[i*2+1];
        
        all_points_2f.push_back(pt);
    }
    
    std::vector<unsigned char> inliner_mask(pos.size() / 4);
    cv::Mat H;
    
    H = cv::findHomography(match_kpts2, match_kpts1, CV_RANSAC, 1, inliner_mask);
    
    
    if(H.empty())
    {
        return empty;
    }
    
    cv::perspectiveTransform(all_points_2f, trans_points_2f, H);
    
//    std::cout << H.at<double>(0, 0) << " " << H.at<double>(0, 1) << " " << H.at<double>(0, 2) << " "
//    << H.at<double>(1, 0) << " " << H.at<double>(1, 1) << " " << H.at<double>(1, 2) << " "
//    << H.at<double>(2, 0) << " " << H.at<double>(2, 1) << " " << H.at<double>(2, 2) << std::endl;
    
    float min_x, max_x;
    float min_y, max_y;
    
    min_x = this->queryImgHeight;
    max_x = 0.0;
    min_y = this->queryImgWidth;
    max_y = 0.0;
    
    int length = trans_points_2f.size();
    
    for(int i=0; i < match_kpts1.size(); i ++)
    {
        trans_points_2f.push_back(match_kpts1[i]);
    }
    
    for(int i=0; i < trans_points_2f.size(); i ++)
    {
        cv::Point2f pt;
        pt.x = trans_points_2f[i].x;
        pt.y = trans_points_2f[i].y;
        
        if(i < length)
        {
            if(pt.x < min_x)
            {
                min_x = pt.x;
            }
            if(pt.x > max_x)
            {
                max_x = pt.x;
            }
            if(pt.y < min_y)
            {
                min_y = pt.y;
            }
            if(pt.y > max_y)
            {
                max_y = pt.y;
            }
        }
        
        int tw, th;
        
        th = int(pt.x / this->grid_width);
        tw = int(pt.y / this->grid_width);
        
        // if the homography is not valid, the image matches are false
        if(tw <=0 - this->error || tw >= this->grid_w + this->error)
        {
            //continue;
            return empty;
        }
        
        if(th <=0 - this->error || th >= this->grid_h + this->error)
        {
            //continue;
            return empty;
        }
        
        this->numPt[tw*this->grid_h + th] ++;
    }
    
    if(this->font_flag)
    {
        float dw, dh;
        dw = max_x - min_x;
        dh = max_y - min_y;

        if(dw > this->font_size)
        {
            return empty;
        }
        if(dh > this->font_size)
        {
            return empty;
        }
    }
    
    std::vector<int> ty(this->numPt);
    return ty;
}
void rejection::setQueryImgParams(cv::Mat image, matches *pMatches)
{
    this->queryImgWidth = image.rows; // actually the image rows
    this->queryImgHeight = image.cols; // actually the image cols
    this->image = image.clone();
    this->pMatches = pMatches;
    
    this->grid_w = this->queryImgWidth / this->grid_width;
    this->grid_h = this->queryImgHeight / this->grid_width;
}
