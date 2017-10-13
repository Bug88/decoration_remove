//
//  matches.cpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/27.
//  Copyright © 2017年 willard. All rights reserved.
//

#include "matches.hpp"

matches::matches()
{
    pMatchedPos = nullptr;
    pPos = nullptr;
    pIndexDet = nullptr;
    pDes = nullptr;
    pIndex = nullptr;
    NN = 0;
    featNum = 0;
}
matches::~matches()
{
    NN = 0;
    featNum = 0;
    if(pMatchedPos != nullptr)
    {
        delete []pMatchedPos;
    }
    if(pPos != nullptr)
    {
        delete []pPos;
    }
    if(pIndexDet != nullptr)
    {
        delete []pIndexDet;
    }
    if(pDes != nullptr)
    {
        delete []pDes;
    }
    if(pIndex != nullptr)
    {
        delete []pIndex;
    }
}
void matches::setParams(int n, index_details* idxDt, cvflann::Index<cvflann::L2<float>>
                        *idx)
{
    NN = n;
    this->pIndexDet = idxDt;
    this->pIndex = idx;
    
    return;
}
cv::Mat matches::extractQueryFeats(cv::Mat imageT)
{
    cv::Mat image = imageT.clone();
    cv::Ptr<cv::Feature2D> surf;
    vector<cv::KeyPoint> kpts1;
    cv::Mat dsp1;
    //cv::Mat image;
    //image = imread(filename, CV_LOAD_IMAGE_COLOR);
    
    std::cout << image.size() << std::endl;
    surf = cv::xfeatures2d::SIFT::create();
    surf->detectAndCompute(image, cv::Mat(), kpts1, dsp1);
    
    int numPt = 0;
    int dim = this->pIndexDet->d;
    
    this->pDes = new float [kpts1.size()*dim];
    this->pPos = new float [kpts1.size()*2];
    
    for(auto i=kpts1.begin(); i!=kpts1.end(); i++)
    {
        for(int j=0; j < dim; j ++)
        {
            this->pDes[numPt*dim+j] = dsp1.at<float>(numPt, j);
        }
        
        this->pPos[numPt*2+0] = i->pt.x;
        this->pPos[numPt*2+1] = i->pt.y;
        
        cv::circle(image, i->pt, 1, cv::Scalar(255, 0, 0));
        
        numPt ++;
    }
    
    this->featNum = numPt;
    
    return image;
}
int matches::stepIdx(float * Data, int length)
{
    float *pDelta = new float[length];
    
    float max = 0.0;
    int pid = 0.0;
    for(int i=0; i < length-1; i ++)
    {
        pDelta[i] = Data[i+1] - Data[i];
        
        if(pDelta[i] > max)
        {
            max = pDelta[i];
            pid = i;
        }
    }
    
    delete []pDelta;
    
    return pid;
}
void matches::printMatchedImgs()
{
    int j = 0;
    for(auto i=vMatchedPos.begin(); i != vMatchedPos.end(); i ++)
    {
        if(i->size() > 4.0)
        {
            std::cout << "the matched feature points: " << std::endl;
            std::cout << float(i->size()) / 4.0 << this->pIndexDet->pFilename[j] << std::endl;
        }
        
        j ++ ;
    }
}
void matches::performQuery()
{
    int imgNum;
    imgNum = this->pIndexDet->imgNum;
    float *pScore = new float[imgNum];
    
    for(int i=0; i < imgNum; i ++)
    {
        pScore[i] = 0;
    }
    
    //std::vector<std::vector<float>> vMatchedPos;
    for(int i=0; i < imgNum; i ++)
    {
        std::vector<float> imgMat;
        vMatchedPos.push_back(imgMat);
    }
    // sanity check
    
    int nn = this->NN;
    //int k = this->featNum;
    
    int *I = nullptr; //new long[k * nn];
    float *D = nullptr; //new float[k * nn];
    
    // flann knn search
    //index.knnsearch(nn, this->pDes, k, D, I);
    cvflann::Matrix<int> indices(new int[this->featNum * nn], this->featNum, nn);
    cvflann::Matrix<float> dists(new float[this->featNum * nn], this->featNum, nn);
    cvflann::Matrix<float> query(this->pDes, this->featNum, this->pIndexDet->d);
    
    //this->pIndex.knnSearch();
    this->pIndex->knnSearch(query, indices, dists, nn, cvflann::SearchParams(256));
    
    I = indices.data;
    D = dists.data;
    
    int K = this->NN;//this->pIndexDet->k;
    int featNum = this->featNum;//this->pIndexDet->featNum;
    int k = this->NN;
    
    int *pImgID = this->pIndexDet->pImgIDs;
    
    // compute the score for each image
    for(int i=0; i < featNum; i ++)
    {
        int idx;
        //idx = this->stepIdx(&D[i*k], nn);
        idx = nn;
        
        if(idx > K)
        {
            idx = K;
        }
        
        if(D[i*k+0] > this->match_dis)
        {
            continue;
        }
        
        for(int j=0; j < idx; j ++)
        {
            int c_img;
            c_img = int(pImgID[I[i*k+j]]);
            pScore[c_img] = pScore[c_img] + 1;
        }
        
        for(int j=0; j < idx; j ++)
        {
            int c_idx, c_imgID;
            c_idx = I[i*k+j];
            c_imgID = int(pImgID[c_idx+0]);
            float x1, y1, x2, y2;
            x1 = this->pPos[i*2+0];
            y1 = this->pPos[i*2+1];
            x2 = this->pIndexDet->pFeatPos[c_idx*2+0];
            y2 = this->pIndexDet->pFeatPos[c_idx*2+1];
            vMatchedPos[c_imgID].push_back(x1);
            vMatchedPos[c_imgID].push_back(y1);
            vMatchedPos[c_imgID].push_back(x2);
            vMatchedPos[c_imgID].push_back(y2);
        }
    }
    
    
    delete [] I;
    delete [] D;
    
    delete []pScore;
}
