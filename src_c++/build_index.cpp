//
//  build_index.cpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/26.
//  Copyright © 2017年 willard. All rights reserved.
//

#include "build_index.hpp"

//#include <../faiss-master/IndexFlat.h>
//#include <../faiss-master/IndexIVFPQ.h>

index_details::index_details()
{
    this->d = 128;
    this->pFeatsD = new float[this->maxImgNum*this->maxFeatNum*this->d];
    this->pFeatNum = new int[this->maxImgNum];
    this->pImgIDs = new int[this->maxImgNum*this->maxFeatNum];
    this->pFeatPos = new float[this->maxImgNum*this->maxFeatNum*2];
    this->pFilename = new std::string [this->maxImgNum];
    
}
index_details::~index_details()
{
    if(this->pFeatsD != nullptr)
    {
        delete [] this->pFeatsD;
    }
    if(this->pFeatPos != nullptr)
    {
        delete []pFeatPos;
    }
    if(this->pImgIDs != nullptr)
    {
        delete []pImgIDs;
    }
    if(this->pFeatNum != nullptr)
    {
        delete []pFeatNum;
    }
    if(this->pFilename != nullptr)
    {
        delete []pFilename;
    }
}
void index_details::save_to_file(std::string filepath)
{
    ofstream outfile(filepath);
    
    outfile << this->d << " " << this->dataDir << " " << this->maxImgNum << " "
    << this->maxFeatNum << " " << this->imgNum << " " << this->featNum << " "
    << this->database_dir << " " << this->sub_dir << " " << this->filenames << " ";
    
    for(int i=0; i < this->imgNum; i ++)
    {
        outfile << this->pFilename[i] << " ";
    }
    
    for(int i=0; i < this->imgNum; i ++)
    {
        outfile << this->pFeatNum[i] << " ";
    }
    
    for(int i=0; i < this->featNum; i ++)
    {
        outfile << this->pFeatPos[2*i+0] << " "
        << this->pFeatPos[2*i+1] << " ";
    }
    
    for(int i=0; i < this->featNum; i ++)
    {
        outfile << this->pImgIDs[i] << " ";
    }
    
    int fn = 0;
    for(int i=0; i < this->imgNum; i ++)
    {
        for(int k=0; k < this->pFeatNum[i]; k ++)
        {
            for(int j=0; j < this->d; j ++)
            {
                outfile << this->pFeatsD[(fn+k)*this->d + j] << " ";
            }
        }
        fn += this->pFeatNum[i];
    }
    
    outfile.close();
}
void index_details::load_from_file(std::string filepath)
{
    if(this->pFeatsD != nullptr)
    {
        delete [] this->pFeatsD;
    }
    if(this->pFeatPos != nullptr)
    {
        delete []pFeatPos;
    }
    if(this->pImgIDs != nullptr)
    {
        delete []pImgIDs;
    }
    if(this->pFeatNum != nullptr)
    {
        delete []pFeatNum;
    }
    if(this->pFilename != nullptr)
    {
        delete []pFilename;
    }
    
    ifstream infile(filepath);
    char ct;
    infile >> this->d;
    infile >> this->dataDir;
    int mn;
    infile >> mn;
    infile >> mn;
    infile >> this->imgNum;
    infile >> this->featNum;
    infile >> this->database_dir;
    infile >> this->sub_dir;
    infile >> this->filenames;
    
    this->pFilename = new std::string[this->imgNum];
    for(int i=0; i < this->imgNum; i ++)
    {
        infile >> this->pFilename[i];
    }
    
    this->pFeatNum = new int[this->imgNum];
    for(int i=0; i < this->imgNum; i ++)
    {
        infile >> this->pFeatNum[i];
    }
    
    this->pFeatPos = new float[this->featNum*2];
    for(int i=0; i < this->featNum; i ++)
    {
        infile >> this->pFeatPos[2*i+0];
        infile >> this->pFeatPos[2*i+1];
    }
    
    this->pImgIDs = new int[this->featNum];
    for(int i=0; i < this->featNum; i ++)
    {
        infile >> this->pImgIDs[i];
    }
    
    this->pFeatsD = new float[this->featNum*this->d];
    int fn = 0;
    for(int i=0; i < this->imgNum; i ++)
    {
        for(int k=0; k < this->pFeatNum[i]; k ++)
        {
            for(int j=0; j < this->d; j ++)
            {
                infile >> this->pFeatsD[(fn+k)*this->d + j];
            }
        }
        fn += this->pFeatNum[i];
    }
    
    
    infile.close();
}
void index_details::set_database_info(std::string database_dir, std::string sub_dir, std::string filenames)
{
    this->dataDir = database_dir;
    this->sub_dir = sub_dir;
    this->filenames = filenames;
    this->database_dir = database_dir;
}
void index_details::build_index()
{
    //index_details details;
    float * pFeatsD = this->pFeatsD;
    int *pFeatNum = this->pFeatNum;
    int *pImgID = this->pImgIDs;
    float *pFeatPos = this->pFeatPos;
    std::string *pFilename = this->pFilename;
    
    int numPt, dim;
    dim = this->d;
    numPt = 0;
    
    // build index for the database
    ifstream input;
    input.open(this->dataDir + this->filenames, std::ios::in);
    std::string filename;
    
    int imgN = 0;
    while(input >> filename)            // load in all the images
    {
        cv::Ptr<cv::Feature2D> surf;
        vector<cv::KeyPoint> kpts1;
        cv::Mat dsp1;
        surf = cv::xfeatures2d::SIFT::create();
        
        cv::Mat image;
        std::cout << this->dataDir + this->sub_dir + filename << std::endl;
        image = cv::imread(this->dataDir + this->sub_dir + filename, CV_LOAD_IMAGE_COLOR);
        surf->detectAndCompute(image, cv::Mat(), kpts1, dsp1);
        
        int qt = 0;
        for(auto i=kpts1.begin(); i!=kpts1.end(); i++)
        {
            for(int j=0; j < dim; j ++)
            {
                pFeatsD[numPt*dim+j] = dsp1.at<float>(qt, j);  // record all the features
            }
            qt ++;
            pFeatPos[numPt*2+0] = i->pt.x; // record all the feature positions
            pFeatPos[numPt*2+1] = i->pt.y;
            
            pImgID[numPt] = imgN; // record the image id for each feature
            
            numPt ++;
        }
        
        pFeatNum[imgN] = int(kpts1.size()); // record the feature number
        pFilename[imgN] = filename; // record the file name
        
        imgN ++;
    }
    
    this->imgNum = imgN;
    this->featNum = numPt;
    
    return;
}
void index_details::set_index_params(int dim)
{
    this->d = dim;
}


