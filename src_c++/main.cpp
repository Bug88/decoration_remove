
#include <iostream>
#include <fstream>
#include "pairwise_validation.hpp"
//#include "ColorCorrelogram.h"
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

typedef struct _TestCase {
    std::string file1;
    std::string file2;
    int is_match;
}TestCase;

int LoadTestCase(std::string& filename, std::vector<TestCase>* cases) {
    if(cases == NULL) {
        std::cout << "case is NULL" << std::endl;
        return -1;
    }
    FILE* fp = fopen(filename.c_str(), "r");
    if(fp == NULL) {
        std::cout << "Failed in opening the testcase file: " << filename << std::endl;
        return -2;
    }
    TestCase unit;
    char id1[64];
    char id2[64];
    int is_match;
    while(!feof(fp)) {
        if(fscanf(fp, "%s %s %d", id1, id2, &is_match) == 3) {
            unit.file1 = std::string(id1);
            unit.file2 = std::string(id2);
            unit.is_match = is_match;
            cases->push_back(unit);
        }else {
            break;
        }
    }
    fclose(fp);
    return 0;
}

int LoadFileContent(std::string& filename, std::vector<char>* content) {
    std::ifstream ifs(filename.c_str());
    content->clear();
    content->assign((std::istreambuf_iterator<char>(ifs)),
                    (std::istreambuf_iterator<char>()));
    if (content->size() <= 0) {
        std::cout << "\nFailed in opening & reading file:\t" << filename << std::endl;
        return -1;
    }
    return 0;
    
    //    std::ifstream file("img.jpg");
    //    std::vector<char> data;
    //
    //    file >> std::noskipws;
    //    std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(), std::back_inserter(data));
}

// video validator with multiple frames

extern void calColorCorrelogram(const IplImage * img,double * correlogram);

typedef struct
{
    double feature[64]={0.0};
} globalFeature;

std::vector<globalFeature> Get_video_feats(std::vector<cv::Mat> video)
{
    std::vector<globalFeature> videoFeats;
    long nFrame = video.size();
    
    for(int i=0; i < nFrame; i ++)
    {
        IplImage img = video[i];
        globalFeature pDes;
        
        if(img.width != 0)
        {
            cv::Mat dst = cv::Mat::zeros(128, 128, CV_8UC3);
            
            //std::cout << video[i].size() << " " << dst.size() << std::endl;
            
            resize(video[i], dst, dst.size());
            
            img = dst;
            
            IplImage* gray = cvCreateImage(cvGetSize(&img),IPL_DEPTH_8U,1);
            cvCvtColor(&img, gray, CV_RGB2GRAY);
            
            calColorCorrelogram(gray, pDes.feature);
            
            cvReleaseImage(&gray);
        }
        //cvReleaseImage(&img2);
        videoFeats.push_back(pDes);
    }
    
    return videoFeats;
}


std::vector<int> Unique_Frames(std::vector<cv::Mat> all_frames, std::vector<globalFeature> video_feats)
{
    std::vector<int> unique_frame_ids;
    //std::vector<cv::Mat> unique_frames;
    std::vector<globalFeature> unique_features;
    
    for(int i=0; i < all_frames.size(); i ++)
    {
        globalFeature cur_ft = video_feats[i];
        
        bool eq = true;
        
        for(int j=0; j < unique_frame_ids.size(); j ++)
        {
            globalFeature q_ft = unique_features[j];
            
            double dis = 0.0;
            
            for(int m=0; m < 64; m ++)
            {
                dis += (cur_ft.feature[m] - q_ft.feature[m])*(cur_ft.feature[m]-q_ft.feature[m]);
            }
            
            if(dis <= 0.01)
            {
                eq = false;
                break;
            }
            else
            {
                eq = true;
            }

        }
        
        if(eq)
        {
            //unique_frames.push_back(all_frames[i].clone());
            unique_features.push_back(cur_ft);
            unique_frame_ids.push_back(i);
        }
    }
    
    //return unique_frame_ids;
    
    std::vector<int> unique_frame_ids_2;

    for(int i=0; i < unique_frame_ids.size(); i ++)
    {
        double md = 0.0;
        int idx = unique_frame_ids[i];
        globalFeature t = video_feats[idx];
        int value = 0;
        for(int m=0; m < 64; m ++)
        {
            md += t.feature[m];
            if(t.feature[m] > 0.0)
            {
                value ++;
            }
        }

        if(value > 32)
        {
            unique_frame_ids_2.push_back(idx);
        }
    }

    return unique_frame_ids_2;
}

std::vector<int> Cross_frame_validation(std::vector<globalFeature> video_1, std::vector<globalFeature> video_2)
{
    long nFrame_1, nFrame_2;
    nFrame_1 = video_1.size();
    nFrame_2 = video_2.size();
    
    std::vector<int> matches;
    
    for(int i=0; i < nFrame_1; i ++)
    {
        
        globalFeature pDes_1 = video_1[i];
        
        std::vector<std::pair<float, int>> dis;
        
        for(int j=0; j < nFrame_2; j ++)
        {
            globalFeature pDes_2 = video_2[j];
            
            // compute the similarity between frames
            float c_dis = 0;
            for(int m=0; m < 64; m ++)
            {
                float t_dis = pDes_1.feature[m] - pDes_2.feature[m];
                c_dis += t_dis * t_dis;
            }
            
            //std::cout << c_dis << std::endl;
            dis.push_back(std::pair<float, int>(c_dis, j));
            
        }
        
        std::sort(dis.begin(), dis.end(),
                  [](const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
                  {return lhs.first < rhs.first;});
        
        matches.push_back(dis[0].second);
       
    }
    
    return matches;
}

int main(int argc, const char * argv[]) {
    
    long id1, id2, label;
    char c;
    int Thred = 4;
    int max_frames = 10;
    int mframeNum = 20;
    int boundframeNum = 100;
    
    for(int mf=5; mf <=5; mf += 5)
    {
        max_frames = mf;
        
        for(int td = 2; td <=2; td += 2)
        {
            long long time_start, time_end;
            time_start = cvGetTickCount();
            
            Thred = td;
            std::ifstream infile;
            std::ofstream outfile;
            std::string dir = "/Users/liuzhen-mac/Desktop/video_dir/data_valid/";
            infile.open(dir + "dup.txt", std::ios::in);
            outfile.open(dir + std::to_string(mf) + "_" + std::to_string(td) + "_" + "videos_result.txt", std::ios::out);
            
            
            while (infile >> id1 >> c >> id2 >> c >> label)
            {
                std::string video_name_1 = std::to_string(id1);
                std::string video_name_2 = std::to_string(id2);
                std::string video_path_1 = dir + video_name_1 + std::string(".mp4");
                std::string video_path_2 = dir + video_name_2 + std::string(".mp4");
                
                cv::VideoCapture capture_1(video_path_1);
                cv::VideoCapture capture_2(video_path_2);
                
                if(!capture_1.isOpened())
                {
                    std::cout << "failed to open video!" << std::endl;
                }
                if(!capture_2.isOpened())
                {
                    std::cout << "failed to open video!" << std::endl;
                }
                
                
                long nFrame_1=static_cast<long>(capture_1.get(CV_CAP_PROP_FRAME_COUNT));
                long nFrame_2=static_cast<long>(capture_2.get(CV_CAP_PROP_FRAME_COUNT));
                
                cv::Mat frame_1, frame_2;
                
                int matched_frame_num = 0;
                
                std::vector<cv::Mat> all_frames_1, all_frames_2;
                std::vector<cv::Mat> unique_frames_1, unique_frames_2;
                
                long extFrame = mframeNum;
                
                if(nFrame_1 > boundframeNum)
                {
                    extFrame = int( nFrame_1 / mframeNum);
                }
                else
                {
                    extFrame = 1;
                }
                
                float poins = 1 / float(extFrame);
                int q = 0;
                for(int i=0; i < nFrame_1; i ++)
                {
                    capture_1 >> frame_1;
                    if(i % extFrame == 0)
                    {
                        
                        all_frames_1.push_back(frame_1.clone());
//                        IplImage img = all_frames_1[q];
//
//                        cvNamedWindow("test");
//                        cvShowImage("test", &img);
//                        cvWaitKey();
//                        cvDestroyWindow("test");
//
//                        q = q + 1;
                    }
                }
                
                //std::cout << all_frames_1.size() << std::endl;
                
                std::vector<globalFeature> videoFeats_1;
                videoFeats_1 = Get_video_feats(all_frames_1);
                
                //std::cout << videoFeats_1.size() << std::endl;
                
                std::vector<int> cur_ids;
                cur_ids = Unique_Frames(all_frames_1, videoFeats_1);
                
                std::vector<globalFeature> unique_feats_1;
                
                for(int i=0; i < cur_ids.size(); i ++)
                {
                    unique_frames_1.push_back(all_frames_1[cur_ids[i]].clone());
                    unique_feats_1.push_back(videoFeats_1[cur_ids[i]]);
                    
//                    IplImage img = unique_frames_1[i];
//
//                    cvNamedWindow("test");
//                    cvShowImage("test", &img);
//                    cvWaitKey();
//                    cvDestroyWindow("test");
                    
                }
                
                //std::cout << unique_frames_1.size() << " " << cur_ids.size() << std::endl;
                
                extFrame = mframeNum;
                
                if(nFrame_2 > boundframeNum)
                {
                    extFrame = int( nFrame_2 / mframeNum);
                }
                else
                {
                    extFrame = 1;
                }
                
                poins = 1 / float(extFrame);
                
                //std::cout << extFrame << " " << std::endl;
                 q = 0;
                for(int i=0; i < nFrame_2; i ++)
                {
                    capture_2 >> frame_2;
                    if(i%extFrame == 0)
                    {
                        all_frames_2.push_back(frame_2.clone());
                        
//                        IplImage img = all_frames_2[q];
//                        
//                        cvNamedWindow("test");
//                        cvShowImage("test", &img);
//                        cvWaitKey();
//                        cvDestroyWindow("test");
//                        
//                        q = q + 1;

                        
                    }
                }
                
                std::vector<globalFeature> videoFeats_2;
                videoFeats_2 = Get_video_feats(all_frames_2);
                
                cur_ids = Unique_Frames(all_frames_2, videoFeats_2);
                
                std::vector<globalFeature> unique_feats_2;
                
                for(int i=0; i < cur_ids.size(); i ++)
                {
                    unique_frames_2.push_back(all_frames_2[cur_ids[i]].clone());
                    unique_feats_2.push_back(videoFeats_2[cur_ids[i]]);
                    
//                    IplImage img = unique_frames_2[i];
//                    
//                    cvNamedWindow("test");
//                    cvShowImage("test", &img);
//                    cvWaitKey();
//                    cvDestroyWindow("test");
                    
                    
                }
                
                std::cout << all_frames_1.size() << " " << all_frames_2.size() << std::endl;
                std::cout << nFrame_1 << " " << nFrame_2 << std::endl;
                
                nFrame_1 = unique_frames_1.size();
                nFrame_2 = unique_frames_2.size();
                long nFrame = nFrame_1 > nFrame_2 ? nFrame_1 : nFrame_2;
                
                nFrame = max_frames < nFrame ? max_frames : nFrame;
                
                if(nFrame_1 == 0 or nFrame_2 == 0)
                {
                    nFrame = 0;
                }
                
                std::cout << nFrame_1 << " " << nFrame_2 << std::endl;
                
                std::vector<cv::Mat> u1;
                std::vector<cv::Mat> u2;
                std::vector<globalFeature> g1;
                std::vector<globalFeature> g2;
                
                if(nFrame_1 < nFrame_2)
                {
                    u1.swap(unique_frames_2);
                    u2.swap(unique_frames_1);
                    g1.swap(unique_feats_2);
                    g2.swap(unique_feats_1);
                }
                else
                {
                    u1.swap(unique_frames_1);
                    u2.swap(unique_frames_2);
                    g1.swap(unique_feats_1);
                    g2.swap(unique_feats_2);
                }
                
                std::vector<int> matches;
                
                if(nFrame > 0)
                {
                    matches = Cross_frame_validation(g1, g2);
                }
                else
                {
                    std::cout << id1 << " " << id2 << " " << std::endl;
                }
                
                for(int i=0; i < nFrame; i++)
                {
                    //capture_1 >> frame_1;
                    //capture_2 >> frame_2;
                    
                    //frame_1 = unique_frames_1[i];
                    //frame_2 = unique_frames_2[matches[i]];
                    //frame_2 = unique_frames_2[i];
                    frame_1 = u1[i];
                    frame_2 = u2[matches[i]];
                    
                    std::vector<int> compression_params;
                    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
                    compression_params.push_back(100);
                    std::string file1 = dir + video_name_1 + std::string("_0.jpg");
                    std::string file2 = dir + video_name_2 + std::string("_0.jpg");
                    
                    cv::imwrite(file1, frame_1, compression_params);
                    cv::imwrite(file2, frame_2, compression_params);
                    
                    
                    TestCase FLAGS_testcase;
                    FLAGS_testcase.file1 = file1;
                    FLAGS_testcase.file2 = file2;
                    FLAGS_testcase.is_match = -1;
                    
                    std::vector<TestCase> cases;
                    cases.push_back(FLAGS_testcase);
                    
                    PairWiseValidation* v = PairWiseValidationFactory::Create("");
                    int test_pass, test_fail;
                    test_pass = test_fail = 0;
                    int is_match = -1;
                    for(int i = 0; i < cases.size(); i++) {
                        std::vector<char> buf1, buf2;
                        is_match = -1;
                        if(LoadFileContent(cases[i].file1, &buf1)) {
                            std::cout << "Failed in loading " << cases[i].file1 << std::endl;
                            //return -1;
                            continue;
                        }
                        if(LoadFileContent(cases[i].file2, &buf2)) {
                            std::cout << "Failed in loading " << cases[i].file2 << std::endl;
                            //return -2;
                            continue;
                        }
                        v->IsMatch(buf1, buf2, &is_match);
                        printf("\nimages matched: %s\n", is_match>0?"yes":"no");
                        
                        if(is_match > 0)
                        {
                            matched_frame_num ++;
                        }
                    }
                    delete v;
                    //return 0;
                }
                
                capture_1.release();
                capture_2.release();
                
                //Thred = int(0.4 * nFrame);
                if(nFrame == 1)
                {
                    if(matched_frame_num >= 1)
                    {
                        printf("\nvideos are matched!\n\n");
                        outfile << id1 << "," << id2 << "," << 1 << std::endl;
                    }
                    else
                    {
                        printf("\nvideos are not matched!\n\n");
                        outfile << id1 << "," << id2 << "," << 2 << std::endl;
                    }
                }
                else if(nFrame)
                {
                    if(matched_frame_num >= Thred)
                    {
                        printf("\nvideos are matched!\n\n");
                        outfile << id1 << "," << id2 << "," << 1 << std::endl;
                    }
                    else
                    {
                        printf("\nvideos are not matched!\n\n");
                        outfile << id1 << "," << id2 << "," << 2 << std::endl;
                    }
                }
                else
                {
                    printf("\nvideos are not matched!\n\n");
                    outfile << id1 << "," << id2 << "," << 2 << std::endl;
                }
                
                
                time_end = cvGetTickCount();
                
                std::cout << "time needed: " << double(time_end - time_start) << std::endl;
                
            }
            
            infile.close();
            outfile.close();
            
        }
        
    }
    
    return 0;
}




//int main(int argc, const char * argv[]) {
//
//    TestCase FLAGS_testcase;
//    FLAGS_testcase.file1 = "/Users/liuzhen-mac/Desktop/bdcases/3156146307_2.jpg";
//    FLAGS_testcase.file2 = "/Users/liuzhen-mac/Desktop/bdcases/3160892623_2.jpg";
//    FLAGS_testcase.is_match = -1;
//
//    std::vector<TestCase> cases;
//    cases.push_back(FLAGS_testcase);
//
//    PairWiseValidation* v = PairWiseValidationFactory::Create("");
//    int test_pass, test_fail;
//    test_pass = test_fail = 0;
//    int is_match = -1;
//    for(int i = 0; i < cases.size(); i++) {
//        std::vector<char> buf1, buf2;
//        is_match = -1;
//        if(LoadFileContent(cases[i].file1, &buf1)) {
//            std::cout << "Failed in loading " << cases[i].file1 << std::endl;
//            return -1;
//        }
//        if(LoadFileContent(cases[i].file2, &buf2)) {
//            std::cout << "Failed in loading " << cases[i].file2 << std::endl;
//            return -2;
//        }
//        v->IsMatch(buf1, buf2, &is_match);
//        printf("\nimages matched: %s\n", is_match>0?"yes":"no");
//    }
//    delete v;
//    return 0;
//}

