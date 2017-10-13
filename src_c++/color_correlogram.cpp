//
//  color_correlogram.cpp
//  pairwise_matching_opencv3
//
//  Created by liuzhen-mac on 2017/9/12.
//  Copyright © 2017年 willard. All rights reserved.
//

#include <stdio.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>

#define uchar unsigned char
#define DISTANCE 8
#define GRAM_SIZE 64

void calColorCorrelogram(const IplImage * img,double * correlogram)
{
    for(int i=0; i < GRAM_SIZE; i ++)
    {
        correlogram[i] = 0.0;
    }
	if(NULL == img)
		return ;
	if(NULL == correlogram)
		return ;

	if(img->nChannels > 1)
		return ;

	//生成一个数组，每个点的范围是0-7
	int width = img->width;
	int height = img->height;
	int * mark = new int[width * height]; 
	int step = img->widthStep/sizeof(uchar);
	uchar * data = (uchar *)img->imageData;

	for (int i = 0;i < height;++i)
		for (int j = 0;j < width;++j)
			mark[i * width + j] = int(8 * data[i * width + j]  * 1.0 /256); 

	//计算相关图 DISTANCE = 8
	long grade,hist;
	long sum = 0;
	for (int i = 0;i < height - DISTANCE + 1;++i)
	{
		for (int j = 0;j < width - DISTANCE + 1;++j)
		{
			//获取当前点的灰度级，目前分8个等级 
			hist = mark[i * width + j];
			for (int k = 1;k < DISTANCE;k++)
			{
				for (int r = 1;r < DISTANCE;r++)
				{
					if(mark[(i + k) * width + j + r] == hist)
					{
						grade = k > r ? k : r;//获取距离---水平与垂直距离中较大的
						correlogram[hist * DISTANCE + grade - 1]++;
						sum++;
					}
				}
			}
		}
	}
	if(sum <= 0)
		return ;

	//计算特征向量 GRAM_SIZE = 64
	for(int i = 0;i < GRAM_SIZE;i++)
		correlogram[i] = correlogram[i] / sum;

	delete[] mark;
    
    return ;
}
