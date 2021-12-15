#pragma once
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include<conio.h>
#include<string.h>
#include <opencv2/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/opencv.hpp>
#include<iostream>
#include<fstream>

#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

//#include <glew.h>
//#include <glut.h>
//#include <cuda_gl_interop.h> 
// 
using namespace std;
using namespace cv;
//Frame Average Kernel
void frameAvg(unsigned short* dev_multiFrameBuff, float* dev_displayBuff, int width, int height, int numberOfFrames, int frameNum);
__global__ void avgKernel(unsigned short* src_Buffer, float* dst_Buffer, int frame_num, int num_Frames, int frameSize);

//int GpuVec(const float* A, const float* x, float* y, const int row, const int col);
void writeMatToFile(cv::Mat m, const char* filename);

void enface(float* d_A, float* d_x, float* d_y, const int row, const int col);