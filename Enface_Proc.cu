#include"Header_Enface.h"
void frameAvg(unsigned short* dev_multiFrameBuff,float* dev_displayBuff, int width, int height, int numberOfFrames, int frameNum)
{
	int numThreadsPerBlock = 256;

	dim3 dimBlockX(numThreadsPerBlock);
	dim3 dimGridX((width * height)/dimBlockX.x);
	avgKernel << <dimGridX, dimBlockX >> > (dev_multiFrameBuff, dev_displayBuff, frameNum, numberOfFrames, width * height);
	
}	

 


void writeMatToFile(cv::Mat m, const char* filename)
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;  return;
	}

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			fout << m.at<float>(i, j) << "\t";
		}
		fout << endl;
	}

	fout.close();
}

void enface(float* d_A, float* d_x, float*d_y, const int row, const int col)
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	float alf = 0.00102;
	float beta = 0;
	stat = cublasCreate(&handle);
	stat = cublasSgemv(handle, CUBLAS_OP_T, col, row, &alf, d_A, col, d_x, 1, &beta, d_y, 1);//swap col and row
	

	


	cublasDestroy(handle);
}