#include"Header_Enface.h"

__global__ void avgKernel(unsigned short* src_Buffer, float* dst_Buffer, int frame_num, int num_Frames, int frameSize)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	__shared__ unsigned short tempVal;
	tempVal = 0;
	for (int i = 0; i < num_Frames; i++)
	{
		tempVal += src_Buffer[(frame_num + i) * frameSize + idx];
	}

	dst_Buffer[idx] = (float)tempVal / num_Frames;
	 
}



 