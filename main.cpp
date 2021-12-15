#include "Header_Enface.h"
//GLuint pbo;
//GLuint tex;
//struct cudaGraphicsResource* cuda_resource;
//***********************************************************
//  initCUDADevice uses CUDA commands to initiate the CUDA
//  enabled graphics card. This is prior to resource mapping,
//  and rendering.
//***********************************************************

//void initCUDADevice() {
//
//	cudaGLSetGLDevice;
//
//}

int main(int arg,char **argv)
{
	char* filename = new char[50];
	filename = "BMImage_975x500x384.bin";
	int host_fileLen;
	cudaError_t err = cudaSuccess;
	//Read File
	FILE* file = fopen(filename, "rb");
	if (file == NULL)
	{
		printf("Unable to Open File\n");
		exit(1);
	}
	fseek(file, 0, SEEK_END);
	host_fileLen = ftell(file) / (int)sizeof(unsigned short);
	printf("Host Memory File Length:%d\n", host_fileLen);
	rewind(file);

	//Memory alloction for Host array
	unsigned short* h_Image_Stack;
	h_Image_Stack = (unsigned short*)malloc(host_fileLen * sizeof(unsigned short));
	fread(h_Image_Stack, 1, host_fileLen * sizeof(unsigned short), file);
	fclose(file);


	//Check all the Image Values
	//for (int i = 0; i < host_fileLen; i++)
	//{
		//printf("%2u\n", h_Image_Stack[i]);
	//}

	//Memory Allocation for Device Array

	unsigned short* d_volumeBuffer = NULL;
	 

	int volumeWidth = 500;
	int bscanHeight = 975;
	int frames = 384;

	// Device Memory Allocation for Input Volume Frames 
	err = cudaMalloc((void**)&d_volumeBuffer, volumeWidth * bscanHeight * frames * sizeof(unsigned short));
	cudaMemset(d_volumeBuffer, 0, volumeWidth * bscanHeight * frames * sizeof(unsigned short));

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Memory(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//

	err = cudaMemcpy(d_volumeBuffer, h_Image_Stack, volumeWidth * bscanHeight * frames * sizeof(unsigned short), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector  from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	//Average Frames

	//Device Memory allocation for Averaged data
	float* d_Average = NULL;
	err = cudaMalloc((void**)&d_Average, volumeWidth * bscanHeight*128* sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Memory(error code % s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//cudaMemset(d_Average, 0, volumeWidth * bscanHeight * (frames/3) * sizeof(unsigned short));
	int ctr = 0;

	float* y_gpu = NULL;
	err = cudaMalloc((void**)&y_gpu, volumeWidth* 128 * sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device Memory(error code % s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	 
	for (int i = 0; i < frames; i += 3)
	{
			
	frameAvg(&d_volumeBuffer[i * volumeWidth * bscanHeight], &d_Average[ctr * volumeWidth * bscanHeight], volumeWidth, bscanHeight, 3,0);
	ctr++;
			
	}


	//Enface Projection
		
	float* x = new float[bscanHeight];
	for (int i = 0; i < bscanHeight; i++)
	{
		x[i] = 1;
	}


	cudaError_t cudastat;
	cublasStatus_t stat;
	int size = bscanHeight * volumeWidth*128;
	//cublasHandle_t handle;
	 
	float* d_x;  //device vector
	float* d_y;  //device result
 
	cudastat = cudaMalloc((void**)&d_x, bscanHeight * sizeof(float));
	cudastat = cudaMalloc((void**)&d_y, volumeWidth * 128*sizeof(float));


	cudaMemcpy(d_x, x, sizeof(float) * bscanHeight, cudaMemcpyHostToDevice);   //copy x to device d_x


	 

	for (int i = 0; i < 128; i++)
	{

		enface(&d_Average[i * bscanHeight * volumeWidth], d_x, &d_y[i*volumeWidth], volumeWidth,bscanHeight);
		//printf("Hi");	
	}



	//Device to Host Memory Transfer of Average Data 
		
	float* h_Average = (float*)malloc(volumeWidth * 128 * sizeof(float));
	cudaMemcpy(h_Average, d_y, volumeWidth * 128 * sizeof(float), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < 10; i++)
	//{
	//	printf("%f\n", h_Average[i]);
	//}

	const char* fileName = "C:\\Users\\AMD-PC-09\\source\\repos\\Enface_Projection\\TEST.txt";
	Mat Image(128,volumeWidth, CV_32FC1);
	memcpy(Image.data, h_Average, 128 * volumeWidth *sizeof(float));
	//writeMatToFile(Image, fileName);
	 
	double min2, max2;
	minMaxLoc(Image, &min2, &max2);

	double divVal = 255 / max2;
	Mat ConvImage = Image * divVal;
	ConvImage.convertTo(ConvImage, CV_8UC1);
	Mat data_Image(128, 500, CV_8UC1);
	data_Image = ConvImage.clone();
	imshow("Image",data_Image);
	waitKey(0);

	/*int OpenGL = 0;
	if (OpenGL == 1)
	{*/
		//OpenGL Functions
		 // ====================================================================================
		// allocate the PBO, texture, and CUDA resource

		 // Clear Color and Depth Buffers
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//// Reset transformations
		//glLoadIdentity();

		//// ====================================================================================
		//// initiate GPU by setting it correctly 
		//initCUDADevice();


		//// Generate a buffer ID called a PBO (Pixel Buffer Object)
		//glGenBuffers(1, &pbo);// Generate a buffer ID called a PBO (Pixel Buffer Object)

		//// Make this the current UNPACK buffer (OpenGL is state-based)
		//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		//// Allocate data for the buffer. 4-channel 8-bit image
		//glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(unsigned char) * data_Image.rows * data_Image.cols, NULL, GL_STREAM_DRAW);
		//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		//cudaGraphicsGLRegisterBuffer(&cuda_resource, pbo, cudaGraphicsMapFlagsNone);
		//// create the texture object

		//// enable 2D texturing
		//glEnable(GL_TEXTURE_2D);

		//// generate and bind the texture    
		//glGenTextures(1, &tex);
		//glBindTexture(GL_TEXTURE_2D, tex);

		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

		//// put flipped.data at the end for cpu rendering 
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, data_Image.cols, data_Image.rows, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);

		//// put tex at the end for cpu rendering 
		//glBindTexture(GL_TEXTURE_2D, 0);




		//// copy OpenCV flipped image data into the device pointer

		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//unsigned char* dev_inp;

		////gpuErrchk( cudaMalloc((void**)&dev_inp, sizeof(unsigned char)*flipped.rows*flipped.cols) );

		//cudaGraphicsMapResources(1, &cuda_resource, 0);

		//size_t size1;
		//cudaGraphicsResourceGetMappedPointer((void**)&dev_inp, &size1, cuda_resource);

		//cudaMemcpy(dev_inp, data_Image.data, sizeof(unsigned char) * data_Image.rows * data_Image.cols, cudaMemcpyHostToDevice);

		//cudaGraphicsUnmapResources(1, &cuda_resource, 0);

		//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		////
		//glBindTexture(GL_TEXTURE_2D, tex);

		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data_Image.cols, data_Image.rows, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

		//cudaGraphicsUnregisterResource(cuda_resource);
		//cudaThreadSynchronize();

	//}






	free(h_Average);
	cudaFree(d_Average);
	cudaFree(d_x);
	cudaFree(d_y);
	return 0;

}



