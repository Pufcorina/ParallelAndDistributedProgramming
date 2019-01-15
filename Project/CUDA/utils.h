#ifndef __CVL_UTILS_H__
#define __CVL_UTILS_H__

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <cmath>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "timer.h"

cv::Mat imageRGBA;
cv::Mat imageFilter;

uchar4        *d_rgbaImage__;
unsigned char *d_filterImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}


//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **filterImage,
	uchar4 **d_rgbaImage, unsigned char **d_filterImage,
	const std::string &filename) {
	//make sure the context initializes ok
	checkCudaErrors(cudaFree(0));

	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);  // CV_BGR2GRAY

	//allocate memory for the output
	imageFilter.create(image.rows, image.cols, CV_8UC1);

	//This shouldn't ever happen given the way the images are created
	//at least based upon my limited understanding of OpenCV, but better to check
	if (!imageRGBA.isContinuous() || !imageFilter.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*filterImage = imageFilter.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();
	//allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_filterImage, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_filterImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

	//copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_filterImage__ = *d_filterImage;
}

void postProcess(const std::string& output_file, unsigned char* data_ptr) {
	cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);

	//output the image
	cv::imwrite(output_file.c_str(), output);
}

void cleanupCuda()
{
	//cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_filterImage__);
	cudaDeviceReset();
}

#endif