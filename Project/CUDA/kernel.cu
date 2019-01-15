
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include "utils.h"
#include "timer.h"
#include "grayscale.cuh"


void processUsingCudaGrayscale(std::string input_file, std::string output_file);
void processUsingCudaGaussianBlur(std::string input_file, std::string output_file);


int main()
{
	std::string input_file1 = "../data/animal-beagle-canine-2048x1174.jpg";
	std::string input_file2 = "../data/pexels-photo-640x336.jpeg";
	std::string input_file3 = "../data/pexels-photo-1280x733.jpeg";
	std::string output_cuda_file_grayscale1 = "../data/output_cuda_grayscale1.png";
	std::string output_cuda_file_grayscale2 = "../data/output_cuda_grayscale2.png";
	std::string output_cuda_file_grayscale3 = "../data/output_cuda_grayscale3.png";

	std::string output_cuda_file_gaussianBlur1 = "../data/output_cuda_gaussianBlur1.png";
	std::string output_cuda_file_gaussianBlur2 = "../data/output_cuda_gaussianBlur2.png";
	std::string output_cuda_file_gaussianBlur3 = "../data/output_cuda_gaussianBlur3.png";

	processUsingCudaGrayscale(input_file1, output_cuda_file_grayscale1);
	processUsingCudaGrayscale(input_file2, output_cuda_file_grayscale2);
	processUsingCudaGrayscale(input_file3, output_cuda_file_grayscale3);

	processUsingCudaGaussianBlur(input_file1, output_cuda_file_gaussianBlur1);
	processUsingCudaGaussianBlur(input_file2, output_cuda_file_gaussianBlur2);
	processUsingCudaGaussianBlur(input_file3, output_cuda_file_gaussianBlur3);

	cleanupCuda();

    return 0;
}

void processUsingCudaGrayscale(std::string input_file, std::string output_file) {
	// pointers to images in CPU's memory (h_) and GPU's memory (d_)
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;

	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	GpuTimer timer;
	timer.Start();
	// here is where the conversion actually happens
	rgbaToGreyscaleCuda(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int err = printf("Implemented CUDA code ran in: %f msecs.\n", timer.Elapsed());

	if (err < 0) {
		//Couldn't print!
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}

	size_t numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	//check results and output the grey image
	postProcess(output_file, h_greyImage);
}

void processUsingCudaGaussianBlur(std::string input_file, std::string output_file) {
	// pointers to images in CPU's memory (h_) and GPU's memory (d_)
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;

	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	GpuTimer timer;
	timer.Start();
	// here is where the conversion actually happens
	rgbaToGaussianBlurCuda(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
	timer.Stop();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int err = printf("Implemented CUDA code ran in: %f msecs.\n", timer.Elapsed());

	if (err < 0) {
		//Couldn't print!
		std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
		exit(1);
	}

	size_t numPixels = numRows()*numCols();
	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	//check results and output the grey image
	postProcess(output_file, h_greyImage);
}