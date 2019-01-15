// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Green, and Blue is in it.
//The 'A' stands for Alpha and is used for transparency; it will be
//ignored in this project

//Each channel Red, Blue, Green, and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color. This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

#ifndef GRAYSCALE_H__
#define GRAYSCALE_H__

#include <stdio.h>
#include "utils.h"

__global__
void rgbaToGreyscaleCudaKernel(const uchar4* const rgbaImage,
	unsigned char* const greyImage,
	const int numRows, const int numCols)
{
	//First create a mapping from the 2D block and grid locations
	//to an absolute 2D location in the image, then use that to
	//calculate a 1D offset
	const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;

	if (pointIndex < numRows*numCols) { // this is necessary only if too many threads are started
		uchar4 const imagePoint = rgbaImage[pointIndex];
		greyImage[pointIndex] = .299f*imagePoint.x + .587f*imagePoint.y + .114f*imagePoint.z;
	}
}


// CUDA kernel which is run in parallel by many GPU threads.
__global__
void rgbaToGaussianBlurCudaKernel(const uchar4* const rgbaImage,
	unsigned char* const gaussianBlurImage,
	const int numRows, const int numCols, const int depth)
{
	//First create a mapping from the 2D block and grid locations
	//to an absolute 2D location in the image, then use that to
	//calculate a 1D offset
	const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;

	if (pointIndex < numRows*numCols) { // this is necessary only if too many threads are started
		int leftUpPoint = pointIndex - depth / 2 - depth / 2 * numCols;

		unsigned char summPoint = 0;
		for (int i = 0; i < depth; i++)
			for (int j = 0; j < depth; j++)
			{
				int currentIndex = leftUpPoint + numCols * i + j;
				if (currentIndex > 0 && currentIndex < numCols * numRows) {
					uchar4 const imagePoint = rgbaImage[currentIndex];
					summPoint += .299f*imagePoint.x + .587f*imagePoint.y + .114f*imagePoint.z;
				}
			}
		summPoint = summPoint / (depth * depth);
		
		
		gaussianBlurImage[pointIndex] = summPoint;
	}
}


// Parallel implementation for running on GPU using multiple threads.
void rgbaToGreyscaleCuda(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
	unsigned char* const d_greyImage, const size_t numRows, const size_t numCols)
{
	const int blockThreadSize = 512;
	const int numberOfBlocks = 1 + ((numRows*numCols - 1) / blockThreadSize); // a/b rounded up
	const dim3 blockSize(blockThreadSize, 1, 1);
	const dim3 gridSize(numberOfBlocks, 1, 1);
	rgbaToGreyscaleCudaKernel << <gridSize, blockSize >> > (d_rgbaImage, d_greyImage, numRows, numCols);
}

// Parallel implementation for running on GPU using multiple threads.
void rgbaToGaussianBlurCuda(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
	unsigned char* const d_greyImage, const size_t numRows, const size_t numCols)
{
	const int blockThreadSize = 512;
	const int numberOfBlocks = 1 + ((numRows*numCols - 1) / blockThreadSize); // a/b rounded up
	const dim3 blockSize(blockThreadSize, 1, 1);
	const dim3 gridSize(numberOfBlocks, 1, 1);
	rgbaToGaussianBlurCudaKernel << <gridSize, blockSize >> > (d_rgbaImage, d_greyImage, numRows, numCols, 200);
}

#endif  /* GRAYSCALE_H__ */