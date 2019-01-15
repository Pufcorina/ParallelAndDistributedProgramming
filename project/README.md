# Gaussian Blur and Grayscale image filtering

Team members: @Pufcorina and @ChisIulia

## Goal

  Applying a simple filter on an image (e.g., a convolution/linear filter). In our case we choose 2 types of filters: Gaussian Blur and Grayscale.

## Requirement

  Each project will have 2 implementations: one with "regular" threads or tasks/futures, and one distributed (possibly, but not required, using MPI). A third implementation, using OpenCL or CUDA, can be made for a bonus.

## Computer Specification

* CPU: Intel Core i7-7500U, 2.90GHz
* RAM: 8 GB

## Algorithms

  In the following section, we will present a brief description for both the Gaussian blur and Grayscale algorithms.
  The first step is to read our image and obtain for each pixel its corresponding RGB's values. After that, we have 3 matrices of pixels for each colour model. The second step is applying effectively the image filtering algorithm presented in the next subsections.

### Gaussian blur

  In convolution, two mathematical functions are combined to produce a third function. In image processing functions are usually called kernels. A kernel is nothing more than a (square) array of pixels (a small image so to speak). Usually, the values in the kernel add up to one. This is to make sure no energy is added or removed from the image after the operation.

  Specifically, a Gaussian kernel (used for Gaussian blur) is a square array of pixels where the pixel values correspond to the values of a Gaussian curve (in 2D).

  In our case the Gaussian kernel is the average of a matrix of size depending on the deep of the blur, having in the center our corresponding pixel.

  <img src="https://developer.apple.com/library/archive/documentation/Performance/Conceptual/vImage/Art/kernel_convolution.jpg" align="center"
       title="Kernel convolution" width="300" height="300">


### Grayscale

  One approach would be to use the average method which is the most simple one. We just have to take the average of three colours. Since it's an RGB image, so it means that we have to add r and g and b then divide it by 3 to get the desired grayscale image.

                  Grayscale = (R + G + B) / 3

  According to this equation, Red has contributed 30\%, Green has contributed 59\% which is greater in all three colours and Blue has contributed 11\%. The average method works but the results were not as expected. We wanted to convert the image into a grayscale, but this turned out to be a rather black image.

#### Problem

  This problem arises due to the fact, that we take the average of the three colours. Since the three different colours have three different wavelengths and have their own contribution to the formation of the image, so we have to take average according to their contribution, not done it averagely using the average method. Right now what we are doing is this, 33\% of Red, 33\% of Green and 33\% of Blue.
  
  We are taking 33\% of each, that means, each of the portions has the same contribution in the image. But in reality, that's not the case. The solution to this has been given by luminosity method.

  Another approach is the weighted method or luminosity method. Now that we have seen the problem that occurs in the average method we offer a solution to that one. Since red colour has more wavelength of all the three colours, and green is the colour that has not only less wavelength than red colour but also green is the colour that gives more soothing effects to the eyes.
  
  It means that we have to decrease the contribution of red colour, and increase the contribution of the green colour, and put the blue colour contribution in between these two.
Son the new equation that form is:

    Grayscale = 0.299 * R + 0.587 * G + 0.114 * B

  According to this equation, Red has contribute 29.9\%, Green has contributed 58.7\% which is greater in all three colours and Blue has contributed 11.4\%.


## Short Description of the Implementation:

* Threads
* Distributed - MPI
* OpenCL / CUDA


### Threads

  We used a thread pool of size 10 and each thread gets a line from the matrix in order to perform the corresponding algorithm. At the end, we wait until all threads finish their task.

### Distributed - MPI

### OpenCL / CUDA


## Performance Tests

| Algorithm                        | 1280x733 | 2048x1174 | 640x336 |
| -------------------------------- |:--------:|:-------:|:---------:|
| Threads         | 128 ms |  416 ms | 64 ms |
| Distributed MPI | ? ms | ? ms | ? ms |
| OpenCL / CUDA   | ? ms | ? ms | ? ms |


## Results

Grayscale             |  Gaussian Blur
:-------------------------:|:-------------------------:
1280x733 | 1280x733
![](https://github.com/Pufcorina/ParallelAndDistributedProgramming/blob/master/project/results/gray_img1280x733.jpg)  |  ![](https://github.com/Pufcorina/ParallelAndDistributedProgramming/blob/master/project/results/blur_img1280x733.jpg)
2048x1174 | 2048x1174
![](https://github.com/Pufcorina/ParallelAndDistributedProgramming/blob/master/project/results/gray_img2048x1174.jpg)  |  ![](https://github.com/Pufcorina/ParallelAndDistributedProgramming/blob/master/project/results/blur_img2048x1174.jpg)
640x336 | 640x336
![](https://github.com/Pufcorina/ParallelAndDistributedProgramming/blob/master/project/results/gray_img640x336.jpg)  |  ![](https://github.com/Pufcorina/ParallelAndDistributedProgramming/blob/master/project/results/blur_img640x336.jpg)


## Conclusion
