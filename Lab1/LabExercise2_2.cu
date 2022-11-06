#include <stdio.h>

int main(void){
    cudaDeviceProp deviceProp;
    if (cudaSuccess != cudaGetDeviceProperties(&deviceProp, 0)) {
        printf("Get device properties failed.\n");
        return 1;
    }
    else{
        printf("Warp size: %d\n", deviceProp.warpSize);
        printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch: %d bytes\n", deviceProp.memPitch);
        return 0;
    }
}