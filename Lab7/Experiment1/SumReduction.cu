#include <cuda_runtime.h>
#include <stdio.h>

// macros
#define N (1<<10)
#define BD 256
#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if(error != cudaSuccess){\
		fprintf(stderr, "Error: %s:%d, ",__FILE__, __LINE__);\
		fprintf(stderr,"code:%d,reason:%s\n",error,\
		cudaGetErrorString(error));\
		exit(1);\
	}\
}
#define rec_init float elapsedTime;\
	cudaEvent_t start, stop;\
	CHECK(cudaEventCreate(&start));\
	CHECK(cudaEventCreate(&stop))
#define rec_start CHECK(cudaEventRecord(start,0))
#define rec_stop CHECK(cudaEventRecord(stop,0));\
	CHECK(cudaEventSynchronize(stop));\
	cudaEventElapsedTime(&elapsedTime,start,stop)
#define rec_pr(s) printf(s" %3.6f ms\n",elapsedTime)

// functions
void initialize(float *a){
	for (int i = 0; i < N; i++)
	{
		a[i] = i + 1;
	}
}

__global__ void sumReduce(float *dev_a,float *dev_sum)
{
	__shared__ float partialSum[BD];
	partialSum[threadIdx.x] = dev_a[blockIdx.x*blockDim.x + threadIdx.x];
	unsigned int t = threadIdx.x;

	for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
	{
		__syncthreads();
		if(t % (2*stride) == 0)
		{
			partialSum[t] += partialSum[t+stride];
		}
	}
	if(t == 0) dev_sum[blockIdx.x] = partialSum[0];
}

int main(int argc,char **argv)
{
	// variable declarition
	int bdimx = BD;
	int gdimx = (N + bdimx -1)/bdimx;
	dim3 block(bdimx);
	dim3 grid(gdimx);

	// declaring input array(a) and sum array(b) for both host and device
	float *a,*b;
	float *dev_a,*dev_sum;
	
	a = (float *)malloc(N*sizeof(float));
	b = (float *)malloc(gdimx*sizeof(float));
	// setup for measure time elapsed
	rec_init;

	// allocate the memory on device
	CHECK(cudaMalloc((void**)&dev_a, N*sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_sum, gdimx*sizeof(float)));

	// initilize array a
	initialize(a);

	rec_start;
		// copying array data to device
		CHECK(cudaMemcpy(dev_a, a, N*sizeof(float),cudaMemcpyHostToDevice));
	rec_stop;

	// printing array size and time elapsed for memory transfer
	printf("Array Size is = %d\n",N);
	rec_pr("Time to do memory transfer of array a, from host to device is");
	
	rec_start;
		//kernel launch
		sumReduce<<<grid,block>>>(dev_a,dev_sum);
		sumReduce<<<1,block>>>(dev_sum,dev_sum);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy(b,dev_sum, sizeof(float),cudaMemcpyDeviceToHost));
	rec_stop;
	rec_pr("Time to do sum reduction is");
	printf("Sum = %f\n",b[0]);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_a);
	cudaFree(dev_sum);
	return 0;
}
