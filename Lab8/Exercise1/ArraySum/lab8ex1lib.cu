#include "lab8ex1lib.cuh"
#define X (1<<10)
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

void initialize(float *a){
	for (int i = 0; i < N; i++)
	{
		// a[i] = 1;
		if(i&1) a[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/X));
		else a[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));
	}
}