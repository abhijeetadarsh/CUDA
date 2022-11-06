#include "lab8ex1lib.cuh"
// correct sum = 2780.8248727321624756
// error sum without sorting 0.0001657009124756
// error sum with sorting -0.0000784397125244
using namespace std;

void show(float *a, int size){
	for (int i = 0; i < size; i++)
	{
		printf("%.16f ",a[i]);
	}
	printf("\n");
}

int main(){
    int bdimx = BD;
	int gdimx = (N + bdimx -1)/bdimx;
	dim3 block(bdimx);
	dim3 grid(gdimx);


	rec_init;
	printf("Array Size is = %d\n",N);


	float *a,*b;
	float *dev_a,*dev_b;
	a = (float *)malloc(N*sizeof(float));
	b = (float *)malloc(gdimx*sizeof(float));
	CHECK(cudaMalloc((void**)&dev_a, N*sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_b, gdimx*sizeof(float)));
	srand(1);
	initialize(a);
	show(a,N);
	CHECK(cudaMemcpy(dev_a, a, N*sizeof(float),cudaMemcpyHostToDevice));
	rec_start;
		sumReduce<<<grid,block>>>(dev_a,dev_b);
		sumReduce<<<1,block>>>(dev_b,dev_b);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy(b,dev_b, sizeof(float),cudaMemcpyDeviceToHost));
	rec_stop;
	rec_pr("Time to do sum of not sorted array is");
	printf("Sum = %.16f\n",b[0]);


	float *c,*d;
	float *dev_c,*dev_d;
	c = (float *)malloc(N*sizeof(float));
	d = (float *)malloc(gdimx*sizeof(float));
	CHECK(cudaMalloc((void**)&dev_c, N*sizeof(float)));
	CHECK(cudaMalloc((void**)&dev_d, gdimx*sizeof(float)));
	srand(1);
	initialize(c);
	sort(c,c+N);
	show(c,N);
	CHECK(cudaMemcpy(dev_c, c, N*sizeof(float),cudaMemcpyHostToDevice));
	rec_start;
		sumReduce<<<grid,block>>>(dev_c,dev_d);
		sumReduce<<<1,block>>>(dev_d,dev_d);
		cudaDeviceSynchronize();
		CHECK(cudaMemcpy(d,dev_d, sizeof(float),cudaMemcpyDeviceToHost));
	rec_stop;
	rec_pr("Time to do sum of sorted array is");
	printf("Sum = %.16f\n",d[0]);


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_d);
	return 0;
}