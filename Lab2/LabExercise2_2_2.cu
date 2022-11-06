#include <cuda_runtime.h>
#include <stdio.h>
#define N 1024
__global__ void fun1(double *a, double *b, double *c)
{
    int i = blockIdx.x;
    if (i < N) c[i] = (a[i] - b[0])*(a[i] - b[0]);
}

__host__ double mean(double *v)
{
    double sum = 0;
    for (int i = 0; i < N; i++)
    {
        sum += v[i];
    }
    return sum/N;
}

int main(int argc, char **argv)
{
    double a[N], b[1], c[N], SD; // a is vector y, b is mean of vector y, c is vector of elements (a[i]-mean)^2
    double *dev_a, *dev_b, *dev_c;

    // allocate the memory on device
    cudaMalloc((void **)&dev_a, N * sizeof(double));
    cudaMalloc((void **)&dev_b, 1 * sizeof(double));
    cudaMalloc((void **)&dev_c, N * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        a[i] = 2*(i+1)+1;
    }
    b[0] = mean(a);
    // Copy data from host to device
    cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, 1 * sizeof(double), cudaMemcpyHostToDevice);
    // kernel launch
    fun1<<<N, 1>>>(dev_a, dev_b, dev_c);
    // Copy result from device to host
    cudaMemcpy(c, dev_c, N * sizeof(double), cudaMemcpyDeviceToHost);
    SD = sqrt(mean(c));

    printf("Standard deviation of y is : %lf\n",SD);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}