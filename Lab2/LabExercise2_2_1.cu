#include <cuda_runtime.h>
#include <stdio.h>
#define N 1024
__global__ void VecDiffGPU(int *a, int *b, int *c)
{
    int i = blockIdx.x;
    if (i < N) c[i] = a[i] - b[i];
}

__host__ double euclideanNorm(int *v)
{
    long long sumOfSq = 0;
    for (int i = 0; i < N; i++)
    {
        sumOfSq += v[i]*v[i];
    }
    return sqrt((double)sumOfSq);
}

int main()
{
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;
    // allocate the memory on device
    cudaMalloc((void **)&dev_a, N * sizeof(int));
    cudaMalloc((void **)&dev_b, N * sizeof(int));
    cudaMalloc((void **)&dev_c, N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        a[i] = (i+1) * (i+1);
        b[i] = 2*(i+1)+1;
    }
    // Copy data from host to device
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    // kernel launch
    VecDiffGPU<<<N, 1>>>(dev_a, dev_b, dev_c);
    // Copy result from device to host
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Distance b/w vector x and vector y is: %lf\n",euclideanNorm(c));
    printf("Euclidean norms of x is %lf and y is %lf\n",euclideanNorm(a),euclideanNorm(b));
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}