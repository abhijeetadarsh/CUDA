#include <stdio.h>
__global__ void kernel ()
{    
    __shared__ int arr[1024 * 12];
    for (int i = 0; i < 1024 * 12; i++)
        arr[i] = 0;
    if (blockIdx.x == 6552)
        printf ("%d, %d, %d\n", blockIdx.x, blockIdx.y, arr[0]);
    return;
}
int main ()
{
    kernel <<<dim3 (6553, 6553), 1>>> ();
    cudaError_t e = cudaGetLastError ();
    printf ("%s\n", cudaGetErrorString (e));
    cudaDeviceReset ();
    return 0;
}