#include <cuda_runtime.h>
#include <iostream>
using namespace std;
int main(){
    int deviceCount = 0, dev = 0, deviceVersion = 0, runtimeVersion = 0;
    cudaGetDeviceCount(&deviceCount);
    cout << "Number of cuda device " << deviceCount << "\n";
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cout << dev << "\n";
    cout << deviceProp.name << endl;
    cout << deviceProp.clockRate * 1e-6 << " GHz\n";
    cout << deviceProp.totalGlobalMem / float(1<<30) << " GB\n";
    cout << deviceProp.warpSize << endl;
    cout << deviceProp.maxThreadsPerBlock << endl;
    cout << deviceProp.maxThreadsPerMultiProcessor << endl;
    cout << deviceProp.maxThreadsDim[0] << endl;
    cout << deviceProp.maxThreadsDim[1] << endl;
    cout << deviceProp.maxThreadsDim[2] << endl;
    cout << deviceProp.maxGridSize[0] << endl;
    cout << deviceProp.maxGridSize[1] << endl;
    cout << deviceProp.maxGridSize[2] << endl;
    cout << deviceProp.memPitch << " bytes\n";
}