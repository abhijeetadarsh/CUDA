#include<cuda_runtime.h>
#include<stdio.h>
#include<sys/time.h>

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double)tp.tv_usec*1e-6);
}

void initialData(float *ip,const int size){
	int i;
	for(i=0;i<size;i++) ip[i]=i;
	return;
}

void displayMatrix(float *A,int nx,int ny){
	int idx;
	for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
			idx=i*ny+j;
			printf(" %f ",A[idx]);
		}
		printf("\n");
	}
	return;
}

// grid 2D block 1D
__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy*nx + ix;
    if (ix < nx && iy < ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}

// grid 2D block 2D
__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatD, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy*nx + ix;
    if (ix < nx && iy < ny)
        MatD[idx] = MatA[idx] + MatB[idx];
}

int main(){
	
	int nx=4;
	int ny=5;
	
	int nxy=nx*ny;
	int nBytes=nxy*sizeof(float);
	printf("Matrix size : nx %d ny %d\n",nx,ny);
	
	
	float *h_A,*h_B,*h_C,*h_D;
	h_A=(float *)malloc(nBytes);
	h_B=(float *)malloc(nBytes);
	h_C=(float *)malloc(nBytes);
	h_D=(float *)malloc(nBytes);
	
	
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	
	// 1. Allocate Device Memory
	float *d_MatA,*d_MatB,*d_MatC,*d_MatD;
	cudaMalloc((void **)&d_MatA, nBytes);
	cudaMalloc((void **)&d_MatB, nBytes);
	cudaMalloc((void **)&d_MatC, nBytes);
	cudaMalloc((void **)&d_MatD, nBytes);
	
	// 2. Transfer Data(Matrices A and B) from host to device
	cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB,h_B,nBytes,cudaMemcpyHostToDevice);

	// 3. Sum two matrices using 2D grid with different block sizes
    // Matrix summation using 2D grid with 1D block
    int dimx = 32;
    dim3 block(dimx);
    dim3 grid((nx + block.x - 1) / block.x,ny);
	
    double iStart = cpuSecond();
	sumMatrixOnGPUMix<<<grid, block>>>(d_MatA,d_MatB,d_MatC,nx,ny);
	cudaDeviceSynchronize();
	double iElaps = cpuSecond() - iStart;

	// 4. Transfer result(Matrix C) from device to host
	cudaMemcpy(h_C,d_MatC,nBytes,cudaMemcpyDeviceToHost);
	// 5. Print the result in matrix format
	displayMatrix(h_C,nx,ny);
	// 6. Show the effect of block size and grid size in terms of total run time.
    printf(" Matrix summation using 2D grid with 1D block\nTime elapsed %lf sec\n\n",iElaps);

    // Matrix summation using 2D grid with 2D block
    int dimx1 = 32;
    int dimy1 = 32;
    dim3 block1(dimx1, dimy1);
    dim3 grid1((nx + block1.x - 1) / block1.x, (ny + block1.y - 1) / block1.y);
	
    iStart = cpuSecond();
	sumMatrixOnGPU2D<<<grid1, block1>>>(d_MatA,d_MatB,d_MatD,nx,ny);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;

	// 4. Transfer result(Matrix C) from device to host
	cudaMemcpy(h_D,d_MatD,nBytes,cudaMemcpyDeviceToHost);
	// 5. Print the result in matrix format
	displayMatrix(h_D,nx,ny);
	// 6. Show the effect of block size and grid size in terms of total run time.
    printf(" Matrix summation using 2D grid with 2D block\nTime elapsed %lf sec\n\n",iElaps);
	

	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);
	

	free(h_A);
	free(h_B);
	free(h_C);
	
	
	cudaDeviceReset();
	
	return 0;
}
