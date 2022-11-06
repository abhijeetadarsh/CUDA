#include <cuda_runtime.h>
#include <stdio.h>
#define N 3
__global__ void MatrixMulKernel(float* MatA, float* MatB, float* MatC,int Width){
	// calculate the row index of the P element and M
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	// calculate the column index of P and N
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if((Row < Width) and (Col < Width)){
		float Pvalue = 0;
		// each thread computes one element of the block sub-matrix
		for(int k=0;k < Width;k++){
			Pvalue += MatA[Row*Width + k]*MatB[k*Width + Col];
		}
		MatC[Row*Width + Col] = Pvalue;
	}
}

void initialData(float* ip,const int size){
	int i;
	for(i=0;i<size;i++) ip[i]=i;
	return;
}

void displayMatrix(float* A,int nx,int ny){
	int idx;
	for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
			idx=i*ny+j;
			printf(" %6.2f ",A[idx]);
		}
		printf("\n");
	}
	return;
}

int main(int argc,char** argv){

	//set up data size of matrix
	int Width = N;
	int nx = Width;
	int ny = Width;
	int nxy = nx*ny;
	int nBytes = nxy*sizeof(float);
	printf("Matrix size:row %d column %d\n",nx,ny);

	//malloc host memory
	float *h_A, *h_B,*h_C;
	h_A=(float *)malloc(nBytes);
	h_B=(float *)malloc(nBytes);
	h_C=(float *)malloc(nBytes);

	//initialize data at host side
	initialData(h_A,nxy);
	initialData(h_B,nxy);

	//malloc device globalmemory
	float *d_MatA,*d_MatB,*d_MatC;
	cudaMalloc((void **)&d_MatA,nBytes);
	cudaMalloc((void **)&d_MatB,nBytes);
	cudaMalloc((void **)&d_MatC,nBytes);

	//transfer data from host to device
	cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB,h_B,nBytes,cudaMemcpyHostToDevice);

	// invoke kernel at host side 
	int bdimx=16;
	int bdimy=16;
	dim3 block(bdimx,bdimy);
	dim3 grid((nx + block.x -1)/block.x, (ny + block.y -1)/block.y);
	MatrixMulKernel<<<grid, block>>>(d_MatA,d_MatB,d_MatC,Width);
	cudaDeviceSynchronize();

	//copy kernel result back to hostside
	cudaMemcpy(h_C,d_MatC,nBytes,cudaMemcpyDeviceToHost);
	printf("Matrix A is =\n");
	displayMatrix(h_A,nx,ny);
	printf("Matrix B is =\n");
	displayMatrix(h_B,nx,ny);
	printf("The product of Matrix A and Matrix B is =\n");
	displayMatrix(h_C,nx,ny);

	//free device global memory
	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);

	//free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	//resetdevice
	cudaDeviceReset();

	return(0);
}
