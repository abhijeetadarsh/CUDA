#include<cuda_runtime.h>
#include<stdio.h>

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

// grid 1D block 1D
__global__ void sumMatrixOnGPU(float *MatA,float *MatB,float *MatC,int nx,int ny){
	unsigned int ix=threadIdx.x+blockIdx.x*blockDim.x;
	
	if(ix<nx)
		for(int iy=0;iy<ny;iy++){
			int idx=iy*nx+ix;
			MatC[idx]=MatA[idx]+MatB[idx];
		}	
}

int main(){
	// set up data size of matrix
	int nx=4;
	int ny=5;
	
	int nxy=nx*ny;
	int nBytes=nxy*sizeof(float);
	printf("Matrix size : nx %d ny %d\n",nx,ny);
	
	// malloc host memory
	float *h_A,*h_B,*h_C;
	h_A=(float *)malloc(nBytes);
	h_B=(float *)malloc(nBytes);
	h_C=(float *)malloc(nBytes);
	
	// initialize data at host side
	initialData(h_A, nxy);
	initialData(h_B, nxy);
	
	// malloc device global memory
	float *d_MatA,*d_MatB,*d_MatC;
	cudaMalloc((void **)&d_MatA, nBytes);
	cudaMalloc((void **)&d_MatB, nBytes);
	cudaMalloc((void **)&d_MatC, nBytes);
	
	// transfer data from host to device
	cudaMemcpy(d_MatA,h_A,nBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB,h_B,nBytes,cudaMemcpyHostToDevice);

	// invoke kernel at host side
	int dimx=32;
	dim3 block(dimx,1);
	dim3 grid((nx+block.x-1)/block.x,1);
	
	sumMatrixOnGPU<<<grid, block>>>(d_MatA,d_MatB,d_MatC,nx,ny);
	cudaDeviceSynchronize();
	
	// copy kernel result back to host side
	cudaMemcpy(h_C,d_MatC,nBytes,cudaMemcpyDeviceToHost);
	displayMatrix(h_C,nx,ny);
	
	// free device global memory
	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);
	
	// free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	
	// reset device
	cudaDeviceReset();
	
	return 0;
}
