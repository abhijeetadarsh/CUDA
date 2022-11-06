#include <stdio.h>
#include <cuda_runtime.h>
#include "lab6ex1lib.cuh"

matrix::matrix() : rows(0), cols(0), device_pointer(NULL), host_pointer(NULL){};
matrix::matrix(int r, int c)
{
	rows = r;
	cols = c;
	memAllocInBoth();
}
matrix::~matrix()
{
	// printf("<>");
	if (device_pointer != NULL)
		cudaFree(device_pointer);
	if (host_pointer != NULL)
		free(host_pointer);
}
void matrix::memAllocInBoth()
{
	host_pointer = (double *)malloc(rows * cols * sizeof(double));
	cudaMalloc(&device_pointer, rows * cols * sizeof(double));
}
void matrix::display()
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			printf("\033[1;32m%*.*lf", widthField, precisionField, host_pointer[i * cols + j]);
		printf("\033[0m\n");
	}
}

void matrix::sdisplay()
{
	printf("[");
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%.0lf", host_pointer[i * cols + j]);
			if (j != cols - 1)
				printf(",");
		}
		if (i != rows - 1)
			printf(";");
	}
	printf("]\n");
}
void matrix::init_rand()
{
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			host_pointer[i * cols + j] = rand() % 10 - 4;
	H2D();
}
void matrix::H2D() { cudaMemcpy(device_pointer, host_pointer, rows * cols * sizeof(double), cudaMemcpyHostToDevice); }
void matrix::D2H() { cudaMemcpy(host_pointer, device_pointer, rows * cols * sizeof(double), cudaMemcpyDeviceToHost); }

__device__ void withoutTILE(double *d_MatA, double *d_MatB, double *d_MatC, int rows, int width, int cols)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < rows && j < cols)
	{
		double dotp = 0;
		for (int k = 0; k < width; k++)
			dotp += d_MatA[i * width + k] * d_MatB[k * cols + j];
		d_MatC[i * cols + j] = dotp;
	}
}

__device__ void withTILE(double *d_MatA, double *d_MatB, double *d_MatC, int Width)
{
	// Shared memory allocation
	__shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	double Pvalue = 0;
	for (int ph = 0; ph < Width / TILE_WIDTH; ++ph)
	{
		// Collaborative loading of A and B tiles into shared memory
		Mds[ty][tx] = d_MatA[Row * Width + ph * TILE_WIDTH + tx];
		Nds[ty][tx] = d_MatB[(ph * TILE_WIDTH + ty) * Width + Col];
		__syncthreads();
		// dot product using shared memory
		for (int k = 0; k < TILE_WIDTH; ++k)
		{
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	d_MatC[Row * Width + Col] = Pvalue;
}