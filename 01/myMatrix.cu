#include <stdio.h>
#include <cuda_runtime.h>

// macros:
#define widthField 8
#define precisionField 0

// forward declaration:
class Matrix;
__global__ void init_GPU(double *p, int rows, int cols);
__global__ void mult_GPU(double *d_MatA, double *d_MatB, double *d_MatC, int rows, int x, int cols);

class Matrix
{
public:
	int rows, cols;
	double *device_pointer, *host_pointer;
	// constructor
	Matrix() : rows(0), cols(0), device_pointer(NULL), host_pointer(NULL){};
	Matrix(int r, int c)
	{
		rows = r;
		cols = c;
		memAllocInBoth();
	}
	Matrix(const Matrix &M)
	{
		rows = M.rows;
		cols = M.cols;
		cudaMalloc(&device_pointer, rows * cols * sizeof(double));
		cudaMemcpy(device_pointer, M.device_pointer, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
		host_pointer = (double *)(malloc(rows * cols * sizeof(double)));
		memcpy(host_pointer, M.host_pointer, rows * cols * sizeof(double));
		return;
	}
	void memAllocInBoth()
	{
		host_pointer = (double *)malloc(rows * cols * sizeof(double));
		cudaMalloc(&device_pointer, rows * cols * sizeof(double));
	}
    void display(){
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                printf("%*.*lf", widthField, precisionField, host_pointer[i * cols + j]);
            }
            printf("\n");
        }
    }
	void init()
	{
		dim3 block(1);
		dim3 grid(rows, cols);
		init_GPU<<<grid, block>>>(device_pointer, rows, cols);
		cudaDeviceSynchronize();
		D2H();
	}
	void H2D()
	{
		cudaMemcpy(device_pointer, host_pointer, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
	}
	void D2H()
	{
		cudaMemcpy(host_pointer, device_pointer, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
	}
	// operator overload
	Matrix operator*(const Matrix &M)
	{
		if (cols != M.rows)
		{
			printf("Multiplication not valid\n");
			return Matrix();
		}
		Matrix product(rows, M.cols);
		dim3 block(1);
		dim3 grid(rows, M.cols);
		mult_GPU<<<grid, block>>>(device_pointer, M.device_pointer, product.device_pointer, rows, cols, M.cols);
		cudaDeviceSynchronize();
		product.D2H();
		return product;
	}
	Matrix operator=(Matrix &M)
	{
		rows = M.rows;
		cols = M.cols;
		memAllocInBoth();
		cudaMemcpy(device_pointer, M.device_pointer, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
		memcpy(host_pointer, M.host_pointer, rows * cols * sizeof(double));
		return *this;
	}
	~Matrix()
	{
		if (device_pointer != NULL)
			cudaFree(device_pointer);
		if (host_pointer != NULL)
			free(host_pointer);
	}
};
__global__ void init_GPU(double *p, int rows, int cols)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < rows && j < cols)
		p[i * cols + j] = (double)(i * cols + j);
	return;
}
__global__ void mult_GPU(double *d_MatA, double *d_MatB, double *d_MatC, int rows, int x, int cols)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i < rows && j < cols)
	{
		double dotp = 0;
		for (int k = 0; k < x; k++)
			dotp += d_MatA[i * x + k] * d_MatB[k * cols + j];
		d_MatC[i * cols + j] = dotp;
	}
	return;
}