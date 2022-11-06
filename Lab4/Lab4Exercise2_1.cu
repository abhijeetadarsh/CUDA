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
	Matrix(int r, int c) //: Matrix()
	{
		rows = r;
		cols = c;
		memAllocInBoth();
	}
	Matrix(const Matrix &M)
	{
// #if SHOW_FUNCTION_CALLS == 1
// 		printf("\033[90mMatrix (const Matrix &M)\033[m\n");
// #endif
		rows = M.rows;
		cols = M.cols;
		cudaMalloc(&device_pointer, rows * cols * sizeof(double));
		cudaMemcpy(device_pointer, M.device_pointer, rows * cols * sizeof(double), cudaMemcpyDeviceToDevice);
		host_pointer = (double *)(malloc(rows * cols * sizeof(double)));
		memcpy(host_pointer, M.host_pointer, rows * cols * sizeof(double));
		return;
	}
// 	Matrix(Matrix &&M)
// 	{
// #if SHOW_FUNCTION_CALLS == 1
// 		printf("\033[90mMatrix (Matrix &&M)\033[m\n");
// #endif
// 		rows = M.rows;
// 		cols = M.cols;
// 		device_pointer = M.device_pointer;
// 		host_pointer = M.host_pointer;
// 		M.rows = M.cols = 0;
// 		M.device_pointer = M.host_pointer = NULL;
// 		return;
// 	}
	void memAllocInBoth()
	{
		host_pointer = (double *)malloc(rows * cols * sizeof(double));
		cudaMalloc(&device_pointer, rows * cols * sizeof(double));
	}
// 	void display()
// 	{
// 		if (NULL == host_pointer)
// 		{
// #if WARNINGS == 1
// 			printf("\nIn function \'\e[33mprint_matrix_yu\e[m\':\n\e[35mwarning:\e[m \'m\' is (null)\n");
// #endif
// 			return;
// 		}
// #define BUFFER_SIZE 128
// 		int *max_width_arr = (int *)(malloc(cols * sizeof(int)));
// 		char **mat_of_strs = (char **)malloc(rows * cols * sizeof(char *));
// 		char *str;
// 		int width;
// 		for (size_t i = 0; i < cols; i++)
// 		{
// 			max_width_arr[i] = 1;
// 			for (size_t j = 0; j < rows; j++)
// 			{
// 				str = (char *)malloc(BUFFER_SIZE * sizeof(char));
// 				width = snprintf(str, BUFFER_SIZE, "%.*lf", precisionField, host_pointer[j * cols + i]);
// 				str = (char *)realloc(str, ((size_t)(width + 1)) * sizeof(char));
// 				mat_of_strs[j * cols + i] = str;
// 				if (max_width_arr[i] < width)
// 					max_width_arr[i] = width;
// 			}
// 		}
// 		for (size_t i = 0; i < rows; i++)
// 		{
// 			printf("\033[1;32m\xb3\033[m");
// 			for (size_t j = 0; j < cols; j++)
// 			{
// 				width = strlen(mat_of_strs[i * cols + j]);
// 				for (int x = 0; x < max_width_arr[j] - width; x++)
// 					printf(" ");
// 				printf("%s", mat_of_strs[i * cols + j]);
// 				if (j != (cols - 1))
// 					printf(" ");
// 			}
// 			printf("\033[1;32m\xb3\033[m");
// 			printf("\n");
// 		}
// 		for (size_t i = 0; i < rows; i++)
// 			for (size_t j = 0; j < cols; j++)
// 				free(mat_of_strs[i * cols + j]);
// 		free(mat_of_strs);
// 		free(max_width_arr);
// 		return;
// 	}
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
	// Matrix operator=(Matrix &&M)
	// {
	// 	rows = M.rows;
	// 	cols = M.cols;
	// 	device_pointer = M.device_pointer;
	// 	host_pointer = M.host_pointer;
	// 	return *this;
	// }
	// distructor
	~Matrix()
	{
		// printf("\033[31m%p;%p\033[m\n", device_pointer, host_pointer);
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
int main()
{
	Matrix A(4, 5), B(5, 6), C(6, 7);
	A.init(), B.init(), C.init();
	printf("MATRIX A\n");
	A.display();
	printf("MATRIX B\n");
	B.display();
	printf("MATRIX C\n");
	C.display();
	Matrix ABC = A * B * C;
	printf("MATRIX A x B x C\n");
	ABC.display();
	cudaDeviceReset();
	return 0;
}