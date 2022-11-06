#include <stdio.h>
#include "lab5ex1lib.cuh"

#define N 4

// This global function compute the square of matrix C which is sum of matrix A and B
__global__ void driver(double *A, double *B, double *C, int rows, int cols){
    add_matrix_GPU(A, B, C, rows, cols);
}

int main()
{
    srand(time(NULL));
    matrix A(N, N), B(N, N), C(N, N), D(N, N);

    A.init_rand();
    printf("Matrix A:\n");
    A.display();

    B.init_rand();
    printf("Matrix B:\n");
    B.display();

    dim3 block(1);
    dim3 grid(N, N);
    driver<<<grid, block>>>(A.device_pointer, B.device_pointer, C.device_pointer, C.rows, C.cols);
    cudaDeviceSynchronize();
    C.D2H();

    mul_matrix_GPU<<<grid, block>>>(C.device_pointer, C.device_pointer, D.device_pointer, C.rows, C.cols, C.cols);
    cudaDeviceSynchronize();
    D.D2H();

    printf("Matrix C:\n");
    C.display();
    printf("Matrix D:\n");
    D.display();

    return 0;
}