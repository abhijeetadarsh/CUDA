#include <stdio.h>
#include "lab5ex2lib.cuh"

#define M 3
#define N 4

// This global function compute the square of matrix C which is sum of matrix A and B
__global__ void driver(double *A, double *B, int cols, int rows){
    trans_matrix_GPU(A, B, cols, rows);
}

int main()
{
    srand(time(NULL));
    matrix A(M, N), TA(N, M), C(M, M);

    A.init_rand();
    printf("Matrix A:\n");
    A.display();

    dim3 block(1);
    dim3 grid(N, M);
    driver<<<grid, block>>>(A.device_pointer, TA.device_pointer, A.cols, A.rows);
    cudaDeviceSynchronize();
    TA.D2H();

    mul_matrix_GPU<<<grid, block>>>(A.device_pointer, TA.device_pointer, C.device_pointer, A.rows, A.cols, TA.cols);
    cudaDeviceSynchronize();
    C.D2H();

    printf("Matrix TA:\n");
    TA.display();
    printf("Matrix A x TA:\n");
    C.display();

    return 0;
}