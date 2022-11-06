#include <iostream>
#include <cuda_runtime.h>
#include "lab6ex1lib.cuh"

// macros
#define M 4
#define N 4

using namespace std;

__global__ void mult(double *d_MatA, double *d_MatB, double *d_MatC, int rows, int width, int cols, int option)
{
    if (option == 1)
        withTILE(d_MatA, d_MatB, d_MatC, width);
    else
        withoutTILE(d_MatA, d_MatB, d_MatC, rows, width, cols);
}
void instruction()
{
    cout << "Enter the option number\n";
    cout << "1. Multiply two matrices A and B using tiled algorithm in GPU.\n";
    cout << "2. Multiply two matrices A and B without tiled algorithm in GPU.\n";
    cout << "0. TO EXIT\n";
}

int main()
{
    int i;
    matrix A(M, N), B(N, M);

    A.init_rand();
    cout << "Matrix A:\n";
    A.display();

    B.init_rand();
    cout << "Matrix B:\n";
    B.display();

    instruction();
    while (cin >> i)
    {
        switch (i)
        {
        case 1:
        {
            matrix C(M,M);
            cout << "A x B with TILE ALGORITHM.\n";
            dim3 block1(TILE_WIDTH, TILE_WIDTH);
            dim3 grid1(ceil((double)C.cols/TILE_WIDTH),ceil((double)C.rows/TILE_WIDTH));
            mult<<<grid1, block1>>>(A.device_pointer, B.device_pointer, C.device_pointer,A.rows, A.cols, B.cols,1);
            cudaDeviceSynchronize();
            C.D2H();
            C.display();
        }
        break;
        case 2:
        {
            matrix C(M,M);
            cout << "A x B without TILE ALGORITHM.\n";
            dim3 block2(M,N);
            dim3 grid2(1);
            mult<<<grid2,block2>>>(A.device_pointer, B.device_pointer, C.device_pointer, A.rows, A.cols, B.cols,2);
            cudaDeviceSynchronize();
            C.D2H();
            C.display();            
        }
        break;
        case 0:
            cout << "\033[1;31mExiting...\033[0m\n";
            cudaDeviceReset();
            return 0;
            break;
        default:
            cout << "\033[1;33mInvalid! Option Try again\033[0m\n";
            break;
        }
    }
}