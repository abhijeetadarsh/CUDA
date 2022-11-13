#define M 8
__global__ void MatrixMulKernel(float *MatA, float *MatB, float *MatC)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < M) and (Col < M))
    {
        float Pvalue = 0;
        for (int k = 0; k < M; k++)
        {
            Pvalue += MatA[Row * M + k] * MatB[k * M + Col];
        }
        MatC[Row * M + Col] = Pvalue;
    }
}

int main(){}