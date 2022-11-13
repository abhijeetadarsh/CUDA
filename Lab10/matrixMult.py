import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

MATRIX_LEN = 8

mat_a = numpy.random.randn(MATRIX_LEN, MATRIX_LEN).astype(numpy.float32)
mat_b = numpy.random.randn(MATRIX_LEN, MATRIX_LEN).astype(numpy.float32)
mat_c = numpy.empty_like(mat_a)

dev_a = cuda.mem_alloc(mat_a.nbytes)
cuda.memcpy_htod(dev_a, mat_a)
dev_b = cuda.mem_alloc(mat_b.nbytes)
cuda.memcpy_htod(dev_b, mat_b)
dev_c = cuda.mem_alloc(mat_c.nbytes)
cuda.memcpy_htod(dev_c, mat_c)

kernel_code_template="""
__global__ void MatrixMulKernel(float *MatA, float *MatB, float *MatC)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < %(MATRIX_LEN)s) and (Col < %(MATRIX_LEN)s))
    {
        float Pvalue = 0;
        for (int k = 0; k < %(MATRIX_LEN)s; k++)
        {
            Pvalue += MatA[Row * %(MATRIX_LEN)s + k] * MatB[k * %(MATRIX_LEN)s + Col];
        }
        MatC[Row * %(MATRIX_LEN)s + Col] = Pvalue;
    }
}
"""
kernel_code = kernel_code_template % {
    'MATRIX_LEN': MATRIX_LEN 
    }
mod = compiler.SourceModule(kernel_code)
mulFunc=mod.get_function("MatrixMulKernel")
mulFunc(
    dev_a, dev_b, dev_c,
    block=(1, 1, 1),
    grid=(MATRIX_LEN, MATRIX_LEN, 1)
    )

cuda.memcpy_dtoh(mat_c, dev_c)

print("Matrix A:")
print(mat_a)
print("Matrix B:")
print(mat_b)
print("Product Matrix:")
print(mat_c)
