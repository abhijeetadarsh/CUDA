import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

TILE_WIDTH = 2;
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

source_module = SourceModule("""
__global__ void tiledMatrixMulKernel(float *mat1, float *mat2, float *mat3, int width) {
        __shared__ float mds[2][2];
        __shared__ float nds[2][2];
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * 2 + ty;
        int col = bx * 2 + tx;

        float pvalue = 0;
        for(int ph = 0; ph < width / 2; ph++) {
                mds[ty][tx] = mat1[row * width + ph * {0} + tx];
                nds[ty][tx] = mat2[(ph * 2 + ty) * width + col];
                __syncthreads();

                for(int k = 0; k < 2; k++) {
                        pvalue += mds[ty][k] * nds[k][tx];
                }
                __syncthreads();
        }
        mat3[row * width + col] = pvalue;
}
""")

tiled_matrix_multiplication_function = source_module.get_function("tiledMatrixMulKernel")
tiled_matrix_multiplication_function(dev_a, dev_b, dev_c, MATRIX_LEN, block=(1, 1, 1), grid=(MATRIX_LEN, MATRIX_LEN, 1))

cuda.memcpy_dtoh(mat_c, dev_c)

print("Matrix A:")
print(mat_a)
print("Matrix B:")
print(mat_b)
print("Product Matrix:")
print(mat_c)
