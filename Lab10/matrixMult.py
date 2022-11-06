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

source_module = SourceModule("""
__global__ void MatrixMulKernel(float* MatA, float* MatB, float* MatC,int Width){
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	int Col = blockIdx.x*blockDim.x+threadIdx.x;
	if((Row < Width) and (Col < Width)){
		float Pvalue = 0;
		for(int k=0;k < Width;k++){
			Pvalue += MatA[Row*Width + k]*MatB[k*Width + Col];
		}
		MatC[Row*Width + Col] = Pvalue;
	}
}
""")

tiled_matrix_multiplication_function = source_module.get_function("MatrixMulKernel")
tiled_matrix_multiplication_function(dev_a, dev_b, dev_c, MATRIX_LEN, block=(1, 1, 1), grid=(MATRIX_LEN, MATRIX_LEN, 1))

cuda.memcpy_dtoh(mat_c, dev_c)

print("Matrix A:")
print(mat_a)
print("Matrix B:")
print(mat_b)
print("Product Matrix:")
print(mat_c)
