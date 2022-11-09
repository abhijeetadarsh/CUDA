import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(5,5).astype(numpy.float32))
a_doubled = (2 * a_gpu).get()
print("ORIGINAL MATRIX")
print(a_doubled)
print("DOUBLED MATRIX AFTER PyCUDA EXECUTION USING GPUARRAY CALL")
print(a_gpu)

from numbapro import guvectorize
import numpy as np

@guvectorize(['void(int64[:,:], int64[:,:], int64[:,:])'], '(m,n),(n,p)->(m,p)')

def matmul(A, B, C):
	m, n = A.shape
	n, p = B.shape
	for i in range(p):
		C[i, j] = 0;
		for k in range(n):
			C[i, j] += A[i, j] * B[k, j]

dim = 10
A = np.random.randint(dim, size=(dim, dim))
B = np.random.randint(dim, size=(dim, dim))

C = matmul(A, B)
print("INPUT MATRIX A")
print(":\n%s"%A)
print("INPUT MATRIX B")
print(":\n%s"%B)
print("RESULT MATRIX C = A*B")
print(":\n%s"%C)
