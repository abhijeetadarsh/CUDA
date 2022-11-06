// Static shmem calculation for convenience (Int 16x16 matrix)
#define SHMEM_SIZE 16*16*4

__global__ void tileMatrixMul(int *a,int *b,int *c,int n,int tile_size){
    // two statically-sized pieces of shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // shorten these parameter for clean re-use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // calculate global row and column positions for this thread
    int row = by * tile_size + ty;
}