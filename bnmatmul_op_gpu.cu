#if GOOGLE_CUDA
#define EIGEN_USE_GPU

//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <cublas_v2.h>
#include "bnmatmul_op.h"

#define BLOCK_SIZE 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace tensorflow {
// 32 single float array ->  32 bits unsigned int
__device__ unsigned int concatenate(const float* array, int size)
{
    unsigned int rvalue=0;
    unsigned int sign;
    
    for (int i = 0; i < size; i++)
    {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }
    
    return rvalue;
}

// n is the original dimension
// launch this like
// the bigger the block size, the better
// dim3 bDim(blockSize,blockSize);
// dim3 gDim( m/32 / blockSize + 1, n/ stride / blockSize + 1);
//concatenate_rows_kernel<<<gDim, bDim>>>(fA, Aconc, m, n, stride);
// n is the original dimension
__global__ void concatenate_rows_kernel(const float *a, unsigned int *b, int m, int n, int stride)
{ 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int realCol = col * 32;
    if ( realCol >= n ) 
        return;
    int startRow = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
    int totalBCol = (n+31)/32;
    int size = 32;
    if ( realCol > (n - size) ) 
        size = n - realCol;
    for ( int s = 0; s < stride; s++ ) {
        int realRow = startRow + s;
        if ( realRow >= m )
            return;
        int offset = realRow * totalBCol + col;
        int offsetF = realRow * n + realCol;
        b[offset] = concatenate(&a[offsetF], size);
    }
}

// m and n are the rows and cols of the transposed matrix, not the original
// to transpose, use
//      float const alpha(1.0);
//      float const beta(0.0);
//      cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, N, K, &alpha, fB, K, &beta, fB, N, BT, N );
__global__ void concatenate_cols_T_kernel(const float *a, unsigned int *b, int m, int n, int stride)
{ 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int realCol = col * 32;
    int startRow = (blockIdx.y * blockDim.y + threadIdx.y) * stride;
    if ( realCol >= n )
        return;
    int size = 32;
    if ( realCol > (n - size) ) 
        size = n - realCol;
    for ( int s = 0; s < stride; s++ ) {
        int realRow = startRow + s;
        if ( realRow >= m )
            return;
        //int offset = realRow  * totalBCol + col ;
        int offset = col * m + realRow;
        int offsetF = realRow * n + realCol;
        //int offsetF = col * m + realRow;
        b[offset] = concatenate(&a[offsetF], size);
    }
}

// n is the original dimension
// launch this like
// dim3 bDim(blockSize,blockSize);
// dim3 gDim2(n /stride/ blockSize +1 , k / 32 / blockSize + 1);
//concatenate_cols_kernel<<<gDim2, bDim>>>(fB, Bconc, n, k, stride);
__global__ void concatenate_cols_kernel(const float *a, unsigned int *b, int n, int k, int stride)
{   

    int startCol = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int realRow = row * 32;
    float *array = new float[32];
    for ( int s =0; s < stride; ++s )
    {
        int realCol = startCol + s;
        if ( realCol >= k || realRow >= n ) {
            delete[] array;
            return;
        }
        int offset = row * k + realCol;
        int size = 32;
        if ( realRow > (n - size) ) 
            size = n - realRow;
        for ( int i= 0; i < size; i++ ) 
        {
            array[i] = a[(realRow + i) * k + realCol];
        } 
        b[offset] = concatenate(array, size);
        /*
        if ( realCol == 32 ) {
            printf("size to take %d is %d\n", row, size);
            printf("row %d %u \n", (row), b[offset]);
        }*/
        //printf("processing %d with size %d, starting %.2f, result %d\n", offset, size, a[(realRow*k + realCol)], b[offset]);
    }
    delete[] array;
}


// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
// launch like this
// tiled memory requires constant value
//dim3 blockDim(16, 16);
//dim3 gridDim(N / 16 + 1, N / 16 + 1);
//xnor_gemm<<<gridDim, blockDim>>>(Aconc, Bconc, fC, m, n, k);
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k) 
{
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    int realRow = blockRow*BLOCK_SIZE + row;
    int realCol = blockCol*BLOCK_SIZE+ col;

    // Each thread block computes one sub-matrix Csub of C
    //float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];
    float* Csub = &C[blockDim.y * k * blockRow + blockDim.y * blockCol];

    int ndex = int((n + 32 - 1)/32);
    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int Cvalue = 0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (ndex + BLOCK_SIZE-1) / BLOCK_SIZE; ++i) {
        int ibs = BLOCK_SIZE * i;
    
        // Get sub-matrix Asub of A
        unsigned int* Asub = &A[BLOCK_SIZE * blockRow * ndex + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        unsigned int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        if ( (ibs + col) < ndex && realRow < m )  {
            As[row][col] = Asub[row*ndex+col];
        } else {
            As[row][col] = 0;
        }

        if ( (ibs + row) < ndex && realCol < k ) {
            Bs[row][col] = Bsub[row*k+col];
        } else {
            Bs[row][col] = 0;
        }
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            Cvalue += __popc(As[row][j]^Bs[j][col]);
        }
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
    if( realCol < k && realRow < m) {
        Csub[row*k+col] = -(2*(float)Cvalue-n);
    }
}

void tobinstr(unsigned int value, int bitsCount, char* output)
{
    int i;
    output[bitsCount] = '\0';
    for (i = bitsCount - 1; i >= 0; --i, value >>= 1)
    {
        output[i] = (value & 1) + '0';
    }
}
void print2i(unsigned int *a, int m, int n) 
{
    for (int i =0; i< m; i++) {
        for ( int j=0; j <n ; j++ ) {
            //printf("%u ", a[i*n + j]);
            char output[33];
            tobinstr(a[i*n +j], 32, output); 
            printf("%s ", output);
        }
        printf("\n");
    }
}
void print2f(const float *a, int m, int n)
{
    for (int i =0; i< m; i++) {
        for ( int j=0; j <n ; j++ ) {
            printf("%.1f ", a[i*n + j]);
        }
        printf("\n");
    }
}

// CUDA likes to use m * k and k *n. but here I am using m*n and n*k for dimension
void XNORGemmKernelDevice(OpKernelContext *ctx, const float* in0, const float* in1, const int m, const int n, const int k, float* out) 
{
    auto d = ctx->eigen_device<Eigen::GpuDevice>();
    unsigned int* binary_in0 = reinterpret_cast<unsigned int*>(d.allocate(m * (n+32-1)/32 * sizeof(unsigned int)));
    unsigned int* binary_in1 = reinterpret_cast<unsigned int*>(d.allocate((n+32-1)/32 * k * sizeof(unsigned int)));
    float *in1T = reinterpret_cast<float *>(d.allocate(n * k * sizeof(float)));

    // smaller stride, larger blocksize helps
    int stride = 4;
    // 64 blows up? the benchmark program seems to run fine
    int blockSize = 32;
    dim3 bDim(blockSize,blockSize);
    dim3 gDim( n/sizeof(unsigned int) / blockSize + 1, m/ stride / blockSize + 1);
    concatenate_rows_kernel<<<gDim, bDim, 0, d.stream()>>>(in0, binary_in0, m, n, stride);
    gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );
    //print2f(result2, m, n);

    // this is slower than the tranpose + concatenate cols_T due to memory access latency
    //dim3 bDim2(blockSize,blockSize);
    //dim3 gDim2(k / stride / blockSize + 1, n /sizeof(unsigned int)/ blockSize +1 );
    //concatenate_cols_kernel<<<gDim2, bDim2, 0, d.stream()>>>(in1, binary_in1, n, k, stride);
    
    //auto stream = ctx->op_device_context()->stream();
    //stream->ThenBlasSgeam( CUBLAS_OP_T, CUBLAS_OP_N, n, k, &alpha, in1, k, &beta, in1, n, in1T, n );
    float const alpha(1.0);
    float const beta(0.0);
    // this handle slows down the execution
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, d.stream());
    cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, n, k, &alpha, in1, k, &beta, in1, n, in1T, n );
    dim3 bDim2(blockSize,blockSize);
    dim3 gDim2(n /sizeof(unsigned int)/ blockSize +1 , k / stride / blockSize + 1);
    concatenate_cols_T_kernel<<<gDim2, bDim2, 0, d.stream()>>>(in1T, binary_in1, k, n, stride);
    gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(k / BLOCK_SIZE + 1, m / BLOCK_SIZE + 1);
    xnor_gemm<<<gridDim, blockDim,0,d.stream()>>>(binary_in0, binary_in1, out, m, n, k);
    d.deallocate(binary_in0);
    d.deallocate(binary_in1);
    d.deallocate(in1T);
    cublasDestroy(handle);
}

}
#endif
