// Small CUDA kernels used by CuDssBridge's BiCGStab path. Compiled by nvcc
// and linked into libCuDssJNI alongside the C++ sources. Each kernel is
// a single stream-bound launch; sync is done by the caller via
// cudaStreamSynchronize.

#include <cuda_runtime.h>

// y[i] -= d[i] * x[i]
__global__ static void axmyKernel (
   int n, const double* __restrict__ d,
   const double* __restrict__ x, double* __restrict__ y) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) {
      y[i] -= d[i] * x[i];
   }
}

extern "C" void axmy_launch (
   int n, const double* d, const double* x, double* y,
   cudaStream_t stream) {
   const int threads = 256;
   const int blocks  = (n + threads - 1) / threads;
   axmyKernel<<<blocks, threads, 0, stream>>> (n, d, x, y);
}

// diag[i] = A[i,i], reading from CSR (rowOffs, colIdxs, vals).
__global__ static void extractDiagKernel (
   int n,
   const int*    __restrict__ rowOffs,
   const int*    __restrict__ colIdxs,
   const double* __restrict__ vals,
   double*       __restrict__ diag) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i >= n) return;
   int rowStart = rowOffs[i];
   int rowEnd   = rowOffs[i + 1];
   double d = 0.0;
   for (int p = rowStart; p < rowEnd; p++) {
      if (colIdxs[p] == i) {
         d = vals[p];
         break;
      }
   }
   diag[i] = d;
}

extern "C" void extractDiag_launch (
   int n, const int* rowOffs, const int* colIdxs, const double* vals,
   double* diag, cudaStream_t stream) {
   const int threads = 256;
   const int blocks  = (n + threads - 1) / threads;
   extractDiagKernel<<<blocks, threads, 0, stream>>>(
      n, rowOffs, colIdxs, vals, diag);
}
