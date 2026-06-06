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

// vals[i] = 0 for all CRS entries.
__global__ static void zeroValuesKernel (
   int nnz, double* __restrict__ vals) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < nnz) {
      vals[i] = 0.0;
   }
}

extern "C" void zeroValues_launch (
   int nnz, double* vals, cudaStream_t stream) {
   const int threads = 256;
   const int blocks  = (nnz + threads - 1) / threads;
   zeroValuesKernel<<<blocks, threads, 0, stream>>> (nnz, vals);
}

// crsVals[slots[i]] += scale * addVals[i].
__global__ static void scatterAddValuesKernel (
   int nvals,
   const int*    __restrict__ slots,
   const double* __restrict__ addVals,
   double scale,
   double*       __restrict__ crsVals) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < nvals) {
      atomicAdd (&crsVals[slots[i]], scale * addVals[i]);
   }
}

extern "C" void scatterAddValues_launch (
   int nvals, const int* slots, const double* addVals,
   double scale, double* crsVals, cudaStream_t stream) {
   const int threads = 256;
   const int blocks  = (nvals + threads - 1) / threads;
   scatterAddValuesKernel<<<blocks, threads, 0, stream>>>(
      nvals, slots, addVals, scale, crsVals);
}

// For each 3-DOF block i, add scale*masses[i] to the three diagonal CRS
// slots. A slot value < 0 is ignored, allowing callers to pass filtered maps.
__global__ static void addScaledDiagonal3Kernel (
   int nblocks,
   const int*    __restrict__ diagSlots,
   const double* __restrict__ masses,
   double scale,
   double*       __restrict__ crsVals) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < nblocks) {
      double v = scale * masses[i];
      int base = 3 * i;
      int s0 = diagSlots[base];
      int s1 = diagSlots[base + 1];
      int s2 = diagSlots[base + 2];
      if (s0 >= 0) {
         atomicAdd (&crsVals[s0], v);
      }
      if (s1 >= 0) {
         atomicAdd (&crsVals[s1], v);
      }
      if (s2 >= 0) {
         atomicAdd (&crsVals[s2], v);
      }
   }
}

extern "C" void addScaledDiagonal3_launch (
   int nblocks, const int* diagSlots, const double* masses,
   double scale, double* crsVals, cudaStream_t stream) {
   const int threads = 256;
   const int blocks  = (nblocks + threads - 1) / threads;
   addScaledDiagonal3Kernel<<<blocks, threads, 0, stream>>>(
      nblocks, diagSlots, masses, scale, crsVals);
}

// For each 3x3 block i, add globalScale*blockScales[i]*blockVals[9*i+j]
// to crsVals[blockSlots[9*i+j]]. A slot value < 0 is ignored.
__global__ static void addScaledBlock3Kernel (
   int nblocks,
   const int*    __restrict__ blockSlots,
   const double* __restrict__ blockVals,
   const double* __restrict__ blockScales,
   double globalScale,
   double*       __restrict__ crsVals) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < nblocks) {
      int base = 9 * i;
      double s = globalScale * blockScales[i];
      for (int j = 0; j < 9; j++) {
         int slot = blockSlots[base + j];
         if (slot >= 0) {
            atomicAdd (&crsVals[slot], s * blockVals[base + j]);
         }
      }
   }
}

extern "C" void addScaledBlock3_launch (
   int nblocks, const int* blockSlots, const double* blockVals,
   const double* blockScales, double globalScale, double* crsVals,
   cudaStream_t stream) {
   const int threads = 256;
   const int blocks  = (nblocks + threads - 1) / threads;
   addScaledBlock3Kernel<<<blocks, threads, 0, stream>>>(
      nblocks, blockSlots, blockVals, blockScales, globalScale, crsVals);
}

// Computes FemUtilities.addMaterialStiffness(K, gi, D, sig, gj, dv) for
// each descriptor and scatters the resulting 3x3 block to CRS slots.
__global__ static void addMaterialStiffness3Kernel (
   int nblocks,
   const int*    __restrict__ blockSlots,
   const double* __restrict__ gis,
   const double* __restrict__ gjs,
   const double* __restrict__ Ds,
   const double* __restrict__ sigmas,
   const double* __restrict__ dvs,
   double globalScale,
   double*       __restrict__ crsVals) {
   int b = blockIdx.x * blockDim.x + threadIdx.x;
   if (b >= nblocks) {
      return;
   }
   const double* D = Ds + 36*b;
   const double* sig = sigmas + 6*b;
   int vbase = 3*b;
   int sbase = 9*b;

   double gix = gis[vbase];
   double giy = gis[vbase + 1];
   double giz = gis[vbase + 2];
   double gjx = gjs[vbase] * dvs[b];
   double gjy = gjs[vbase + 1] * dvs[b];
   double gjz = gjs[vbase + 2] * dvs[b];

   double dm00 = D[0]*gjx  + D[3]*gjy  + D[5]*gjz;
   double dm01 = D[1]*gjy  + D[3]*gjx  + D[4]*gjz;
   double dm02 = D[2]*gjz  + D[4]*gjy  + D[5]*gjx;

   double dm10 = D[6]*gjx  + D[9]*gjy  + D[11]*gjz;
   double dm11 = D[7]*gjy  + D[9]*gjx  + D[10]*gjz;
   double dm12 = D[8]*gjz  + D[10]*gjy + D[11]*gjx;

   double dm20 = D[12]*gjx + D[15]*gjy + D[17]*gjz;
   double dm21 = D[13]*gjy + D[15]*gjx + D[16]*gjz;
   double dm22 = D[14]*gjz + D[16]*gjy + D[17]*gjx;

   double dm30 = D[18]*gjx + D[21]*gjy + D[23]*gjz;
   double dm31 = D[19]*gjy + D[21]*gjx + D[22]*gjz;
   double dm32 = D[20]*gjz + D[22]*gjy + D[23]*gjx;

   double dm40 = D[24]*gjx + D[27]*gjy + D[29]*gjz;
   double dm41 = D[25]*gjy + D[27]*gjx + D[28]*gjz;
   double dm42 = D[26]*gjz + D[28]*gjy + D[29]*gjx;

   double dm50 = D[30]*gjx + D[33]*gjy + D[35]*gjz;
   double dm51 = D[31]*gjy + D[33]*gjx + D[34]*gjz;
   double dm52 = D[32]*gjz + D[34]*gjy + D[35]*gjx;

   double K[9];
   K[0] = gix*dm00 + giy*dm30 + giz*dm50;
   K[1] = gix*dm01 + giy*dm31 + giz*dm51;
   K[2] = gix*dm02 + giy*dm32 + giz*dm52;
   K[3] = giy*dm10 + gix*dm30 + giz*dm40;
   K[4] = giy*dm11 + gix*dm31 + giz*dm41;
   K[5] = giy*dm12 + gix*dm32 + giz*dm42;
   K[6] = giz*dm20 + giy*dm40 + gix*dm50;
   K[7] = giz*dm21 + giy*dm41 + gix*dm51;
   K[8] = giz*dm22 + giy*dm42 + gix*dm52;

   double sx = sig[0]*gjs[vbase] + sig[3]*gjs[vbase + 1] +
      sig[5]*gjs[vbase + 2];
   double sy = sig[3]*gjs[vbase] + sig[1]*gjs[vbase + 1] +
      sig[4]*gjs[vbase + 2];
   double sz = sig[5]*gjs[vbase] + sig[4]*gjs[vbase + 1] +
      sig[2]*gjs[vbase + 2];
   double Kg = (gix*sx + giy*sy + giz*sz) * dvs[b];
   K[0] += Kg;
   K[4] += Kg;
   K[8] += Kg;

   for (int j = 0; j < 9; j++) {
      int slot = blockSlots[sbase + j];
      if (slot >= 0) {
         atomicAdd (&crsVals[slot], globalScale * K[j]);
      }
   }
}

extern "C" void addMaterialStiffness3_launch (
   int nblocks, const int* blockSlots, const double* gis,
   const double* gjs, const double* Ds, const double* sigmas,
   const double* dvs, double globalScale, double* crsVals,
   cudaStream_t stream) {
   const int threads = 256;
   const int blocks  = (nblocks + threads - 1) / threads;
   addMaterialStiffness3Kernel<<<blocks, threads, 0, stream>>>(
      nblocks, blockSlots, gis, gjs, Ds, sigmas, dvs, globalScale, crsVals);
}

__device__ static void addMaterialStiffness3Block (
   const int* slots, const double* gi, const double* D,
   const double* sig, const double* gjRaw, double dv, double globalScale,
   double* crsVals) {

   double gix = gi[0];
   double giy = gi[1];
   double giz = gi[2];
   double gjx = gjRaw[0] * dv;
   double gjy = gjRaw[1] * dv;
   double gjz = gjRaw[2] * dv;

   double dm00 = D[0]*gjx  + D[3]*gjy  + D[5]*gjz;
   double dm01 = D[1]*gjy  + D[3]*gjx  + D[4]*gjz;
   double dm02 = D[2]*gjz  + D[4]*gjy  + D[5]*gjx;

   double dm10 = D[6]*gjx  + D[9]*gjy  + D[11]*gjz;
   double dm11 = D[7]*gjy  + D[9]*gjx  + D[10]*gjz;
   double dm12 = D[8]*gjz  + D[10]*gjy + D[11]*gjx;

   double dm20 = D[12]*gjx + D[15]*gjy + D[17]*gjz;
   double dm21 = D[13]*gjy + D[15]*gjx + D[16]*gjz;
   double dm22 = D[14]*gjz + D[16]*gjy + D[17]*gjx;

   double dm30 = D[18]*gjx + D[21]*gjy + D[23]*gjz;
   double dm31 = D[19]*gjy + D[21]*gjx + D[22]*gjz;
   double dm32 = D[20]*gjz + D[22]*gjy + D[23]*gjx;

   double dm40 = D[24]*gjx + D[27]*gjy + D[29]*gjz;
   double dm41 = D[25]*gjy + D[27]*gjx + D[28]*gjz;
   double dm42 = D[26]*gjz + D[28]*gjy + D[29]*gjx;

   double dm50 = D[30]*gjx + D[33]*gjy + D[35]*gjz;
   double dm51 = D[31]*gjy + D[33]*gjx + D[34]*gjz;
   double dm52 = D[32]*gjz + D[34]*gjy + D[35]*gjx;

   double K[9];
   K[0] = gix*dm00 + giy*dm30 + giz*dm50;
   K[1] = gix*dm01 + giy*dm31 + giz*dm51;
   K[2] = gix*dm02 + giy*dm32 + giz*dm52;
   K[3] = giy*dm10 + gix*dm30 + giz*dm40;
   K[4] = giy*dm11 + gix*dm31 + giz*dm41;
   K[5] = giy*dm12 + gix*dm32 + giz*dm42;
   K[6] = giz*dm20 + giy*dm40 + gix*dm50;
   K[7] = giz*dm21 + giy*dm41 + gix*dm51;
   K[8] = giz*dm22 + giy*dm42 + gix*dm52;

   double sx = sig[0]*gjRaw[0] + sig[3]*gjRaw[1] + sig[5]*gjRaw[2];
   double sy = sig[3]*gjRaw[0] + sig[1]*gjRaw[1] + sig[4]*gjRaw[2];
   double sz = sig[5]*gjRaw[0] + sig[4]*gjRaw[1] + sig[2]*gjRaw[2];
   double Kg = (gix*sx + giy*sy + giz*sz) * dv;
   K[0] += Kg;
   K[4] += Kg;
   K[8] += Kg;

   for (int j = 0; j < 9; j++) {
      int slot = slots[j];
      if (slot >= 0) {
         atomicAdd (&crsVals[slot], globalScale * K[j]);
      }
   }
}

__global__ static void addMaterialStiffness3ElementKernel (
   int nelems,
   const int*    __restrict__ elemNodeCounts,
   const int*    __restrict__ elemPairOffsets,
   const int*    __restrict__ elemIpOffsets,
   const int*    __restrict__ elemGradOffsets,
   const int*    __restrict__ pairNodeIdxs,
   const int*    __restrict__ blockSlots,
   const double* __restrict__ grads,
   const double* __restrict__ Ds,
   const double* __restrict__ sigmas,
   const double* __restrict__ dvs,
   double globalScale,
   double*       __restrict__ crsVals) {

   int e = blockIdx.x;
   if (e >= nelems) {
      return;
   }
   int nnodes = elemNodeCounts[e];
   int pair0 = elemPairOffsets[e];
   int npairs = elemPairOffsets[e + 1] - pair0;
   int ip0 = elemIpOffsets[e];
   int nips = elemIpOffsets[e + 1] - ip0;
   int grad0 = elemGradOffsets[e];
   int total = nips * npairs;

   for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
      int ip = idx / npairs;
      int pair = idx - ip * npairs;
      int pidx = pair0 + pair;
      int i = pairNodeIdxs[2*pidx];
      int j = pairNodeIdxs[2*pidx + 1];
      int ipidx = ip0 + ip;
      int gbase = grad0 + ip * nnodes;
      const double* gi = grads + 3*(gbase + i);
      const double* gj = grads + 3*(gbase + j);
      addMaterialStiffness3Block (
         blockSlots + 9*pidx, gi, Ds + 36*ipidx, sigmas + 6*ipidx,
         gj, dvs[ipidx], globalScale, crsVals);
   }
}

extern "C" void addMaterialStiffness3Element_launch (
   int nelems, const int* elemNodeCounts, const int* elemPairOffsets,
   const int* elemIpOffsets, const int* elemGradOffsets,
   const int* pairNodeIdxs, const int* blockSlots, const double* grads,
   const double* Ds, const double* sigmas, const double* dvs,
   double globalScale, double* crsVals, cudaStream_t stream) {
   const int threads = 256;
   addMaterialStiffness3ElementKernel<<<nelems, threads, 0, stream>>>(
      nelems, elemNodeCounts, elemPairOffsets, elemIpOffsets,
      elemGradOffsets, pairNodeIdxs, blockSlots, grads, Ds, sigmas, dvs,
      globalScale, crsVals);
}

__device__ static void fillLinearElasticD (
   double E, double nu, double* D) {

   for (int i=0; i<36; i++) {
      D[i] = 0.0;
   }
   double a = E / (1.0 + nu);
   double dia = (1.0 - nu) / (1.0 - 2.0*nu) * a;
   double mu = 0.5 * a;
   double off = nu / (1.0 - 2.0*nu) * a;

   D[0] = dia; D[1] = off; D[2] = off;
   D[6] = off; D[7] = dia; D[8] = off;
   D[12] = off; D[13] = off; D[14] = dia;
   D[21] = mu;
   D[28] = mu;
   D[35] = mu;
}

__global__ static void addLinearElasticStiffness3ElementKernel (
   int nelems,
   const int*    __restrict__ elemNodeCounts,
   const int*    __restrict__ elemPairOffsets,
   const int*    __restrict__ elemIpOffsets,
   const int*    __restrict__ elemGradOffsets,
   const int*    __restrict__ pairNodeIdxs,
   const int*    __restrict__ blockSlots,
   const double* __restrict__ elemParams,
   const double* __restrict__ grads,
   const double* __restrict__ dvs,
   double globalScale,
   double*       __restrict__ crsVals) {

   int e = blockIdx.x;
   if (e >= nelems) {
      return;
   }
   int nnodes = elemNodeCounts[e];
   int pair0 = elemPairOffsets[e];
   int npairs = elemPairOffsets[e + 1] - pair0;
   int ip0 = elemIpOffsets[e];
   int nips = elemIpOffsets[e + 1] - ip0;
   int grad0 = elemGradOffsets[e];
   int total = nips * npairs;

   double D[36];
   double sig[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
   double E = elemParams[2*e];
   double nu = elemParams[2*e + 1];
   fillLinearElasticD (E, nu, D);

   for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
      int ip = idx / npairs;
      int pair = idx - ip * npairs;
      int pidx = pair0 + pair;
      int i = pairNodeIdxs[2*pidx];
      int j = pairNodeIdxs[2*pidx + 1];
      int ipidx = ip0 + ip;
      int gbase = grad0 + ip * nnodes;
      const double* gi = grads + 3*(gbase + i);
      const double* gj = grads + 3*(gbase + j);
      addMaterialStiffness3Block (
         blockSlots + 9*pidx, gi, D, sig, gj, dvs[ipidx],
         globalScale, crsVals);
   }
}

extern "C" void addLinearElasticStiffness3Element_launch (
   int nelems, const int* elemNodeCounts, const int* elemPairOffsets,
   const int* elemIpOffsets, const int* elemGradOffsets,
   const int* pairNodeIdxs, const int* blockSlots,
   const double* elemParams, const double* grads, const double* dvs,
   double globalScale, double* crsVals, cudaStream_t stream) {
   const int threads = 256;
   addLinearElasticStiffness3ElementKernel<<<nelems, threads, 0, stream>>>(
      nelems, elemNodeCounts, elemPairOffsets, elemIpOffsets,
      elemGradOffsets, pairNodeIdxs, blockSlots, elemParams, grads, dvs,
      globalScale, crsVals);
}

__global__ static void addDilationalStiffness3ElementKernel (
   int nelems,
   const int*    __restrict__ elemNodeCounts,
   const int*    __restrict__ elemPressureCounts,
   const int*    __restrict__ elemPairOffsets,
   const int*    __restrict__ elemConstraintOffsets,
   const int*    __restrict__ elemRinvOffsets,
   const int*    __restrict__ pairNodeIdxs,
   const int*    __restrict__ blockSlots,
   const double* __restrict__ constraints,
   const double* __restrict__ rinvs,
   double globalScale,
   double*       __restrict__ crsVals) {

   int e = blockIdx.x;
   if (e >= nelems) {
      return;
   }
   int np = elemPressureCounts[e];
   if (np <= 0) {
      return;
   }
   int pair0 = elemPairOffsets[e];
   int npairs = elemPairOffsets[e + 1] - pair0;
   int c0 = elemConstraintOffsets[e];
   int r0 = elemRinvOffsets[e];

   for (int pair = threadIdx.x; pair < npairs; pair += blockDim.x) {
      int pidx = pair0 + pair;
      int i = pairNodeIdxs[2*pidx];
      int j = pairNodeIdxs[2*pidx + 1];
      int ci0 = c0 + i*np;
      int cj0 = c0 + j*np;

      double K[9];
      for (int k=0; k<9; k++) {
         K[k] = 0.0;
      }
      for (int a=0; a<np; a++) {
         const double* ci = constraints + 3*(ci0 + a);
         for (int b=0; b<np; b++) {
            double r = rinvs[r0 + a*np + b];
            const double* cj = constraints + 3*(cj0 + b);
            K[0] += ci[0]*r*cj[0];
            K[1] += ci[0]*r*cj[1];
            K[2] += ci[0]*r*cj[2];
            K[3] += ci[1]*r*cj[0];
            K[4] += ci[1]*r*cj[1];
            K[5] += ci[1]*r*cj[2];
            K[6] += ci[2]*r*cj[0];
            K[7] += ci[2]*r*cj[1];
            K[8] += ci[2]*r*cj[2];
         }
      }
      const int* slots = blockSlots + 9*pidx;
      for (int k=0; k<9; k++) {
         int slot = slots[k];
         if (slot >= 0) {
            atomicAdd (&crsVals[slot], globalScale*K[k]);
         }
      }
   }
}

extern "C" void addDilationalStiffness3Element_launch (
   int nelems, const int* elemNodeCounts, const int* elemPressureCounts,
   const int* elemPairOffsets, const int* elemConstraintOffsets,
   const int* elemRinvOffsets, const int* pairNodeIdxs,
   const int* blockSlots, const double* constraints, const double* rinvs,
   double globalScale, double* crsVals, cudaStream_t stream) {
   const int threads = 256;
   addDilationalStiffness3ElementKernel<<<nelems, threads, 0, stream>>>(
      nelems, elemNodeCounts, elemPressureCounts, elemPairOffsets,
      elemConstraintOffsets, elemRinvOffsets, pairNodeIdxs, blockSlots,
      constraints, rinvs, globalScale, crsVals);
}
