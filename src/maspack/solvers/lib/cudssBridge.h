// cudssBridge: thin C++ wrapper around NVIDIA cuDSS that holds the GPU-side
// state of a single sparse direct factorization between calls. Designed to
// be driven from Java via CuDssJNI.cc. Indices are 0-based.
//
// Lifetime:
//   CuDssBridge b;
//   b.init();
//   b.setPattern(n, nnz, rowOffs0Based, colIdxs0Based, mtype);
//   b.analyze();
//   b.factor(vals);            // first factor after analyze: full factorization
//   b.factor(vals2);           // subsequent: CUDSS_PHASE_REFACTORIZATION
//   b.solve(b_host, x_host);
//   b.dispose();
//
// Status return values:
//   0  success
//  <0  error (see CUDSS_BRIDGE_ERR_* constants)
//
// On error, getLastErrorMessage() returns a static C string suitable for
// throwing as a Java exception.

#ifndef CUDSS_BRIDGE_H
#define CUDSS_BRIDGE_H

#include <cuda_runtime.h>
#include "cudss.h"

// Matrix-type constants exchanged with Java. These mirror the values used by
// maspack.matrix.Matrix so we don't have to interpret bit flags in C++:
//   CUDSS_BRIDGE_MT_GENERAL    -> CUDSS_MTYPE_GENERAL,   CUDSS_MVIEW_FULL
//   CUDSS_BRIDGE_MT_SYMMETRIC  -> CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER
//   CUDSS_BRIDGE_MT_SPD        -> CUDSS_MTYPE_SPD,       CUDSS_MVIEW_UPPER
#define CUDSS_BRIDGE_MT_GENERAL    0
#define CUDSS_BRIDGE_MT_SYMMETRIC  1
#define CUDSS_BRIDGE_MT_SPD        2

#define CUDSS_BRIDGE_OK                       0
#define CUDSS_BRIDGE_ERR_INIT                -1
#define CUDSS_BRIDGE_ERR_CUDA_ALLOC          -2
#define CUDSS_BRIDGE_ERR_CUDA_COPY           -3
#define CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR    -4
#define CUDSS_BRIDGE_ERR_ANALYSIS            -5
#define CUDSS_BRIDGE_ERR_FACTORIZATION       -6
#define CUDSS_BRIDGE_ERR_SOLVE               -7
#define CUDSS_BRIDGE_ERR_STATE               -8
#define CUDSS_BRIDGE_ERR_UNKNOWN_MTYPE       -9

class CuDssBridge {
public:
   CuDssBridge();
   ~CuDssBridge();

   // Acquire CUDA + cuDSS handles. Safe to call once. Subsequent calls are
   // no-ops and return OK.
   int init();

   // Replace any previous pattern. Allocates device buffers for indices,
   // values, and one RHS/solution vector. After this call the bridge holds:
   //   row offsets and column indices on the device (immutable until next
   //   call to setPattern), uninitialized value/x/b buffers, and an
   //   analyzed=false flag.
   //
   // rowOffs has length n+1; colIdxs and the future vals[] both have length
   // nnz. Indices must be 0-based.
   int setPattern (int n, int nnz,
                   const int* rowOffs, const int* colIdxs,
                   int mtypeFlag);

   // Run CUDSS_PHASE_ANALYSIS on the current pattern.
   int analyze();

   // Copy vals[] host->device and run CUDSS_PHASE_FACTORIZATION (first call
   // after analyze) or CUDSS_PHASE_REFACTORIZATION (subsequent calls with
   // the same pattern). vals[] length must equal nnz from setPattern.
   int factor (const double* vals);

   // Copy b[] H->D, run CUDSS_PHASE_SOLVE, copy x[] D->H. Both arrays length n.
   int solve (const double* b, double* x);

   // Release all native resources. Safe to call multiple times.
   void dispose();

   int getN()   const { return myN; }
   int getNnz() const { return myNnz; }

   // Returns a static C string describing the most recent failure, or
   // nullptr if the last call succeeded. Caller must not free.
   const char* getLastErrorMessage() const { return myLastErr; }

private:
   bool myInitialized;
   bool myHasPattern;
   bool myAnalyzed;
   bool myFactored;

   int myN;
   int myNnz;
   int myMtypeFlag;

   cudssHandle_t myHandle;
   cudssConfig_t myConfig;
   cudssData_t   myData;
   cudaStream_t  myStream;

   cudssMatrix_t myMatA;
   cudssMatrix_t myMatX;
   cudssMatrix_t myMatB;
   bool myMatADesc;
   bool myMatXDesc;
   bool myMatBDesc;

   int*    myRowOffsD;
   int*    myColIdxsD;
   double* myValsD;
   double* myBVecD;
   double* myXVecD;

   const char* myLastErr;

   void destroyMatrixDescriptors();
   void freeDeviceBuffers();
   bool setMtype (int flag, cudssMatrixType_t& mtype, cudssMatrixViewType_t& mview);
};

#endif // CUDSS_BRIDGE_H
