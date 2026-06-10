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
#include <cusparse.h>
#include <cublas_v2.h>
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
#define CUDSS_BRIDGE_ERR_CUSPARSE           -10
#define CUDSS_BRIDGE_ERR_CUBLAS             -11
#define CUDSS_BRIDGE_ERR_ITER_BREAKDOWN     -12
#define CUDSS_BRIDGE_ERR_ITER_NO_CONVERGE   -13

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

   // Device-side numeric assembly support. clearDeviceValues() zeros the
   // persistent CSR values buffer. addDeviceValues() performs
   // valsD[slots[i]] += scale * addVals[i] on the CUDA stream. After device
   // assembly, factorDeviceValues() factors the current device values without
   // copying a host CRS value array.
   int clearDeviceValues();
   int addDeviceValues (
      const int* slots, const double* addVals, int nvals, double scale);
   int addScaledDiagonal3DeviceValues (
      const int* diagSlots, const double* masses, int nblocks, double scale);
   int addScaledBlock3DeviceValues (
      const int* blockSlots, const double* blockVals,
      const double* blockScales, int nblocks, double scale);
   int addMaterialStiffness3DeviceValues (
      const int* blockSlots, const double* gis, const double* gjs,
      const double* Ds, const double* sigmas, const double* dvs,
      int nblocks, double scale);
   int addMaterialStiffness3ElementDeviceValues (
      const int* elemNodeCounts, const int* elemPairOffsets,
      const int* elemIpOffsets, const int* elemGradOffsets,
      const int* pairNodeIdxs, const int* blockSlots, const double* grads,
      const double* Ds, const double* sigmas, const double* dvs,
      int nelems, double scale);
   int addLinearElasticStiffness3ElementDeviceValues (
      const int* elemNodeCounts, const int* elemPairOffsets,
      const int* elemIpOffsets, const int* elemGradOffsets,
      const int* pairNodeIdxs, const int* blockSlots,
      const double* elemParams, const double* grads, const double* dvs,
      int nelems, double scale);
   int addLinearElasticStiffness3ElementGeometryDeviceValues (
      const int* elemNodeCounts, const int* elemNodeOffsets,
      const int* elemPairOffsets, const int* elemIpOffsets,
      const int* elemNaturalGradOffsets, const int* pairNodeIdxs,
      const int* blockSlots, const int* nodeDims,
      const double* nodeTransforms, const double* elemParams,
      const double* elemNodePositions, const double* naturalGrads,
      const double* ipWeights, int nelems, double scale);
   int addDilationalStiffness3ElementDeviceValues (
      const int* elemNodeCounts, const int* elemPressureCounts,
      const int* elemPairOffsets, const int* elemConstraintOffsets,
      const int* elemRinvOffsets, const int* pairNodeIdxs,
      const int* blockSlots, const double* constraints, const double* rinvs,
      int nelems, double scale);
   int factorDeviceValues();

   // Copy the current device-side CSR values buffer back to host (length nnz).
   // Debug / verification aid: lets callers compare GPU-assembled CSR values
   // against a CPU reference assembly. Returns OK or an error code.
   int getDeviceValues (double* vals);

   // Copy b[] H->D, run CUDSS_PHASE_SOLVE, copy x[] D->H. Both arrays length n.
   int solve (const double* b, double* x);

   // Device sparse matrix-vector product y = A*x, where A is the matrix
   // currently resident in the device CSR values buffer (myValsD) over the
   // analyzed pattern. Host x[] is copied H->D, the SpMV runs entirely on the
   // device (via the same cuSPARSE machinery used by the BiCGStab path), and
   // the result y[] is copied D->H. Both arrays length n. Used to evaluate the
   // velocity-Jacobian J*v term on the GPU (assemble J_v into the device values
   // buffer, then multiply by v) so the host never assembles J_v. For symmetric
   // upper-only storage the same U + U^T - diag(U) reconstruction as applyA is
   // used; for CUDSS_BRIDGE_MT_GENERAL a single SpMV suffices.
   int multiply (const double* x, double* y);

   // Multi-RHS solve. B and X are column-major dense matrices of size n x nrhs
   // stored as contiguous host arrays of length n*nrhs. The bridge grows its
   // multi-RHS device buffers lazily; the first call with nrhs > 1 allocates
   // them, and subsequent calls with a larger nrhs grow them. Calls with
   // nrhs == 1 are allowed (equivalent to single-RHS solve but slightly less
   // efficient due to descriptor rebuild).
   int solveMulti (int nrhs, const double* B, double* X);

   // Preconditioned BiCGStab iterative solve. Uses the current cuDSS factor
   // as a left preconditioner; the matrix-vector product A*x is performed via
   // cuSPARSE SpMV using the values supplied via the `vals` argument (which
   // also gets pushed to the device-side values buffer so subsequent factor()
   // calls see the same values).
   //
   // tolRel: target relative residual ||A x - b|| / ||b||
   // maxIter: hard cap on BiCGStab iterations
   //
   // Returns: number of iterations on success (>= 1, <= maxIter)
   //          0 if the initial residual was already below tol (rare)
   //          negative on failure (no convergence, breakdown, or backend error)
   //
   // For symmetric matrices stored upper-only (CUDSS_BRIDGE_MT_SYMMETRIC /
   // _SPD), A*x is computed as
   //     A x = U x + U^T x - diag(U) x
   // where U is the upper-triangular CSR. For CUDSS_BRIDGE_MT_GENERAL the
   // standard SpMV is used.
   int iterativeSolveBiCGStab (
      const double* vals, const double* b, double* x,
      double tolRel, int maxIter, int* outIters);

   // Release all native resources. Safe to call multiple times.
   void dispose();

   int getN()   const { return myN; }
   int getNnz() const { return myNnz; }

   // Returns a static C string describing the most recent failure, or
   // nullptr if the last call succeeded. Caller must not free.
   const char* getLastErrorMessage() const { return myLastErr; }

   // Per-phase timing toggle. When enabled, the bridge prints lines like
   //   [cudss-timing] factor:           45.2 ms
   //   [cudss-timing] bicgstab(iter=2): 12.3 ms (1.5 ms/iter)
   // to stderr for each major operation. Off by default; enable via
   // Java CuDssSolver.setTimingEnabled(true) or by setting the
   // CUDSS_BRIDGE_TIMING env var.
   static void setTimingEnabled (bool on);
   static bool timingEnabled();

   // Update cuDSS's CUDSS_CONFIG_IR_N_STEPS for this bridge's config.
   // The default of 4 is set in init(). MurtyMechSolver temporarily
   // sets this to 0 during specific solves where PARDISO would also
   // disable refinement (mirrors PARDISO.setMaxRefinementSteps(0)).
   int setIterativeRefinementSteps (int n);

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

   // Multi-RHS scratch state. Allocated lazily on first solveMulti() call,
   // grown if a larger nrhs is requested. Independent of the single-RHS
   // buffers above so we don't disturb that state.
   double*       myBMatD;
   double*       myXMatD;
   int           myMultiNrhs;     // current capacity in number of RHS cols
   cudssMatrix_t myMatBMulti;
   cudssMatrix_t myMatXMulti;
   bool          myMatBMultiDesc;
   bool          myMatXMultiDesc;

   // BiCGStab + SpMV scratch state, allocated lazily on first
   // iterativeSolveBiCGStab() call. Owned by this bridge; freed on dispose
   // and on setPattern (since n may change).
   bool                  myIterInited;
   cusparseHandle_t      myCuSparse;
   cublasHandle_t        myCuBlas;
   cusparseSpMatDescr_t  myCsrDesc;        // CSR view of (myValsD,
                                           // myColIdxsD, myRowOffsD)
   cusparseDnVecDescr_t  mySpmvInVec;      // bound to itVecBuf[i] per call
   cusparseDnVecDescr_t  mySpmvOutVec;     // bound to itVecBuf[j] per call
   void*                 mySpmvBuffer;     // workspace for cusparseSpMV
   size_t                mySpmvBufferBytes;
   // 8 device vectors of length n used by BiCGStab.
   double*               myItR;
   double*               myItRhat;
   double*               myItP;
   double*               myItV;
   double*               myItS;
   double*               myItT;
   double*               myItY;
   double*               myItZ;
   double*               myItX;   // BiCGStab solution accumulator;
                                  // distinct from myXVecD which cuDSS
                                  // overwrites on every preconditioner solve
   // Diagonal entries d_i = A[i,i], cached after first iterativeSolve call.
   // Used to correct the (U + U^T) double-count of the diagonal in symmetric
   // SpMV. Length n.
   double*               myItDiag;
   bool                  myItDiagDirty;    // true if vals changed since last
                                           // extraction

   const char* myLastErr;

   void destroyMatrixDescriptors();
   void destroyMultiDescriptors();
   void freeDeviceBuffers();
   void freeMultiBuffers();
   void freeIterativeBuffers();
   int  ensureIterativeState();             // lazy init of cuSPARSE/cuBLAS/buffers
   int  applyA (const double* xD, double* yD);  // y = A * x, dispatches on mtype
   int  extractDiagonal();                  // populate myItDiag from myValsD
   bool setMtype (int flag, cudssMatrixType_t& mtype, cudssMatrixViewType_t& mview);
};

#endif // CUDSS_BRIDGE_H
