#include "cudssBridge.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

namespace {

inline bool cudaOk (cudaError_t e) { return e == cudaSuccess; }
inline bool dssOk  (cudssStatus_t s) { return s == CUDSS_STATUS_SUCCESS; }
inline bool spOk   (cusparseStatus_t s) { return s == CUSPARSE_STATUS_SUCCESS; }
inline bool blOk   (cublasStatus_t s)   { return s == CUBLAS_STATUS_SUCCESS; }

// Process-wide timing toggle. Set via setTimingEnabled() from Java or by
// CUDSS_BRIDGE_TIMING env var at first init().
static bool g_timing = false;
static bool g_timing_initialized = false;

// Returns wall time in milliseconds.
inline double nowMs() {
   timespec ts;
   clock_gettime (CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1.0e3 + ts.tv_nsec * 1.0e-6;
}

} // anon

void CuDssBridge::setTimingEnabled (bool on) {
   g_timing = on;
   g_timing_initialized = true;
}

bool CuDssBridge::timingEnabled() {
   if (!g_timing_initialized) {
      const char* env = std::getenv ("CUDSS_BRIDGE_TIMING");
      if (env && env[0] && env[0] != '0') g_timing = true;
      g_timing_initialized = true;
   }
   return g_timing;
}

int CuDssBridge::setIterativeRefinementSteps (int n) {
   if (!myInitialized || !myConfig) {
      myLastErr = "setIterativeRefinementSteps called before init";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   if (!dssOk (cudssConfigSet (
          myConfig, CUDSS_CONFIG_IR_N_STEPS, &n, sizeof(int)))) {
      myLastErr = "cudssConfigSet(IR_N_STEPS) failed";
      return CUDSS_BRIDGE_ERR_INIT;
   }
   myLastErr = nullptr;
   return CUDSS_BRIDGE_OK;
}

CuDssBridge::CuDssBridge()
   : myInitialized(false),
     myHasPattern(false),
     myAnalyzed(false),
     myFactored(false),
     myN(0),
     myNnz(0),
     myMtypeFlag(CUDSS_BRIDGE_MT_GENERAL),
     myHandle(nullptr),
     myConfig(nullptr),
     myData(nullptr),
     myStream(nullptr),
     myMatA(nullptr),
     myMatX(nullptr),
     myMatB(nullptr),
     myMatADesc(false),
     myMatXDesc(false),
     myMatBDesc(false),
     myRowOffsD(nullptr),
     myColIdxsD(nullptr),
     myValsD(nullptr),
     myBVecD(nullptr),
     myXVecD(nullptr),
     myBMatD(nullptr),
     myXMatD(nullptr),
     myMultiNrhs(0),
     myMatBMulti(nullptr),
     myMatXMulti(nullptr),
     myMatBMultiDesc(false),
     myMatXMultiDesc(false),
     myIterInited(false),
     myCuSparse(nullptr),
     myCuBlas(nullptr),
     myCsrDesc(nullptr),
     mySpmvInVec(nullptr),
     mySpmvOutVec(nullptr),
     mySpmvBuffer(nullptr),
     mySpmvBufferBytes(0),
     myItR(nullptr), myItRhat(nullptr), myItP(nullptr), myItV(nullptr),
     myItS(nullptr), myItT(nullptr), myItY(nullptr), myItZ(nullptr),
     myItX(nullptr),
     myItDiag(nullptr),
     myItDiagDirty(true),
     myLastErr(nullptr)
{}

CuDssBridge::~CuDssBridge() {
   dispose();
}

int CuDssBridge::init() {
   if (myInitialized) {
      myLastErr = nullptr;
      return CUDSS_BRIDGE_OK;
   }
   if (!cudaOk (cudaStreamCreate (&myStream))) {
      myLastErr = "cudaStreamCreate failed (no CUDA-capable device?)";
      return CUDSS_BRIDGE_ERR_INIT;
   }
   if (!dssOk (cudssCreate (&myHandle))) {
      cudaStreamDestroy (myStream);
      myStream = nullptr;
      myLastErr = "cudssCreate failed";
      return CUDSS_BRIDGE_ERR_INIT;
   }
   if (!dssOk (cudssSetStream (myHandle, myStream))) {
      cudssDestroy (myHandle);
      myHandle = nullptr;
      cudaStreamDestroy (myStream);
      myStream = nullptr;
      myLastErr = "cudssSetStream failed";
      return CUDSS_BRIDGE_ERR_INIT;
   }
   if (!dssOk (cudssConfigCreate (&myConfig))) {
      cudssDestroy (myHandle);
      myHandle = nullptr;
      cudaStreamDestroy (myStream);
      myStream = nullptr;
      myLastErr = "cudssConfigCreate failed";
      return CUDSS_BRIDGE_ERR_INIT;
   }
   // Enable cuDSS's built-in iterative refinement (2 steps by default).
   // This applies residual-correction passes after each solve, tightening
   // the result to PARDISO-comparable precision without the cost of a
   // full refactor. Not load-bearing for the FE path (matrix is mild and
   // single solve is already accurate); important for symmetric-indefinite
   // KKT systems where the LDL pivot path is otherwise less precise than
   // PARDISO's. Failure is non-fatal: log and continue without IR.
   {
      int irSteps = 4;
      cudssStatus_t s = cudssConfigSet (
         myConfig, CUDSS_CONFIG_IR_N_STEPS, &irSteps, sizeof(int));
      if (!dssOk (s)) {
         std::fprintf (stderr,
            "[cudssBridge] cudssConfigSet(IR_N_STEPS) returned %d -- "
            "continuing without iterative refinement\n", (int)s);
      }
   }
   if (!dssOk (cudssDataCreate (myHandle, &myData))) {
      cudssConfigDestroy (myConfig);
      myConfig = nullptr;
      cudssDestroy (myHandle);
      myHandle = nullptr;
      cudaStreamDestroy (myStream);
      myStream = nullptr;
      myLastErr = "cudssDataCreate failed";
      return CUDSS_BRIDGE_ERR_INIT;
   }
   myInitialized = true;
   myLastErr = nullptr;
   return CUDSS_BRIDGE_OK;
}

bool CuDssBridge::setMtype (int flag,
                            cudssMatrixType_t&     mtype,
                            cudssMatrixViewType_t& mview) {
   switch (flag) {
      case CUDSS_BRIDGE_MT_GENERAL:
         mtype = CUDSS_MTYPE_GENERAL;   mview = CUDSS_MVIEW_FULL;  return true;
      case CUDSS_BRIDGE_MT_SYMMETRIC:
         mtype = CUDSS_MTYPE_SYMMETRIC; mview = CUDSS_MVIEW_UPPER; return true;
      case CUDSS_BRIDGE_MT_SPD:
         mtype = CUDSS_MTYPE_SPD;       mview = CUDSS_MVIEW_UPPER; return true;
      default:
         return false;
   }
}

void CuDssBridge::destroyMatrixDescriptors() {
   if (myMatADesc) { cudssMatrixDestroy (myMatA); myMatADesc = false; myMatA = nullptr; }
   if (myMatXDesc) { cudssMatrixDestroy (myMatX); myMatXDesc = false; myMatX = nullptr; }
   if (myMatBDesc) { cudssMatrixDestroy (myMatB); myMatBDesc = false; myMatB = nullptr; }
}

void CuDssBridge::destroyMultiDescriptors() {
   if (myMatXMultiDesc) {
      cudssMatrixDestroy (myMatXMulti);
      myMatXMultiDesc = false;
      myMatXMulti = nullptr;
   }
   if (myMatBMultiDesc) {
      cudssMatrixDestroy (myMatBMulti);
      myMatBMultiDesc = false;
      myMatBMulti = nullptr;
   }
}

void CuDssBridge::freeDeviceBuffers() {
   if (myRowOffsD) { cudaFree (myRowOffsD); myRowOffsD = nullptr; }
   if (myColIdxsD) { cudaFree (myColIdxsD); myColIdxsD = nullptr; }
   if (myValsD)    { cudaFree (myValsD);    myValsD    = nullptr; }
   if (myBVecD)    { cudaFree (myBVecD);    myBVecD    = nullptr; }
   if (myXVecD)    { cudaFree (myXVecD);    myXVecD    = nullptr; }
}

void CuDssBridge::freeMultiBuffers() {
   destroyMultiDescriptors();
   if (myBMatD) { cudaFree (myBMatD); myBMatD = nullptr; }
   if (myXMatD) { cudaFree (myXMatD); myXMatD = nullptr; }
   myMultiNrhs = 0;
}

void CuDssBridge::freeIterativeBuffers() {
   if (mySpmvInVec)  { cusparseDestroyDnVec (mySpmvInVec);  mySpmvInVec  = nullptr; }
   if (mySpmvOutVec) { cusparseDestroyDnVec (mySpmvOutVec); mySpmvOutVec = nullptr; }
   if (myCsrDesc)    { cusparseDestroySpMat (myCsrDesc);    myCsrDesc    = nullptr; }
   if (mySpmvBuffer) { cudaFree (mySpmvBuffer); mySpmvBuffer = nullptr; }
   mySpmvBufferBytes = 0;
   if (myCuSparse) { cusparseDestroy (myCuSparse); myCuSparse = nullptr; }
   if (myCuBlas)   { cublasDestroy   (myCuBlas);   myCuBlas   = nullptr; }
   if (myItR)    { cudaFree (myItR);    myItR    = nullptr; }
   if (myItRhat) { cudaFree (myItRhat); myItRhat = nullptr; }
   if (myItP)    { cudaFree (myItP);    myItP    = nullptr; }
   if (myItV)    { cudaFree (myItV);    myItV    = nullptr; }
   if (myItS)    { cudaFree (myItS);    myItS    = nullptr; }
   if (myItT)    { cudaFree (myItT);    myItT    = nullptr; }
   if (myItY)    { cudaFree (myItY);    myItY    = nullptr; }
   if (myItZ)    { cudaFree (myItZ);    myItZ    = nullptr; }
   if (myItX)    { cudaFree (myItX);    myItX    = nullptr; }
   if (myItDiag) { cudaFree (myItDiag); myItDiag = nullptr; }
   myIterInited  = false;
   myItDiagDirty = true;
}

int CuDssBridge::setPattern (int n, int nnz,
                             const int* rowOffs, const int* colIdxs,
                             int mtypeFlag) {
   if (!myInitialized) {
      myLastErr = "setPattern called before init";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   if (n <= 0 || nnz <= 0) {
      myLastErr = "setPattern: n and nnz must be positive";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   cudssMatrixType_t     mtype;
   cudssMatrixViewType_t mview;
   if (!setMtype (mtypeFlag, mtype, mview)) {
      myLastErr = "unknown matrix-type flag passed to setPattern";
      return CUDSS_BRIDGE_ERR_UNKNOWN_MTYPE;
   }

   // Discard any prior pattern/factorization state. cuDSS doesn't support
   // changing the pattern of an existing cudssMatrix_t in-place; destroy and
   // rebuild. Multi-RHS and iterative-refinement scratch buffers are tied to
   // n (and to the CSR layout for SpMV), so drop them here too.
   destroyMatrixDescriptors();
   freeDeviceBuffers();
   freeMultiBuffers();
   freeIterativeBuffers();
   myHasPattern = false;
   myAnalyzed   = false;
   myFactored   = false;
   if (myData) {
      cudssDataDestroy (myHandle, myData);
      myData = nullptr;
      if (!dssOk (cudssDataCreate (myHandle, &myData))) {
         myLastErr = "cudssDataCreate failed during setPattern";
         return CUDSS_BRIDGE_ERR_INIT;
      }
   }

   const size_t rowBytes = (size_t)(n + 1) * sizeof(int);
   const size_t colBytes = (size_t)nnz     * sizeof(int);
   const size_t valBytes = (size_t)nnz     * sizeof(double);
   const size_t vecBytes = (size_t)n       * sizeof(double);

   if (!cudaOk (cudaMalloc (&myRowOffsD, rowBytes)) ||
       !cudaOk (cudaMalloc (&myColIdxsD, colBytes)) ||
       !cudaOk (cudaMalloc (&myValsD,    valBytes)) ||
       !cudaOk (cudaMalloc (&myBVecD,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myXVecD,    vecBytes))) {
      freeDeviceBuffers();
      myLastErr = "cudaMalloc failed in setPattern";
      return CUDSS_BRIDGE_ERR_CUDA_ALLOC;
   }
   if (!cudaOk (cudaMemcpyAsync (myRowOffsD, rowOffs, rowBytes,
                                 cudaMemcpyHostToDevice, myStream)) ||
       !cudaOk (cudaMemcpyAsync (myColIdxsD, colIdxs, colBytes,
                                 cudaMemcpyHostToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying CSR pattern";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }

   if (!dssOk (cudssMatrixCreateCsr (
          &myMatA, n, n, nnz,
          myRowOffsD, /*rowEnd*/ nullptr,
          myColIdxsD, myValsD,
          CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
          mtype, mview, CUDSS_BASE_ZERO))) {
      myLastErr = "cudssMatrixCreateCsr failed";
      return CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR;
   }
   myMatADesc = true;

   if (!dssOk (cudssMatrixCreateDn (
          &myMatX, n, 1, n, myXVecD, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR))) {
      myLastErr = "cudssMatrixCreateDn failed (x)";
      return CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR;
   }
   myMatXDesc = true;

   if (!dssOk (cudssMatrixCreateDn (
          &myMatB, n, 1, n, myBVecD, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR))) {
      myLastErr = "cudssMatrixCreateDn failed (b)";
      return CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR;
   }
   myMatBDesc = true;

   myN          = n;
   myNnz        = nnz;
   myMtypeFlag  = mtypeFlag;
   myHasPattern = true;
   myLastErr    = nullptr;
   return CUDSS_BRIDGE_OK;
}

int CuDssBridge::analyze() {
   if (!myInitialized) {
      myLastErr = "analyze called before init";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   if (!myHasPattern) {
      myLastErr = "analyze called before setPattern";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   if (!dssOk (cudssExecute (myHandle, CUDSS_PHASE_ANALYSIS,
                             myConfig, myData,
                             myMatA, myMatX, myMatB))) {
      myLastErr = "CUDSS_PHASE_ANALYSIS failed";
      return CUDSS_BRIDGE_ERR_ANALYSIS;
   }
   if (!cudaOk (cudaStreamSynchronize (myStream))) {
      myLastErr = "cudaStreamSynchronize failed after analyze";
      return CUDSS_BRIDGE_ERR_ANALYSIS;
   }
   myAnalyzed = true;
   myFactored = false;
   myLastErr  = nullptr;
   return CUDSS_BRIDGE_OK;
}

int CuDssBridge::factor (const double* vals) {
   if (!myAnalyzed) {
      myLastErr = "factor called before analyze";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   const bool tm = timingEnabled();
   const double t0 = tm ? nowMs() : 0;
   const size_t valBytes = (size_t)myNnz * sizeof(double);
   if (!cudaOk (cudaMemcpyAsync (myValsD, vals, valBytes,
                                 cudaMemcpyHostToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying matrix values";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   double tCopyEnd = 0;
   if (tm) {
      cudaStreamSynchronize (myStream);
      tCopyEnd = nowMs();
   }
   const cudssPhase_t phase = myFactored ? CUDSS_PHASE_REFACTORIZATION
                                         : CUDSS_PHASE_FACTORIZATION;
   if (!dssOk (cudssExecute (myHandle, phase, myConfig, myData,
                             myMatA, myMatX, myMatB))) {
      myLastErr = (phase == CUDSS_PHASE_REFACTORIZATION)
                  ? "CUDSS_PHASE_REFACTORIZATION failed"
                  : "CUDSS_PHASE_FACTORIZATION failed";
      return CUDSS_BRIDGE_ERR_FACTORIZATION;
   }
   if (!cudaOk (cudaStreamSynchronize (myStream))) {
      myLastErr = "cudaStreamSynchronize failed after factor";
      return CUDSS_BRIDGE_ERR_FACTORIZATION;
   }
   if (tm) {
      double t1 = nowMs();
      std::fprintf (stderr,
         "[cudss-timing] %s n=%d nnz=%d: H2D vals=%.2fms %s=%.2fms total=%.2fms\n",
         myFactored ? "refactor" : "factor",
         myN, myNnz,
         tCopyEnd - t0,
         myFactored ? "REFACTOR" : "FACTOR",
         t1 - tCopyEnd,
         t1 - t0);
   }
   myFactored    = true;
   myItDiagDirty = true;   // device values changed; diag cache invalid
   myLastErr     = nullptr;
   return CUDSS_BRIDGE_OK;
}

int CuDssBridge::solve (const double* b, double* x) {
   if (!myFactored) {
      myLastErr = "solve called before factor";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   const bool tm = timingEnabled();
   const double t0 = tm ? nowMs() : 0;
   const size_t vecBytes = (size_t)myN * sizeof(double);
   if (!cudaOk (cudaMemcpyAsync (myBVecD, b, vecBytes,
                                 cudaMemcpyHostToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying RHS";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   double tH2D = 0;
   if (tm) { cudaStreamSynchronize (myStream); tH2D = nowMs(); }
   if (!dssOk (cudssExecute (myHandle, CUDSS_PHASE_SOLVE,
                             myConfig, myData,
                             myMatA, myMatX, myMatB))) {
      myLastErr = "CUDSS_PHASE_SOLVE failed";
      return CUDSS_BRIDGE_ERR_SOLVE;
   }
   double tSolve = 0;
   if (tm) { cudaStreamSynchronize (myStream); tSolve = nowMs(); }
   if (!cudaOk (cudaMemcpyAsync (x, myXVecD, vecBytes,
                                 cudaMemcpyDeviceToHost, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying solution";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   if (!cudaOk (cudaStreamSynchronize (myStream))) {
      myLastErr = "cudaStreamSynchronize failed after solve";
      return CUDSS_BRIDGE_ERR_SOLVE;
   }
   if (tm) {
      double t1 = nowMs();
      std::fprintf (stderr,
         "[cudss-timing] solve n=%d: H2D b=%.2fms SOLVE=%.2fms "
         "D2H x=%.2fms total=%.2fms\n",
         myN, tH2D - t0, tSolve - tH2D, t1 - tSolve, t1 - t0);
   }
   myLastErr = nullptr;
   return CUDSS_BRIDGE_OK;
}

int CuDssBridge::solveMulti (int nrhs, const double* B, double* X) {
   if (!myFactored) {
      myLastErr = "solveMulti called before factor";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   if (nrhs <= 0) {
      myLastErr = "solveMulti: nrhs must be positive";
      return CUDSS_BRIDGE_ERR_STATE;
   }

   const size_t blockBytes = (size_t)myN * (size_t)nrhs * sizeof(double);

   // Grow multi-RHS buffers if needed. We grow only upwards; if a caller
   // ever asks for fewer columns later we just reuse the larger allocation
   // and pass nrhs through to the descriptor.
   if (nrhs > myMultiNrhs) {
      destroyMultiDescriptors();
      if (myBMatD) { cudaFree (myBMatD); myBMatD = nullptr; }
      if (myXMatD) { cudaFree (myXMatD); myXMatD = nullptr; }
      if (!cudaOk (cudaMalloc (&myBMatD, blockBytes)) ||
          !cudaOk (cudaMalloc (&myXMatD, blockBytes))) {
         freeMultiBuffers();
         myLastErr = "cudaMalloc failed for multi-RHS buffers";
         return CUDSS_BRIDGE_ERR_CUDA_ALLOC;
      }
      myMultiNrhs = nrhs;
   } else {
      // Buffers are large enough; just need to rebuild descriptors if the
      // current nrhs differs from the descriptor's nrhs.
      destroyMultiDescriptors();
   }

   if (!dssOk (cudssMatrixCreateDn (
          &myMatBMulti, myN, nrhs, myN, myBMatD,
          CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR))) {
      myLastErr = "cudssMatrixCreateDn failed for multi-RHS B";
      return CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR;
   }
   myMatBMultiDesc = true;
   if (!dssOk (cudssMatrixCreateDn (
          &myMatXMulti, myN, nrhs, myN, myXMatD,
          CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR))) {
      myLastErr = "cudssMatrixCreateDn failed for multi-RHS X";
      return CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR;
   }
   myMatXMultiDesc = true;

   if (!cudaOk (cudaMemcpyAsync (myBMatD, B, blockBytes,
                                 cudaMemcpyHostToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying multi-RHS B";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   if (!dssOk (cudssExecute (myHandle, CUDSS_PHASE_SOLVE,
                             myConfig, myData,
                             myMatA, myMatXMulti, myMatBMulti))) {
      myLastErr = "CUDSS_PHASE_SOLVE failed (multi-RHS)";
      return CUDSS_BRIDGE_ERR_SOLVE;
   }
   if (!cudaOk (cudaMemcpyAsync (X, myXMatD, blockBytes,
                                 cudaMemcpyDeviceToHost, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying multi-RHS X";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   if (!cudaOk (cudaStreamSynchronize (myStream))) {
      myLastErr = "cudaStreamSynchronize failed after multi-RHS solve";
      return CUDSS_BRIDGE_ERR_SOLVE;
   }
   myLastErr = nullptr;
   return CUDSS_BRIDGE_OK;
}

// ---------------------------------------------------------------------------
// BiCGStab support
// ---------------------------------------------------------------------------

extern "C" void axmy_launch (
   int n, const double* d, const double* x, double* y, cudaStream_t stream);
extern "C" void extractDiag_launch (
   int n, const int* rowOffs, const int* colIdxs, const double* vals,
   double* diag, cudaStream_t stream);

int CuDssBridge::ensureIterativeState() {
   if (myIterInited) return CUDSS_BRIDGE_OK;
   if (!myHasPattern) {
      myLastErr = "iterativeSolve called before setPattern";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   // Create handles bound to our stream.
   if (!spOk (cusparseCreate (&myCuSparse))) {
      myLastErr = "cusparseCreate failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }
   if (!spOk (cusparseSetStream (myCuSparse, myStream))) {
      myLastErr = "cusparseSetStream failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }
   if (!blOk (cublasCreate (&myCuBlas))) {
      myLastErr = "cublasCreate failed";
      return CUDSS_BRIDGE_ERR_CUBLAS;
   }
   if (!blOk (cublasSetStream (myCuBlas, myStream))) {
      myLastErr = "cublasSetStream failed";
      return CUDSS_BRIDGE_ERR_CUBLAS;
   }

   // CSR descriptor pointing at the same device buffers used by cuDSS.
   // SpMV reads (rowOffs, colIdxs, vals); these are updated before each
   // iterative solve via cudaMemcpy.
   if (!spOk (cusparseCreateCsr (
          &myCsrDesc, myN, myN, myNnz,
          myRowOffsD, myColIdxsD, myValsD,
          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))) {
      myLastErr = "cusparseCreateCsr failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }

   // Dummy dense vector descriptors; we'll re-point them per call via
   // cusparseDnVecSetValues. Initial values pointer doesn't matter.
   double* dummy = nullptr;
   if (!cudaOk (cudaMalloc (&dummy, sizeof(double)))) {
      myLastErr = "cudaMalloc tiny placeholder failed";
      return CUDSS_BRIDGE_ERR_CUDA_ALLOC;
   }
   if (!spOk (cusparseCreateDnVec (
                 &mySpmvInVec, myN, dummy, CUDA_R_64F))) {
      cudaFree (dummy);
      myLastErr = "cusparseCreateDnVec in failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }
   if (!spOk (cusparseCreateDnVec (
                 &mySpmvOutVec, myN, dummy, CUDA_R_64F))) {
      cudaFree (dummy);
      myLastErr = "cusparseCreateDnVec out failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }
   cudaFree (dummy);

   // Allocate BiCGStab work vectors.
   const size_t vecBytes = (size_t)myN * sizeof(double);
   if (!cudaOk (cudaMalloc (&myItR,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myItRhat, vecBytes)) ||
       !cudaOk (cudaMalloc (&myItP,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myItV,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myItS,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myItT,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myItY,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myItZ,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myItX,    vecBytes)) ||
       !cudaOk (cudaMalloc (&myItDiag, vecBytes))) {
      freeIterativeBuffers();
      myLastErr = "cudaMalloc BiCGStab work vectors failed";
      return CUDSS_BRIDGE_ERR_CUDA_ALLOC;
   }

   // Query and allocate cusparseSpMV buffer (take max of TRANS and
   // NON_TRANS sizes so the same buffer serves both).
   double one  = 1.0;
   double zero = 0.0;
   size_t bufN = 0, bufT = 0;
   cusparseDnVecSetValues (mySpmvInVec,  myItR);
   cusparseDnVecSetValues (mySpmvOutVec, myItS);
   if (!spOk (cusparseSpMV_bufferSize (
          myCuSparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
          &one, myCsrDesc, mySpmvInVec, &zero, mySpmvOutVec,
          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufN))) {
      freeIterativeBuffers();
      myLastErr = "cusparseSpMV_bufferSize NON_TRANS failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }
   if (!spOk (cusparseSpMV_bufferSize (
          myCuSparse, CUSPARSE_OPERATION_TRANSPOSE,
          &one, myCsrDesc, mySpmvInVec, &zero, mySpmvOutVec,
          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufT))) {
      freeIterativeBuffers();
      myLastErr = "cusparseSpMV_bufferSize TRANS failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }
   mySpmvBufferBytes = (bufN > bufT) ? bufN : bufT;
   if (mySpmvBufferBytes > 0) {
      if (!cudaOk (cudaMalloc (&mySpmvBuffer, mySpmvBufferBytes))) {
         freeIterativeBuffers();
         myLastErr = "cudaMalloc SpMV workspace failed";
         return CUDSS_BRIDGE_ERR_CUDA_ALLOC;
      }
   }

   myIterInited = true;
   myItDiagDirty = true;
   myLastErr = nullptr;
   return CUDSS_BRIDGE_OK;
}

int CuDssBridge::extractDiagonal() {
   if (!myItDiagDirty) return CUDSS_BRIDGE_OK;
   extractDiag_launch (myN, myRowOffsD, myColIdxsD, myValsD,
                       myItDiag, myStream);
   if (cudaPeekAtLastError() != cudaSuccess) {
      myLastErr = "extractDiag kernel launch failed";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   myItDiagDirty = false;
   return CUDSS_BRIDGE_OK;
}

// y = A * x. xD and yD must be distinct device pointers, length myN.
// For symmetric upper-only storage we compute
//   y = U*x          (non-trans)
//   y += U^T*x       (trans, beta=1)
//   y -= diag(U) .* x  (kernel)
// For general storage, a single non-trans SpMV suffices.
int CuDssBridge::applyA (const double* xD, double* yD) {
   double one  = 1.0;
   double zero = 0.0;
   if (!spOk (cusparseDnVecSetValues (mySpmvInVec,  (void*)xD))) {
      myLastErr = "cusparseDnVecSetValues in failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }
   if (!spOk (cusparseDnVecSetValues (mySpmvOutVec, (void*)yD))) {
      myLastErr = "cusparseDnVecSetValues out failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }
   if (!spOk (cusparseSpMV (
          myCuSparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
          &one, myCsrDesc, mySpmvInVec, &zero, mySpmvOutVec,
          CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, mySpmvBuffer))) {
      myLastErr = "cusparseSpMV (non-trans) failed";
      return CUDSS_BRIDGE_ERR_CUSPARSE;
   }

   if (myMtypeFlag == CUDSS_BRIDGE_MT_SYMMETRIC ||
       myMtypeFlag == CUDSS_BRIDGE_MT_SPD) {
      // Add U^T x and subtract the doubled diagonal.
      if (!spOk (cusparseSpMV (
             myCuSparse, CUSPARSE_OPERATION_TRANSPOSE,
             &one, myCsrDesc, mySpmvInVec, &one, mySpmvOutVec,
             CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, mySpmvBuffer))) {
         myLastErr = "cusparseSpMV (transpose) failed";
         return CUDSS_BRIDGE_ERR_CUSPARSE;
      }
      int st = extractDiagonal();
      if (st != CUDSS_BRIDGE_OK) return st;
      axmy_launch (myN, myItDiag, xD, yD, myStream);
      if (cudaPeekAtLastError() != cudaSuccess) {
         myLastErr = "axmy kernel launch failed";
         return CUDSS_BRIDGE_ERR_CUDA_COPY;
      }
   }
   return CUDSS_BRIDGE_OK;
}

int CuDssBridge::iterativeSolveBiCGStab (
   const double* vals, const double* b, double* x,
   double tolRel, int maxIter, int* outIters) {
   if (!myFactored) {
      myLastErr = "iterativeSolve called before factor";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   const bool tm = timingEnabled();
   const double tT0 = tm ? nowMs() : 0;
   int s = ensureIterativeState();
   if (s != CUDSS_BRIDGE_OK) return s;

   // The cuDSS-config-driven iterative refinement enabled by
   // CUDSS_CONFIG_IR_N_STEPS reads the current matrix values
   // (myValsD) to compute its residual. During BiCGStab we push the
   // NEW matrix values to myValsD while reusing the STALE factor in
   // myData, so cuDSS-side IR sees a mismatched system and produces
   // garbage. Disable IR for the duration of this call and restore
   // it on exit.
   int savedIrSteps = 4;
   int zeroIrSteps  = 0;
   cudssConfigSet (myConfig, CUDSS_CONFIG_IR_N_STEPS,
                   &zeroIrSteps, sizeof(int));
   struct RestoreIr {
      cudssConfig_t cfg; int n;
      ~RestoreIr() { cudssConfigSet (cfg, CUDSS_CONFIG_IR_N_STEPS,
                                     &n, sizeof(int)); }
   } restoreIr { myConfig, savedIrSteps };

   const int n = myN;
   const size_t vecBytes = (size_t)n * sizeof(double);

   // Push current A values to the device (cuSPARSE will read myValsD,
   // which is the same buffer cuDSS used during factor; the iteration
   // intentionally uses these CURRENT values while the preconditioner
   // uses the STALE factor). Diagonal cache becomes invalid.
   if (!cudaOk (cudaMemcpyAsync (myValsD, vals, (size_t)myNnz * sizeof(double),
                                 cudaMemcpyHostToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync (vals) failed in iterativeSolve";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   myItDiagDirty = true;

   // Initial guess x0 = M^-1 b (preconditioner applied once). This is
   // already very close to the true solution when the matrix has only
   // changed slightly since the factorization, so BiCGStab typically
   // needs few iterations to refine it.
   //
   // Push b to myBVecD, run cuDSS solve, result lands in myXVecD = x0.
   if (!cudaOk (cudaMemcpyAsync (myBVecD, b, vecBytes,
                                 cudaMemcpyHostToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync (b -> myBVecD) failed";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   if (!dssOk (cudssExecute (myHandle, CUDSS_PHASE_SOLVE,
                             myConfig, myData,
                             myMatA, myMatX, myMatB))) {
      myLastErr = "cuDSS warm-start solve failed";
      return CUDSS_BRIDGE_ERR_SOLVE;
   }
   // Copy x0 into our dedicated accumulator; myXVecD will be clobbered
   // by every preconditioner solve inside the loop.
   if (!cudaOk (cudaMemcpyAsync (myItX, myXVecD, vecBytes,
                                 cudaMemcpyDeviceToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync (x0 -> myItX) failed";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   // r = b - A x0. Compute A x0 into myItR, then myItR = b - myItR.
   s = applyA (myItX, myItR);
   if (s != CUDSS_BRIDGE_OK) return s;
   double minusOne = -1.0;
   double one      =  1.0;
   if (!blOk (cublasDscal (myCuBlas, n, &minusOne, myItR, 1))) {
      myLastErr = "cublasDscal failed in r init";
      return CUDSS_BRIDGE_ERR_CUBLAS;
   }
   if (!blOk (cublasDaxpy (myCuBlas, n, &one, myBVecD, 1, myItR, 1))) {
      myLastErr = "cublasDaxpy failed in r init";
      return CUDSS_BRIDGE_ERR_CUBLAS;
   }

   // Initial residual norm.
   double bNorm = 0.0;
   double rNorm = 0.0;
   if (!blOk (cublasDnrm2 (myCuBlas, n, myBVecD, 1, &bNorm)) ||
       !blOk (cublasDnrm2 (myCuBlas, n, myItR,   1, &rNorm))) {
      myLastErr = "cublasDnrm2 failed";
      return CUDSS_BRIDGE_ERR_CUBLAS;
   }
   if (bNorm == 0.0) bNorm = 1.0;
   if (rNorm / bNorm <= tolRel) {
      // Already converged after one preconditioner solve.
      if (!cudaOk (cudaMemcpyAsync (x, myItX, vecBytes,
                                    cudaMemcpyDeviceToHost, myStream))) {
         myLastErr = "cudaMemcpyAsync (x out) failed";
         return CUDSS_BRIDGE_ERR_CUDA_COPY;
      }
      if (!cudaOk (cudaStreamSynchronize (myStream))) {
         myLastErr = "cudaStreamSynchronize failed";
         return CUDSS_BRIDGE_ERR_SOLVE;
      }
      if (outIters) *outIters = 1;
      myLastErr = nullptr;
      return CUDSS_BRIDGE_OK;
   }

   // r_hat = r (shadow residual, fixed throughout the loop).
   if (!cudaOk (cudaMemcpyAsync (myItRhat, myItR, vecBytes,
                                 cudaMemcpyDeviceToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync (r_hat) failed";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }

   double rho_prev = 1.0;
   double alpha    = 1.0;
   double omega    = 1.0;

   // p = 0, v = 0
   if (!cudaOk (cudaMemsetAsync (myItP, 0, vecBytes, myStream)) ||
       !cudaOk (cudaMemsetAsync (myItV, 0, vecBytes, myStream))) {
      myLastErr = "cudaMemsetAsync (p,v init) failed";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }

   const double eps = 1e-30;
   int iter;
   for (iter = 1; iter <= maxIter; iter++) {
      double rho;
      if (!blOk (cublasDdot (myCuBlas, n, myItRhat, 1, myItR, 1, &rho))) {
         myLastErr = "cublasDdot (rho) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      if (std::fabs (rho) < eps) {
         myLastErr = "BiCGStab breakdown: rho ~ 0";
         return CUDSS_BRIDGE_ERR_ITER_BREAKDOWN;
      }
      // beta = (rho / rho_prev) * (alpha / omega)
      double beta = (rho / rho_prev) * (alpha / omega);
      // p = r + beta * (p - omega * v)
      //   step 1: p = p - omega * v
      double negOmega = -omega;
      if (!blOk (cublasDaxpy (myCuBlas, n, &negOmega, myItV, 1, myItP, 1))) {
         myLastErr = "cublasDaxpy (p step 1) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      //   step 2: p = beta * p
      if (!blOk (cublasDscal (myCuBlas, n, &beta, myItP, 1))) {
         myLastErr = "cublasDscal (p step 2) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      //   step 3: p = p + r
      if (!blOk (cublasDaxpy (myCuBlas, n, &one, myItR, 1, myItP, 1))) {
         myLastErr = "cublasDaxpy (p step 3) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      // y = M^-1 * p   (cuDSS solve: input myItP -> output myItY)
      if (!cudaOk (cudaMemcpyAsync (myBVecD, myItP, vecBytes,
                                    cudaMemcpyDeviceToDevice, myStream))) {
         myLastErr = "cudaMemcpyAsync (p -> myBVecD) failed";
         return CUDSS_BRIDGE_ERR_CUDA_COPY;
      }
      if (!dssOk (cudssExecute (myHandle, CUDSS_PHASE_SOLVE,
                                myConfig, myData,
                                myMatA, myMatX, myMatB))) {
         myLastErr = "cuDSS solve failed in BiCGStab (y = M^-1 p)";
         return CUDSS_BRIDGE_ERR_SOLVE;
      }
      if (!cudaOk (cudaMemcpyAsync (myItY, myXVecD, vecBytes,
                                    cudaMemcpyDeviceToDevice, myStream))) {
         myLastErr = "cudaMemcpyAsync (y from myXVecD) failed";
         return CUDSS_BRIDGE_ERR_CUDA_COPY;
      }
      // v = A * y
      s = applyA (myItY, myItV);
      if (s != CUDSS_BRIDGE_OK) return s;
      // alpha = rho / (r_hat, v)
      double rhv;
      if (!blOk (cublasDdot (myCuBlas, n, myItRhat, 1, myItV, 1, &rhv))) {
         myLastErr = "cublasDdot (r_hat, v) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      if (std::fabs (rhv) < eps) {
         myLastErr = "BiCGStab breakdown: (r_hat,v) ~ 0";
         return CUDSS_BRIDGE_ERR_ITER_BREAKDOWN;
      }
      alpha = rho / rhv;
      // s_vec = r - alpha * v
      //   start by s_vec = r, then s_vec -= alpha * v
      if (!cudaOk (cudaMemcpyAsync (myItS, myItR, vecBytes,
                                    cudaMemcpyDeviceToDevice, myStream))) {
         myLastErr = "cudaMemcpyAsync (s = r) failed";
         return CUDSS_BRIDGE_ERR_CUDA_COPY;
      }
      double negAlpha = -alpha;
      if (!blOk (cublasDaxpy (myCuBlas, n, &negAlpha, myItV, 1, myItS, 1))) {
         myLastErr = "cublasDaxpy (s = r - alpha v) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      // Early-exit if ||s|| small.
      double sNorm;
      if (!blOk (cublasDnrm2 (myCuBlas, n, myItS, 1, &sNorm))) {
         myLastErr = "cublasDnrm2 (s) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      if (sNorm / bNorm <= tolRel) {
         // x += alpha * y
         if (!blOk (cublasDaxpy (myCuBlas, n, &alpha, myItY, 1, myItX, 1))) {
            myLastErr = "cublasDaxpy (x += alpha y, early) failed";
            return CUDSS_BRIDGE_ERR_CUBLAS;
         }
         break;
      }
      // z = M^-1 * s_vec
      if (!cudaOk (cudaMemcpyAsync (myBVecD, myItS, vecBytes,
                                    cudaMemcpyDeviceToDevice, myStream))) {
         myLastErr = "cudaMemcpyAsync (s -> myBVecD) failed";
         return CUDSS_BRIDGE_ERR_CUDA_COPY;
      }
      if (!dssOk (cudssExecute (myHandle, CUDSS_PHASE_SOLVE,
                                myConfig, myData,
                                myMatA, myMatX, myMatB))) {
         myLastErr = "cuDSS solve failed in BiCGStab (z = M^-1 s)";
         return CUDSS_BRIDGE_ERR_SOLVE;
      }
      if (!cudaOk (cudaMemcpyAsync (myItZ, myXVecD, vecBytes,
                                    cudaMemcpyDeviceToDevice, myStream))) {
         myLastErr = "cudaMemcpyAsync (z from myXVecD) failed";
         return CUDSS_BRIDGE_ERR_CUDA_COPY;
      }
      // t = A * z
      s = applyA (myItZ, myItT);
      if (s != CUDSS_BRIDGE_OK) return s;
      // omega = (t, s) / (t, t)
      double ts, tt;
      if (!blOk (cublasDdot (myCuBlas, n, myItT, 1, myItS, 1, &ts)) ||
          !blOk (cublasDdot (myCuBlas, n, myItT, 1, myItT, 1, &tt))) {
         myLastErr = "cublasDdot (t,s)/(t,t) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      if (tt < eps) {
         myLastErr = "BiCGStab breakdown: (t,t) ~ 0";
         return CUDSS_BRIDGE_ERR_ITER_BREAKDOWN;
      }
      omega = ts / tt;
      // x += alpha * y + omega * z
      if (!blOk (cublasDaxpy (myCuBlas, n, &alpha, myItY, 1, myItX, 1)) ||
          !blOk (cublasDaxpy (myCuBlas, n, &omega, myItZ, 1, myItX, 1))) {
         myLastErr = "cublasDaxpy (x update) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      // r = s - omega * t
      if (!cudaOk (cudaMemcpyAsync (myItR, myItS, vecBytes,
                                    cudaMemcpyDeviceToDevice, myStream))) {
         myLastErr = "cudaMemcpyAsync (r = s) failed";
         return CUDSS_BRIDGE_ERR_CUDA_COPY;
      }
      double negOmega2 = -omega;
      if (!blOk (cublasDaxpy (myCuBlas, n, &negOmega2, myItT, 1, myItR, 1))) {
         myLastErr = "cublasDaxpy (r = s - omega t) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      // Convergence check.
      if (!blOk (cublasDnrm2 (myCuBlas, n, myItR, 1, &rNorm))) {
         myLastErr = "cublasDnrm2 (r) failed";
         return CUDSS_BRIDGE_ERR_CUBLAS;
      }
      if (rNorm / bNorm <= tolRel) {
         break;
      }
      if (std::fabs (omega) < eps) {
         myLastErr = "BiCGStab breakdown: omega ~ 0";
         return CUDSS_BRIDGE_ERR_ITER_BREAKDOWN;
      }
      rho_prev = rho;
   }

   if (iter > maxIter) {
      myLastErr = "BiCGStab did not converge within iteration limit";
      return CUDSS_BRIDGE_ERR_ITER_NO_CONVERGE;
   }

   // Copy x back from our dedicated accumulator.
   if (!cudaOk (cudaMemcpyAsync (x, myItX, vecBytes,
                                 cudaMemcpyDeviceToHost, myStream))) {
      myLastErr = "cudaMemcpyAsync (x out final) failed";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   if (!cudaOk (cudaStreamSynchronize (myStream))) {
      myLastErr = "cudaStreamSynchronize failed at end of iterativeSolve";
      return CUDSS_BRIDGE_ERR_SOLVE;
   }
   if (outIters) *outIters = iter;
   if (tm) {
      double tT1 = nowMs();
      double total = tT1 - tT0;
      std::fprintf (stderr,
         "[cudss-timing] bicgstab n=%d iters=%d: total=%.2fms (%.2fms/iter)\n",
         myN, iter, total, total / std::max (1, iter));
   }
   myLastErr = nullptr;
   return CUDSS_BRIDGE_OK;
}

void CuDssBridge::dispose() {
   destroyMatrixDescriptors();
   freeDeviceBuffers();
   freeMultiBuffers();
   freeIterativeBuffers();
   if (myData)   { cudssDataDestroy (myHandle, myData); myData = nullptr; }
   if (myConfig) { cudssConfigDestroy (myConfig);       myConfig = nullptr; }
   if (myHandle) { cudssDestroy (myHandle);             myHandle = nullptr; }
   if (myStream) { cudaStreamDestroy (myStream);        myStream = nullptr; }
   myInitialized = false;
   myHasPattern  = false;
   myAnalyzed    = false;
   myFactored    = false;
   myN           = 0;
   myNnz         = 0;
}
