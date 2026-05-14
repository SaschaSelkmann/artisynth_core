#include "cudssBridge.h"

#include <cstddef>
#include <cstdio>

namespace {

inline bool cudaOk (cudaError_t e) { return e == cudaSuccess; }
inline bool dssOk  (cudssStatus_t s) { return s == CUDSS_STATUS_SUCCESS; }

} // anon

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
   // rebuild. Multi-RHS scratch is tied to n; drop it too since the new
   // pattern may have a different n.
   destroyMatrixDescriptors();
   freeDeviceBuffers();
   freeMultiBuffers();
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
          CUDA_R_32I, CUDA_R_64F,
          mtype, mview, CUDSS_BASE_ZERO))) {
      myLastErr = "cudssMatrixCreateCsr failed";
      return CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR;
   }
   myMatADesc = true;

   if (!dssOk (cudssMatrixCreateDn (
          &myMatX, n, 1, n, myXVecD, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR))) {
      myLastErr = "cudssMatrixCreateDn failed (x)";
      return CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR;
   }
   myMatXDesc = true;

   if (!dssOk (cudssMatrixCreateDn (
          &myMatB, n, 1, n, myBVecD, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR))) {
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
   const size_t valBytes = (size_t)myNnz * sizeof(double);
   if (!cudaOk (cudaMemcpyAsync (myValsD, vals, valBytes,
                                 cudaMemcpyHostToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying matrix values";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
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
   myFactored = true;
   myLastErr  = nullptr;
   return CUDSS_BRIDGE_OK;
}

int CuDssBridge::solve (const double* b, double* x) {
   if (!myFactored) {
      myLastErr = "solve called before factor";
      return CUDSS_BRIDGE_ERR_STATE;
   }
   const size_t vecBytes = (size_t)myN * sizeof(double);
   if (!cudaOk (cudaMemcpyAsync (myBVecD, b, vecBytes,
                                 cudaMemcpyHostToDevice, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying RHS";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   if (!dssOk (cudssExecute (myHandle, CUDSS_PHASE_SOLVE,
                             myConfig, myData,
                             myMatA, myMatX, myMatB))) {
      myLastErr = "CUDSS_PHASE_SOLVE failed";
      return CUDSS_BRIDGE_ERR_SOLVE;
   }
   if (!cudaOk (cudaMemcpyAsync (x, myXVecD, vecBytes,
                                 cudaMemcpyDeviceToHost, myStream))) {
      myLastErr = "cudaMemcpyAsync failed copying solution";
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   if (!cudaOk (cudaStreamSynchronize (myStream))) {
      myLastErr = "cudaStreamSynchronize failed after solve";
      return CUDSS_BRIDGE_ERR_SOLVE;
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
          CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR))) {
      myLastErr = "cudssMatrixCreateDn failed for multi-RHS B";
      return CUDSS_BRIDGE_ERR_CUDSS_DESCRIPTOR;
   }
   myMatBMultiDesc = true;
   if (!dssOk (cudssMatrixCreateDn (
          &myMatXMulti, myN, nrhs, myN, myXMatD,
          CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR))) {
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

void CuDssBridge::dispose() {
   destroyMatrixDescriptors();
   freeDeviceBuffers();
   freeMultiBuffers();
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
