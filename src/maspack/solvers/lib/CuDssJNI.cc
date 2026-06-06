// JNI dispatch for maspack.solvers.CuDssSolver. Each native method maps to a
// thin wrapper around the matching CuDssBridge method. Bridge state is
// passed back to Java as an opaque jlong (a CuDssBridge*).

#include "maspack_solvers_CuDssSolver.h"
#include "cudssBridge.h"

#include <cstdio>
#include <cstring>

static inline CuDssBridge* asBridge (jlong handle) {
   return reinterpret_cast<CuDssBridge*> (handle);
}

JNIEXPORT jlong JNICALL Java_maspack_solvers_CuDssSolver_doInit
  (JNIEnv* /*env*/, jclass /*cls*/) {
   CuDssBridge* b = new CuDssBridge();
   if (b->init() != CUDSS_BRIDGE_OK) {
      delete b;
      return 0L;
   }
   return reinterpret_cast<jlong> (b);
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doSetPattern
  (JNIEnv* env, jclass /*cls*/,
   jlong handle, jint n, jint nnz,
   jintArray rowOffs, jintArray colIdxs, jint mtype) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;

   jint* rowOffsP = env->GetIntArrayElements (rowOffs, nullptr);
   jint* colIdxsP = env->GetIntArrayElements (colIdxs, nullptr);
   if (!rowOffsP || !colIdxsP) {
      if (rowOffsP) env->ReleaseIntArrayElements (rowOffs, rowOffsP, JNI_ABORT);
      if (colIdxsP) env->ReleaseIntArrayElements (colIdxs, colIdxsP, JNI_ABORT);
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = b->setPattern ((int)n, (int)nnz,
                               (const int*)rowOffsP, (const int*)colIdxsP,
                               (int)mtype);
   // JNI_ABORT: we didn't modify the contents, no need to copy back
   env->ReleaseIntArrayElements (rowOffs, rowOffsP, JNI_ABORT);
   env->ReleaseIntArrayElements (colIdxs, colIdxsP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doAnalyze
  (JNIEnv* /*env*/, jclass /*cls*/, jlong handle) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   return (jint)b->analyze();
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doFactor
  (JNIEnv* env, jclass /*cls*/, jlong handle, jdoubleArray vals) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   jdouble* valsP = env->GetDoubleArrayElements (vals, nullptr);
   if (!valsP) return CUDSS_BRIDGE_ERR_CUDA_COPY;
   int status = b->factor ((const double*)valsP);
   env->ReleaseDoubleArrayElements (vals, valsP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doClearDeviceValues
  (JNIEnv* /*env*/, jclass /*cls*/, jlong handle) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   return (jint)b->clearDeviceValues();
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doAddDeviceValues
  (JNIEnv* env, jclass /*cls*/, jlong handle,
   jintArray slots, jdoubleArray vals, jint nvals, jdouble scale) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   jint* slotsP = env->GetIntArrayElements (slots, nullptr);
   jdouble* valsP = env->GetDoubleArrayElements (vals, nullptr);
   if (!slotsP || !valsP) {
      if (slotsP) env->ReleaseIntArrayElements (slots, slotsP, JNI_ABORT);
      if (valsP) env->ReleaseDoubleArrayElements (vals, valsP, JNI_ABORT);
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = b->addDeviceValues (
      (const int*)slotsP, (const double*)valsP, (int)nvals, (double)scale);
   env->ReleaseIntArrayElements (slots, slotsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (vals, valsP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL
Java_maspack_solvers_CuDssSolver_doAddScaledDiagonal3DeviceValues
  (JNIEnv* env, jclass /*cls*/, jlong handle,
   jintArray diagSlots, jdoubleArray masses, jint nblocks, jdouble scale) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   jint* diagSlotsP = env->GetIntArrayElements (diagSlots, nullptr);
   jdouble* massesP = env->GetDoubleArrayElements (masses, nullptr);
   if (!diagSlotsP || !massesP) {
      if (diagSlotsP) {
         env->ReleaseIntArrayElements (diagSlots, diagSlotsP, JNI_ABORT);
      }
      if (massesP) {
         env->ReleaseDoubleArrayElements (masses, massesP, JNI_ABORT);
      }
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = b->addScaledDiagonal3DeviceValues (
      (const int*)diagSlotsP, (const double*)massesP,
      (int)nblocks, (double)scale);
   env->ReleaseIntArrayElements (diagSlots, diagSlotsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (masses, massesP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL
Java_maspack_solvers_CuDssSolver_doAddScaledBlock3DeviceValues
  (JNIEnv* env, jclass /*cls*/, jlong handle,
   jintArray blockSlots, jdoubleArray blockVals, jdoubleArray blockScales,
   jint nblocks, jdouble scale) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   jint* blockSlotsP = env->GetIntArrayElements (blockSlots, nullptr);
   jdouble* blockValsP = env->GetDoubleArrayElements (blockVals, nullptr);
   jdouble* blockScalesP =
      env->GetDoubleArrayElements (blockScales, nullptr);
   if (!blockSlotsP || !blockValsP || !blockScalesP) {
      if (blockSlotsP) {
         env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
      }
      if (blockValsP) {
         env->ReleaseDoubleArrayElements (blockVals, blockValsP, JNI_ABORT);
      }
      if (blockScalesP) {
         env->ReleaseDoubleArrayElements (
            blockScales, blockScalesP, JNI_ABORT);
      }
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = b->addScaledBlock3DeviceValues (
      (const int*)blockSlotsP, (const double*)blockValsP,
      (const double*)blockScalesP, (int)nblocks, (double)scale);
   env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (blockVals, blockValsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (blockScales, blockScalesP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL
Java_maspack_solvers_CuDssSolver_doAddMaterialStiffness3DeviceValues
  (JNIEnv* env, jclass /*cls*/, jlong handle,
   jintArray blockSlots, jdoubleArray gis, jdoubleArray gjs,
   jdoubleArray Ds, jdoubleArray sigmas, jdoubleArray dvs,
   jint nblocks, jdouble scale) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   jint* blockSlotsP = env->GetIntArrayElements (blockSlots, nullptr);
   jdouble* gisP = env->GetDoubleArrayElements (gis, nullptr);
   jdouble* gjsP = env->GetDoubleArrayElements (gjs, nullptr);
   jdouble* DsP = env->GetDoubleArrayElements (Ds, nullptr);
   jdouble* sigmasP = env->GetDoubleArrayElements (sigmas, nullptr);
   jdouble* dvsP = env->GetDoubleArrayElements (dvs, nullptr);
   if (!blockSlotsP || !gisP || !gjsP || !DsP || !sigmasP || !dvsP) {
      if (blockSlotsP) {
         env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
      }
      if (gisP) env->ReleaseDoubleArrayElements (gis, gisP, JNI_ABORT);
      if (gjsP) env->ReleaseDoubleArrayElements (gjs, gjsP, JNI_ABORT);
      if (DsP) env->ReleaseDoubleArrayElements (Ds, DsP, JNI_ABORT);
      if (sigmasP) {
         env->ReleaseDoubleArrayElements (sigmas, sigmasP, JNI_ABORT);
      }
      if (dvsP) env->ReleaseDoubleArrayElements (dvs, dvsP, JNI_ABORT);
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = b->addMaterialStiffness3DeviceValues (
      (const int*)blockSlotsP, (const double*)gisP, (const double*)gjsP,
      (const double*)DsP, (const double*)sigmasP, (const double*)dvsP,
      (int)nblocks, (double)scale);
   env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (gis, gisP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (gjs, gjsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (Ds, DsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (sigmas, sigmasP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (dvs, dvsP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL
Java_maspack_solvers_CuDssSolver_doAddMaterialStiffness3ElementDeviceValues
  (JNIEnv* env, jclass /*cls*/, jlong handle,
   jintArray elemNodeCounts, jintArray elemPairOffsets,
   jintArray elemIpOffsets, jintArray elemGradOffsets,
   jintArray pairNodeIdxs, jintArray blockSlots, jdoubleArray grads,
   jdoubleArray Ds, jdoubleArray sigmas, jdoubleArray dvs,
   jint nelems, jdouble scale) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   jint* elemNodeCountsP =
      env->GetIntArrayElements (elemNodeCounts, nullptr);
   jint* elemPairOffsetsP =
      env->GetIntArrayElements (elemPairOffsets, nullptr);
   jint* elemIpOffsetsP =
      env->GetIntArrayElements (elemIpOffsets, nullptr);
   jint* elemGradOffsetsP =
      env->GetIntArrayElements (elemGradOffsets, nullptr);
   jint* pairNodeIdxsP = env->GetIntArrayElements (pairNodeIdxs, nullptr);
   jint* blockSlotsP = env->GetIntArrayElements (blockSlots, nullptr);
   jdouble* gradsP = env->GetDoubleArrayElements (grads, nullptr);
   jdouble* DsP = env->GetDoubleArrayElements (Ds, nullptr);
   jdouble* sigmasP = env->GetDoubleArrayElements (sigmas, nullptr);
   jdouble* dvsP = env->GetDoubleArrayElements (dvs, nullptr);
   if (!elemNodeCountsP || !elemPairOffsetsP || !elemIpOffsetsP ||
       !elemGradOffsetsP || !pairNodeIdxsP || !blockSlotsP || !gradsP ||
       !DsP || !sigmasP || !dvsP) {
      if (elemNodeCountsP) {
         env->ReleaseIntArrayElements (
            elemNodeCounts, elemNodeCountsP, JNI_ABORT);
      }
      if (elemPairOffsetsP) {
         env->ReleaseIntArrayElements (
            elemPairOffsets, elemPairOffsetsP, JNI_ABORT);
      }
      if (elemIpOffsetsP) {
         env->ReleaseIntArrayElements (
            elemIpOffsets, elemIpOffsetsP, JNI_ABORT);
      }
      if (elemGradOffsetsP) {
         env->ReleaseIntArrayElements (
            elemGradOffsets, elemGradOffsetsP, JNI_ABORT);
      }
      if (pairNodeIdxsP) {
         env->ReleaseIntArrayElements (
            pairNodeIdxs, pairNodeIdxsP, JNI_ABORT);
      }
      if (blockSlotsP) {
         env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
      }
      if (gradsP) env->ReleaseDoubleArrayElements (grads, gradsP, JNI_ABORT);
      if (DsP) env->ReleaseDoubleArrayElements (Ds, DsP, JNI_ABORT);
      if (sigmasP) {
         env->ReleaseDoubleArrayElements (sigmas, sigmasP, JNI_ABORT);
      }
      if (dvsP) env->ReleaseDoubleArrayElements (dvs, dvsP, JNI_ABORT);
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = b->addMaterialStiffness3ElementDeviceValues (
      (const int*)elemNodeCountsP, (const int*)elemPairOffsetsP,
      (const int*)elemIpOffsetsP, (const int*)elemGradOffsetsP,
      (const int*)pairNodeIdxsP, (const int*)blockSlotsP,
      (const double*)gradsP, (const double*)DsP,
      (const double*)sigmasP, (const double*)dvsP,
      (int)nelems, (double)scale);
   env->ReleaseIntArrayElements (
      elemNodeCounts, elemNodeCountsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemPairOffsets, elemPairOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemIpOffsets, elemIpOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemGradOffsets, elemGradOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (pairNodeIdxs, pairNodeIdxsP, JNI_ABORT);
   env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (grads, gradsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (Ds, DsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (sigmas, sigmasP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (dvs, dvsP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL
Java_maspack_solvers_CuDssSolver_doAddLinearElasticStiffness3ElementDeviceValues
  (JNIEnv* env, jclass /*cls*/, jlong handle,
   jintArray elemNodeCounts, jintArray elemPairOffsets,
   jintArray elemIpOffsets, jintArray elemGradOffsets,
   jintArray pairNodeIdxs, jintArray blockSlots,
   jdoubleArray elemParams, jdoubleArray grads, jdoubleArray dvs,
   jint nelems, jdouble scale) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   jint* elemNodeCountsP =
      env->GetIntArrayElements (elemNodeCounts, nullptr);
   jint* elemPairOffsetsP =
      env->GetIntArrayElements (elemPairOffsets, nullptr);
   jint* elemIpOffsetsP =
      env->GetIntArrayElements (elemIpOffsets, nullptr);
   jint* elemGradOffsetsP =
      env->GetIntArrayElements (elemGradOffsets, nullptr);
   jint* pairNodeIdxsP = env->GetIntArrayElements (pairNodeIdxs, nullptr);
   jint* blockSlotsP = env->GetIntArrayElements (blockSlots, nullptr);
   jdouble* elemParamsP =
      env->GetDoubleArrayElements (elemParams, nullptr);
   jdouble* gradsP = env->GetDoubleArrayElements (grads, nullptr);
   jdouble* dvsP = env->GetDoubleArrayElements (dvs, nullptr);
   if (!elemNodeCountsP || !elemPairOffsetsP || !elemIpOffsetsP ||
       !elemGradOffsetsP || !pairNodeIdxsP || !blockSlotsP ||
       !elemParamsP || !gradsP || !dvsP) {
      if (elemNodeCountsP) {
         env->ReleaseIntArrayElements (
            elemNodeCounts, elemNodeCountsP, JNI_ABORT);
      }
      if (elemPairOffsetsP) {
         env->ReleaseIntArrayElements (
            elemPairOffsets, elemPairOffsetsP, JNI_ABORT);
      }
      if (elemIpOffsetsP) {
         env->ReleaseIntArrayElements (
            elemIpOffsets, elemIpOffsetsP, JNI_ABORT);
      }
      if (elemGradOffsetsP) {
         env->ReleaseIntArrayElements (
            elemGradOffsets, elemGradOffsetsP, JNI_ABORT);
      }
      if (pairNodeIdxsP) {
         env->ReleaseIntArrayElements (
            pairNodeIdxs, pairNodeIdxsP, JNI_ABORT);
      }
      if (blockSlotsP) {
         env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
      }
      if (elemParamsP) {
         env->ReleaseDoubleArrayElements (
            elemParams, elemParamsP, JNI_ABORT);
      }
      if (gradsP) env->ReleaseDoubleArrayElements (grads, gradsP, JNI_ABORT);
      if (dvsP) env->ReleaseDoubleArrayElements (dvs, dvsP, JNI_ABORT);
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = b->addLinearElasticStiffness3ElementDeviceValues (
      (const int*)elemNodeCountsP, (const int*)elemPairOffsetsP,
      (const int*)elemIpOffsetsP, (const int*)elemGradOffsetsP,
      (const int*)pairNodeIdxsP, (const int*)blockSlotsP,
      (const double*)elemParamsP, (const double*)gradsP,
      (const double*)dvsP, (int)nelems, (double)scale);
   env->ReleaseIntArrayElements (
      elemNodeCounts, elemNodeCountsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemPairOffsets, elemPairOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemIpOffsets, elemIpOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemGradOffsets, elemGradOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (pairNodeIdxs, pairNodeIdxsP, JNI_ABORT);
   env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (elemParams, elemParamsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (grads, gradsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (dvs, dvsP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL
Java_maspack_solvers_CuDssSolver_doAddDilationalStiffness3ElementDeviceValues
  (JNIEnv* env, jclass /*cls*/, jlong handle,
   jintArray elemNodeCounts, jintArray elemPressureCounts,
   jintArray elemPairOffsets, jintArray elemConstraintOffsets,
   jintArray elemRinvOffsets, jintArray pairNodeIdxs,
   jintArray blockSlots, jdoubleArray constraints, jdoubleArray rinvs,
   jint nelems, jdouble scale) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   jint* elemNodeCountsP =
      env->GetIntArrayElements (elemNodeCounts, nullptr);
   jint* elemPressureCountsP =
      env->GetIntArrayElements (elemPressureCounts, nullptr);
   jint* elemPairOffsetsP =
      env->GetIntArrayElements (elemPairOffsets, nullptr);
   jint* elemConstraintOffsetsP =
      env->GetIntArrayElements (elemConstraintOffsets, nullptr);
   jint* elemRinvOffsetsP =
      env->GetIntArrayElements (elemRinvOffsets, nullptr);
   jint* pairNodeIdxsP = env->GetIntArrayElements (pairNodeIdxs, nullptr);
   jint* blockSlotsP = env->GetIntArrayElements (blockSlots, nullptr);
   jdouble* constraintsP =
      env->GetDoubleArrayElements (constraints, nullptr);
   jdouble* rinvsP = env->GetDoubleArrayElements (rinvs, nullptr);
   if (!elemNodeCountsP || !elemPressureCountsP || !elemPairOffsetsP ||
       !elemConstraintOffsetsP || !elemRinvOffsetsP || !pairNodeIdxsP ||
       !blockSlotsP || !constraintsP || !rinvsP) {
      if (elemNodeCountsP) {
         env->ReleaseIntArrayElements (
            elemNodeCounts, elemNodeCountsP, JNI_ABORT);
      }
      if (elemPressureCountsP) {
         env->ReleaseIntArrayElements (
            elemPressureCounts, elemPressureCountsP, JNI_ABORT);
      }
      if (elemPairOffsetsP) {
         env->ReleaseIntArrayElements (
            elemPairOffsets, elemPairOffsetsP, JNI_ABORT);
      }
      if (elemConstraintOffsetsP) {
         env->ReleaseIntArrayElements (
            elemConstraintOffsets, elemConstraintOffsetsP, JNI_ABORT);
      }
      if (elemRinvOffsetsP) {
         env->ReleaseIntArrayElements (
            elemRinvOffsets, elemRinvOffsetsP, JNI_ABORT);
      }
      if (pairNodeIdxsP) {
         env->ReleaseIntArrayElements (
            pairNodeIdxs, pairNodeIdxsP, JNI_ABORT);
      }
      if (blockSlotsP) {
         env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
      }
      if (constraintsP) {
         env->ReleaseDoubleArrayElements (
            constraints, constraintsP, JNI_ABORT);
      }
      if (rinvsP) {
         env->ReleaseDoubleArrayElements (rinvs, rinvsP, JNI_ABORT);
      }
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = b->addDilationalStiffness3ElementDeviceValues (
      (const int*)elemNodeCountsP, (const int*)elemPressureCountsP,
      (const int*)elemPairOffsetsP, (const int*)elemConstraintOffsetsP,
      (const int*)elemRinvOffsetsP, (const int*)pairNodeIdxsP,
      (const int*)blockSlotsP, (const double*)constraintsP,
      (const double*)rinvsP, (int)nelems, (double)scale);
   env->ReleaseIntArrayElements (
      elemNodeCounts, elemNodeCountsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemPressureCounts, elemPressureCountsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemPairOffsets, elemPairOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemConstraintOffsets, elemConstraintOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (
      elemRinvOffsets, elemRinvOffsetsP, JNI_ABORT);
   env->ReleaseIntArrayElements (pairNodeIdxs, pairNodeIdxsP, JNI_ABORT);
   env->ReleaseIntArrayElements (blockSlots, blockSlotsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (constraints, constraintsP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (rinvs, rinvsP, JNI_ABORT);
   return (jint)status;
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doFactorDeviceValues
  (JNIEnv* /*env*/, jclass /*cls*/, jlong handle) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   return (jint)b->factorDeviceValues();
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doSolve
  (JNIEnv* env, jclass /*cls*/, jlong handle, jdoubleArray b, jdoubleArray x) {
   CuDssBridge* br = asBridge (handle);
   if (!br) return CUDSS_BRIDGE_ERR_STATE;
   jdouble* bP = env->GetDoubleArrayElements (b, nullptr);
   jdouble* xP = env->GetDoubleArrayElements (x, nullptr);
   if (!bP || !xP) {
      if (bP) env->ReleaseDoubleArrayElements (b, bP, JNI_ABORT);
      if (xP) env->ReleaseDoubleArrayElements (x, xP, JNI_ABORT);
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = br->solve ((const double*)bP, (double*)xP);
   // 0: commit changes to the x array
   env->ReleaseDoubleArrayElements (b, bP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (x, xP, 0);
   return (jint)status;
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doSolveMulti
  (JNIEnv* env, jclass /*cls*/, jlong handle, jint nrhs,
   jdoubleArray B, jdoubleArray X) {
   CuDssBridge* br = asBridge (handle);
   if (!br) return CUDSS_BRIDGE_ERR_STATE;
   jdouble* bP = env->GetDoubleArrayElements (B, nullptr);
   jdouble* xP = env->GetDoubleArrayElements (X, nullptr);
   if (!bP || !xP) {
      if (bP) env->ReleaseDoubleArrayElements (B, bP, JNI_ABORT);
      if (xP) env->ReleaseDoubleArrayElements (X, xP, JNI_ABORT);
      return CUDSS_BRIDGE_ERR_CUDA_COPY;
   }
   int status = br->solveMulti ((int)nrhs, (const double*)bP, (double*)xP);
   env->ReleaseDoubleArrayElements (B, bP, JNI_ABORT);
   env->ReleaseDoubleArrayElements (X, xP, 0);
   return (jint)status;
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doIterativeSolve
  (JNIEnv* env, jclass /*cls*/, jlong handle,
   jdoubleArray vals, jdoubleArray b, jdoubleArray x,
   jdouble tolRel, jint maxIter) {
   CuDssBridge* br = asBridge (handle);
   if (!br) return -1;

   jdouble* vp = env->GetDoubleArrayElements (vals, nullptr);
   jdouble* bp = env->GetDoubleArrayElements (b,    nullptr);
   jdouble* xp = env->GetDoubleArrayElements (x,    nullptr);
   if (!vp || !bp || !xp) {
      if (vp) env->ReleaseDoubleArrayElements (vals, vp, JNI_ABORT);
      if (bp) env->ReleaseDoubleArrayElements (b,    bp, JNI_ABORT);
      if (xp) env->ReleaseDoubleArrayElements (x,    xp, JNI_ABORT);
      return -1;
   }
   int iters = 0;
   int status = br->iterativeSolveBiCGStab (
      (const double*)vp, (const double*)bp, (double*)xp,
      (double)tolRel, (int)maxIter, &iters);
   env->ReleaseDoubleArrayElements (vals, vp, JNI_ABORT);
   env->ReleaseDoubleArrayElements (b,    bp, JNI_ABORT);
   // Commit x back to Java.
   env->ReleaseDoubleArrayElements (x,    xp, (status == 0) ? 0 : JNI_ABORT);
   // Return iteration count on success, negative status on failure.
   return (jint)((status == 0) ? iters : status);
}

JNIEXPORT void JNICALL Java_maspack_solvers_CuDssSolver_doDispose
  (JNIEnv* /*env*/, jclass /*cls*/, jlong handle) {
   CuDssBridge* b = asBridge (handle);
   if (b) {
      delete b; // destructor calls dispose()
   }
}

JNIEXPORT jstring JNICALL Java_maspack_solvers_CuDssSolver_doGetLastError
  (JNIEnv* env, jclass /*cls*/, jlong handle) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return nullptr;
   const char* msg = b->getLastErrorMessage();
   if (!msg) return nullptr;
   return env->NewStringUTF (msg);
}

JNIEXPORT void JNICALL Java_maspack_solvers_CuDssSolver_doSetTimingEnabled
  (JNIEnv* /*env*/, jclass /*cls*/, jboolean on) {
   CuDssBridge::setTimingEnabled (on == JNI_TRUE);
}

JNIEXPORT jint JNICALL Java_maspack_solvers_CuDssSolver_doSetIrSteps
  (JNIEnv* /*env*/, jclass /*cls*/, jlong handle, jint n) {
   CuDssBridge* b = asBridge (handle);
   if (!b) return CUDSS_BRIDGE_ERR_STATE;
   return (jint) b->setIterativeRefinementSteps ((int)n);
}

JNIEXPORT jstring JNICALL Java_maspack_solvers_CuDssSolver_doGetVersion
  (JNIEnv* env, jclass /*cls*/) {
   int major = 0, minor = 0, patch = 0;
   cudssGetProperty (MAJOR_VERSION, &major);
   cudssGetProperty (MINOR_VERSION, &minor);
   cudssGetProperty (PATCH_LEVEL,   &patch);
   char buf[64];
   std::snprintf (buf, sizeof(buf), "%d.%d.%d", major, minor, patch);
   return env->NewStringUTF (buf);
}
