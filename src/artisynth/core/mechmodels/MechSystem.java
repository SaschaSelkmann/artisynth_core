/**
 * Copyright (c) 2014, by the Authors: John E Lloyd (UBC)
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package artisynth.core.mechmodels;

import java.util.*;

import artisynth.core.modelbase.StepAdjustment;
import maspack.matrix.*;
import maspack.spatialmotion.FrictionInfo;
import maspack.spatialmotion.RigidBodyConstraint.MotionType;
import maspack.util.IntHolder;
import maspack.util.DataBuffer;

/**
 * Interface to a second order mechanical system that can be integrated by 
 * a variety of integrators.
 */
public interface MechSystem {

   /**
    * Context supplied to optional GPU assembly hooks. It exposes the current
    * solve matrix structure together with a block-to-CRS slot map that can be
    * reused while the system structure version remains unchanged.
    */
   public static class GpuAssemblyContext {
      private SparseNumberedBlockMatrix myMatrix;
      private SparseNumberedBlockMatrix.CrsBlockSlotMap mySlotMap;
      private double[] myCrsValues;
      private int[] myCrsColIdxs;
      private int[] myCrsRowOffs;
      private int[] myZeroBasedCrsColIdxs;
      private int[] myZeroBasedCrsRowOffs;
      private int[] myCrsValueSlots;
      private double[] myCrsValueContributions;
      private int myNumCrsValueContributions;
      private int[] myScaledDiagonal3Slots;
      private double[] myScaledDiagonal3Values;
      private int myNumScaledDiagonal3Contributions;
      private int[] myScaledBlock3Slots;
      private double[] myScaledBlock3Values;
      private double[] myScaledBlock3Scales;
      private int myNumScaledBlock3Contributions;
      private int[] myMaterialStiffness3Slots;
      private double[] myMaterialStiffness3Gis;
      private double[] myMaterialStiffness3Gjs;
      private double[] myMaterialStiffness3Ds;
      private double[] myMaterialStiffness3Sigmas;
      private double[] myMaterialStiffness3Dvs;
      private int myNumMaterialStiffness3Contributions;
      private int[] myMaterialStiffness3ElemNodeCounts;
      private int[] myMaterialStiffness3ElemPairOffsets;
      private int[] myMaterialStiffness3ElemIpOffsets;
      private int[] myMaterialStiffness3ElemGradOffsets;
      private int[] myMaterialStiffness3PairNodeIdxs;
      private int[] myMaterialStiffness3ElemBlockSlots;
      private double[] myMaterialStiffness3ElemGrads;
      private double[] myMaterialStiffness3ElemDs;
      private double[] myMaterialStiffness3ElemSigmas;
      private double[] myMaterialStiffness3ElemDvs;
      private int myNumMaterialStiffness3ElemContributions;
      private int myNumMaterialStiffness3ElemPairs;
      private int myNumMaterialStiffness3ElemIps;
      private int myNumMaterialStiffness3ElemGradVecs;
      private int[] myLinearElasticStiffness3ElemNodeCounts;
      private int[] myLinearElasticStiffness3ElemPairOffsets;
      private int[] myLinearElasticStiffness3ElemIpOffsets;
      private int[] myLinearElasticStiffness3ElemGradOffsets;
      private int[] myLinearElasticStiffness3PairNodeIdxs;
      private int[] myLinearElasticStiffness3ElemBlockSlots;
      private double[] myLinearElasticStiffness3ElemParams;
      private double[] myLinearElasticStiffness3ElemGrads;
      private double[] myLinearElasticStiffness3ElemDvs;
      private int myNumLinearElasticStiffness3ElemContributions;
      private int myNumLinearElasticStiffness3ElemPairs;
      private int myNumLinearElasticStiffness3ElemIps;
      private int myNumLinearElasticStiffness3ElemGradVecs;
      private int[] myLinearElasticGeometry3ElemNodeCounts;
      private int[] myLinearElasticGeometry3ElemNodeOffsets;
      private int[] myLinearElasticGeometry3ElemPairOffsets;
      private int[] myLinearElasticGeometry3ElemIpOffsets;
      private int[] myLinearElasticGeometry3ElemNaturalGradOffsets;
      private int[] myLinearElasticGeometry3PairNodeIdxs;
      private int[] myLinearElasticGeometry3ElemBlockSlots;
      // Per-element-node master-slave reduction data: target block dimension
      // (3 free / 6 slave) and the 6x3 transform T (I3 free / H slave),
      // row-major in an 18-double stride. See addReducedMaterialStiffness3Block.
      private int[] myLinearElasticGeometry3NodeDims;
      private double[] myLinearElasticGeometry3NodeTransforms;
      private double[] myLinearElasticGeometry3ElemParams;
      private double[] myLinearElasticGeometry3ElemNodePositions;
      private double[] myLinearElasticGeometry3NaturalGrads;
      private double[] myLinearElasticGeometry3IpWeights;
      private int myNumLinearElasticGeometry3ElemContributions;
      private int myNumLinearElasticGeometry3ElemNodes;
      private int myNumLinearElasticGeometry3ElemPairs;
      private int myNumLinearElasticGeometry3ElemIps;
      private int myNumLinearElasticGeometry3ElemNaturalGradVecs;
      private int[] myDilationalStiffness3ElemNodeCounts;
      private int[] myDilationalStiffness3ElemPressureCounts;
      private int[] myDilationalStiffness3ElemPairOffsets;
      private int[] myDilationalStiffness3ElemConstraintOffsets;
      private int[] myDilationalStiffness3ElemRinvOffsets;
      private int[] myDilationalStiffness3PairNodeIdxs;
      private int[] myDilationalStiffness3ElemBlockSlots;
      private double[] myDilationalStiffness3ElemConstraints;
      private double[] myDilationalStiffness3ElemRinvs;
      private int myNumDilationalStiffness3ElemContributions;
      private int myNumDilationalStiffness3ElemPairs;
      private int myNumDilationalStiffness3ElemConstraints;
      private int myNumDilationalStiffness3ElemRinvs;
      private int myStructureVersion;

      public GpuAssemblyContext (
         SparseNumberedBlockMatrix matrix,
         SparseNumberedBlockMatrix.CrsBlockSlotMap slotMap,
         int structureVersion) {
         myMatrix = matrix;
         mySlotMap = slotMap;
         myStructureVersion = structureVersion;
         int numVals = slotMap.numVals();
         myCrsValues = new double[numVals];
         myCrsColIdxs = new int[numVals];
         myCrsRowOffs = new int[slotMap.rowSize()+1];
         matrix.getCRSIndices (
            myCrsColIdxs, myCrsRowOffs, slotMap.getPartition(),
            slotMap.rowSize(), slotMap.colSize());
         myCrsValueSlots = new int[0];
         myCrsValueContributions = new double[0];
         myNumCrsValueContributions = 0;
         myScaledDiagonal3Slots = new int[0];
         myScaledDiagonal3Values = new double[0];
         myNumScaledDiagonal3Contributions = 0;
         myScaledBlock3Slots = new int[0];
         myScaledBlock3Values = new double[0];
         myScaledBlock3Scales = new double[0];
         myNumScaledBlock3Contributions = 0;
         myMaterialStiffness3Slots = new int[0];
         myMaterialStiffness3Gis = new double[0];
         myMaterialStiffness3Gjs = new double[0];
         myMaterialStiffness3Ds = new double[0];
         myMaterialStiffness3Sigmas = new double[0];
         myMaterialStiffness3Dvs = new double[0];
         myNumMaterialStiffness3Contributions = 0;
         myMaterialStiffness3ElemNodeCounts = new int[0];
         myMaterialStiffness3ElemPairOffsets = new int[] { 0 };
         myMaterialStiffness3ElemIpOffsets = new int[] { 0 };
         myMaterialStiffness3ElemGradOffsets = new int[] { 0 };
         myMaterialStiffness3PairNodeIdxs = new int[0];
         myMaterialStiffness3ElemBlockSlots = new int[0];
         myMaterialStiffness3ElemGrads = new double[0];
         myMaterialStiffness3ElemDs = new double[0];
         myMaterialStiffness3ElemSigmas = new double[0];
         myMaterialStiffness3ElemDvs = new double[0];
         myNumMaterialStiffness3ElemContributions = 0;
         myNumMaterialStiffness3ElemPairs = 0;
         myNumMaterialStiffness3ElemIps = 0;
         myNumMaterialStiffness3ElemGradVecs = 0;
         myLinearElasticStiffness3ElemNodeCounts = new int[0];
         myLinearElasticStiffness3ElemPairOffsets = new int[] { 0 };
         myLinearElasticStiffness3ElemIpOffsets = new int[] { 0 };
         myLinearElasticStiffness3ElemGradOffsets = new int[] { 0 };
         myLinearElasticStiffness3PairNodeIdxs = new int[0];
         myLinearElasticStiffness3ElemBlockSlots = new int[0];
         myLinearElasticStiffness3ElemParams = new double[0];
         myLinearElasticStiffness3ElemGrads = new double[0];
         myLinearElasticStiffness3ElemDvs = new double[0];
         myNumLinearElasticStiffness3ElemContributions = 0;
         myNumLinearElasticStiffness3ElemPairs = 0;
         myNumLinearElasticStiffness3ElemIps = 0;
         myNumLinearElasticStiffness3ElemGradVecs = 0;
         myLinearElasticGeometry3ElemNodeCounts = new int[0];
         myLinearElasticGeometry3ElemNodeOffsets = new int[] { 0 };
         myLinearElasticGeometry3ElemPairOffsets = new int[] { 0 };
         myLinearElasticGeometry3ElemIpOffsets = new int[] { 0 };
         myLinearElasticGeometry3ElemNaturalGradOffsets = new int[] { 0 };
         myLinearElasticGeometry3PairNodeIdxs = new int[0];
         myLinearElasticGeometry3ElemBlockSlots = new int[0];
         myLinearElasticGeometry3NodeDims = new int[0];
         myLinearElasticGeometry3NodeTransforms = new double[0];
         myLinearElasticGeometry3ElemParams = new double[0];
         myLinearElasticGeometry3ElemNodePositions = new double[0];
         myLinearElasticGeometry3NaturalGrads = new double[0];
         myLinearElasticGeometry3IpWeights = new double[0];
         myNumLinearElasticGeometry3ElemContributions = 0;
         myNumLinearElasticGeometry3ElemNodes = 0;
         myNumLinearElasticGeometry3ElemPairs = 0;
         myNumLinearElasticGeometry3ElemIps = 0;
         myNumLinearElasticGeometry3ElemNaturalGradVecs = 0;
         myDilationalStiffness3ElemNodeCounts = new int[0];
         myDilationalStiffness3ElemPressureCounts = new int[0];
         myDilationalStiffness3ElemPairOffsets = new int[] { 0 };
         myDilationalStiffness3ElemConstraintOffsets = new int[] { 0 };
         myDilationalStiffness3ElemRinvOffsets = new int[] { 0 };
         myDilationalStiffness3PairNodeIdxs = new int[0];
         myDilationalStiffness3ElemBlockSlots = new int[0];
         myDilationalStiffness3ElemConstraints = new double[0];
         myDilationalStiffness3ElemRinvs = new double[0];
         myNumDilationalStiffness3ElemContributions = 0;
         myNumDilationalStiffness3ElemPairs = 0;
         myNumDilationalStiffness3ElemConstraints = 0;
         myNumDilationalStiffness3ElemRinvs = 0;
      }

      public SparseNumberedBlockMatrix getMatrix() {
         return myMatrix;
      }

      public SparseNumberedBlockMatrix.CrsBlockSlotMap getSlotMap() {
         return mySlotMap;
      }

      public double[] getCrsValues() {
         return myCrsValues;
      }

      public void clearCrsValues() {
         Arrays.fill (myCrsValues, 0);
      }

      public void clearCrsValueContributions() {
         myNumCrsValueContributions = 0;
         myNumScaledDiagonal3Contributions = 0;
         myNumScaledBlock3Contributions = 0;
         myNumMaterialStiffness3Contributions = 0;
         myNumMaterialStiffness3ElemContributions = 0;
         myNumMaterialStiffness3ElemPairs = 0;
         myNumMaterialStiffness3ElemIps = 0;
         myNumMaterialStiffness3ElemGradVecs = 0;
         myNumLinearElasticStiffness3ElemContributions = 0;
         myNumLinearElasticStiffness3ElemPairs = 0;
         myNumLinearElasticStiffness3ElemIps = 0;
         myNumLinearElasticStiffness3ElemGradVecs = 0;
         myNumLinearElasticGeometry3ElemContributions = 0;
         myNumLinearElasticGeometry3ElemNodes = 0;
         myNumLinearElasticGeometry3ElemPairs = 0;
         myNumLinearElasticGeometry3ElemIps = 0;
         myNumLinearElasticGeometry3ElemNaturalGradVecs = 0;
         myNumDilationalStiffness3ElemContributions = 0;
         myNumDilationalStiffness3ElemPairs = 0;
         myNumDilationalStiffness3ElemConstraints = 0;
         myNumDilationalStiffness3ElemRinvs = 0;
         myMaterialStiffness3ElemPairOffsets[0] = 0;
         myMaterialStiffness3ElemIpOffsets[0] = 0;
         myMaterialStiffness3ElemGradOffsets[0] = 0;
         myLinearElasticStiffness3ElemPairOffsets[0] = 0;
         myLinearElasticStiffness3ElemIpOffsets[0] = 0;
         myLinearElasticStiffness3ElemGradOffsets[0] = 0;
         myLinearElasticGeometry3ElemNodeOffsets[0] = 0;
         myLinearElasticGeometry3ElemPairOffsets[0] = 0;
         myLinearElasticGeometry3ElemIpOffsets[0] = 0;
         myLinearElasticGeometry3ElemNaturalGradOffsets[0] = 0;
         myDilationalStiffness3ElemPairOffsets[0] = 0;
         myDilationalStiffness3ElemConstraintOffsets[0] = 0;
         myDilationalStiffness3ElemRinvOffsets[0] = 0;
      }

      public void addCrsValueContribution (int slot, double value) {
         if (value == 0) {
            return;
         }
         ensureCrsValueContributionCapacity (myNumCrsValueContributions+1);
         myCrsValueSlots[myNumCrsValueContributions] = slot;
         myCrsValueContributions[myNumCrsValueContributions] = value;
         myNumCrsValueContributions++;
         myCrsValues[slot] += value;
      }

      public void addScaledDiagonal3CrsValueContribution (
         int slot0, int slot1, int slot2, double value) {
         if (value == 0) {
            return;
         }
         ensureScaledDiagonal3ContributionCapacity (
            myNumScaledDiagonal3Contributions+1);
         int idx = 3*myNumScaledDiagonal3Contributions;
         myScaledDiagonal3Slots[idx] = slot0;
         myScaledDiagonal3Slots[idx+1] = slot1;
         myScaledDiagonal3Slots[idx+2] = slot2;
         myScaledDiagonal3Values[myNumScaledDiagonal3Contributions] = value;
         myNumScaledDiagonal3Contributions++;
         if (slot0 >= 0) {
            myCrsValues[slot0] += value;
         }
         if (slot1 >= 0) {
            myCrsValues[slot1] += value;
         }
         if (slot2 >= 0) {
            myCrsValues[slot2] += value;
         }
      }

      public void addScaledBlock3CrsValueContribution (
         int blkNum, double scale, Matrix3d K) {

         if (blkNum == -1 || K == null || scale == 0) {
            return;
         }
         if (!mySlotMap.hasBlockSlots (blkNum)) {
            return;
         }
         ensureScaledBlock3ContributionCapacity (
            myNumScaledBlock3Contributions+1);
         int slotIdx = 9*myNumScaledBlock3Contributions;
         int valIdx = slotIdx;
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 0, 0);
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 0, 1);
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 0, 2);
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 1, 0);
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 1, 1);
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 1, 2);
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 2, 0);
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 2, 1);
         myScaledBlock3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 2, 2);
         myScaledBlock3Values[valIdx++] = K.m00;
         myScaledBlock3Values[valIdx++] = K.m01;
         myScaledBlock3Values[valIdx++] = K.m02;
         myScaledBlock3Values[valIdx++] = K.m10;
         myScaledBlock3Values[valIdx++] = K.m11;
         myScaledBlock3Values[valIdx++] = K.m12;
         myScaledBlock3Values[valIdx++] = K.m20;
         myScaledBlock3Values[valIdx++] = K.m21;
         myScaledBlock3Values[valIdx++] = K.m22;
         myScaledBlock3Scales[myNumScaledBlock3Contributions] = scale;
         int base = 9*myNumScaledBlock3Contributions;
         for (int i=0; i<9; i++) {
            int slot = myScaledBlock3Slots[base+i];
            if (slot >= 0) {
               myCrsValues[slot] += scale*myScaledBlock3Values[base+i];
            }
         }
         myNumScaledBlock3Contributions++;
      }

      public void addMaterialStiffness3CrsValueContribution (
         int blkNum, Vector3d gi, Matrix6d D, SymmetricMatrix3d sig,
         Vector3d gj, double dv) {

         addMaterialStiffness3CrsValueContribution (
            blkNum, gi, D, sig, gj, dv, 1.0);
      }

      public void addMaterialStiffness3CrsValueContribution (
         int blkNum, Vector3d gi, Matrix6d D, SymmetricMatrix3d sig,
         Vector3d gj, double dv, double scale) {

         if (blkNum == -1 || gi == null || D == null || sig == null ||
             gj == null || dv == 0 || scale == 0) {
            return;
         }
         if (!mySlotMap.hasBlockSlots (blkNum)) {
            return;
         }
         double scaledDv = scale*dv;
         ensureMaterialStiffness3ContributionCapacity (
            myNumMaterialStiffness3Contributions+1);
         int slotIdx = 9*myNumMaterialStiffness3Contributions;
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 0, 0);
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 0, 1);
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 0, 2);
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 1, 0);
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 1, 1);
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 1, 2);
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 2, 0);
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 2, 1);
         myMaterialStiffness3Slots[slotIdx++] =
            mySlotMap.getBlockValueSlot (blkNum, 2, 2);

         int vecIdx = 3*myNumMaterialStiffness3Contributions;
         myMaterialStiffness3Gis[vecIdx] = gi.x;
         myMaterialStiffness3Gis[vecIdx+1] = gi.y;
         myMaterialStiffness3Gis[vecIdx+2] = gi.z;
         myMaterialStiffness3Gjs[vecIdx] = gj.x;
         myMaterialStiffness3Gjs[vecIdx+1] = gj.y;
         myMaterialStiffness3Gjs[vecIdx+2] = gj.z;

         int dIdx = 36*myNumMaterialStiffness3Contributions;
         myMaterialStiffness3Ds[dIdx++] = D.m00;
         myMaterialStiffness3Ds[dIdx++] = D.m01;
         myMaterialStiffness3Ds[dIdx++] = D.m02;
         myMaterialStiffness3Ds[dIdx++] = D.m03;
         myMaterialStiffness3Ds[dIdx++] = D.m04;
         myMaterialStiffness3Ds[dIdx++] = D.m05;
         myMaterialStiffness3Ds[dIdx++] = D.m10;
         myMaterialStiffness3Ds[dIdx++] = D.m11;
         myMaterialStiffness3Ds[dIdx++] = D.m12;
         myMaterialStiffness3Ds[dIdx++] = D.m13;
         myMaterialStiffness3Ds[dIdx++] = D.m14;
         myMaterialStiffness3Ds[dIdx++] = D.m15;
         myMaterialStiffness3Ds[dIdx++] = D.m20;
         myMaterialStiffness3Ds[dIdx++] = D.m21;
         myMaterialStiffness3Ds[dIdx++] = D.m22;
         myMaterialStiffness3Ds[dIdx++] = D.m23;
         myMaterialStiffness3Ds[dIdx++] = D.m24;
         myMaterialStiffness3Ds[dIdx++] = D.m25;
         myMaterialStiffness3Ds[dIdx++] = D.m30;
         myMaterialStiffness3Ds[dIdx++] = D.m31;
         myMaterialStiffness3Ds[dIdx++] = D.m32;
         myMaterialStiffness3Ds[dIdx++] = D.m33;
         myMaterialStiffness3Ds[dIdx++] = D.m34;
         myMaterialStiffness3Ds[dIdx++] = D.m35;
         myMaterialStiffness3Ds[dIdx++] = D.m40;
         myMaterialStiffness3Ds[dIdx++] = D.m41;
         myMaterialStiffness3Ds[dIdx++] = D.m42;
         myMaterialStiffness3Ds[dIdx++] = D.m43;
         myMaterialStiffness3Ds[dIdx++] = D.m44;
         myMaterialStiffness3Ds[dIdx++] = D.m45;
         myMaterialStiffness3Ds[dIdx++] = D.m50;
         myMaterialStiffness3Ds[dIdx++] = D.m51;
         myMaterialStiffness3Ds[dIdx++] = D.m52;
         myMaterialStiffness3Ds[dIdx++] = D.m53;
         myMaterialStiffness3Ds[dIdx++] = D.m54;
         myMaterialStiffness3Ds[dIdx++] = D.m55;

         int sigIdx = 6*myNumMaterialStiffness3Contributions;
         myMaterialStiffness3Sigmas[sigIdx] = sig.m00;
         myMaterialStiffness3Sigmas[sigIdx+1] = sig.m11;
         myMaterialStiffness3Sigmas[sigIdx+2] = sig.m22;
         myMaterialStiffness3Sigmas[sigIdx+3] = sig.m01;
         myMaterialStiffness3Sigmas[sigIdx+4] = sig.m12;
         myMaterialStiffness3Sigmas[sigIdx+5] = sig.m02;
         myMaterialStiffness3Dvs[
            myNumMaterialStiffness3Contributions] = scaledDv;

         addMaterialStiffness3ToCrsValues (
            9*myNumMaterialStiffness3Contributions, gi, D, sig, gj,
            scaledDv);
         myNumMaterialStiffness3Contributions++;
      }

      private void addMaterialStiffness3ToCrsValues (
         int slotBase, Vector3d gi, Matrix6d D, SymmetricMatrix3d sig,
         Vector3d gj, double dv) {

         double gjx = gj.x*dv;
         double gjy = gj.y*dv;
         double gjz = gj.z*dv;

         double dm00 = D.m00*gjx + D.m03*gjy + D.m05*gjz;
         double dm01 = D.m01*gjy + D.m03*gjx + D.m04*gjz;
         double dm02 = D.m02*gjz + D.m04*gjy + D.m05*gjx;

         double dm10 = D.m10*gjx + D.m13*gjy + D.m15*gjz;
         double dm11 = D.m11*gjy + D.m13*gjx + D.m14*gjz;
         double dm12 = D.m12*gjz + D.m14*gjy + D.m15*gjx;

         double dm20 = D.m20*gjx + D.m23*gjy + D.m25*gjz;
         double dm21 = D.m21*gjy + D.m23*gjx + D.m24*gjz;
         double dm22 = D.m22*gjz + D.m24*gjy + D.m25*gjx;

         double dm30 = D.m30*gjx + D.m33*gjy + D.m35*gjz;
         double dm31 = D.m31*gjy + D.m33*gjx + D.m34*gjz;
         double dm32 = D.m32*gjz + D.m34*gjy + D.m35*gjx;

         double dm40 = D.m40*gjx + D.m43*gjy + D.m45*gjz;
         double dm41 = D.m41*gjy + D.m43*gjx + D.m44*gjz;
         double dm42 = D.m42*gjz + D.m44*gjy + D.m45*gjx;

         double dm50 = D.m50*gjx + D.m53*gjy + D.m55*gjz;
         double dm51 = D.m51*gjy + D.m53*gjx + D.m54*gjz;
         double dm52 = D.m52*gjz + D.m54*gjy + D.m55*gjx;

         double gix = gi.x;
         double giy = gi.y;
         double giz = gi.z;
         double[] K = new double[9];
         K[0] = gix*dm00 + giy*dm30 + giz*dm50;
         K[1] = gix*dm01 + giy*dm31 + giz*dm51;
         K[2] = gix*dm02 + giy*dm32 + giz*dm52;
         K[3] = giy*dm10 + gix*dm30 + giz*dm40;
         K[4] = giy*dm11 + gix*dm31 + giz*dm41;
         K[5] = giy*dm12 + gix*dm32 + giz*dm42;
         K[6] = giz*dm20 + giy*dm40 + gix*dm50;
         K[7] = giz*dm21 + giy*dm41 + gix*dm51;
         K[8] = giz*dm22 + giy*dm42 + gix*dm52;

         double Kg = (
            gi.x*(sig.m00*gj.x + sig.m01*gj.y + sig.m02*gj.z) +
            gi.y*(sig.m10*gj.x + sig.m11*gj.y + sig.m12*gj.z) +
            gi.z*(sig.m20*gj.x + sig.m21*gj.y + sig.m22*gj.z))*dv;
         K[0] += Kg;
         K[4] += Kg;
         K[8] += Kg;
         for (int i=0; i<9; i++) {
            int slot = myMaterialStiffness3Slots[slotBase+i];
            if (slot >= 0) {
               myCrsValues[slot] += K[i];
            }
         }
      }

      private void ensureCrsValueContributionCapacity (int cap) {
         if (myCrsValueSlots.length < cap) {
            int newCap = Math.max (cap, Math.max (64, 2*myCrsValueSlots.length));
            myCrsValueSlots = Arrays.copyOf (myCrsValueSlots, newCap);
            myCrsValueContributions =
               Arrays.copyOf (myCrsValueContributions, newCap);
         }
      }

      private void ensureScaledDiagonal3ContributionCapacity (int cap) {
         if (myScaledDiagonal3Values.length < cap) {
            int newCap =
               Math.max (cap, Math.max (64, 2*myScaledDiagonal3Values.length));
            myScaledDiagonal3Slots =
               Arrays.copyOf (myScaledDiagonal3Slots, 3*newCap);
            myScaledDiagonal3Values =
               Arrays.copyOf (myScaledDiagonal3Values, newCap);
         }
      }

      private void ensureScaledBlock3ContributionCapacity (int cap) {
         if (myScaledBlock3Scales.length < cap) {
            int newCap =
               Math.max (cap, Math.max (64, 2*myScaledBlock3Scales.length));
            myScaledBlock3Slots =
               Arrays.copyOf (myScaledBlock3Slots, 9*newCap);
            myScaledBlock3Values =
               Arrays.copyOf (myScaledBlock3Values, 9*newCap);
            myScaledBlock3Scales =
               Arrays.copyOf (myScaledBlock3Scales, newCap);
         }
      }

      private void ensureMaterialStiffness3ContributionCapacity (int cap) {
         if (myMaterialStiffness3Dvs.length < cap) {
            int newCap =
               Math.max (cap, Math.max (64, 2*myMaterialStiffness3Dvs.length));
            myMaterialStiffness3Slots =
               Arrays.copyOf (myMaterialStiffness3Slots, 9*newCap);
            myMaterialStiffness3Gis =
               Arrays.copyOf (myMaterialStiffness3Gis, 3*newCap);
            myMaterialStiffness3Gjs =
               Arrays.copyOf (myMaterialStiffness3Gjs, 3*newCap);
            myMaterialStiffness3Ds =
               Arrays.copyOf (myMaterialStiffness3Ds, 36*newCap);
            myMaterialStiffness3Sigmas =
               Arrays.copyOf (myMaterialStiffness3Sigmas, 6*newCap);
            myMaterialStiffness3Dvs =
               Arrays.copyOf (myMaterialStiffness3Dvs, newCap);
         }
      }

      public void addMaterialStiffness3ElementCrsValueContributions (
         int[] elemNodeCounts, int[] elemPairOffsets, int[] elemIpOffsets,
         int[] elemGradOffsets, int[] pairNodeIdxs, int[] blockSlots,
         double[] grads, double[] Ds, double[] sigmas, double[] dvs,
         int nelems) {

         if (nelems == 0) {
            return;
         }
         int npairs = elemPairOffsets[nelems];
         int nips = elemIpOffsets[nelems];
         int ngrads = elemGradOffsets[nelems];
         if (npairs == 0 || nips == 0 || ngrads == 0) {
            return;
         }
         ensureMaterialStiffness3ElementCapacity (
            myNumMaterialStiffness3ElemContributions + nelems,
            myNumMaterialStiffness3ElemPairs + npairs,
            myNumMaterialStiffness3ElemIps + nips,
            myNumMaterialStiffness3ElemGradVecs + ngrads);

         int elemBase = myNumMaterialStiffness3ElemContributions;
         int pairBase = myNumMaterialStiffness3ElemPairs;
         int ipBase = myNumMaterialStiffness3ElemIps;
         int gradBase = myNumMaterialStiffness3ElemGradVecs;

         System.arraycopy (
            elemNodeCounts, 0, myMaterialStiffness3ElemNodeCounts,
            elemBase, nelems);
         for (int i=1; i<=nelems; i++) {
            myMaterialStiffness3ElemPairOffsets[elemBase+i] =
               pairBase + elemPairOffsets[i];
            myMaterialStiffness3ElemIpOffsets[elemBase+i] =
               ipBase + elemIpOffsets[i];
            myMaterialStiffness3ElemGradOffsets[elemBase+i] =
               gradBase + elemGradOffsets[i];
         }
         System.arraycopy (
            pairNodeIdxs, 0, myMaterialStiffness3PairNodeIdxs,
            2*pairBase, 2*npairs);
         System.arraycopy (
            blockSlots, 0, myMaterialStiffness3ElemBlockSlots,
            9*pairBase, 9*npairs);
         System.arraycopy (
            grads, 0, myMaterialStiffness3ElemGrads, 3*gradBase, 3*ngrads);
         System.arraycopy (
            Ds, 0, myMaterialStiffness3ElemDs, 36*ipBase, 36*nips);
         System.arraycopy (
            sigmas, 0, myMaterialStiffness3ElemSigmas, 6*ipBase, 6*nips);
         System.arraycopy (
            dvs, 0, myMaterialStiffness3ElemDvs, ipBase, nips);

         myNumMaterialStiffness3ElemContributions += nelems;
         myNumMaterialStiffness3ElemPairs += npairs;
         myNumMaterialStiffness3ElemIps += nips;
         myNumMaterialStiffness3ElemGradVecs += ngrads;
      }

      private void ensureMaterialStiffness3ElementCapacity (
         int nelems, int npairs, int nips, int ngrads) {

         if (myMaterialStiffness3ElemNodeCounts.length < nelems) {
            int newCap = Math.max (
               nelems,
               Math.max (64, 2*myMaterialStiffness3ElemNodeCounts.length));
            myMaterialStiffness3ElemNodeCounts =
               Arrays.copyOf (myMaterialStiffness3ElemNodeCounts, newCap);
            myMaterialStiffness3ElemPairOffsets =
               Arrays.copyOf (myMaterialStiffness3ElemPairOffsets, newCap+1);
            myMaterialStiffness3ElemIpOffsets =
               Arrays.copyOf (myMaterialStiffness3ElemIpOffsets, newCap+1);
            myMaterialStiffness3ElemGradOffsets =
               Arrays.copyOf (myMaterialStiffness3ElemGradOffsets, newCap+1);
         }
         if (myMaterialStiffness3PairNodeIdxs.length < 2*npairs) {
            int newCap = Math.max (
               npairs, Math.max (64, myMaterialStiffness3PairNodeIdxs.length));
            while (newCap < npairs) {
               newCap *= 2;
            }
            myMaterialStiffness3PairNodeIdxs =
               Arrays.copyOf (myMaterialStiffness3PairNodeIdxs, 2*newCap);
            myMaterialStiffness3ElemBlockSlots =
               Arrays.copyOf (myMaterialStiffness3ElemBlockSlots, 9*newCap);
         }
         if (myMaterialStiffness3ElemDvs.length < nips) {
            int newCap = Math.max (
               nips, Math.max (64, 2*myMaterialStiffness3ElemDvs.length));
            myMaterialStiffness3ElemDs =
               Arrays.copyOf (myMaterialStiffness3ElemDs, 36*newCap);
            myMaterialStiffness3ElemSigmas =
               Arrays.copyOf (myMaterialStiffness3ElemSigmas, 6*newCap);
            myMaterialStiffness3ElemDvs =
               Arrays.copyOf (myMaterialStiffness3ElemDvs, newCap);
         }
         if (myMaterialStiffness3ElemGrads.length < 3*ngrads) {
            int newCap = Math.max (
               ngrads, Math.max (64, myMaterialStiffness3ElemGrads.length));
            while (newCap < ngrads) {
               newCap *= 2;
            }
            myMaterialStiffness3ElemGrads =
               Arrays.copyOf (myMaterialStiffness3ElemGrads, 3*newCap);
         }
      }

      public void addLinearElasticStiffness3ElementCrsValueContributions (
         int[] elemNodeCounts, int[] elemPairOffsets, int[] elemIpOffsets,
         int[] elemGradOffsets, int[] pairNodeIdxs, int[] blockSlots,
         double[] elemParams, double[] grads, double[] dvs, int nelems) {

         if (nelems == 0) {
            return;
         }
         int npairs = elemPairOffsets[nelems];
         int nips = elemIpOffsets[nelems];
         int ngrads = elemGradOffsets[nelems];
         if (npairs == 0 || nips == 0 || ngrads == 0) {
            return;
         }
         ensureLinearElasticStiffness3ElementCapacity (
            myNumLinearElasticStiffness3ElemContributions + nelems,
            myNumLinearElasticStiffness3ElemPairs + npairs,
            myNumLinearElasticStiffness3ElemIps + nips,
            myNumLinearElasticStiffness3ElemGradVecs + ngrads);

         int elemBase = myNumLinearElasticStiffness3ElemContributions;
         int pairBase = myNumLinearElasticStiffness3ElemPairs;
         int ipBase = myNumLinearElasticStiffness3ElemIps;
         int gradBase = myNumLinearElasticStiffness3ElemGradVecs;

         System.arraycopy (
            elemNodeCounts, 0, myLinearElasticStiffness3ElemNodeCounts,
            elemBase, nelems);
         System.arraycopy (
            elemParams, 0, myLinearElasticStiffness3ElemParams,
            2*elemBase, 2*nelems);
         for (int i=1; i<=nelems; i++) {
            myLinearElasticStiffness3ElemPairOffsets[elemBase+i] =
               pairBase + elemPairOffsets[i];
            myLinearElasticStiffness3ElemIpOffsets[elemBase+i] =
               ipBase + elemIpOffsets[i];
            myLinearElasticStiffness3ElemGradOffsets[elemBase+i] =
               gradBase + elemGradOffsets[i];
         }
         System.arraycopy (
            pairNodeIdxs, 0, myLinearElasticStiffness3PairNodeIdxs,
            2*pairBase, 2*npairs);
         System.arraycopy (
            blockSlots, 0, myLinearElasticStiffness3ElemBlockSlots,
            9*pairBase, 9*npairs);
         System.arraycopy (
            grads, 0, myLinearElasticStiffness3ElemGrads,
            3*gradBase, 3*ngrads);
         System.arraycopy (
            dvs, 0, myLinearElasticStiffness3ElemDvs, ipBase, nips);

         myNumLinearElasticStiffness3ElemContributions += nelems;
         myNumLinearElasticStiffness3ElemPairs += npairs;
         myNumLinearElasticStiffness3ElemIps += nips;
         myNumLinearElasticStiffness3ElemGradVecs += ngrads;
      }

      private void ensureLinearElasticStiffness3ElementCapacity (
         int nelems, int npairs, int nips, int ngrads) {

         if (myLinearElasticStiffness3ElemNodeCounts.length < nelems) {
            int newCap = Math.max (
               nelems,
               Math.max (
                  64, 2*myLinearElasticStiffness3ElemNodeCounts.length));
            myLinearElasticStiffness3ElemNodeCounts =
               Arrays.copyOf (
                  myLinearElasticStiffness3ElemNodeCounts, newCap);
            myLinearElasticStiffness3ElemPairOffsets =
               Arrays.copyOf (
                  myLinearElasticStiffness3ElemPairOffsets, newCap+1);
            myLinearElasticStiffness3ElemIpOffsets =
               Arrays.copyOf (
                  myLinearElasticStiffness3ElemIpOffsets, newCap+1);
            myLinearElasticStiffness3ElemGradOffsets =
               Arrays.copyOf (
                  myLinearElasticStiffness3ElemGradOffsets, newCap+1);
            myLinearElasticStiffness3ElemParams =
               Arrays.copyOf (
                  myLinearElasticStiffness3ElemParams, 2*newCap);
         }
         if (myLinearElasticStiffness3PairNodeIdxs.length < 2*npairs) {
            int newCap = Math.max (
               npairs,
               Math.max (64, myLinearElasticStiffness3PairNodeIdxs.length));
            while (newCap < npairs) {
               newCap *= 2;
            }
            myLinearElasticStiffness3PairNodeIdxs =
               Arrays.copyOf (
                  myLinearElasticStiffness3PairNodeIdxs, 2*newCap);
            myLinearElasticStiffness3ElemBlockSlots =
               Arrays.copyOf (
                  myLinearElasticStiffness3ElemBlockSlots, 9*newCap);
         }
         if (myLinearElasticStiffness3ElemDvs.length < nips) {
            int newCap = Math.max (
               nips,
               Math.max (64, 2*myLinearElasticStiffness3ElemDvs.length));
            myLinearElasticStiffness3ElemDvs =
               Arrays.copyOf (myLinearElasticStiffness3ElemDvs, newCap);
         }
         if (myLinearElasticStiffness3ElemGrads.length < 3*ngrads) {
            int newCap = Math.max (
               ngrads,
               Math.max (64, myLinearElasticStiffness3ElemGrads.length));
            while (newCap < ngrads) {
               newCap *= 2;
            }
            myLinearElasticStiffness3ElemGrads =
               Arrays.copyOf (myLinearElasticStiffness3ElemGrads, 3*newCap);
         }
      }

      public void addLinearElasticStiffness3ElementGeometryCrsValueContributions (
         int[] elemNodeCounts, int[] elemNodeOffsets, int[] elemPairOffsets,
         int[] elemIpOffsets, int[] elemNaturalGradOffsets,
         int[] pairNodeIdxs, int[] blockSlots, int[] nodeDims,
         double[] nodeTransforms, double[] elemParams,
         double[] elemNodePositions, double[] naturalGrads,
         double[] ipWeights, int nelems) {

         if (nelems == 0) {
            return;
         }
         int nnodes = elemNodeOffsets[nelems];
         int npairs = elemPairOffsets[nelems];
         int nips = elemIpOffsets[nelems];
         int ngrads = elemNaturalGradOffsets[nelems];
         if (nnodes == 0 || npairs == 0 || nips == 0 || ngrads == 0) {
            return;
         }
         ensureLinearElasticStiffness3ElementGeometryCapacity (
            myNumLinearElasticGeometry3ElemContributions + nelems,
            myNumLinearElasticGeometry3ElemNodes + nnodes,
            myNumLinearElasticGeometry3ElemPairs + npairs,
            myNumLinearElasticGeometry3ElemIps + nips,
            myNumLinearElasticGeometry3ElemNaturalGradVecs + ngrads);

         int elemBase = myNumLinearElasticGeometry3ElemContributions;
         int nodeBase = myNumLinearElasticGeometry3ElemNodes;
         int pairBase = myNumLinearElasticGeometry3ElemPairs;
         int ipBase = myNumLinearElasticGeometry3ElemIps;
         int gradBase = myNumLinearElasticGeometry3ElemNaturalGradVecs;

         System.arraycopy (
            elemNodeCounts, 0, myLinearElasticGeometry3ElemNodeCounts,
            elemBase, nelems);
         System.arraycopy (
            elemParams, 0, myLinearElasticGeometry3ElemParams,
            2*elemBase, 2*nelems);
         for (int i=1; i<=nelems; i++) {
            myLinearElasticGeometry3ElemNodeOffsets[elemBase+i] =
               nodeBase + elemNodeOffsets[i];
            myLinearElasticGeometry3ElemPairOffsets[elemBase+i] =
               pairBase + elemPairOffsets[i];
            myLinearElasticGeometry3ElemIpOffsets[elemBase+i] =
               ipBase + elemIpOffsets[i];
            myLinearElasticGeometry3ElemNaturalGradOffsets[elemBase+i] =
               gradBase + elemNaturalGradOffsets[i];
         }
         System.arraycopy (
            pairNodeIdxs, 0, myLinearElasticGeometry3PairNodeIdxs,
            2*pairBase, 2*npairs);
         System.arraycopy (
            blockSlots, 0, myLinearElasticGeometry3ElemBlockSlots,
            36*pairBase, 36*npairs);
         System.arraycopy (
            nodeDims, 0, myLinearElasticGeometry3NodeDims,
            nodeBase, nnodes);
         System.arraycopy (
            nodeTransforms, 0, myLinearElasticGeometry3NodeTransforms,
            18*nodeBase, 18*nnodes);
         System.arraycopy (
            elemNodePositions, 0, myLinearElasticGeometry3ElemNodePositions,
            3*nodeBase, 3*nnodes);
         System.arraycopy (
            naturalGrads, 0, myLinearElasticGeometry3NaturalGrads,
            3*gradBase, 3*ngrads);
         System.arraycopy (
            ipWeights, 0, myLinearElasticGeometry3IpWeights, ipBase, nips);

         myNumLinearElasticGeometry3ElemContributions += nelems;
         myNumLinearElasticGeometry3ElemNodes += nnodes;
         myNumLinearElasticGeometry3ElemPairs += npairs;
         myNumLinearElasticGeometry3ElemIps += nips;
         myNumLinearElasticGeometry3ElemNaturalGradVecs += ngrads;
      }

      private void ensureLinearElasticStiffness3ElementGeometryCapacity (
         int nelems, int nnodes, int npairs, int nips, int ngrads) {

         if (myLinearElasticGeometry3ElemNodeCounts.length < nelems) {
            int newCap = Math.max (
               nelems,
               Math.max (
                  64, 2*myLinearElasticGeometry3ElemNodeCounts.length));
            myLinearElasticGeometry3ElemNodeCounts =
               Arrays.copyOf (
                  myLinearElasticGeometry3ElemNodeCounts, newCap);
            myLinearElasticGeometry3ElemNodeOffsets =
               Arrays.copyOf (
                  myLinearElasticGeometry3ElemNodeOffsets, newCap+1);
            myLinearElasticGeometry3ElemPairOffsets =
               Arrays.copyOf (
                  myLinearElasticGeometry3ElemPairOffsets, newCap+1);
            myLinearElasticGeometry3ElemIpOffsets =
               Arrays.copyOf (
                  myLinearElasticGeometry3ElemIpOffsets, newCap+1);
            myLinearElasticGeometry3ElemNaturalGradOffsets =
               Arrays.copyOf (
                  myLinearElasticGeometry3ElemNaturalGradOffsets, newCap+1);
            myLinearElasticGeometry3ElemParams =
               Arrays.copyOf (
                  myLinearElasticGeometry3ElemParams, 2*newCap);
         }
         if (myLinearElasticGeometry3ElemNodePositions.length < 3*nnodes) {
            int newCap = Math.max (
               nnodes,
               Math.max (
                  64, myLinearElasticGeometry3ElemNodePositions.length));
            while (newCap < nnodes) {
               newCap *= 2;
            }
            myLinearElasticGeometry3ElemNodePositions =
               Arrays.copyOf (
                  myLinearElasticGeometry3ElemNodePositions, 3*newCap);
            myLinearElasticGeometry3NodeDims =
               Arrays.copyOf (
                  myLinearElasticGeometry3NodeDims, newCap);
            myLinearElasticGeometry3NodeTransforms =
               Arrays.copyOf (
                  myLinearElasticGeometry3NodeTransforms, 18*newCap);
         }
         if (myLinearElasticGeometry3PairNodeIdxs.length < 2*npairs) {
            int newCap = Math.max (
               npairs,
               Math.max (64, myLinearElasticGeometry3PairNodeIdxs.length));
            while (newCap < npairs) {
               newCap *= 2;
            }
            myLinearElasticGeometry3PairNodeIdxs =
               Arrays.copyOf (
                  myLinearElasticGeometry3PairNodeIdxs, 2*newCap);
            myLinearElasticGeometry3ElemBlockSlots =
               Arrays.copyOf (
                  myLinearElasticGeometry3ElemBlockSlots, 36*newCap);
         }
         if (myLinearElasticGeometry3IpWeights.length < nips) {
            int newCap = Math.max (
               nips,
               Math.max (64, 2*myLinearElasticGeometry3IpWeights.length));
            myLinearElasticGeometry3IpWeights =
               Arrays.copyOf (myLinearElasticGeometry3IpWeights, newCap);
         }
         if (myLinearElasticGeometry3NaturalGrads.length < 3*ngrads) {
            int newCap = Math.max (
               ngrads,
               Math.max (64, myLinearElasticGeometry3NaturalGrads.length));
            while (newCap < ngrads) {
               newCap *= 2;
            }
            myLinearElasticGeometry3NaturalGrads =
               Arrays.copyOf (myLinearElasticGeometry3NaturalGrads, 3*newCap);
         }
      }

      public void addDilationalStiffness3ElementCrsValueContributions (
         int[] elemNodeCounts, int[] elemPressureCounts,
         int[] elemPairOffsets, int[] elemConstraintOffsets,
         int[] elemRinvOffsets, int[] pairNodeIdxs, int[] blockSlots,
         double[] constraints, double[] rinvs, int nelems) {

         if (nelems == 0) {
            return;
         }
         int npairs = elemPairOffsets[nelems];
         int nconstraints = elemConstraintOffsets[nelems];
         int nrinvs = elemRinvOffsets[nelems];
         if (npairs == 0 || nconstraints == 0 || nrinvs == 0) {
            return;
         }
         ensureDilationalStiffness3ElementCapacity (
            myNumDilationalStiffness3ElemContributions + nelems,
            myNumDilationalStiffness3ElemPairs + npairs,
            myNumDilationalStiffness3ElemConstraints + nconstraints,
            myNumDilationalStiffness3ElemRinvs + nrinvs);

         int elemBase = myNumDilationalStiffness3ElemContributions;
         int pairBase = myNumDilationalStiffness3ElemPairs;
         int constraintBase = myNumDilationalStiffness3ElemConstraints;
         int rinvBase = myNumDilationalStiffness3ElemRinvs;

         System.arraycopy (
            elemNodeCounts, 0, myDilationalStiffness3ElemNodeCounts,
            elemBase, nelems);
         System.arraycopy (
            elemPressureCounts, 0, myDilationalStiffness3ElemPressureCounts,
            elemBase, nelems);
         for (int i=1; i<=nelems; i++) {
            myDilationalStiffness3ElemPairOffsets[elemBase+i] =
               pairBase + elemPairOffsets[i];
            myDilationalStiffness3ElemConstraintOffsets[elemBase+i] =
               constraintBase + elemConstraintOffsets[i];
            myDilationalStiffness3ElemRinvOffsets[elemBase+i] =
               rinvBase + elemRinvOffsets[i];
         }
         System.arraycopy (
            pairNodeIdxs, 0, myDilationalStiffness3PairNodeIdxs,
            2*pairBase, 2*npairs);
         System.arraycopy (
            blockSlots, 0, myDilationalStiffness3ElemBlockSlots,
            9*pairBase, 9*npairs);
         System.arraycopy (
            constraints, 0, myDilationalStiffness3ElemConstraints,
            3*constraintBase, 3*nconstraints);
         System.arraycopy (
            rinvs, 0, myDilationalStiffness3ElemRinvs, rinvBase, nrinvs);

         myNumDilationalStiffness3ElemContributions += nelems;
         myNumDilationalStiffness3ElemPairs += npairs;
         myNumDilationalStiffness3ElemConstraints += nconstraints;
         myNumDilationalStiffness3ElemRinvs += nrinvs;
      }

      private void ensureDilationalStiffness3ElementCapacity (
         int nelems, int npairs, int nconstraints, int nrinvs) {

         if (myDilationalStiffness3ElemNodeCounts.length < nelems) {
            int newCap = Math.max (
               nelems,
               Math.max (64, 2*myDilationalStiffness3ElemNodeCounts.length));
            myDilationalStiffness3ElemNodeCounts =
               Arrays.copyOf (myDilationalStiffness3ElemNodeCounts, newCap);
            myDilationalStiffness3ElemPressureCounts =
               Arrays.copyOf (
                  myDilationalStiffness3ElemPressureCounts, newCap);
            myDilationalStiffness3ElemPairOffsets =
               Arrays.copyOf (
                  myDilationalStiffness3ElemPairOffsets, newCap+1);
            myDilationalStiffness3ElemConstraintOffsets =
               Arrays.copyOf (
                  myDilationalStiffness3ElemConstraintOffsets, newCap+1);
            myDilationalStiffness3ElemRinvOffsets =
               Arrays.copyOf (
                  myDilationalStiffness3ElemRinvOffsets, newCap+1);
         }
         if (myDilationalStiffness3PairNodeIdxs.length < 2*npairs) {
            int newCap = Math.max (
               npairs, Math.max (64, myDilationalStiffness3PairNodeIdxs.length));
            while (newCap < npairs) {
               newCap *= 2;
            }
            myDilationalStiffness3PairNodeIdxs =
               Arrays.copyOf (myDilationalStiffness3PairNodeIdxs, 2*newCap);
            myDilationalStiffness3ElemBlockSlots =
               Arrays.copyOf (
                  myDilationalStiffness3ElemBlockSlots, 9*newCap);
         }
         if (myDilationalStiffness3ElemConstraints.length <
             3*nconstraints) {
            int newCap = Math.max (
               nconstraints,
               Math.max (64, myDilationalStiffness3ElemConstraints.length));
            while (newCap < nconstraints) {
               newCap *= 2;
            }
            myDilationalStiffness3ElemConstraints =
               Arrays.copyOf (
                  myDilationalStiffness3ElemConstraints, 3*newCap);
         }
         if (myDilationalStiffness3ElemRinvs.length < nrinvs) {
            int newCap = Math.max (
               nrinvs,
               Math.max (64, 2*myDilationalStiffness3ElemRinvs.length));
            myDilationalStiffness3ElemRinvs =
               Arrays.copyOf (myDilationalStiffness3ElemRinvs, newCap);
         }
      }

      public int numCrsValueContributions() {
         return myNumCrsValueContributions;
      }

      public int[] getCrsValueContributionSlots() {
         return myCrsValueSlots;
      }

      public double[] getCrsValueContributions() {
         return myCrsValueContributions;
      }

      public int numScaledDiagonal3Contributions() {
         return myNumScaledDiagonal3Contributions;
      }

      public int[] getScaledDiagonal3ContributionSlots() {
         return myScaledDiagonal3Slots;
      }

      public double[] getScaledDiagonal3Contributions() {
         return myScaledDiagonal3Values;
      }

      public int numScaledBlock3Contributions() {
         return myNumScaledBlock3Contributions;
      }

      public int[] getScaledBlock3ContributionSlots() {
         return myScaledBlock3Slots;
      }

      public double[] getScaledBlock3Contributions() {
         return myScaledBlock3Values;
      }

      public double[] getScaledBlock3ContributionScales() {
         return myScaledBlock3Scales;
      }

      public int numMaterialStiffness3Contributions() {
         return myNumMaterialStiffness3Contributions;
      }

      public int numMaterialStiffness3ElementContributions() {
         return myNumMaterialStiffness3ElemContributions;
      }

      public int numMaterialStiffness3ElementPairs() {
         return myNumMaterialStiffness3ElemPairs;
      }

      public int numMaterialStiffness3ElementIps() {
         return myNumMaterialStiffness3ElemIps;
      }

      public int[] getMaterialStiffness3ElementNodeCounts() {
         return myMaterialStiffness3ElemNodeCounts;
      }

      public int[] getMaterialStiffness3ElementPairOffsets() {
         return myMaterialStiffness3ElemPairOffsets;
      }

      public int[] getMaterialStiffness3ElementIpOffsets() {
         return myMaterialStiffness3ElemIpOffsets;
      }

      public int[] getMaterialStiffness3ElementGradOffsets() {
         return myMaterialStiffness3ElemGradOffsets;
      }

      public int[] getMaterialStiffness3ElementPairNodeIdxs() {
         return myMaterialStiffness3PairNodeIdxs;
      }

      public int[] getMaterialStiffness3ElementBlockSlots() {
         return myMaterialStiffness3ElemBlockSlots;
      }

      public double[] getMaterialStiffness3ElementGrads() {
         return myMaterialStiffness3ElemGrads;
      }

      public double[] getMaterialStiffness3ElementDs() {
         return myMaterialStiffness3ElemDs;
      }

      public double[] getMaterialStiffness3ElementSigmas() {
         return myMaterialStiffness3ElemSigmas;
      }

      public double[] getMaterialStiffness3ElementDvs() {
         return myMaterialStiffness3ElemDvs;
      }

      public int numLinearElasticStiffness3ElementContributions() {
         return myNumLinearElasticStiffness3ElemContributions;
      }

      public int numLinearElasticStiffness3ElementPairs() {
         return myNumLinearElasticStiffness3ElemPairs;
      }

      public int numLinearElasticStiffness3ElementIps() {
         return myNumLinearElasticStiffness3ElemIps;
      }

      public int[] getLinearElasticStiffness3ElementNodeCounts() {
         return myLinearElasticStiffness3ElemNodeCounts;
      }

      public int[] getLinearElasticStiffness3ElementPairOffsets() {
         return myLinearElasticStiffness3ElemPairOffsets;
      }

      public int[] getLinearElasticStiffness3ElementIpOffsets() {
         return myLinearElasticStiffness3ElemIpOffsets;
      }

      public int[] getLinearElasticStiffness3ElementGradOffsets() {
         return myLinearElasticStiffness3ElemGradOffsets;
      }

      public int[] getLinearElasticStiffness3ElementPairNodeIdxs() {
         return myLinearElasticStiffness3PairNodeIdxs;
      }

      public int[] getLinearElasticStiffness3ElementBlockSlots() {
         return myLinearElasticStiffness3ElemBlockSlots;
      }

      public double[] getLinearElasticStiffness3ElementParams() {
         return myLinearElasticStiffness3ElemParams;
      }

      public double[] getLinearElasticStiffness3ElementGrads() {
         return myLinearElasticStiffness3ElemGrads;
      }

      public double[] getLinearElasticStiffness3ElementDvs() {
         return myLinearElasticStiffness3ElemDvs;
      }

      public int numLinearElasticStiffness3ElementGeometryContributions() {
         return myNumLinearElasticGeometry3ElemContributions;
      }

      public int numLinearElasticStiffness3ElementGeometryPairs() {
         return myNumLinearElasticGeometry3ElemPairs;
      }

      public int numLinearElasticStiffness3ElementGeometryIps() {
         return myNumLinearElasticGeometry3ElemIps;
      }

      public int[] getLinearElasticStiffness3ElementGeometryNodeCounts() {
         return myLinearElasticGeometry3ElemNodeCounts;
      }

      public int[] getLinearElasticStiffness3ElementGeometryNodeOffsets() {
         return myLinearElasticGeometry3ElemNodeOffsets;
      }

      public int[] getLinearElasticStiffness3ElementGeometryPairOffsets() {
         return myLinearElasticGeometry3ElemPairOffsets;
      }

      public int[] getLinearElasticStiffness3ElementGeometryIpOffsets() {
         return myLinearElasticGeometry3ElemIpOffsets;
      }

      public int[] getLinearElasticStiffness3ElementGeometryNaturalGradOffsets() {
         return myLinearElasticGeometry3ElemNaturalGradOffsets;
      }

      public int[] getLinearElasticStiffness3ElementGeometryPairNodeIdxs() {
         return myLinearElasticGeometry3PairNodeIdxs;
      }

      public int[] getLinearElasticStiffness3ElementGeometryBlockSlots() {
         return myLinearElasticGeometry3ElemBlockSlots;
      }

      public int[] getLinearElasticStiffness3ElementGeometryNodeDims() {
         return myLinearElasticGeometry3NodeDims;
      }

      public double[] getLinearElasticStiffness3ElementGeometryNodeTransforms() {
         return myLinearElasticGeometry3NodeTransforms;
      }

      public double[] getLinearElasticStiffness3ElementGeometryParams() {
         return myLinearElasticGeometry3ElemParams;
      }

      public double[] getLinearElasticStiffness3ElementGeometryNodePositions() {
         return myLinearElasticGeometry3ElemNodePositions;
      }

      public double[] getLinearElasticStiffness3ElementGeometryNaturalGrads() {
         return myLinearElasticGeometry3NaturalGrads;
      }

      public double[] getLinearElasticStiffness3ElementGeometryIpWeights() {
         return myLinearElasticGeometry3IpWeights;
      }

      public int numDilationalStiffness3ElementContributions() {
         return myNumDilationalStiffness3ElemContributions;
      }

      public int numDilationalStiffness3ElementPairs() {
         return myNumDilationalStiffness3ElemPairs;
      }

      public int numDilationalStiffness3ElementConstraints() {
         return myNumDilationalStiffness3ElemConstraints;
      }

      public int[] getDilationalStiffness3ElementNodeCounts() {
         return myDilationalStiffness3ElemNodeCounts;
      }

      public int[] getDilationalStiffness3ElementPressureCounts() {
         return myDilationalStiffness3ElemPressureCounts;
      }

      public int[] getDilationalStiffness3ElementPairOffsets() {
         return myDilationalStiffness3ElemPairOffsets;
      }

      public int[] getDilationalStiffness3ElementConstraintOffsets() {
         return myDilationalStiffness3ElemConstraintOffsets;
      }

      public int[] getDilationalStiffness3ElementRinvOffsets() {
         return myDilationalStiffness3ElemRinvOffsets;
      }

      public int[] getDilationalStiffness3ElementPairNodeIdxs() {
         return myDilationalStiffness3PairNodeIdxs;
      }

      public int[] getDilationalStiffness3ElementBlockSlots() {
         return myDilationalStiffness3ElemBlockSlots;
      }

      public double[] getDilationalStiffness3ElementConstraints() {
         return myDilationalStiffness3ElemConstraints;
      }

      public double[] getDilationalStiffness3ElementRinvs() {
         return myDilationalStiffness3ElemRinvs;
      }

      public int[] getMaterialStiffness3ContributionSlots() {
         return myMaterialStiffness3Slots;
      }

      public double[] getMaterialStiffness3Gis() {
         return myMaterialStiffness3Gis;
      }

      public double[] getMaterialStiffness3Gjs() {
         return myMaterialStiffness3Gjs;
      }

      public double[] getMaterialStiffness3Ds() {
         return myMaterialStiffness3Ds;
      }

      public double[] getMaterialStiffness3Sigmas() {
         return myMaterialStiffness3Sigmas;
      }

      public double[] getMaterialStiffness3Dvs() {
         return myMaterialStiffness3Dvs;
      }

      public boolean hasCpuGeneratedMatrixContributions() {
         return (myNumCrsValueContributions > 0 ||
                 myNumScaledDiagonal3Contributions > 0 ||
                 myNumScaledBlock3Contributions > 0 ||
                 myNumMaterialStiffness3Contributions > 0 ||
                 myNumMaterialStiffness3ElemContributions > 0 ||
                 myNumLinearElasticStiffness3ElemContributions > 0 ||
                 myNumLinearElasticGeometry3ElemContributions > 0 ||
                 myNumDilationalStiffness3ElemContributions > 0);
      }

      public boolean hasHostAssembledCrsValueContributions() {
         return myNumCrsValueContributions > 0;
      }

      public boolean hasHostAssembledBlock3Contributions() {
         return myNumScaledBlock3Contributions > 0;
      }

      public boolean hasHostEvaluatedMaterialDescriptors() {
         return (myNumMaterialStiffness3Contributions > 0 ||
                 myNumMaterialStiffness3ElemContributions > 0 ||
                 myNumDilationalStiffness3ElemContributions > 0);
      }

      public String getContributionSourceSummary() {
         if (hasHostAssembledCrsValueContributions() ||
             hasHostAssembledBlock3Contributions()) {
            return "hostMatrixValuesToDeviceCrs";
         }
         else if (hasHostEvaluatedMaterialDescriptors()) {
            return "hostMaterialDescriptorsToGpuKernels";
         }
         else if (myNumLinearElasticStiffness3ElemContributions > 0) {
            return "hostGeometryDescriptorsToGpuKernels";
         }
         else if (myNumLinearElasticGeometry3ElemContributions > 0) {
            return "hostReferenceGeometryToGpuKernels";
         }
         else if (myNumScaledDiagonal3Contributions > 0) {
            return "hostScalarDescriptorsToGpuKernels";
         }
         else {
            return "deviceGeneratedValues";
         }
      }

      public String getContributionSummary() {
         return String.format (
            "generic=%d diag3=%d block3=%d material3=%d "+
            "materialElem3=%d/%d/%d linearElem3=%d/%d/%d "+
            "linearGeomElem3=%d/%d/%d dilationElem3=%d/%d/%d",
            myNumCrsValueContributions,
            myNumScaledDiagonal3Contributions,
            myNumScaledBlock3Contributions,
            myNumMaterialStiffness3Contributions,
            myNumMaterialStiffness3ElemContributions,
            myNumMaterialStiffness3ElemIps,
            myNumMaterialStiffness3ElemPairs,
            myNumLinearElasticStiffness3ElemContributions,
            myNumLinearElasticStiffness3ElemIps,
            myNumLinearElasticStiffness3ElemPairs,
            myNumLinearElasticGeometry3ElemContributions,
            myNumLinearElasticGeometry3ElemIps,
            myNumLinearElasticGeometry3ElemPairs,
            myNumDilationalStiffness3ElemContributions,
            myNumDilationalStiffness3ElemConstraints,
            myNumDilationalStiffness3ElemPairs);
      }

      /**
       * Returns 1-based CRS column indices, matching the public maspack CRS
       * export convention.
       */
      public int[] getCrsColIdxs() {
         return myCrsColIdxs;
      }

      /**
       * Returns 1-based CRS row offsets, matching the public maspack CRS export
       * convention.
       */
      public int[] getCrsRowOffs() {
         return myCrsRowOffs;
      }

      public int[] getZeroBasedCrsColIdxs() {
         if (myZeroBasedCrsColIdxs == null) {
            myZeroBasedCrsColIdxs = myCrsColIdxs.clone();
            for (int i=0; i<myZeroBasedCrsColIdxs.length; i++) {
               myZeroBasedCrsColIdxs[i]--;
            }
         }
         return myZeroBasedCrsColIdxs;
      }

      public int[] getZeroBasedCrsRowOffs() {
         if (myZeroBasedCrsRowOffs == null) {
            myZeroBasedCrsRowOffs = myCrsRowOffs.clone();
            for (int i=0; i<myZeroBasedCrsRowOffs.length; i++) {
               myZeroBasedCrsRowOffs[i]--;
            }
         }
         return myZeroBasedCrsRowOffs;
      }

      public int getStructureVersion() {
         return myStructureVersion;
      }
   }

   /** 
    * Flag passed to {@link #updateConstraints updateConstraints()}
    * indicating that contact information should be computed.
    */
   public static final int COMPUTE_CONTACTS = 0x01;

   /** 
    * Flag passed to {@link #updateConstraints updateConstraints()} indicating
    * that contact information should be updated. This means that there has
    * only been an adjustment in position and existing contacts between bodies
    * should be maintained, though possibly with a modified set of constraints.
    */
   public static final int UPDATE_CONTACTS = 0x02;

   /**
    * Contains information for a single constraint direction
    */
   public class ConstraintInfo {

      // Note: distance to the constraint surface should be signed in such a
      // way that solving the constraint equation G dx = -dist will produce
      // an impulse that moves the system back towards the constraint surface.
      public double dist;      // distance to the constraint surface.
      public double compliance;// inverse stiffness; 0 implies rigid constraint
      public double damping;   // damping; only used if compliance > 0
      public double force;     // used for computing non-linear compliance
      public boolean coordLimit; // is limit constraint for a joint coordinate
      
      // motionType: for some constraint types, indicates the type
      // of motion associated with this constraint; otherwise is null
      public MotionType motionType;
   };

   /**
    * Returns the current structure version of this system. The structure
    * version should be incremented whenever the number and arrangement
    * of active and parametric components changes, implying that
    * any mass and solve matrices should be be rebuilt.
    * 
    * @return current structure version number for this system
    */
   public int getStructureVersion();

   /**
    * Returns the size of the position state for all active components.
    * This should remain unchanged as long as the current structure
    * version (returned by {@link #getStructureVersion getStructureVersion()} 
    * remains unchanged.
    * 
    * @return size of the active position state
    */
   public int getActivePosStateSize();

   /**
    * Gets the current position state for all active components. This is stored
    * in the vector <code>q</code> (which will be sized appropriately by the
    * system).
    * 
    * @param q 
    * vector in which the state is stored
    */
   public void getActivePosState (VectorNd q);

   /**
    * Sets the current position state for all active components. This is
    * supplied in the vector <code>q</code>, whose size should be greater than
    * or equal to the value returned by {@link #getActivePosStateSize
    * getActivePosStateSize()}.
    * 
    * @param q
    * vector supplying the state information
    */
   public void setActivePosState (VectorNd q);

   /**
    * Returns the size of the velocity state for all active components.
    * This should remain unchanged as long as the current structure
    * version (returned by {@link #getStructureVersion getStructureVersion()} 
    * remains unchanged.
    * 
    * @return size of the velocity state
    */
   public int getActiveVelStateSize();

   /**
    * Gets the current velocity state for all active components. This is stored
    * in the vector <code>u</code> (which will be sized appropriately by the
    * system).
    * 
    * @param u
    * vector in which the state is stored
    */
   public void getActiveVelState (VectorNd u);

   /**
    * Sets the current velocity state for all active components. This is
    * supplied in the vector <code>u</code>, whose size should be greater than
    * or equal to the value returned by {@link #getActiveVelStateSize
    * getActiveVelStateSize()}.
    * 
    * @param u
    * vector supplying the state information
    */
   public void setActiveVelState (VectorNd u);

   /**
    * Returns the size of the position state for all parametric components.
    * This should remain unchanged as long as the current structure
    * version (returned by {@link #getStructureVersion getStructureVersion()} 
    * remains unchanged.
    * 
    * @return size of the parametric position state
    */
   public int getParametricPosStateSize();

   /**
    * Obtains the desired target position for all parametric components at a
    * particular time.  This is stored in the vector <code>q</code> (which will
    * be sized appropriately by the system).
    *
    * The system interpolates between the current parametric position values
    * and those desired for the end of the time step, using a parameter
    * <code>s</code> defined on the interval [0,1].  In particular, specifying
    * s = 0 yields the current parametric positions, and s = 1 yields the
    * positions desired for the end of the step. The actual time step size
    * <code>h</code> is also provided, as this may be needed by some
    * interpolation methods.
    *
    * <p>The interpolation uses the current parametric positions, and possibly
    * the current parametric velocities as well.  Hence this method should be
    * called before either of these are changed using {@link
    * #setParametricPosState setParametricPosState()} or {@link
    * #setParametricVelState setParametricVelState()}.
    * 
    * @param q
    * vector returning the state information
    * @param s
    * specifies time relative to the current time step
    * @param h
    * time step size
    */
   public void getParametricPosTarget (VectorNd q, double s, double h);

   /**
    * Gets the current position state for all parametric components. This is
    * stored in the vector <code>q</code> (which will be sized appropriately by
    * the system).
    * 
    * @param q
    * vector in which the state is stored
    */
   public void getParametricPosState (VectorNd q);

   /**
    * Sets the current position state for all parametric components. This is
    * supplied in the vector <code>q</code>, whose size should be greater than
    * or equal to the value returned by {@link #getParametricPosStateSize
    * getParametricPosStateSize()}.
    * 
    * @param q
    * vector supplying the state information
    */
   public void setParametricPosState (VectorNd q);

   /**
    * Returns the size of the velocity state for all parametric components.
    * This should remain unchanged as long as the current structure
    * version (returned by {@link #getStructureVersion getStructureVersion()} 
    * remains unchanged.
    * 
    * @return size of the parametric velocity state
    */
   public int getParametricVelStateSize();

   /**
    * Obtains the desired target velocity for all parametric components at a
    * particular time.  This is stored in the vector <code>u</code> (which will
    * be sized appropriately by the system).
    *
    * The system interpolates between the current parametric velocity values
    * and those desired for the end of the time step, using a parameter
    * <code>s</code> defined on the interval [0,1].  In particular, specifying
    * s = 0 yields the current parametric velocities, and s = 1 yields the
    * velocities desired for the end of the step. The actual time step size
    * <code>h</code> is also provided, as this may be needed by some
    * interpolation methods.
    *
    * <p>The interpolation uses the current parametric velocities, and possibly
    * the current parametric positions as well.  Hence this method should be
    * called before either of these are changed using {@link
    * #setParametricPosState setParametricPosState()} or {@link
    * #setParametricVelState setParametricVelState()}.
    * 
    * @param u
    * vector returning the parametric target velocites
    * @param s
    * specifies time relative to the current time step
    * @param h
    * time step size
    */
   public void getParametricVelTarget (VectorNd u, double s, double h);

   /**
    * Gets the current velocity state for all parametric components. This is
    * stored in the vector <code>u</code> (which will be sized appropriately by
    * the system).
    * 
    * @param u
    * vector in which state is stored
    */
   public void getParametricVelState (VectorNd u);

   /**
    * Sets the current velocity state for all parametric components. This is
    * supplied in the vector <code>u</code>, whose size should be greater than
    * or equal to the value returned by {@link #getParametricVelStateSize
    * getParametricVelStateSize()}.
    * 
    * @param u
    * vector supplying the state information
    */
   public void setParametricVelState (VectorNd u);

   /**
    * Sets the forces associated with parametric components. This is supplied
    * in the vector <code>f</code>, whose size should be greater or equal to
    * the value returned by {@link #getParametricVelStateSize 
    * getParametricVelStateSize()}.
    * 
    * @param f
    * vector supplying the force information
    */
   public void setParametricForces (VectorNd f);

   /**
    * Gets the forces associated with parametric components. This is returned
    * in the vector <code>f</code> (which will be sized appropriately by the
    * system).
    * 
    * @param f
    * vector in which to return the force information
    */
   public void getParametricForces (VectorNd f);

   /**
    * Gets the current value of the position derivative for all active
    * components. This is stored in the vector <code>dxdt</code> (which will be
    * sized appropriately by the system).
    * 
    * @param dxdt
    * vector in which the derivative is stored
    * @param t
    * current time value
    */
   public void getActivePosDerivative (VectorNd dxdt, double t);

   /**
    * Returns the generalized forces acting on all the active components in this
    * system. This is stored in the vector <code>f</code> (which will be
    * sized appropriately by the system).
    * 
    * @param f
    * vector in which the forces are stored
    */
   public void getActiveForces (VectorNd f);

   /**
    * Sets the generalized forces acting on all the active components in this
    * system. The values are specifies in the vector <code>f</code>, whose size 
    * should be greater or equal to the value returned by 
    * {@link #getActiveVelStateSize getActiveVelStateSize()}.
    * 
    * @param f
    * vector specifying the forces to be set
    */
   public void setActiveForces (VectorNd f);

   /** 
    * Builds a mass matrix for this system. This is done by adding blocks of an
    * appropriate type to the sparse block matrix <code>M</code>. On input,
    * <code>M</code> should be empty with zero size; it will be sized
    * appropriately by the system.
    *
    * <p>This method returns <code>true</code> if the mass matrix is constant;
    * i.e., does not vary with time. It does not place actual values in the
    * matrix; that must be done by calling {@link #getMassMatrix 
    * getMassMatrix()}.
    *
    * <p>A new mass matrix should be built whenever the system's structure
    * version (as returned by {@link #getStructureVersion 
    * getStructureVersion()}) changes.
    * 
    * @param M matrix in which the mass matrix will be built
    * @return true if the mass matrix is constant.
    */
   public boolean buildMassMatrix (SparseNumberedBlockMatrix M);

   /** 
    * Sets <code>M</code> to the current value of the mass matrix for this
    * system, evaluated at time <code>t</code>. <code>M</code> should
    * have been previously created with a call to
    * {@link #buildMassMatrix buildMassMatrix()}. The current mass forces
    * are returned in the vector <code>f</code>, which should have a
    * size greater or equal to the size of <code>M</code>.
    * The mass forces (also known as the fictitious forces)
    * are given by
    * <pre>
    * - dM/dt u
    * </pre>
    * where <code>u</code> is the current system velocity.
    * <code>f</code> will be sized appropriately by the system.
    *
    * @param M returns the mass matrix
    * @param f returns the mass forces
    * @param t current time
    */
   public void getMassMatrix (
      SparseNumberedBlockMatrix M, VectorNd f, double t);

   /** 
    * Sets <code>Minv</code> to the inverse of the mass matrix <code>M</code>.
    * <code>Minv</code> should have been previously created with a call to
    * {@link #buildMassMatrix buildMassMatrix()}.
    *
    * <p> This method assumes that <code>M</code> is block diagonal and hence
    * <code>Minv</code> has the same block structure. Although it is possible
    * to compute <code>Minv</code> by simply inverting each block of
    * <code>M</code>, the special structure of each mass block may permit
    * the system to do this in a highly optimized way.
    * 
    * @param Minv returns the inverted mass matrix
    * @param M mass matrix to invert
    */   
   public void getInverseMassMatrix (SparseBlockMatrix Minv, SparseBlockMatrix M);

   public void mulInverseMass (SparseBlockMatrix M, VectorNd a, VectorNd f);

   /** 
    * Builds a solve matrix for this system. This is done by adding blocks of
    * an appropriate type to the sparse block matrix <code>S</code>. On input,
    * <code>S</code> should be empty with zero size; it will be sized
    * appropriately by the system.  The resulting matrix should have all the
    * blocks required to store any combination of the mass matrix, the
    * force-position Jacobian, and the force-velocity Jacobian.
    * 
    * This method does not place actual values in the matrix; that must be done
    * by adding a mass matrix to it, or by calling {@link #addVelJacobian 
    * addVelJacobian()} or
    * {@link #addPosJacobian addPosJacobian()}.
    *
    * <p>A new solve matrix should be built whenever the system's structure
    * version (as returned by {@link #getStructureVersion 
    * getStructureVersion()}) changes.
    * 
    * @param S matrix in which the solve matrix will be built
    */
   public void buildSolveMatrix (SparseNumberedBlockMatrix S);

   /**
    * Returns information about the solve matrix for this system. This consists
    * of an or-ed set of flags, including {@link 
    * maspack.matrix.Matrix#SYMMETRIC SYMMETRIC} or {@link 
    * maspack.matrix.Matrix#SYMMETRIC POSITIVE_DEFINITE},
    * which aid in determining the best way to solve the matrix.
    * 
    * @return type information for the solve matrix
    */
   public int getSolveMatrixType();

   /**
    * Returns the current number of active components in this system.
    * This should remain unchanged as long as the current structure
    * version (returned by {@link #getStructureVersion getStructureVersion()} 
    * remains unchanged.
    * 
    * @return number of active components
    */
   public int numActiveComponents();

   /**
    * Returns the current number of parametric components in this system.
    * This should remain unchanged as long as the current structure
    * version (returned by {@link #getStructureVersion getStructureVersion()} 
    * remains unchanged.
    * 
    * @return number of parametric components
    */
   public int numParametricComponents();

   /**
    * Adds the current force-velocity Jacobian, scaled by <code>h</code>, to
    * the matrix <code>S</code>, which should have been previously created with
    * a call to {@link #buildSolveMatrix buildSolveMatrix()}. 
    * Addition fictitious forces associated
    * with the Jacobian can be optionally returned in the vector
    * <code>f</code>, which will be sized appropriately by the system.
    * 
    * @param S
    * matrix to which scaled Jacobian is to be added
    * @param f
    * if non-null, returns fictitious forces associated with the Jacobian
    * @param h
    * scale factor for the Jacobian
    */
   public void addVelJacobian (SparseNumberedBlockMatrix S, VectorNd f, double h);

   /**
    * Attempts to add the current force-velocity Jacobian using a GPU assembly
    * implementation. Implementations should return {@code true} only if they
    * handled the entire requested contribution and updated {@code f} in the
    * same way as {@link #addVelJacobian addVelJacobian()} when {@code f} is
    * non-null. The default implementation reports that GPU assembly is not
    * available, leaving callers to use the CPU path.
    *
    * @param S
    * matrix to which scaled Jacobian is to be added
    * @param f
    * if non-null, returns fictitious forces associated with the Jacobian
    * @param h
    * scale factor for the Jacobian
    * @return {@code true} if GPU assembly handled the contribution
    */
   public default boolean addGpuVelJacobian (
      SparseNumberedBlockMatrix S, VectorNd f, double h) {
      return false;
   }

   /**
    * Attempts to add the current force-velocity Jacobian using a GPU assembly
    * implementation with access to CRS slot metadata. The default method
    * delegates to the legacy GPU hook, allowing existing implementations to
    * remain source-compatible.
    *
    * @param context GPU assembly context for the current solve matrix
    * @param f if non-null, returns fictitious Jacobian forces
    * @param h scale factor for the Jacobian
    * @return {@code true} if GPU assembly handled the contribution
    */
   public default boolean addGpuVelJacobian (
      GpuAssemblyContext context, VectorNd f, double h) {
      return addGpuVelJacobian (context.getMatrix(), f, h);
   }

   /**
    * Optionally assembles the scaled force-velocity Jacobian directly into the
    * CRS values contained in {@code context}. This is intended for validation
    * and future GPU/direct-CRS solve paths; the default implementation reports
    * that direct CRS assembly is not available.
    *
    * @param context GPU assembly context containing CRS values and slot map
    * @param h scale factor for the Jacobian
    * @return {@code true} if the full system contribution was assembled
    */
   public default boolean assembleGpuVelJacobianCrsValues (
      GpuAssemblyContext context, double h) {
      return false;
   }

   /**
    * Optionally assembles the scaled force-velocity Jacobian directly into
    * CRS values while also returning fictitious Jacobian forces.
    *
    * @param context GPU assembly context containing CRS values and slot map
    * @param f if non-null, returns fictitious Jacobian forces
    * @param h scale factor for the Jacobian
    * @return {@code true} if the full system contribution was assembled
    */
   public default boolean assembleGpuVelJacobianCrsValues (
      GpuAssemblyContext context, VectorNd f, double h) {
      if (f != null) {
         f.setSize (context.getSlotMap().rowSize());
         f.setZero();
      }
      return assembleGpuVelJacobianCrsValues (context, h);
   }

   /**
    * Optionally assembles the scaled force-velocity Jacobian as CRS
    * slot/value contributions. Implementations should add contributions using
    * {@link GpuAssemblyContext#addCrsValueContribution}.
    */
   public default boolean assembleGpuVelJacobianCrsValueContributions (
      GpuAssemblyContext context, VectorNd f, double h) {
      return false;
   }

   /**
    * Adds the current force-position Jacobian, scaled by <code>h</code>, to
    * the matrix <code>S</code>, which should have been previously created with
    * a call to {@link #buildSolveMatrix buildSolveMatrix()}.  Addition
    * fictitious forces associated with the Jacobian can be optionally returned
    * in the vector <code>f</code>, which will be sized appropriately by the
    * system.
    * 
    * @param S
    * matrix to which scaled Jacobian is to be added
    * @param f
    * if non-null, returns fictitious forces associated with the Jacobian
    * @param h
    * scale factor for the Jacobian
    */
   public void addPosJacobian (SparseNumberedBlockMatrix S, VectorNd f, double h);

   /**
    * Attempts to add the current force-position Jacobian using a GPU assembly
    * implementation. Implementations should return {@code true} only if they
    * handled the entire requested contribution and updated {@code f} in the
    * same way as {@link #addPosJacobian addPosJacobian()} when {@code f} is
    * non-null. The default implementation reports that GPU assembly is not
    * available, leaving callers to use the CPU path.
    *
    * @param S
    * matrix to which scaled Jacobian is to be added
    * @param f
    * if non-null, returns fictitious forces associated with the Jacobian
    * @param h
    * scale factor for the Jacobian
    * @return {@code true} if GPU assembly handled the contribution
    */
   public default boolean addGpuPosJacobian (
      SparseNumberedBlockMatrix S, VectorNd f, double h) {
      return false;
   }

   /**
    * Attempts to add the current force-position Jacobian using a GPU assembly
    * implementation with access to CRS slot metadata. The default method
    * delegates to the legacy GPU hook, allowing existing implementations to
    * remain source-compatible.
    *
    * @param context GPU assembly context for the current solve matrix
    * @param f if non-null, returns fictitious Jacobian forces
    * @param h scale factor for the Jacobian
    * @return {@code true} if GPU assembly handled the contribution
    */
   public default boolean addGpuPosJacobian (
      GpuAssemblyContext context, VectorNd f, double h) {
      return addGpuPosJacobian (context.getMatrix(), f, h);
   }

   /**
    * Optionally assembles the scaled force-position Jacobian directly into the
    * CRS values contained in {@code context}. This is intended for validation
    * and future GPU/direct-CRS solve paths; the default implementation reports
    * that direct CRS assembly is not available.
    *
    * @param context GPU assembly context containing CRS values and slot map
    * @param h scale factor for the Jacobian
    * @return {@code true} if the full system contribution was assembled
    */
   public default boolean assembleGpuPosJacobianCrsValues (
      GpuAssemblyContext context, double h) {
      return false;
   }

   /**
    * Optionally assembles the scaled force-position Jacobian directly into
    * CRS values while also returning fictitious Jacobian forces.
    *
    * @param context GPU assembly context containing CRS values and slot map
    * @param f if non-null, returns fictitious Jacobian forces
    * @param h scale factor for the Jacobian
    * @return {@code true} if the full system contribution was assembled
    */
   public default boolean assembleGpuPosJacobianCrsValues (
      GpuAssemblyContext context, VectorNd f, double h) {
      if (f != null) {
         f.setSize (context.getSlotMap().rowSize());
         f.setZero();
      }
      return assembleGpuPosJacobianCrsValues (context, h);
   }

   /**
    * Optionally assembles the scaled force-position Jacobian as CRS
    * slot/value contributions. Implementations should add contributions using
    * {@link GpuAssemblyContext#addCrsValueContribution}.
    */
   public default boolean assembleGpuPosJacobianCrsValueContributions (
      GpuAssemblyContext context, VectorNd f, double h) {
      return false;
   }

   /**
    * Queries whether or not the matrix structure of the bilateral constraints
    * returned by this system is constant for a given structure version.
    * 
    * @return {@code true} if bilateral constraints have a constant structure
    */
   public boolean isBilateralStructureConstant();   
   
   /** 
    * Obtains the transpose of the current bilateral constraint matrix G for
    * this system. This is built and stored in <code>GT</code>. On input,
    * <code>GT</code> should be empty with zero size; it will be sized
    * appropriately by the system. The derivative term is returned
    * in <code>dg</code>; this is given by
    * <pre>
    * dG/dt u
    * </pre>
    * where <code>u</code> is the current system velocity.
    * <code>dg</code> will also be sized appropriately by the system.
    *
    * @param GT returns the transpose of G
    * @param dg if non-null, returns the derivative term for G
    */
   public void getBilateralConstraints (SparseBlockMatrix GT, VectorNd dg);

   /** 
    * Obtains information for all the constraint directions returned
    * by the most recent call to {@link #getBilateralConstraints 
    * getBilateralConstraints()}.
    * The information is returned in the array <code>ginfo</code>,
    * which should contain preallocated
    * {@link artisynth.core.mechmodels.MechSystem.ConstraintInfo ConstraintInfo}
    * structures and should have a length greater or equal to
    * the column size of <code>GT</code> returned by
    * {@link #getBilateralConstraints getBilateralConstraints()}.
    *
    * @param ginfo
    * Array of
    * {@link artisynth.core.mechmodels.MechSystem.ConstraintInfo ConstraintInfo}
    * objects used to return the constraint information.
    */
   public void getBilateralInfo (ConstraintInfo[] ginfo);

   /** 
    * Supplies to the system the most recently computed bilateral constraint
    * forces. These are supplied in the form {@code lam*s}, where {@code lam}
    * is a vector of impulses and {@code s} is the inverse of the step size
    * used in the computation. <code>lam</code>, which should have a size
    * greater or equal to the column size of <code>GT</code> returned by {@link
    * #getBilateralConstraints getBilateralConstraints()}.
    * 
    * @param lam
    * When scaled by {@code s}, gives the bilateral constraint forces 
    * being supplied to the system.
    * @param s 
    * Scaling factor to be applied to the {@code lam}.
    */
   public void setBilateralForces (VectorNd lam, double s);

   /** 
    * Returns from the system the most recently computed bilateral constraint
    * forces.  These are stored in the vector <code>lam</code>, which should
    * have a size greater or equal to the column size of <code>GT</code>
    * returned by {@link #getBilateralConstraints getBilateralConstraints()}.  
    * For constraints which
    * where present in the previous solve step, the force values should equal
    * those which were set by the prevous call to {@link
    * #setBilateralForces setBilateralForces()}.  
    * Otherwise, values can be estimated from
    * previous force values (where appropriate), or supplied as 0.
    * 
    * @param lam
    * Bilateral constraint forces being returned from the system.
    */
   public void getBilateralForces (VectorNd lam);

   /** 
    * Obtains the transpose of the current unilateral constraint matrix N for this
    * system. This is built and stored in <code>NT</code>. On input,
    * <code>NT</code> should be empty with zero size; it will be sized
    * appropriately by the system. The derivative term is returned
    * in <code>dn</code>; this is given by
    * <pre>
    * dn/dt u
    * </pre>
    * where <code>u</code> is the current system velocity.
    * <code>dn</code> will also be sized appropriately by the system.
    *
    * @param NT returns the transpose of N
    * @param dn if non-null. returns the derivative term for N
    */
   public void getUnilateralConstraints (SparseBlockMatrix NT, VectorNd dn);

   /** 
    * Obtains information for all the constraint directions returned
    * by the most recent call to {@link #getUnilateralConstraints 
    * getUnilateralConstraints()}.
    * The information is returned in the array <code>ninfo</code>,
    * which should contain preallocated
    * {@link artisynth.core.mechmodels.MechSystem.ConstraintInfo ConstraintInfo}
    * structures and should have a length greater or equal to
    * the column size of <code>NT</code> returned by
    * {@link #getUnilateralConstraints getUnilateralConstraints()}.
    *
    * @param ninfo
    * Array of
    * {@link artisynth.core.mechmodels.MechSystem.ConstraintInfo ConstraintInfo}
    * objects used to return the constraint information.
    */
   public void getUnilateralInfo (ConstraintInfo[] ninfo);

   /** 
    * Supplies to the system the most recently computed unilateral constraint
    * forces. These are supplied in the form {@code the*s}, where {@code the}
    * is a vector of impulses and {@code s} is the inverse of the step size
    * used in the computation. <code>the</code> should have a size greater or
    * equal to the column size of <code>NT</code> returned by {@link
    * #getUnilateralConstraints getUnilateralConstraints()}.
    * 
    * @param the
    * When scaled by {@code s}, gives the unilateral constraint forces 
    * being supplied to the system.
    * @param s 
    * Scaling factor to be applied to the {@code the}.
    */
   public void setUnilateralForces (VectorNd the, double s);

   /** 
    * Returns from the system the most recently computed unilateral constraint
    * forces.  These are stored in the vector <code>the</code>, which should
    * have a size greater or equal to the column size of <code>NT</code>
    * returned by {@link #getUnilateralConstraints getUnilateralConstraints()}.
    * For constraints which where present in the previous solve step, the
    * force values should equal those which were set by the previous call to
    * {@link #setUnilateralForces setUnilateralForces()}.  Otherwise,
    * values can be estimated from previous force values (where appropriate),
    * or supplied as 0.
    * 
    * @param the
    * Unilateral constraint forces being returned from the system.
    */
   public void getUnilateralForces (VectorNd the);
   
   public int setUnilateralState (VectorNi state, int idx);
   
   public int getUnilateralState (VectorNi state, int idx);

   /** 
    * Returns that maximum number of friction constraint set that may be added by
    * the method {@link #getFrictionConstraints getFrictionConstraints()}.
    * This is used to size the <code>finfo</code> array supplied to that
    * method.
    *
    * @return maximum friction constraint sets
    */
   public int maxFrictionConstraintSets();

   /** 
    * Obtains the transpose of the current friction constraint matrix D for
    * this system. This is built and stored in <code>DT</code>. On input,
    * <code>DT</code> should be empty with zero size; it will be sized
    * appropriately by the system.
    *
    * <p>Each column block in <code>DT</code> describes a friction constraint
    * set associated with one unilateral or bilateral constraint. Information
    * about each friction constraint set is returned in the array
    * <code>finfo</code>, which should contain preallocated 
    * {@link maspack.spatialmotion.FrictionInfo FrictionInfo}
    * structures and should have a length greater or 
    * equal to the value returned by
    * {@link #maxFrictionConstraintSets FrictionConstraintSets()}.
    *
    * @param DT returns the transpose of D
    * @param finfo returns information for each friction constraint set
    * @param prune limit DT entries to those for which the corresponding
    * contact forces are {@code > 0}
    * @return number of friction entries
    */
   public int getFrictionConstraints (
      SparseBlockMatrix DT, ArrayList<FrictionInfo> finfo, boolean prune);

   /** 
    * Supplies to the system the most recently computed friction constraint
    * forces. These are supplied in the form {@code phi*s}, where {@code phi}
    * is a vector of impulses and {@code s} is the inverse of the step size
    * used in the computation. <code>phi</code> should have a size greater or
    * equal to the column size of <code>DT</code> returned by {@link
    * #getFrictionConstraints getFrictionConstraints()}.
    * 
    * @param phi When scaled by {@code s}, gives the friction constraint forces
    * being supplied to the system.
    * @param s 
    * Scaling factor to be applied to the {@code phi}.
    */
   public void setFrictionForces (VectorNd phi, double s);
   
   /** 
    * Returns from the system the most recently computed friction constraint
    * forces.  These are stored in the vector <code>phi</code>, which should
    * have a size greater or equal to the column size of <code>DT</code>
    * returned by {@link #getFrictionConstraints getFrictionConstraints()}.
    * For constraints which where present in the previous solve step, the force
    * values should equal those which were set by the previous call to {@link
    * #setFrictionForces setFrictionForces()}.  Otherwise, values can be
    * estimated from previous force values (where appropriate), or supplied as
    * 0.
    * 
    * @param phi
    * Unilateral constraint forces being returned from the system.
    */
   public void getFrictionForces (VectorNd phi);
   
   public int setFrictionState (VectorNi state, int idx);

   public int getFrictionState (VectorNi state, int idx);

   /**
    * Computes an adjustment to the active positions of the system (stored
    * in the vector <code>q</code>) by applying
    * a velocity <code>u</code> for time
    * <code>h</code>.  Where velocity equals position derivative,
    * this corresponds to computing
    * <pre>
    * q += h u.
    * </pre>
    * In other situations, such as where position is orientation expressed as a
    * quaternion and velocity is angular velocity, the system should perform
    * the analagous computation.
    *
    * <code>q</code> should have a size greater or equal to the value
    * returned by {@link #getActivePosStateSize getActivePosStateSize()}, and
    * <code>u</code> should have a size greater or equal to the value
    * returned by {@link #getActiveVelStateSize getActiveVelStateSize()}.
    * 
    * @param q positions to be adjusted
    * @param h length of time to apply velocity
    * @param u velocity to be applied
    */
   public void addActivePosImpulse (VectorNd q, double h, VectorNd u);

   /** 
    * Updates the constraints associated with this system to be consistent with
    * the current position and indicated time. This method should be called
    * between any change to position values and any call that obtains
    * constraint information (such as {@link #getBilateralConstraints
    * getBilateralConstraints()}.
    *
    * <p>Because contact computations are expensive, the constraints associated
    * with contact should only be computed if the flags {@link
    * #COMPUTE_CONTACTS} or {@link #UPDATE_CONTACTS} are specified. The former
    * calls for contact information to be computed from scratch, while the
    * latter calls for contact information to be modified to reflect changes in
    * body positions, while preserving the general contact state between
    * bodies.
    *
    * <p>In the process of updating the constraints, the system may determine
    * that a smaller step size is needed.  This is particulary true when
    * contact calculations show an unacceptable level of interpenetration.
    * A smaller step size can be recommended If so, it can indicate this through
    * the optional argument <code>stepAdjust</code>, if present.
    * 
    * @param t current time
    * @param stepAdjust
    * (optional) can be used to indicate whether the current advance
    * should be redone with a smaller step size.
    * @param flags information flags
    * @return true if the system contains constraints
    */
   public boolean updateConstraints (
      double t, StepAdjustment stepAdjust, int flags);
   
   /**
    * Updates all internal forces associated with this system to be
    * consistent with the current position and velocity and the indicated time
    * <code>t</code>. This method should be called between any change to
    * position or velocity values (such as through
    * {@link #setActivePosState setActivePosState()} or
    * {@link #setActiveVelState setActiveVelState()},
    * and any call to {@link #getActiveForces getActiveForces()}.
    *
    * <p>In the process of updating the forces, the system may determine that a
    * smaller step size is needed.  If so, it can indicate this through the
    * argument optional <code>stepAdjust</code>, if present.
    *
    * @param t
    * current time
    */
   public void updateForces (double t);

   public void advanceAuxState (double t0, double t1);

   public void getAuxAdvanceState (DataBuffer buf);

   public void setAuxAdvanceState (DataBuffer buf);

   public int getAuxVarStateSize();

   public void getAuxVarState (VectorNd w);

   public void setAuxVarState (VectorNd w);

   public void getAuxVarDerivative (VectorNd dwdt);
   
}
