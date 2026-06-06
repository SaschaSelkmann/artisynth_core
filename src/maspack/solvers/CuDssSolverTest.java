/**
 * Copyright (c) 2026, by the Authors: ArtiSynth Team
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package maspack.solvers;

import maspack.matrix.Matrix;
import maspack.matrix.Matrix1x1Block;
import maspack.matrix.MatrixNd;
import maspack.matrix.SparseBlockMatrix;
import maspack.matrix.SparseCRSMatrix;
import maspack.matrix.SparseMatrixNd;
import maspack.matrix.SparseNumberedBlockMatrix;
import maspack.matrix.VectorNd;
import maspack.util.RandomGenerator;
import maspack.util.TestException;
import maspack.util.UnitTest;

/**
 * Unit tests for {@link CuDssSolver}. Skips itself with a notice if the
 * cuDSS native library is unavailable on this machine.
 */
public class CuDssSolverTest extends UnitTest {

   private static final double RESIDUAL_TOL = 1e-10;

   // Verify ||A x - b|| / max(1, ||b||) <= tol. Avoids needing a reference
   // solver; the cuDSS solution must satisfy its own input.
   private void checkResidual (
      String label, Matrix M, VectorNd x, VectorNd b, double tol) {
      int n = M.rowSize();
      MatrixNd A = new MatrixNd (n, n);
      for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
            A.set (i, j, M.get (i, j));
         }
      }
      VectorNd Ax = new VectorNd (n);
      A.mul (Ax, x);
      Ax.sub (b);
      double scale = Math.max (1.0, b.norm());
      double rel = Ax.norm() / scale;
      if (rel > tol) {
         throw new TestException (
            label + ": relative residual " + rel + " exceeds " + tol
            + "\n  x = " + x
            + "\n  b = " + b);
      }
   }

   // ---- Test matrix builders ----

   // 5x5 SPD matrix:
   //   [[4 0 1 0 0]
   //    [0 3 2 0 0]
   //    [1 2 5 0 0]
   //    [0 0 0 1 1]
   //    [0 0 0 1 2]]
   private SparseCRSMatrix buildSpd5() {
      SparseCRSMatrix M = new SparseCRSMatrix();
      M.setSize (5, 5);
      double[] vals = { 4, 1,  3, 2,  1, 2, 5,  1, 1,  1, 2 };
      int[] cols    = { 1, 3,  2, 3,  1, 2, 3,  4, 5,  4, 5 };
      int[] rowOffs = { 1, 3, 5, 8, 10, 12 };
      M.setCRSValues (vals, cols, rowOffs, vals.length, 5, Matrix.Partition.Full);
      return M;
   }

   // Same sparsity as buildSpd5() but values scaled — exercises the
   // refactorization fast path.
   private SparseCRSMatrix buildSpd5Scaled (double scale) {
      SparseCRSMatrix M = buildSpd5();
      M.scale (scale);
      return M;
   }

   // 4x4 general nonsymmetric matrix with full sparsity for simplicity:
   //   [[ 4  1  0  0]
   //    [ 2  3  1  0]
   //    [ 0  1  5  2]
   //    [ 0  0  3  6]]
   private SparseCRSMatrix buildGeneral4() {
      SparseCRSMatrix M = new SparseCRSMatrix();
      M.setSize (4, 4);
      double[] vals = { 4, 1,  2, 3, 1,  1, 5, 2,  3, 6 };
      int[] cols    = { 1, 2,  1, 2, 3,  2, 3, 4,  3, 4 };
      int[] rowOffs = { 1, 3, 6, 9, 11 };
      M.setCRSValues (vals, cols, rowOffs, vals.length, 4, Matrix.Partition.Full);
      return M;
   }

   // 3x3 symmetric indefinite (negative eigenvalue):
   //   [[ 1  2  0]
   //    [ 2  1  0]
   //    [ 0  0  3]]
   // Eigenvalues are 3, 3, -1.
   private SparseCRSMatrix buildSymIndef3() {
      SparseCRSMatrix M = new SparseCRSMatrix();
      M.setSize (3, 3);
      double[] vals = { 1, 2,  2, 1,  3 };
      int[] cols    = { 1, 2,  1, 2,  3 };
      int[] rowOffs = { 1, 3, 5, 6 };
      M.setCRSValues (vals, cols, rowOffs, vals.length, 3, Matrix.Partition.Full);
      return M;
   }

   // A 3x3-block SPD matrix using Matrix1x1Blocks. Exercises the
   // SparseNumberedBlockMatrix code path that the FE solver uses.
   private SparseNumberedBlockMatrix buildBlockSpd (int n) {
      int[] sizes = new int[n];
      for (int i = 0; i < n; i++) sizes[i] = 1;
      SparseNumberedBlockMatrix M = new SparseNumberedBlockMatrix (sizes);
      // tridiagonal SPD: 4 on diag, -1 off-diag (strictly diagonally dominant)
      for (int i = 0; i < n; i++) {
         Matrix1x1Block d = new Matrix1x1Block(); d.m00 = 4.0;
         M.addBlock (i, i, d);
         if (i + 1 < n) {
            Matrix1x1Block u = new Matrix1x1Block(); u.m00 = -1.0;
            Matrix1x1Block l = new Matrix1x1Block(); l.m00 = -1.0;
            M.addBlock (i, i + 1, u);
            M.addBlock (i + 1, i, l);
         }
      }
      return M;
   }

   // Build a random RHS reproducibly.
   private VectorNd randomVec (int n) {
      VectorNd v = new VectorNd (n);
      for (int i = 0; i < n; i++) {
         v.set (i, RandomGenerator.get().nextDouble() * 2 - 1);
      }
      return v;
   }

   // ---- Individual tests ----

   private void testSpd5() {
      CuDssSolver s = new CuDssSolver();
      try {
         SparseCRSMatrix M = buildSpd5();
         VectorNd b = new VectorNd (new double[] { 7, 12, 20, 9, 14 });
         VectorNd x = new VectorNd (5);
         s.analyze (M, 5, Matrix.SPD);
         s.factor();
         s.solve (x, b);
         // Construction was designed so x = [1, 2, 3, 4, 5] exactly.
         VectorNd expected = new VectorNd (new double[] { 1, 2, 3, 4, 5 });
         checkNormedEquals ("SPD 5x5 solve", x, expected, RESIDUAL_TOL);
         checkResidual ("SPD 5x5 residual", M, x, b, RESIDUAL_TOL);
      }
      finally { s.dispose(); }
   }

   private void testGeneral4() {
      CuDssSolver s = new CuDssSolver();
      try {
         SparseCRSMatrix M = buildGeneral4();
         VectorNd b = randomVec (4);
         VectorNd x = new VectorNd (4);
         s.analyze (M, 4, Matrix.INDEFINITE);
         s.factor();
         s.solve (x, b);
         checkResidual ("general 4x4", M, x, b, RESIDUAL_TOL);
      }
      finally { s.dispose(); }
   }

   private void testSymIndef3() {
      CuDssSolver s = new CuDssSolver();
      try {
         SparseCRSMatrix M = buildSymIndef3();
         VectorNd b = randomVec (3);
         VectorNd x = new VectorNd (3);
         // Try as INDEFINITE rather than SYMMETRIC: cuDSS's symmetric
         // indefinite path historically requires pivoting we may not have
         // configured. Treating it as a general matrix is always safe.
         s.analyze (M, 3, Matrix.INDEFINITE);
         s.factor();
         s.solve (x, b);
         checkResidual ("sym indef 3x3 (as general)", M, x, b, RESIDUAL_TOL);
      }
      finally { s.dispose(); }
   }

   // After analyze + factor, factor() again with scaled values must hit the
   // refactorization fast path and produce the correctly-scaled solution.
   private void testRefactor() {
      CuDssSolver s = new CuDssSolver();
      try {
         SparseCRSMatrix M = buildSpd5();
         VectorNd b = new VectorNd (new double[] { 7, 12, 20, 9, 14 });
         VectorNd x = new VectorNd (5);
         s.analyze (M, 5, Matrix.SPD);
         s.factor();
         s.solve (x, b);
         checkResidual ("refactor: first solve", M, x, b, RESIDUAL_TOL);

         // Scale values by 2 -> x must scale by 1/2.
         M.scale (2.0);
         s.factor();   // refactorization phase
         s.solve (x, b);
         checkResidual ("refactor: 2x scaled solve", M, x, b, RESIDUAL_TOL);
         VectorNd halfExpected = new VectorNd (new double[] { 0.5, 1, 1.5, 2, 2.5 });
         checkNormedEquals ("refactor: halved x", x, halfExpected, RESIDUAL_TOL);

         // Scale by another 0.25 (= 0.5x original) -> x scales by 2 of original.
         M.scale (0.25);
         s.factor();   // refactorization phase
         s.solve (x, b);
         checkResidual ("refactor: 0.5x original scaled solve",
                        M, x, b, RESIDUAL_TOL);
      }
      finally { s.dispose(); }
   }

   // After analyze, do multiple solves without re-factoring; each must
   // produce a correct residual.
   private void testRepeatedSolveSameFactor() {
      CuDssSolver s = new CuDssSolver();
      try {
         SparseCRSMatrix M = buildSpd5();
         s.analyze (M, 5, Matrix.SPD);
         s.factor();
         for (int k = 0; k < 5; k++) {
            VectorNd b = randomVec (5);
            VectorNd x = new VectorNd (5);
            s.solve (x, b);
            checkResidual ("repeated solve #" + k, M, x, b, RESIDUAL_TOL);
         }
      }
      finally { s.dispose(); }
   }

   // analyze with one matrix, then analyze with a different sparsity/size.
   // Each round must produce a correct residual.
   private void testReanalyze() {
      CuDssSolver s = new CuDssSolver();
      try {
         {
            SparseCRSMatrix M = buildSpd5();
            VectorNd b = randomVec (5);
            VectorNd x = new VectorNd (5);
            s.analyze (M, 5, Matrix.SPD);
            s.factor();
            s.solve (x, b);
            checkResidual ("reanalyze: first SPD 5x5", M, x, b, RESIDUAL_TOL);
         }
         {
            SparseCRSMatrix M = buildGeneral4();
            VectorNd b = randomVec (4);
            VectorNd x = new VectorNd (4);
            s.analyze (M, 4, Matrix.INDEFINITE);
            s.factor();
            s.solve (x, b);
            checkResidual ("reanalyze: second general 4x4", M, x, b, RESIDUAL_TOL);
         }
         {
            SparseCRSMatrix M = buildSpd5();
            VectorNd b = randomVec (5);
            VectorNd x = new VectorNd (5);
            s.analyze (M, 5, Matrix.SPD);
            s.factor();
            s.solve (x, b);
            checkResidual ("reanalyze: third SPD 5x5", M, x, b, RESIDUAL_TOL);
         }
      }
      finally { s.dispose(); }
   }

   // The FE solver hands matrices to DirectSolver as SparseNumberedBlockMatrix.
   // Verify CRS extraction + solve work for that representation.
   private void testBlockMatrix() {
      CuDssSolver s = new CuDssSolver();
      try {
         int n = 12;
         SparseNumberedBlockMatrix M = buildBlockSpd (n);
         VectorNd b = randomVec (n);
         VectorNd x = new VectorNd (n);
         s.analyze (M, n, Matrix.SPD);
         s.factor();
         s.solve (x, b);
         checkResidual ("SparseNumberedBlockMatrix solve",
                        M, x, b, RESIDUAL_TOL);
      }
      finally { s.dispose(); }
   }

   // Also verify SparseMatrixNd works (smaller priority — light test).
   private void testDenseSparseCarrier() {
      CuDssSolver s = new CuDssSolver();
      try {
         int n = 6;
         SparseMatrixNd M = new SparseMatrixNd (n, n);
         for (int i = 0; i < n; i++) {
            M.set (i, i, 4.0);
            if (i + 1 < n) {
               M.set (i, i + 1, -1.0);
               M.set (i + 1, i, -1.0);
            }
         }
         VectorNd b = randomVec (n);
         VectorNd x = new VectorNd (n);
         s.analyze (M, n, Matrix.SPD);
         s.factor();
         s.solve (x, b);
         checkResidual ("SparseMatrixNd solve", M, x, b, RESIDUAL_TOL);
      }
      finally { s.dispose(); }
   }

   // Run multiple full lifecycle iterations on the same solver instance to
   // catch resource leaks or stale state between cycles.
   private void testLifecycleStability() {
      CuDssSolver s = new CuDssSolver();
      try {
         for (int iter = 0; iter < 10; iter++) {
            SparseCRSMatrix M = buildSpd5Scaled (1.0 + 0.1 * iter);
            VectorNd b = randomVec (5);
            VectorNd x = new VectorNd (5);
            s.analyze (M, 5, Matrix.SPD);
            s.factor();
            s.solve (x, b);
            checkResidual (
               "lifecycle iter " + iter, M, x, b, RESIDUAL_TOL);
         }
      }
      finally { s.dispose(); }
   }

   // dispose() must be idempotent.
   private void testIdempotentDispose() {
      CuDssSolver s = new CuDssSolver();
      s.dispose();
      s.dispose();
   }

   // ---- Stage B additions: array-CSR analyze, factor(double[]),
   //      solve(double[],double[]), multi-RHS solve, symmetric mtype ----

   // 5x5 SPD reference, with 0-based CSR upper-triangle as cuDSS sees it.
   // Mirrors the C++ smoke test from Stage 1.
   private static final int[]    SPD5_ROW_OFFS_0 = { 0, 2, 4, 5, 7, 8 };
   private static final int[]    SPD5_COL_IDXS_0 = { 0, 2, 1, 2, 2, 3, 4, 4 };
   private static final double[] SPD5_VALS       = { 4, 1, 3, 2, 5, 1, 1, 2 };
   private static final double[] SPD5_RHS        = { 7, 12, 20, 9, 14 };
   private static final double[] SPD5_EXPECTED   = { 1, 2, 3, 4, 5 };

   // analyze + factor + solve via the raw-array entry points.
   private void testArrayCsrSpdRoundtrip() {
      CuDssSolver s = new CuDssSolver();
      try {
         s.analyze (SPD5_VALS, SPD5_COL_IDXS_0, SPD5_ROW_OFFS_0, 5, Matrix.SPD);
         s.factor (SPD5_VALS);
         double[] x = new double[5];
         s.solve (x, SPD5_RHS);
         for (int i = 0; i < 5; i++) {
            if (Math.abs (x[i] - SPD5_EXPECTED[i]) > RESIDUAL_TOL) {
               throw new TestException (
                  "array-CSR SPD x[" + i + "]=" + x[i] +
                  " expected " + SPD5_EXPECTED[i]);
            }
         }
      }
      finally { s.dispose(); }
   }

   private void testDeviceValueAssembly() {
      CuDssSolver s = new CuDssSolver();
      try {
         s.analyze (SPD5_VALS, SPD5_COL_IDXS_0, SPD5_ROW_OFFS_0, 5, Matrix.SPD);
         int[] slots = new int[SPD5_VALS.length];
         double[] halfVals = new double[SPD5_VALS.length];
         for (int i=0; i<SPD5_VALS.length; i++) {
            slots[i] = i;
            halfVals[i] = 0.5*SPD5_VALS[i];
         }

         s.clearDeviceValues();
         s.addDeviceValues (slots, halfVals, halfVals.length, 1.0);
         s.addDeviceValues (slots, halfVals, halfVals.length, 1.0);
         s.factorDeviceValues();

         double[] x = new double[5];
         s.solve (x, SPD5_RHS);
         for (int i=0; i<5; i++) {
            if (Math.abs (x[i] - SPD5_EXPECTED[i]) > RESIDUAL_TOL) {
               throw new TestException (
                  "device value assembly x[" + i + "]=" + x[i] +
                  " expected " + SPD5_EXPECTED[i]);
            }
         }
      }
      finally { s.dispose(); }
   }

   private void testScaledDiagonal3DeviceContribution() {
      CuDssSolver s = new CuDssSolver();
      try {
         double[] zeroVals = { 0, 0, 0 };
         int[] cols = { 0, 1, 2 };
         int[] rows = { 0, 1, 2, 3 };
         s.analyze (zeroVals, cols, rows, 3, Matrix.SPD);

         int[] diagSlots = { 0, 1, 2 };
         double[] masses = { 2.0 };
         s.clearDeviceValues();
         s.addScaledDiagonal3DeviceValues (diagSlots, masses, 1, 2.0);
         s.factorDeviceValues();

         double[] x = new double[3];
         s.solve (x, new double[] { 4, 8, 12 });
         for (int i=0; i<3; i++) {
            double expected = i + 1;
            if (Math.abs (x[i] - expected) > RESIDUAL_TOL) {
               throw new TestException (
                  "scaled diagonal3 device contribution x[" + i + "]=" +
                  x[i] + " expected " + expected);
            }
         }
      }
      finally { s.dispose(); }
   }

   private void testScaledBlock3DeviceContribution() {
      CuDssSolver s = new CuDssSolver();
      try {
         double[] zeroVals = { 0, 0, 0, 0, 0, 0, 0 };
         int[] cols = { 0, 1, 0, 1, 2, 1, 2 };
         int[] rows = { 0, 2, 5, 7 };
         s.analyze (zeroVals, cols, rows, 3, Matrix.INDEFINITE);

         int[] blockSlots = {
            0, 1, -1,
            2, 3,  4,
           -1, 5,  6
         };
         double[] blockVals = {
            4, 1, 0,
            1, 3, 1,
            0, 1, 2
         };
         double[] blockScales = { 0.5 };
         s.clearDeviceValues();
         s.addScaledBlock3DeviceValues (
            blockSlots, blockVals, blockScales, 1, 2.0);
         s.factorDeviceValues();

         double[] x = new double[3];
         s.solve (x, new double[] { 6, 10, 8 });
         for (int i=0; i<3; i++) {
            double expected = i + 1;
            if (Math.abs (x[i] - expected) > RESIDUAL_TOL) {
               throw new TestException (
                  "scaled block3 device contribution x[" + i + "]=" +
                  x[i] + " expected " + expected);
            }
         }
      }
      finally { s.dispose(); }
   }

   private void testMaterialStiffness3DeviceContribution() {
      CuDssSolver s = new CuDssSolver();
      try {
         double[] zeroVals = { 0, 0, 0 };
         int[] cols = { 0, 1, 2 };
         int[] rows = { 0, 1, 2, 3 };
         s.analyze (zeroVals, cols, rows, 3, Matrix.SPD);

         int[] blockSlots = {
             0, -1, -1,  -1,  1, -1,  -1, -1,  2,
             0, -1, -1,  -1,  1, -1,  -1, -1,  2,
             0, -1, -1,  -1,  1, -1,  -1, -1,  2
         };
         double[] gis = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
         };
         double[] gjs = gis.clone();
         double[] Ds = new double[3*36];
         Ds[0*36 + 0] = 4;
         Ds[1*36 + 7] = 5;
         Ds[2*36 + 14] = 6;
         double[] sigmas = new double[3*6];
         double[] dvs = { 1, 1, 1 };

         s.clearDeviceValues();
         s.addMaterialStiffness3DeviceValues (
            blockSlots, gis, gjs, Ds, sigmas, dvs, 3, 1.0);
         s.factorDeviceValues();

         double[] x = new double[3];
         s.solve (x, new double[] { 4, 10, 18 });
         for (int i=0; i<3; i++) {
            double expected = i + 1;
            if (Math.abs (x[i] - expected) > RESIDUAL_TOL) {
               throw new TestException (
                  "material stiffness3 device contribution x[" + i + "]=" +
                  x[i] + " expected " + expected);
            }
         }
      }
      finally { s.dispose(); }
   }

   private void testMaterialStiffness3ElementDeviceContribution() {
      CuDssSolver s = new CuDssSolver();
      try {
         double[] zeroVals = { 0, 0, 0 };
         int[] cols = { 0, 1, 2 };
         int[] rows = { 0, 1, 2, 3 };
         s.analyze (zeroVals, cols, rows, 3, Matrix.SPD);

         int[] elemNodeCounts = { 3 };
         int[] elemPairOffsets = { 0, 3 };
         int[] elemIpOffsets = { 0, 1 };
         int[] elemGradOffsets = { 0, 3 };
         int[] pairNodeIdxs = {
            0, 0,
            1, 1,
            2, 2
         };
         int[] blockSlots = {
             0, -1, -1,  -1, -1, -1,  -1, -1, -1,
            -1, -1, -1,  -1,  1, -1,  -1, -1, -1,
            -1, -1, -1,  -1, -1, -1,  -1, -1,  2
         };
         double[] grads = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
         };
         double[] Ds = new double[36];
         Ds[0] = 4;
         Ds[7] = 5;
         Ds[14] = 6;
         double[] sigmas = new double[6];
         double[] dvs = { 1 };

         s.clearDeviceValues();
         s.addMaterialStiffness3ElementDeviceValues (
            elemNodeCounts, elemPairOffsets, elemIpOffsets, elemGradOffsets,
            pairNodeIdxs, blockSlots, grads, Ds, sigmas, dvs, 1, 1.0);
         s.factorDeviceValues();

         double[] x = new double[3];
         s.solve (x, new double[] { 4, 10, 18 });
         for (int i=0; i<3; i++) {
            double expected = i + 1;
            if (Math.abs (x[i] - expected) > RESIDUAL_TOL) {
               throw new TestException (
                  "material stiffness3 element device contribution x[" + i +
                  "]=" + x[i] + " expected " + expected);
            }
         }
      }
      finally { s.dispose(); }
   }

   private void testDilationalStiffness3ElementDeviceContribution() {
      CuDssSolver s = new CuDssSolver();
      try {
         double[] zeroVals = { 0, 0, 0 };
         int[] cols = { 0, 1, 2 };
         int[] rows = { 0, 1, 2, 3 };
         s.analyze (zeroVals, cols, rows, 3, Matrix.SPD);

         int[] elemNodeCounts = { 3 };
         int[] elemPressureCounts = { 3 };
         int[] elemPairOffsets = { 0, 3 };
         int[] elemConstraintOffsets = { 0, 9 };
         int[] elemRinvOffsets = { 0, 9 };
         int[] pairNodeIdxs = {
            0, 0,
            1, 1,
            2, 2
         };
         int[] blockSlots = {
             0, -1, -1,  -1, -1, -1,  -1, -1, -1,
            -1, -1, -1,  -1,  1, -1,  -1, -1, -1,
            -1, -1, -1,  -1, -1, -1,  -1, -1,  2
         };
         double[] constraints = {
            1, 0, 0,  0, 0, 0,  0, 0, 0,
            0, 0, 0,  0, 1, 0,  0, 0, 0,
            0, 0, 0,  0, 0, 0,  0, 0, 1
         };
         double[] rinvs = {
            4, 0, 0,
            0, 5, 0,
            0, 0, 6
         };

         s.clearDeviceValues();
         s.addDilationalStiffness3ElementDeviceValues (
            elemNodeCounts, elemPressureCounts, elemPairOffsets,
            elemConstraintOffsets, elemRinvOffsets, pairNodeIdxs, blockSlots,
            constraints, rinvs, 1, 1.0);
         s.factorDeviceValues();

         double[] x = new double[3];
         s.solve (x, new double[] { 4, 10, 18 });
         for (int i=0; i<3; i++) {
            double expected = i + 1;
            if (Math.abs (x[i] - expected) > RESIDUAL_TOL) {
               throw new TestException (
                  "dilational stiffness3 element device contribution x[" + i +
                  "]=" + x[i] + " expected " + expected);
            }
         }
      }
      finally { s.dispose(); }
   }

   private void testLinearElasticStiffness3ElementDeviceContribution() {
      CuDssSolver s = new CuDssSolver();
      try {
         double[] zeroVals = { 0, 0, 0 };
         int[] cols = { 0, 1, 2 };
         int[] rows = { 0, 1, 2, 3 };
         s.analyze (zeroVals, cols, rows, 3, Matrix.SPD);

         int[] elemNodeCounts = { 3 };
         int[] elemPairOffsets = { 0, 3 };
         int[] elemIpOffsets = { 0, 1 };
         int[] elemGradOffsets = { 0, 3 };
         int[] pairNodeIdxs = {
            0, 0,
            1, 1,
            2, 2
         };
         int[] blockSlots = {
             0, -1, -1,  -1, -1, -1,  -1, -1, -1,
            -1, -1, -1,  -1,  1, -1,  -1, -1, -1,
            -1, -1, -1,  -1, -1, -1,  -1, -1,  2
         };
         double[] elemParams = { 4, 0 };
         double[] grads = {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
         };
         double[] dvs = { 1 };

         s.clearDeviceValues();
         s.addLinearElasticStiffness3ElementDeviceValues (
            elemNodeCounts, elemPairOffsets, elemIpOffsets, elemGradOffsets,
            pairNodeIdxs, blockSlots, elemParams, grads, dvs, 1, 1.0);
         s.factorDeviceValues();

         double[] x = new double[3];
         s.solve (x, new double[] { 4, 8, 12 });
         for (int i=0; i<3; i++) {
            double expected = i + 1;
            if (Math.abs (x[i] - expected) > RESIDUAL_TOL) {
               throw new TestException (
                  "linear elastic stiffness3 element device contribution x[" +
                  i + "]=" + x[i] + " expected " + expected);
            }
         }
      }
      finally { s.dispose(); }
   }


   // Refactor with same pattern via array entry points.
   private void testArrayCsrRefactor() {
      CuDssSolver s = new CuDssSolver();
      try {
         s.analyze (SPD5_VALS, SPD5_COL_IDXS_0, SPD5_ROW_OFFS_0, 5, Matrix.SPD);
         s.factor (SPD5_VALS);
         double[] x = new double[5];
         s.solve (x, SPD5_RHS);
         // Scale values 2x: x should halve.
         double[] vals2 = SPD5_VALS.clone();
         for (int i = 0; i < vals2.length; i++) vals2[i] *= 2.0;
         s.factor (vals2);
         s.solve (x, SPD5_RHS);
         for (int i = 0; i < 5; i++) {
            double exp = SPD5_EXPECTED[i] * 0.5;
            if (Math.abs (x[i] - exp) > RESIDUAL_TOL) {
               throw new TestException (
                  "array-CSR refactor x[" + i + "]=" + x[i] + " expected " + exp);
            }
         }
      }
      finally { s.dispose(); }
   }

   // Multi-RHS solve: solving for k right-hand sides at once must produce
   // the same answer as k separate single-RHS solves.
   private void testMultiRhsConsistency() {
      CuDssSolver s = new CuDssSolver();
      try {
         s.analyze (SPD5_VALS, SPD5_COL_IDXS_0, SPD5_ROW_OFFS_0, 5, Matrix.SPD);
         s.factor (SPD5_VALS);

         int n = 5;
         int k = 4;
         double[] B = new double[n * k];
         for (int c = 0; c < k; c++) {
            for (int r = 0; r < n; r++) {
               B[c * n + r] = RandomGenerator.get().nextDouble() * 2 - 1;
            }
         }
         double[] Xbatch = new double[n * k];
         s.solve (Xbatch, B, k);

         // Cross-check each column against a single-RHS solve.
         double[] x = new double[n];
         double[] b = new double[n];
         for (int c = 0; c < k; c++) {
            System.arraycopy (B, c * n, b, 0, n);
            s.solve (x, b);
            for (int r = 0; r < n; r++) {
               double err = Math.abs (Xbatch[c * n + r] - x[r]);
               if (err > RESIDUAL_TOL) {
                  throw new TestException (
                     "multi-RHS col " + c + " row " + r +
                     ": multi=" + Xbatch[c*n+r] + " single=" + x[r]);
               }
            }
         }
      }
      finally { s.dispose(); }
   }

   // Grow the multi-RHS buffers via increasing nrhs, then shrink. The bridge
   // should reuse the larger allocation cleanly.
   private void testMultiRhsGrowShrink() {
      CuDssSolver s = new CuDssSolver();
      try {
         s.analyze (SPD5_VALS, SPD5_COL_IDXS_0, SPD5_ROW_OFFS_0, 5, Matrix.SPD);
         s.factor (SPD5_VALS);
         for (int nrhs : new int[] { 1, 3, 8, 2, 5 }) {
            double[] B = new double[5 * nrhs];
            double[] X = new double[5 * nrhs];
            for (int i = 0; i < B.length; i++) {
               B[i] = RandomGenerator.get().nextDouble() * 2 - 1;
            }
            s.solve (X, B, nrhs);
            // Residual check on each column.
            for (int c = 0; c < nrhs; c++) {
               double[] x = new double[5];
               double[] b = new double[5];
               System.arraycopy (X, c * 5, x, 0, 5);
               System.arraycopy (B, c * 5, b, 0, 5);
               // ||A x - b|| -- assemble A on the fly using the upper-tri data.
               double[] Ax = new double[5];
               // upper-tri storage + symmetric mirror
               for (int row = 0; row < 5; row++) {
                  for (int p = SPD5_ROW_OFFS_0[row]; p < SPD5_ROW_OFFS_0[row+1]; p++) {
                     int col = SPD5_COL_IDXS_0[p];
                     double v = SPD5_VALS[p];
                     Ax[row] += v * x[col];
                     if (col != row) Ax[col] += v * x[row];
                  }
               }
               double err = 0;
               for (int i = 0; i < 5; i++) {
                  err += (Ax[i] - b[i]) * (Ax[i] - b[i]);
               }
               err = Math.sqrt (err);
               if (err > 1e-9) {
                  throw new TestException (
                     "multi-RHS grow/shrink nrhs=" + nrhs + " col=" + c +
                     " residual=" + err);
               }
            }
         }
      }
      finally { s.dispose(); }
   }

   // Symmetric indefinite KKT-shape matrix. Build [[M G^T];[G 0]] where M is
   // SPD and G is rectangular -- the resulting block is symmetric indefinite,
   // exactly what equality KKT produces. Solve with CUDSS_MTYPE_SYMMETRIC
   // (upper-triangle storage).
   //
   // Concrete 5x5 KKT system, M is 3x3 SPD, G is 2x3:
   //   M = [[4 0 1]
   //        [0 3 2]
   //        [1 2 5]]
   //   G = [[1 0 1]
   //        [0 1 1]]
   //   K = [[M     G^T]
   //        [G     0  ]]  (5x5, symmetric indefinite, has both positive and
   //                       negative eigenvalues because the (G G^T) Schur
   //                       complement makes the lower-right block effectively
   //                       negative-definite after elimination).
   //
   // Stored as upper triangle, 0-based.
   private void testSymmetricIndefiniteKkt() {
      CuDssSolver s = new CuDssSolver();
      try {
         // Upper-triangle entries of K, row by row.
         // Row 0: (0,0)=4, (0,2)=1, (0,3)=1
         // Row 1: (1,1)=3, (1,2)=2, (1,4)=1
         // Row 2: (2,2)=5, (2,3)=1, (2,4)=1
         // Row 3: (3,3)=0  (the zero block diagonal). cuDSS needs the
         //                  diagonal present; 0 is OK as long as factor
         //                  doesn't see it as a true zero pivot. To be safe
         //                  add a tiny regularization (-1e-12) — but for this
         //                  test we'll use a regularized variant with -0.1.
         // Row 4: (4,4)=-0.1
         double[] vals = {
            4, 1, 1,        // row 0
            3, 2, 1,        // row 1
            5, 1, 1,        // row 2
            -0.1,           // row 3
            -0.1            // row 4
         };
         int[]    cols = {
            0, 2, 3,
            1, 2, 4,
            2, 3, 4,
            3,
            4
         };
         int[]    rows = { 0, 3, 6, 9, 10, 11 };

         int n = 5;
         double[] b = new double[] { 1, 2, 3, 4, 5 };
         double[] x = new double[n];

         s.analyze (vals, cols, rows, n, Matrix.SYMMETRIC);
         s.factor (vals);
         s.solve (x, b);

         // Compute ||K x - b|| using full symmetric multiply.
         double[] Kx = new double[n];
         for (int r = 0; r < n; r++) {
            for (int p = rows[r]; p < rows[r+1]; p++) {
               int c = cols[p];
               double v = vals[p];
               Kx[r] += v * x[c];
               if (c != r) Kx[c] += v * x[r];
            }
         }
         double err = 0, bn = 0;
         for (int i = 0; i < n; i++) {
            err += (Kx[i] - b[i]) * (Kx[i] - b[i]);
            bn  += b[i] * b[i];
         }
         double rel = Math.sqrt (err) / Math.max (1.0, Math.sqrt (bn));
         if (rel > 1e-9) {
            throw new TestException (
               "symmetric-indefinite KKT solve relative residual = " + rel);
         }
      }
      finally { s.dispose(); }
   }

   @Override
   public void test() {
      if (!CuDssSolver.isAvailable()) {
         System.out.println (
            "CuDssSolverTest: cuDSS native library not available -- skipping");
         return;
      }
      System.out.println ("cuDSS version: " + CuDssSolver.getCuDssVersion());

      testSpd5();
      testGeneral4();
      testSymIndef3();
      testRefactor();
      testRepeatedSolveSameFactor();
      testReanalyze();
      testBlockMatrix();
      testDenseSparseCarrier();
      testLifecycleStability();
      testIdempotentDispose();

      // Stage B additions
      testArrayCsrSpdRoundtrip();
      testDeviceValueAssembly();
      testScaledDiagonal3DeviceContribution();
      testScaledBlock3DeviceContribution();
      testMaterialStiffness3DeviceContribution();
      testMaterialStiffness3ElementDeviceContribution();
      testDilationalStiffness3ElementDeviceContribution();
      testLinearElasticStiffness3ElementDeviceContribution();
      testArrayCsrRefactor();
      testMultiRhsConsistency();
      testMultiRhsGrowShrink();
      testSymmetricIndefiniteKkt();

      // Stage (c) additions: BiCGStab hybrid solve
      testHybridSolveBasic();
      testHybridSolveSpdBigger();
      testHybridSolveArrayPath();
   }

   // After an initial factor, autoFactorAndSolve(tolExp>0) must run
   // BiCGStab using the stale factor as preconditioner and produce a
   // correct answer for the CURRENT (modified) matrix values.
   private void testHybridSolveBasic() {
      CuDssSolver s = new CuDssSolver();
      try {
         SparseCRSMatrix M = buildSpd5();
         VectorNd b = new VectorNd (new double[] { 7, 12, 20, 9, 14 });
         VectorNd x = new VectorNd (5);
         s.analyze (M, 5, Matrix.SPD);
         s.factor();
         s.solve (x, b);

         if (!s.hasAutoIterativeSolving()) {
            throw new TestException (
               "hasAutoIterativeSolving() returned false after analyze(Matrix)");
         }

         // Perturb M so that the stale factor is *close* to A but not exact.
         // The 1.05 scale keeps the matrix SPD and gives BiCGStab something
         // to converge to.
         M.scale (1.05);

         // Pick a different RHS to ensure we're not just re-using cached x.
         VectorNd b2 = new VectorNd (new double[] { 1, 2, 3, 4, 5 });
         VectorNd x2 = new VectorNd (5);
         s.autoFactorAndSolve (x2, b2, /*tolExp=*/10);

         // Verify residual of x2 against the CURRENT M.
         checkResidual ("hybrid basic", M, x2, b2, 1e-9);
      }
      finally { s.dispose(); }
   }

   // Hybrid solve on a larger SPD matrix; BiCGStab should converge quickly
   // because the perturbation is small.
   private void testHybridSolveSpdBigger() {
      CuDssSolver s = new CuDssSolver();
      try {
         int n = 40;
         SparseNumberedBlockMatrix M = buildBlockSpd (n);
         VectorNd b = randomVec (n);
         VectorNd x = new VectorNd (n);

         s.analyze (M, n, Matrix.SPD);
         s.factor();
         s.solve (x, b);
         checkResidual ("hybrid bigger: initial", M, x, b, RESIDUAL_TOL);

         // Perturb the block matrix slightly. SparseNumberedBlockMatrix
         // doesn't expose a global scale, but we can modify a few block
         // values directly. Just add a small diagonal jitter.
         for (int i = 0; i < n; i++) {
            Matrix1x1Block d = (Matrix1x1Block) M.getBlock (i, i);
            d.m00 += 0.001;
         }

         VectorNd b2 = randomVec (n);
         VectorNd x2 = new VectorNd (n);
         s.autoFactorAndSolve (x2, b2, /*tolExp=*/10);
         checkResidual ("hybrid bigger: after perturb", M, x2, b2, 1e-9);
      }
      finally { s.dispose(); }
   }

   // Verify the array-CSR iterativeSolve entry that KKTSolver will use.
   private void testHybridSolveArrayPath() {
      CuDssSolver s = new CuDssSolver();
      try {
         s.analyze (SPD5_VALS, SPD5_COL_IDXS_0, SPD5_ROW_OFFS_0, 5, Matrix.SPD);
         s.factor (SPD5_VALS);

         // Same matrix, different RHS, same factor.
         double[] b2 = { 1, 2, 3, 4, 5 };
         double[] x2 = new double[5];
         int iters = s.iterativeSolve (
            SPD5_VALS, x2, b2, /*tolExp=*/10);
         if (iters <= 0) {
            throw new TestException (
               "iterativeSolve returned " + iters + " (expected positive)"
               + ", native error=" + s.getLastErrorMessage());
         }
         // Verify residual.
         double[] Ax = new double[5];
         for (int row = 0; row < 5; row++) {
            for (int p = SPD5_ROW_OFFS_0[row]; p < SPD5_ROW_OFFS_0[row+1]; p++) {
               int col = SPD5_COL_IDXS_0[p];
               double v = SPD5_VALS[p];
               Ax[row] += v * x2[col];
               if (col != row) Ax[col] += v * x2[row];
            }
         }
         double err = 0, bn = 0;
         for (int i = 0; i < 5; i++) {
            err += (Ax[i] - b2[i]) * (Ax[i] - b2[i]);
            bn  += b2[i] * b2[i];
         }
         double rel = Math.sqrt (err) / Math.max (1.0, Math.sqrt (bn));
         if (rel > 1e-9) {
            throw new TestException (
               "iterativeSolve relative residual " + rel + " too large");
         }
      }
      finally { s.dispose(); }
   }

   public static void main (String[] args) {
      RandomGenerator.setSeed (0x1234);
      new CuDssSolverTest().runtest();
   }
}
