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
   }

   public static void main (String[] args) {
      RandomGenerator.setSeed (0x1234);
      new CuDssSolverTest().runtest();
   }
}
