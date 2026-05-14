/**
 * Copyright (c) 2014, by the Authors: John E Lloyd (UBC)
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package maspack.solvers;

import maspack.matrix.Matrix;
import maspack.matrix.VectorNd;
import maspack.matrix.NumericalException;

/**
 * Common interface for sparse direct solvers used by ArtiSynth.
 *
 * <p>The {@code Matrix}-based {@code analyze(Matrix,int,int)} entry point
 * is the primary public API. The array-based {@code analyze(double[],
 * int[], int[], int, int)} entry point, together with {@code
 * factor(double[])} and the {@code solve(double[],double[][,int])}
 * overloads, exist so callers like {@link KKTSolver} that already build
 * CSR arrays internally can hand them to the solver without going through
 * a {@code Matrix} wrapper. Not every implementation supports the array
 * forms; the default implementations throw {@link
 * UnsupportedOperationException}.
 */
public interface DirectSolver {
   /**
    * Performs prefactor analysis on a specified matrix. The matrix reference is
    * stored and used by later calls to {@link #factor() factor()}. If
    * <code>size</code> is less than the actual matrix size, then the analysis
    * is done on the principal submatrix of M defined by the first
    * <code>size</code> rows and columns.
    * 
    * @param M
    * matrix to analyze
    * @param size
    * size of the matrix to factor.
    * @param type
    * or-ed flags giving information about the matrix type. Typical flags are
    * {@link maspack.matrix.Matrix#SYMMETRIC SYMMETRIC} or
    * {@link maspack.matrix.Matrix#SYMMETRIC POSITIVE_DEFINITE}
    * @throws IllegalArgumentException
    * if the matrix is not square, or the matrix type is not supported by the
    * solver.
    * @throws NumericalException
    * if the analysis failed for numeric reasons.
    */
   public void analyze (Matrix M, int size, int type);

   /**
    * Factors a previously analyzed matrix.
    * 
    * @throws IllegalStateException
    * if no previous call to {@link #analyze analyze} has been made.
    * @throws NumericalException
    * if the factor failed for numeric reasons.
    */
   public void factor();

   /**
    * Factors a matrix. This is equivalent to the two calls
    * 
    * <pre>
    *   analyze (M, M.rowSize(), 0)
    *   factor()
    * </pre>
    * 
    * @param M
    * matrix to factor
    * @throws IllegalArgumentException
    * if the matrix is not square, or general matrices are not supported by the
    * solver.
    * @throws NumericalException
    * if the analysis or factoring failed for numeric reasons.
    */
   public void analyzeAndFactor (Matrix M);

   /**
    * Solves the system
    * 
    * <pre>
    *  M x = b
    * </pre>
    * 
    * where M was specified using previous calls to {@link #analyze analyze} or
    * {@link #analyzeAndFactor(Matrix) factor}.
    * 
    * @param x
    * vector in which result is returned
    * @param b
    * right hand vector of matrix equation
    * @throws NumericalException
    * if the solve failed for numeric reasons.
    * @throws IllegalStateException
    * if no previous call to {@link #analyze analyze} or
    * {@link #analyzeAndFactor(Matrix) factor} has been made.
    */
   public void solve (VectorNd x, VectorNd b);

   /**
    * Factors a previously analyzed matrix M and then solves the system
    * 
    * <pre>
    * M x = b
    * </pre>
    * 
    * This is equivalent to
    * 
    * <pre>
    * factor()
    * solve (x, b) 
    * </pre>
    * 
    * but is included because it may be faster depending on the underlying
    * implementation.
    * 
    * If auto-iterative solving is available (as determined by {@link
    * #hasAutoIterativeSolving hasAutoIterativeSolving}), this method also
    * allows the solver to automatically employ iterative solving using a recent
    * direct factorization as a preconditioner. To enable auto-iterative
    * solving, the argument tolExp should be set to a positive value giving the
    * (negative) exponent of the desired relative residual.
    * 
    * @param x
    * vector in which result is returned
    * @param b
    * right hand vector of matrix equation
    * @param tolExp
    * if positive, enables auto-iterative solving with the specified value
    * giving the (negative) exponent of the desired relative residual.
    * @throws NumericalException
    * if the factoring or solving failed for numeric reasons
    */
   public void autoFactorAndSolve (VectorNd x, VectorNd b, int tolExp);

   /**
    * Returns true if this solver supports automatic iterative solving using a
    * recent directly-factored matrix as a preconditioner. If this feature is
    * available, it may be invoked using the
    * {@link #autoFactorAndSolve autoFactorAndSolve} method.
    * 
    * @return true if auto-iterative solving is available
    */
   public boolean hasAutoIterativeSolving();

   /**
    * Releases all internal resources allocated by this solver.
    */
   public void dispose();

   // -------------------------------------------------------------------
   // Array-CSR entry points used by KKTSolver and similar callers that
   // build CRS arrays themselves rather than going through Matrix.
   // -------------------------------------------------------------------

   /**
    * Performs prefactor analysis on a matrix supplied as CRS arrays. The
    * caller is responsible for matching the implementation's expected
    * index base; PARDISO expects 1-based indices, while cuDSS expects
    * 0-based. Each implementation documents its convention.
    *
    * <p>The default implementation throws {@link
    * UnsupportedOperationException}. Implementations that support direct
    * CRS analyze (PARDISO, cuDSS) override this method.
    *
    * @param vals nonzero values, row-major
    * @param colIdxs column indices for each value
    * @param rowOffs row start offsets, length size+1
    * @param size matrix size (number of rows/columns)
    * @param type matrix type flags ({@link Matrix#INDEFINITE},
    * {@link Matrix#SYMMETRIC}, {@link Matrix#SPD})
    */
   default void analyze (
      double[] vals, int[] colIdxs, int[] rowOffs, int size, int type) {
      throw new UnsupportedOperationException (
         getClass().getSimpleName()
         + " does not support analyze(double[],int[],int[],int,int)");
   }

   /**
    * Re-factors a previously-analyzed matrix using new values, keeping
    * the same sparsity pattern. Implementations may use a refactorization
    * fast path when available.
    *
    * <p>The default implementation throws {@link
    * UnsupportedOperationException}.
    *
    * @param vals new nonzero values, in the same order used by the most
    * recent {@link #analyze(double[],int[],int[],int,int) analyze}
    */
   default void factor (double[] vals) {
      throw new UnsupportedOperationException (
         getClass().getSimpleName() + " does not support factor(double[])");
   }

   /**
    * Solves {@code M x = b} for a previously-factored matrix, using
    * primitive arrays. The default implementation wraps {@code x} and
    * {@code b} as {@link VectorNd} buffers and delegates to {@link
    * #solve(VectorNd,VectorNd)}.
    */
   default void solve (double[] x, double[] b) {
      VectorNd xv = new VectorNd (x.length, x);
      VectorNd bv = new VectorNd (b.length, b);
      solve (xv, bv);
      // VectorNd constructed from a double[] backing array does NOT
      // alias; copy result back.
      for (int i = 0; i < x.length; i++) {
         x[i] = xv.get(i);
      }
   }

   /**
    * Solves {@code M X = B} for multiple right-hand sides, with {@code X}
    * and {@code B} stored column-major (each column is one RHS). The
    * default implementation loops over single-RHS solves; implementations
    * with native multi-RHS support override this for performance.
    *
    * @param X solution columns, length {@code n * nrhs}
    * @param B right-hand side columns, length {@code n * nrhs}
    * @param nrhs number of right-hand sides
    */
   default void solve (double[] X, double[] B, int nrhs) {
      int n = X.length / nrhs;
      double[] x = new double[n];
      double[] b = new double[n];
      for (int k = 0; k < nrhs; k++) {
         System.arraycopy (B, k * n, b, 0, n);
         solve (x, b);
         System.arraycopy (x, 0, X, k * n, n);
      }
   }
}
