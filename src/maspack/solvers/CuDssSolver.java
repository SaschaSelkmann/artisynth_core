/**
 * Copyright (c) 2026, by the Authors: ArtiSynth Team
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package maspack.solvers;

import maspack.fileutil.NativeLibraryException;
import maspack.fileutil.NativeLibraryManager;
import maspack.matrix.ImproperStateException;
import maspack.matrix.Matrix;
import maspack.matrix.Matrix.Partition;
import maspack.matrix.NumericalException;
import maspack.matrix.VectorNd;

/**
 * JNI interface to NVIDIA cuDSS, exposed as a {@link DirectSolver}.
 *
 * <p>This is an optional, experimental backend intended for regular
 * FEM-heavy sparse direct solves on a CUDA-capable GPU. It is not a
 * drop-in replacement for {@link PardisoSolver}: it does not implement
 * iterative refinement, multiple right-hand sides, or KKT-style
 * constrained solves. Use it only for the regular
 * {@code BackwardEuler}-style FE solve path.
 *
 * <p>Typical usage mirrors {@link PardisoSolver}:
 * <pre>
 *    CuDssSolver s = new CuDssSolver();
 *    s.analyze (M, M.rowSize(), Matrix.SPD);
 *    s.factor();
 *    s.solve (x, b);
 *    s.dispose();
 * </pre>
 *
 * <p>After the first {@link #factor()} call following an analyze,
 * subsequent calls to {@link #factor()} with the same sparsity pattern
 * use cuDSS's refactorization phase, which is significantly cheaper than
 * a full factorization. This maps directly onto FE time-stepping where
 * the solve matrix structure stays fixed across steps and only the
 * numeric values change.
 *
 * <p>Indices passed to cuDSS are 0-based. Maspack's public CRS export is
 * 1-based, so this class converts a copy on every {@code analyze()}.
 *
 * <p>You must call {@link #dispose()} when done so that GPU resources
 * are released.
 */
public class CuDssSolver implements DirectSolver {

   // Matches the CUDSS_BRIDGE_MT_* constants in cudssBridge.h
   private static final int MT_GENERAL   = 0;
   private static final int MT_SYMMETRIC = 1;
   private static final int MT_SPD       = 2;

   // Solver state, matching PardisoSolver's convention
   public static final int UNSET    = 0;
   public static final int ANALYZED = 1;
   public static final int FACTORED = 2;

   // Initialization status
   private static final int INIT_UNKNOWN              =  0;
   private static final int INIT_OK                   =  1;
   private static final int ERR_CANT_LOAD_LIBRARIES   = -1;

   // Native library name. NativeLibraryManager resolves this to
   // lib/Linux64/libCuDssJNI.so.0.8.0 (mirroring PardisoJNI's pattern).
   static String nativeLibrary = "CuDssJNI.0.8.0";

   private static int myInitStatus = INIT_UNKNOWN;
   private static String myInitErrMsg;

   private long myHandle;        // pointer to native CuDssBridge
   private int  myState;
   private int  mySize;
   private int  myNumVals;
   private int  myType;
   private Matrix myMatrix;
   private Partition myPart;
   private double[] myVals;
   private int[]    myColIdxs;   // 0-based, length numVals
   private int[]    myRowOffs;   // 0-based, length size+1

   // Native methods — see CuDssJNI.cc
   private static native long   doInit();
   private static native int    doSetPattern (
      long handle, int n, int nnz, int[] rowOffs, int[] colIdxs, int mtype);
   private static native int    doAnalyze (long handle);
   private static native int    doFactor  (long handle, double[] vals);
   private static native int    doSolve   (long handle, double[] b, double[] x);
   private static native int    doSolveMulti (
      long handle, int nrhs, double[] B, double[] X);
   private static native int    doIterativeSolve (
      long handle, double[] vals, double[] b, double[] x,
      double tolRel, int maxIter);
   private static native void   doDispose (long handle);
   private static native String doGetLastError (long handle);
   private static native String doGetVersion();
   private static native void   doSetTimingEnabled (boolean on);
   private static native int    doSetIrSteps (long handle, int n);

   private static synchronized void doLoadLibraries() {
      if (myInitStatus != INIT_UNKNOWN) {
         return;
      }
      try {
         NativeLibraryManager.load (nativeLibrary);
         myInitStatus = INIT_OK;
         // Propagate the timing system property to the native bridge.
         if (Boolean.getBoolean ("artisynth.cudss.timing")) {
            doSetTimingEnabled (true);
         }
      }
      catch (NativeLibraryException | UnsatisfiedLinkError e) {
         myInitErrMsg = e.getMessage();
         myInitStatus = ERR_CANT_LOAD_LIBRARIES;
      }
   }

   /**
    * Programmatic toggle for the per-phase timing diagnostics. Equivalent
    * to launching the JVM with {@code -Dartisynth.cudss.timing=true}.
    * When enabled, each factor / solve / BiCGStab call prints a
    * {@code [cudss-timing]} line to stderr.
    */
   public static void setTimingEnabled (boolean on) {
      if (!isAvailable()) {
         return;
      }
      doSetTimingEnabled (on);
   }

   /**
    * Sets cuDSS's CUDSS_CONFIG_IR_N_STEPS (number of iterative refinement
    * passes applied after each cuDSS solve). Default is 4 (set in
    * the native init). Mirrors {@link PardisoSolver#setMaxRefinementSteps}
    * for the cases where {@link MurtyMechSolver} disables refinement
    * during specific contact/friction solves.
    *
    * @param n number of IR passes; 0 disables IR
    */
   public synchronized void setIterativeRefinementSteps (int n) {
      if (myHandle == 0L) {
         throw new ImproperStateException ("Solver disposed");
      }
      int status = doSetIrSteps (myHandle, n);
      if (status != 0) {
         String detail = doGetLastError (myHandle);
         throw new NumericalException (
            "cuDSS setIterativeRefinementSteps failed"
            + (detail != null ? ": " + detail : ""));
      }
   }

   /**
    * Returns true if the cuDSS native library is available and loaded.
    * If this returns false, all attempts to construct a {@code CuDssSolver}
    * will throw {@link UnsupportedOperationException}.
    */
   public static boolean isAvailable() {
      if (myInitStatus == INIT_UNKNOWN) {
         doLoadLibraries();
      }
      return myInitStatus == INIT_OK;
   }

   /**
    * Returns the cuDSS runtime version (e.g. "0.8.0"), or {@code null} if
    * the native library is unavailable.
    */
   public static String getCuDssVersion() {
      if (!isAvailable()) {
         return null;
      }
      return doGetVersion();
   }

   /**
    * Creates a new CuDssSolver. Throws {@link UnsupportedOperationException}
    * if the native cuDSS library cannot be loaded.
    */
   public CuDssSolver() {
      if (!isAvailable()) {
         throw new UnsupportedOperationException (
            "cuDSS not available: " + myInitErrMsg);
      }
      myHandle = doInit();
      if (myHandle == 0L) {
         throw new UnsupportedOperationException (
            "cuDSS initialization failed (no CUDA-capable device?)");
      }
      myState = UNSET;
      myVals    = new double[0];
      myColIdxs = new int[0];
      myRowOffs = new int[0];
   }

   private static int mtypeFlag (int type) {
      if ((type & Matrix.SYMMETRIC) != 0) {
         if ((type & Matrix.POSITIVE_DEFINITE) != 0) {
            return MT_SPD;
         }
         return MT_SYMMETRIC;
      }
      return MT_GENERAL;
   }

   private static Partition partitionFor (int type) {
      if ((type & Matrix.SYMMETRIC) != 0) {
         return Partition.UpperTriangular;
      }
      return Partition.Full;
   }

   private void ensureBufferCapacity (int size, int numVals) {
      if (myVals.length < numVals) {
         myVals = new double[numVals];
      }
      if (myColIdxs.length < numVals) {
         myColIdxs = new int[numVals];
      }
      if (myRowOffs.length < size + 1) {
         myRowOffs = new int[size + 1];
      }
   }

   // Maspack CRS export is 1-based; cuDSS needs 0-based. Convert in place.
   private static void toZeroBased (int[] rowOffs, int sizePlus1,
                                    int[] colIdxs, int numVals) {
      for (int i = 0; i < sizePlus1; i++) {
         rowOffs[i] -= 1;
      }
      for (int i = 0; i < numVals; i++) {
         colIdxs[i] -= 1;
      }
   }

   private void check (int status, String op) {
      if (status != 0) {
         String detail = doGetLastError (myHandle);
         throw new NumericalException (
            "cuDSS " + op + " failed (status " + status + ")"
            + (detail != null ? ": " + detail : ""));
      }
   }

   /**
    * Returns the most recent native cuDSS bridge error for this solver, or
    * {@code null} if the last native operation succeeded.
    */
   public synchronized String getLastErrorMessage() {
      if (myHandle == 0L) {
         return null;
      }
      return doGetLastError (myHandle);
   }

   @Override
   public synchronized void analyze (Matrix M, int size, int type) {
      if (M.rowSize() != M.colSize()) {
         throw new IllegalArgumentException ("Matrix is not square");
      }
      if (size < 0 || size > M.rowSize()) {
         throw new IllegalArgumentException (
            "Requested size " + size + " is out of bounds");
      }
      Partition part = partitionFor (type);
      int numVals = M.numNonZeroVals (Partition.Full, size, size);
      if (part == Partition.UpperTriangular) {
         numVals -= (numVals - size) / 2;
      }
      ensureBufferCapacity (size, numVals);
      M.getCRSIndices (myColIdxs, myRowOffs, part, size, size);
      M.getCRSValues  (myVals, part, size, size);
      toZeroBased (myRowOffs, size + 1, myColIdxs, numVals);

      // Trim to exact lengths; the native side reads up to nnz from these
      // arrays, but it's safer to pass arrays of the exact size in case
      // future versions add bounds checks.
      int[] rowOffs = myRowOffs;
      int[] colIdxs = myColIdxs;
      if (myRowOffs.length != size + 1) {
         rowOffs = new int[size + 1];
         System.arraycopy (myRowOffs, 0, rowOffs, 0, size + 1);
      }
      if (myColIdxs.length != numVals) {
         colIdxs = new int[numVals];
         System.arraycopy (myColIdxs, 0, colIdxs, 0, numVals);
      }

      check (doSetPattern (myHandle, size, numVals, rowOffs, colIdxs,
                           mtypeFlag (type)),
             "setPattern");
      check (doAnalyze (myHandle), "analyze");

      mySize    = size;
      myNumVals = numVals;
      myType    = type;
      myMatrix  = M;
      myPart    = part;
      myState   = ANALYZED;
   }

   @Override
   public synchronized void factor() {
      if (myState == UNSET || myMatrix == null) {
         throw new ImproperStateException (
            "analyze() or analyzeAndFactor() not previously called");
      }
      myMatrix.getCRSValues (myVals, myPart, mySize, mySize);
      double[] vals = myVals;
      if (myVals.length != myNumVals) {
         vals = new double[myNumVals];
         System.arraycopy (myVals, 0, vals, 0, myNumVals);
      }
      check (doFactor (myHandle, vals), "factor");
      myState = FACTORED;
   }

   @Override
   public void analyzeAndFactor (Matrix M) {
      analyze (M, M.rowSize(), 0);
      factor();
   }

   /**
    * {@inheritDoc}
    *
    * <p>cuDSS expects <b>0-based</b> CRS indices. Callers must supply
    * {@code colIdxs} and {@code rowOffs} in 0-based form (this is the
    * opposite of {@link PardisoSolver#analyze(double[],int[],int[],int,int)},
    * which expects 1-based indices). After this call, follow up with
    * {@link #factor(double[])} to push the actual values; {@link #factor()}
    * (no-args) is not supported because no {@link Matrix} reference is
    * retained.
    */
   @Override
   public synchronized void analyze (
      double[] vals, int[] colIdxs, int[] rowOffs, int size, int type) {
      check (doSetPattern (myHandle, size, rowOffs[size], rowOffs, colIdxs,
                           mtypeFlag (type)),
             "setPattern");
      check (doAnalyze (myHandle), "analyze");
      // We don't retain a Matrix object — factor() (no-args) won't work
      // after this; factor(double[]) is the expected continuation.
      myMatrix  = null;
      mySize    = size;
      myNumVals = rowOffs[size];
      myType    = type;
      myPart    = null;
      myState   = ANALYZED;
   }

   /**
    * {@inheritDoc}
    *
    * <p>The first call after {@link #analyze} does a full factorization;
    * subsequent calls with the same sparsity pattern hit cuDSS's
    * {@code CUDSS_PHASE_REFACTORIZATION} fast path.
    */
   @Override
   public synchronized void factor (double[] vals) {
      if (myState == UNSET) {
         throw new ImproperStateException ("analyze() not previously called");
      }
      check (doFactor (myHandle, vals), "factor");
      myState = FACTORED;
   }

   /**
    * {@inheritDoc}
    *
    * <p>The native side reads exactly {@code mySize} doubles from {@code b}
    * and writes exactly {@code mySize} doubles to {@code x}. Both arrays
    * must have length at least {@code mySize}.
    */
   @Override
   public synchronized void solve (double[] x, double[] b) {
      if (myState != FACTORED) {
         throw new ImproperStateException ("factor() not previously called");
      }
      check (doSolve (myHandle, b, x), "solve");
   }

   /**
    * {@inheritDoc}
    *
    * <p>Both {@code X} and {@code B} are column-major dense matrices of
    * size {@code mySize x nrhs}, stored as contiguous arrays of length
    * {@code mySize * nrhs}. Backed by a dedicated multi-RHS scratch pair
    * on the GPU that is grown lazily.
    */
   @Override
   public synchronized void solve (double[] X, double[] B, int nrhs) {
      if (myState != FACTORED) {
         throw new ImproperStateException ("factor() not previously called");
      }
      if (nrhs <= 0) {
         throw new IllegalArgumentException ("nrhs must be positive");
      }
      if (X.length < mySize * nrhs || B.length < mySize * nrhs) {
         throw new IllegalArgumentException (
            "X and B must have length >= n*nrhs (" + (mySize * nrhs) + ")");
      }
      check (doSolveMulti (myHandle, nrhs, B, X), "solveMulti");
   }

   @Override
   public synchronized void solve (VectorNd x, VectorNd b) {
      if (myState != FACTORED) {
         throw new ImproperStateException (
            "factor() not previously called");
      }
      if (x.size() < mySize) {
         x.setSize (mySize);
      }
      if (b.size() < mySize) {
         throw new IllegalArgumentException (
            "Right-hand side b has size " + b.size()
            + ", expected at least " + mySize);
      }
      // VectorNd.getBuffer() may be longer than size(); the native side
      // reads/writes exactly the first mySize entries.
      check (doSolve (myHandle, b.getBuffer(), x.getBuffer()), "solve");
   }

   /**
    * Equivalent to {@link PardisoSolver#autoFactorAndSolve} for cuDSS.
    * <ul>
    *   <li>If {@code tolExp <= 0} or {@code myState != FACTORED}: do a
    *       full {@link #factor()}+{@link #solve(VectorNd,VectorNd)} pair.
    *       The first call after analyze always lands here.</li>
    *   <li>Else: re-extract current matrix values from the {@code Matrix}
    *       reference stored at analyze time, run BiCGStab preconditioned
    *       by the existing factor, and accept the result if it converges
    *       within {@code maxIter}. If BiCGStab fails (breakdown,
    *       non-convergence), fall back to a real factor + solve.</li>
    * </ul>
    *
    * <p>The matrix used for SpMV during BiCGStab is the <i>current</i>
    * value of the matrix (re-extracted from {@code myMatrix}), while the
    * preconditioner is the <i>stale</i> factor from the previous
    * factor() call. That's the same trick PARDISO's hybrid solve plays.
    */
   @Override
   public synchronized void autoFactorAndSolve (
      VectorNd x, VectorNd b, int tolExp) {
      if (myState == UNSET || myMatrix == null) {
         throw new ImproperStateException (
            "analyze(Matrix) or analyzeAndFactor(Matrix) not previously called");
      }
      if (tolExp <= 0 || myState != FACTORED) {
         factor();
         solve (x, b);
         return;
      }
      // Re-extract current matrix values.
      myMatrix.getCRSValues (myVals, myPart, mySize, mySize);
      double[] vals = myVals;
      if (myVals.length != myNumVals) {
         vals = new double[myNumVals];
         System.arraycopy (myVals, 0, vals, 0, myNumVals);
      }
      // BiCGStab needs the b and x arrays sized to exactly mySize. Most
      // callers pass exact-sized VectorNds (size == buffer length), but
      // VectorNd may have a longer underlying buffer; we work on copies
      // sized to mySize to avoid feeding stale tail bytes to the GPU.
      double[] bArr;
      if (b.size() == mySize && b.getBuffer().length == mySize) {
         bArr = b.getBuffer();
      }
      else {
         bArr = new double[mySize];
         for (int i = 0; i < mySize; i++) bArr[i] = b.get(i);
      }
      double[] xArr = new double[mySize];
      double tolRel = Math.pow (10.0, -tolExp);
      // PARDISO defaults to ~32 CGS iterations before giving up. Use the
      // same budget for parity.
      int maxIter = 32;
      int rc = doIterativeSolve (myHandle, vals, bArr, xArr, tolRel, maxIter);
      if (rc > 0) {
         // Success: rc is iteration count.
         if (x.size() < mySize) x.setSize (mySize);
         for (int i = 0; i < mySize; i++) x.set (i, xArr[i]);
         return;
      }
      // BiCGStab failed (rc <= 0). Fall back to a real factor + solve.
      // This refactors with the current values; the stale factor is
      // replaced.
      factor();
      solve (x, b);
   }

   @Override
   public boolean hasAutoIterativeSolving() {
      // BiCGStab is available for both analyze paths (Matrix-based and
      // array-CSR), distinguished only by which autoFactorAndSolve /
      // iterativeSolve entry point the caller uses.
      return true;
   }

   /**
    * Iterative-solve entry point for callers that build CSR arrays
    * themselves (like {@link KKTSolver}). Pushes {@code vals} as the
    * current matrix values, runs BiCGStab using the existing cuDSS
    * factor as a preconditioner, and writes the solution into {@code x}.
    *
    * <p>Returns the number of BiCGStab iterations on success ({@code > 0}),
    * or a non-positive value on failure. Callers should refactor and
    * fall back to a direct solve if the return value is non-positive.
    */
   @Override
   public synchronized int iterativeSolve (
      double[] vals, double[] x, double[] b, int tolExp) {
      if (myState != FACTORED) {
         return -1;
      }
      double tolRel = Math.pow (10.0, -tolExp);
      int maxIter = 32;
      return doIterativeSolve (myHandle, vals, b, x, tolRel, maxIter);
   }

   @Override
   public synchronized void dispose() {
      if (myHandle != 0L) {
         doDispose (myHandle);
         myHandle = 0L;
      }
      myState = UNSET;
   }

   @Override
   protected void finalize() throws Throwable {
      try {
         dispose();
      }
      finally {
         super.finalize();
      }
   }
}
