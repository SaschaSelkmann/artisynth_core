/**
 * Copyright (c) 2014, by the Authors: John E Lloyd (UBC)
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package maspack.solvers;

import java.io.*;

import maspack.matrix.*;
import maspack.matrix.Matrix.Partition;
import maspack.solvers.*;
import maspack.solvers.KKTSolver.Status;
import maspack.util.*;

public class KKTSolverTest {
   public KKTSolverTest() {
      RandomGenerator.setSeed (0x1234);
   }

   private static double PREC = 1e-8;
   private boolean verbose = false;

   private void setUniqueValues (SparseBlockMatrix M) {
      double value = 1;
      for (int bi=0; bi<M.numBlockRows(); bi++) {
         for (MatrixBlock blk=M.firstBlockInRow(bi);
              blk!=null; blk=blk.next()) {
            for (int i=0; i<blk.rowSize(); i++) {
               for (int j=0; j<blk.colSize(); j++) {
                  if (blk.valueIsNonZero (i, j)) {
                     blk.set (i, j, value++);
                  }
               }
            }
         }
      }
   }

   private class MLCP {
      SparseBlockMatrix M;   // mass matrix
      SparseBlockMatrix GT;  // bilateral constraint matrix, can be null
      SparseBlockMatrix NT;  // unilateral constraint matrix, can be null
      SparseBlockMatrix DT;  // friction constraint matrix, can be null

      VectorNd Rg; // if non-null, regularization for G
      VectorNd Rn; // if non-null, regularization for N
      VectorNd Rd; // if non-null, regularization for D

      VectorNd bf; // rhs for mass matrix, must not be null
      VectorNd bg; // if non-null, rhs for G
      VectorNd bn; // if non-null, rhs for N
      VectorNd bd; // if non-null, rhs for D
      VectorNd flim; // friction limits for D, must not be null if DT != null

      VectorNd vel; // if non-null, used to check velocity solution
      VectorNd lam; // if non-null, used to check lambda solution
      VectorNd the; // if non-null, used to check theta solutuon
      VectorNd phi; // if non-null, used to check phi solution

      String name = "";
   }

   // public void dotest (int sizeM, int rowSizeG, int rowSizeN)
   // {
   // }

   public void test() {
      MatrixNdBlock Mblk = new MatrixNdBlock (6, 6);
      MatrixNdBlock GTblk = new MatrixNdBlock (6, 3);
      GTblk.setRandom();
      Mblk.setRandom();
      Mblk.mulTranspose (Mblk);
      SparseBlockMatrix M = new SparseBlockMatrix();
      M.addBlock (0, 0, Mblk);
      SparseBlockMatrix GT = new SparseBlockMatrix();
      GT.addBlock (0, 0, GTblk);

      SparseBlockMatrix NT = null;

      KKTSolver solver = new KKTSolver();

      VectorNd Rg = new VectorNd (3);
      VectorNd Rn = null;
      VectorNd vel = new VectorNd (6);
      VectorNd bm = new VectorNd (6);
      VectorNd bg = new VectorNd (3);
      VectorNd bn = null;
      VectorNd lam = new VectorNd (3);
      VectorNd the = null;

      bm.setRandom();
      Rg.setRandom();
      Rg.absolute();

      solveAndCheck (
         M, 6, GT, NT, Rg, Rn, bm, bg, bn, vel, lam, the, Matrix.SYMMETRIC);
      solveAndCheck (
         M, 6, GT, NT, Rg, Rn, bm, bg, bn, vel, lam, the, Matrix.INDEFINITE);

      // now try with diagonal M
      VectorNd Mdiag = new VectorNd (6);
      Mdiag.setRandom();
      Mdiag.absolute();
      solveAndCheck (
         Mdiag, 6, GT, NT, Rg, Rn, bm, bg, bn, vel, lam, the, Matrix.INDEFINITE);
      solveAndCheck (
         Mdiag, 6, GT, NT, Rg, Rn, bm, bg, bn, vel, lam, the, Matrix.SPD);

      int numN = 4;
      MatrixNdBlock NTblk = new MatrixNdBlock (6, numN);
      NTblk.setRandom();
      NT = new SparseBlockMatrix();
      NT.addBlock (0, 0, NTblk);
      Rn = new VectorNd (numN);
      bn = new VectorNd (numN);
      the = new VectorNd (numN);
      Rn.setRandom();
      Rn.absolute();
      bn.setRandom();

      solveAndCheck (
         M, 6, GT, NT, Rg, Rn, bm, bg, bn, vel, lam, the, Matrix.SYMMETRIC);
      solveAndCheck (
         M, 6, GT, NT, Rg, Rn, bm, bg, bn, vel, lam, the, Matrix.INDEFINITE);
   }

   private void checkKktMBlockSlotMap (
      SparseNumberedBlockMatrix M, int sizeM, SparseBlockMatrix GT,
      int typeM) {

      VectorNd Rg = new VectorNd (GT.colSize());
      Rg.setRandom();
      KKTSolver solver = new KKTSolver();
      solver.analyze (M, sizeM, GT, Rg, typeM);
      SparseNumberedBlockMatrix.CrsBlockSlotMap map =
         solver.getKktMBlockSlotMap();
      if (map == null) {
         throw new TestException ("KKT M block slot map is null");
      }
      double[] vals = solver.getCRSValuesForTesting();
      int nseen = 0;
      boolean[] seen = new boolean[vals.length];
      Partition part = ((typeM & Matrix.SYMMETRIC) != 0) ?
         Partition.UpperTriangular : Partition.Full;

      for (int bi=0; bi<M.numBlockRows(); bi++) {
         for (MatrixBlock blk=M.firstBlockInRow(bi);
              blk!=null; blk=blk.next()) {
            for (int i=0; i<blk.rowSize(); i++) {
               for (int j=0; j<blk.colSize(); j++) {
                  boolean stored = blk.valueIsNonZero (i, j);
                  if (part == Partition.UpperTriangular) {
                     if (blk.getBlockCol() < blk.getBlockRow()) {
                        stored = false;
                     }
                     else if (blk.getBlockCol() == blk.getBlockRow() && j < i) {
                        stored = false;
                     }
                  }
                  int slot = map.getBlockValueSlot (blk, i, j);
                  if (stored) {
                     if (slot < 0 || slot >= vals.length) {
                        throw new TestException (
                           "bad KKT slot "+slot+" for block entry");
                     }
                     if (seen[slot]) {
                        throw new TestException (
                           "KKT slot "+slot+" mapped more than once");
                     }
                     seen[slot] = true;
                     nseen++;
                     if (vals[slot] != blk.get (i, j)) {
                        throw new TestException (
                           "bad KKT slot value: got "+vals[slot]+
                           ", expected "+blk.get(i,j));
                     }
                  }
                  else if (slot != -1) {
                     throw new TestException (
                        "unstored block entry mapped to KKT slot "+slot);
                  }
               }
            }
         }
      }
      if (nseen != map.numMappedBlockValues()) {
         throw new TestException (
            "saw "+nseen+" mapped KKT M values, expected "+
            map.numMappedBlockValues());
      }
      solver.dispose();
   }

   public void testKktMBlockSlotMap() {
      int[] sizes = new int[] { 2, 3, 1 };
      SparseNumberedBlockMatrix M =
         new SparseNumberedBlockMatrix (sizes, sizes);
      M.addBlock (0, 0, new Matrix2x2Block());
      M.addBlock (0, 1, new MatrixNdBlock (2, 3));
      M.addBlock (1, 0, new MatrixNdBlock (3, 2));
      M.addBlock (1, 1, new MatrixNdBlock (3, 3));
      M.addBlock (1, 2, new MatrixNdBlock (3, 1));
      M.addBlock (2, 2, new MatrixNdBlock (1, 1));
      setUniqueValues (M);

      SparseBlockMatrix GT = new SparseBlockMatrix();
      MatrixNdBlock GT0 = new MatrixNdBlock (2, 2);
      MatrixNdBlock GT1 = new MatrixNdBlock (3, 2);
      MatrixNdBlock GT2 = new MatrixNdBlock (1, 2);
      GT0.setRandom();
      GT1.setRandom();
      GT2.setRandom();
      GT.addBlock (0, 0, GT0);
      GT.addBlock (1, 0, GT1);
      GT.addBlock (2, 0, GT2);

      checkKktMBlockSlotMap (M, M.rowSize(), GT, Matrix.INDEFINITE);
      checkKktMBlockSlotMap (M, M.rowSize(), GT, Matrix.SYMMETRIC);
   }

   private void solveAndCheck (
      Object M, int sizeM, SparseBlockMatrix GT, SparseBlockMatrix NT,
      VectorNd Rg, VectorNd Rn, VectorNd bm, VectorNd bg, VectorNd bn,
      VectorNd vel, VectorNd lam, VectorNd the, int typeM) {
      SparseBlockMatrix DT = null;
      VectorNd mu = null;
      VectorNd bet = null;
      int[] Dref = null;

      int numG = GT.colSize();
      int numN = 0;
      if (NT != null && NT.colSize() > 0) {
         numN = NT.colSize();
      }

      KKTSolver solver = new KKTSolver();
      if (M instanceof SparseBlockMatrix) {
         solver.analyze ((SparseBlockMatrix)M, sizeM, GT, Rg, typeM);
      }
      else {
         solver.analyze ((VectorNd)M, sizeM, GT, Rg);
      }
      Status status;
      if (numN == 0) {
         if (M instanceof SparseBlockMatrix) {
            solver.factor ((SparseBlockMatrix)M, sizeM, GT, Rg);
         }
         else {
            solver.factor ((VectorNd)M, sizeM, GT, Rg);
         }
         status = solver.solve (vel, lam, bm, bg);
      }
      else {
         if (M instanceof SparseBlockMatrix) {
            solver.factor ((SparseBlockMatrix)M, sizeM, GT, Rg, NT, Rn);
         }
         else {
            solver.factor ((VectorNd)M, sizeM, GT, Rg, NT, Rn);
         }
         status = solver.solve (vel, lam, the, bm, bg, bn);
      }
      if (status != Status.SOLVED) {
         throw new TestException ("Could not solve system, status " + status);
      }

      VectorNd bmCheck = new VectorNd (sizeM);
      VectorNd bgCheck = new VectorNd (numG);

      if (checkSolve (M, sizeM, GT, NT, vel, lam, the, bm, bmCheck) > 1e-8) {
         throw new TestException ("bm=" + bm.toString ("%8.3f")
         + ", expected\n   " + bmCheck.toString ("%8.3f"));
      }

      checkComplementarity (
         GT, NT, null, Rg, Rn, bg, bn, null, null, vel, lam, the, null);

      // GT.mulTranspose (bgCheck, vel);
      // for (int i = 0; i < numG; i++) {
      //    bgCheck.add (i, Rg.get(i) * lam.get(i));
      // }
      // if (!bg.epsilonEquals (bgCheck, 1e-8)) {
      //    throw new TestException ("bg=" + bg.toString ("%8.3f")
      //    + ", expected\n   " + bgCheck.toString ("%8.3f"));
      // }
      // if (numN > 0) {
      //    VectorNd wCheck = new VectorNd (numN);
      //    NT.mulTranspose (wCheck, vel);
      //    for (int i = 0; i < numN; i++) {
      //       wCheck.add (i, Rn.get(i) * the.get(i) - bn.get(i));
      //    }
      //    double maxErr = 0;
      //    for (int i = 0; i < numN; i++) {
      //       if (the.get(i) < -maxErr) {
      //          maxErr = -the.get(i);
      //       }
      //       if (wCheck.get(i) < -maxErr) {
      //          maxErr = -wCheck.get(i);
      //       }
      //       double err = Math.abs (wCheck.get(i) * the.get(i));
      //       if (err > maxErr) {
      //          maxErr = err;
      //       }
      //    }
      //    if (maxErr > 1e-8) {
      //       throw new TestException ("w/z error:\nw="
      //       + wCheck.toString ("%8.3f") + "\nz=" + the.toString ("%8.3f"));
      //    }
      // }
   }

   private void checkComplementarity (
      SparseBlockMatrix GT, SparseBlockMatrix NT, SparseBlockMatrix DT,
      VectorNd Rg, VectorNd Rn, VectorNd bg, VectorNd bn, VectorNd bd,
      VectorNd flim, VectorNd vel, VectorNd lam, VectorNd the, VectorNd phi) {
      
      int numG = GT != null ? GT.colSize() : 0;
      int numN = NT != null ? NT.colSize() : 0;
      int numD = DT != null ? DT.colSize() : 0;
      
      if (numG != 0) {
         VectorNd bgCheck = new VectorNd (numG);

         GT.mulTranspose (bgCheck, vel);
         for (int i=0; i<numG; i++) {
            bgCheck.add (i, Rg.get(i)*lam.get(i));
         }
         checkResult ("bg", bg, bgCheck);
      }
      if (numN > 0) {
         VectorNd wCheck = new VectorNd (numN);
         NT.mulTranspose (wCheck, vel);
         for (int i = 0; i < numN; i++) {
            wCheck.add (i, Rn.get(i)*the.get(i) - bn.get(i));
         }
         double tol = Math.max (PREC*(wCheck.infinityNorm()), PREC);
         double maxErr = 0;
         for (int i = 0; i < numN; i++) {
            if (the.get(i) < -maxErr) {
               maxErr = -the.get(i);
            }
            if (wCheck.get(i) < -maxErr) {
               maxErr = -wCheck.get(i);
            }
            double err = Math.abs (wCheck.get(i) * the.get(i));
            if (err > maxErr) {
               maxErr = err;
            }
         }
         if (maxErr > tol) {
            throw new TestException (
               "w/the error:\nw="+wCheck.toString("%8.3f")+
               "\nthe="+the.toString ("%8.3f"));
         }
      }
      if (numD > 0) {
         VectorNd wCheck = new VectorNd (numD);
         DT.mulTranspose (wCheck, vel);
         for (int i = 0; i < numD; i++) {
            wCheck.add (i, -bd.get(i));
         }
         double tol = Math.max (PREC*(wCheck.infinityNorm()), PREC);
         for (int i = 0; i < numD; i++) {
            double fmax = flim.get(i);
            double p = phi.get(i);
            double w = wCheck.get(i);
            // check bounds
            if (p < -fmax-tol || p > fmax+tol) {
               throw new TestException (
                  "phi("+i+")="+p+" out of bounds, fmax="+fmax);
            }
            // check complementarity
            if (w > tol && p > -fmax+tol) {
               throw new TestException (
                  "phi("+i+")="+p+" with w > 0, w="+w+", fmax="+fmax);
            }
            if (w < -tol && p < fmax-tol) {
               throw new TestException (
                  "phi("+i+")="+p+" with w < 0, w="+w+", fmax="+fmax);
            }
            if (p > -fmax+tol && p < fmax-tol && Math.abs(w) > tol) {
               throw new TestException (
                  "w != 0 with p in range, w="+w+", phi="+phi+", fmax="+fmax);
            }
         }
      }
   }

   private void checkResult (String name, VectorNd res, VectorNd chk) {
      double tol = Math.max (PREC*(res.infinityNorm()+chk.infinityNorm()), PREC);
      if (!res.epsilonEquals (chk, tol)) {
         throw new TestException (
            name+"="+res+", expecting "+chk);
      }
   }

   void testFromString (String str) {
      testMLCP (new ReaderTokenizer (new StringReader (str)));
   }

   void testFromFile (String fileName) {
      BufferedReader reader = null;
      try {
         reader = new BufferedReader (new FileReader (fileName));
      }
      catch (Exception e) {
         e.printStackTrace();
         System.exit(1); 
      }
      testMLCP (new ReaderTokenizer (reader));
   }

   void testMLCP (ReaderTokenizer rtok) {
      MLCP mlcp = new MLCP();
      while (true) {
         
         try {
            int tok = rtok.nextToken();
            if (tok != ReaderTokenizer.TT_EOF) {
               rtok.pushBack();
               scanMLCP (rtok, mlcp);
            }
            else {
               break;
            }
         }
         catch (Exception e) {
            e.printStackTrace(); 
            throw new TestException ("Error reading MLCP: " + e);
         }
         testMLCP (mlcp);
      }
   }

   void testMLCP (MLCP mlcp) {

      KKTSolver solver = new KKTSolver();

      if (mlcp.M == null) {
         throw new TestException ("M is null");
      }
      int Mtype = Matrix.INDEFINITE;
      if (mlcp.M.isSymmetric(0)) {
         Mtype = Matrix.SYMMETRIC;
      }
      
      int sizeM = mlcp.M.rowSize();
      VectorNd vel = new VectorNd (sizeM);
      int sizeG = mlcp.GT != null ? mlcp.GT.colSize() : 0;
      
      VectorNd Rg = mlcp.Rg != null ? mlcp.Rg : new VectorNd(sizeG);
      VectorNd bg = mlcp.bg != null ? mlcp.bg : new VectorNd(sizeG);
      VectorNd lam = new VectorNd(sizeG);
      int sizeN = mlcp.NT != null ? mlcp.NT.colSize() : 0;

      VectorNd Rn = mlcp.Rn != null ? mlcp.Rn : new VectorNd(sizeN);
      VectorNd bn = mlcp.bn != null ? mlcp.bn : new VectorNd(sizeN);
      VectorNd the = new VectorNd(sizeN);
      int sizeD = mlcp.DT != null ? mlcp.DT.colSize() : 0;

      VectorNd Rd = mlcp.Rd != null ? mlcp.Rd : new VectorNd(sizeD);
      VectorNd bd = mlcp.bd != null ? mlcp.bd : new VectorNd(sizeD);
      VectorNd phi = new VectorNd(sizeD);
      Status status;

      solver.analyze (mlcp.M, sizeM, mlcp.GT, Rg, Mtype);

      if (sizeN == 0) {
         solver.factor (mlcp.M, sizeM, mlcp.GT, Rg);
         status = solver.solve (vel, lam, mlcp.bf, bg);
      }
      else if (sizeD == 0) {
         solver.factor (mlcp.M, sizeM, mlcp.GT, Rg, mlcp.NT, Rn);
         status = solver.solve (vel, lam, the, mlcp.bf, bg, bn);
      }
      else {
         solver.factor (mlcp.M, sizeM, mlcp.GT, Rg, mlcp.NT, Rn, mlcp.DT, Rd);
         status = solver.solve (
            vel, lam, the, phi, mlcp.bf, bg, bn, bd, mlcp.flim);
      }
      if (status != Status.SOLVED) {
         throw new TestException ("Could not solve system, status " + status);
      }      
      if (mlcp.vel != null) {
         checkResult ("vel", vel, mlcp.vel);
      }
      if (sizeG != 0 && mlcp.lam != null) {
         checkResult ("lam", lam, mlcp.lam);
      }
      if (sizeN != 0 && mlcp.the != null) {
         checkResult ("the", the, mlcp.the);
      }
      if (sizeD != 0 && mlcp.phi != null) {
         checkResult ("phi", phi, mlcp.phi);
      }
      checkComplementarity (
         mlcp.GT, mlcp.NT, mlcp.DT, Rg, Rn, bg, bn, bd, mlcp.flim,
         vel, lam, the, phi);

      // if (sizeN > 0) {
      //    checkResult ("the", the, mlcp.the);
      // }
      // if (sizeD > 0) {
      //    checkResult ("phi", phi, mlcp.phi);
      // }
      if (verbose && mlcp.name != null && mlcp.name.length() > 0) {
         System.out.println (mlcp.name + " OK");
      }
   }      

   SparseBlockMatrix getMatrix (
      ReaderTokenizer rtok, boolean setNull) throws IOException {
      if (setNull) {
         return null;
      }
      else {
         SparseBlockMatrix M = new SparseBlockMatrix();
         M.scanBlocks (rtok);
         return M;
      }
   }

   VectorNd getVector (
      ReaderTokenizer rtok, boolean setNull) throws IOException {
      if (setNull) {
         return null;
      }
      else {
         VectorNd v = new VectorNd();
         v.scan (rtok);
         return v;
      }
   }

   private void checkSize (
      String vname, VectorNd v, String Mname, int size, boolean nullOK)
   throws IOException {

      if (!nullOK && v == null) {
         throw new IOException ("'"+vname+"' must be specified");
      }
      if (v != null && v.size() != size) {
         throw new IOException (
            "'"+vname+"' size = "+v.size()+", "+Mname+" requires "+size);
      }
   }

   private void checkRowSize (
      String Mname, SparseBlockMatrix M, int rsize) throws IOException{
         if (M.rowSize() != rsize) {
            throw new IOException (
               Mname+" row size "+M.rowSize()+" incompatible with M size "+rsize);
         }
   }

   public void scanMLCP (
      ReaderTokenizer rtok, MLCP mlcp) throws IOException {
      
      rtok.scanToken ('[');
      int tok;
      while ((tok = rtok.nextToken()) != ']' && tok != ReaderTokenizer.TT_EOF) {
         boolean setNull = false;
         rtok.pushBack();
         String name = rtok.scanWord();
         rtok.scanToken ('=');
         if (rtok.nextToken() == ReaderTokenizer.TT_WORD &&
             rtok.sval.equals ("null")) {
            setNull = true;
         }
         else {
            rtok.pushBack();
         }
         if (name.equals ("M")) {
            mlcp.M = getMatrix (rtok, setNull);
         }
         else if (name.equals ("GT")) {
            mlcp.GT = getMatrix (rtok, setNull);
         }
         else if (name.equals ("NT")) {
            mlcp.NT = getMatrix (rtok, setNull);
         }
         else if (name.equals ("DT")) {
            mlcp.DT = getMatrix (rtok, setNull);
         }
         else if (name.equals ("Rg")) {
            mlcp.Rg = getVector (rtok, setNull);
         }
         else if (name.equals ("Rn")) {
            mlcp.Rn = getVector (rtok, setNull);
         }
         else if (name.equals ("Rd")) {
            mlcp.Rd = getVector (rtok, setNull);
         }
         else if (name.equals ("bf")) {
            mlcp.bf = getVector (rtok, setNull);
         }
         else if (name.equals ("bg")) {
            mlcp.bg = getVector (rtok, setNull);
         }
         else if (name.equals ("bn")) {
            mlcp.bn = getVector (rtok, setNull);
         }
         else if (name.equals ("bd")) {
            mlcp.bd = getVector (rtok, setNull);
         }
         else if (name.equals ("flim")) {
            mlcp.flim = getVector (rtok, setNull);
         }
         else if (name.equals ("vel")) {
            mlcp.vel = getVector (rtok, setNull);
         }
         else if (name.equals ("lam")) {
            mlcp.lam = getVector (rtok, setNull);
         }
         else if (name.equals ("the")) {
            mlcp.the = getVector (rtok, setNull);
         }
         else if (name.equals ("phi")) {
            mlcp.phi = getVector (rtok, setNull);
         }
         else if (name.equals ("name")) {
            mlcp.name = rtok.scanQuotedString('"');
         }
         else {
            throw new IOException ("unrecognized quantity " + name);
         }
      }
      // check consistency
      int sizeM = mlcp.M.rowSize();
      int sizeG = mlcp.GT != null ? mlcp.GT.colSize() : 0;
      int sizeN = mlcp.NT != null ? mlcp.NT.colSize() : 0;
      int sizeD = mlcp.DT != null ? mlcp.DT.colSize() : 0;

      checkSize ("vel", mlcp.vel, "M", sizeM, true);
      checkSize ("bf", mlcp.bf, "M", sizeM, false);

      if (sizeG != 0) {
         checkRowSize ("GT", mlcp.GT, sizeM);
         checkSize ("bg", mlcp.Rg, "GT", sizeG, true);
         checkSize ("lam", mlcp.lam, "GT", sizeG, true);
      }
      if (sizeN != 0) {
         checkRowSize ("NT", mlcp.NT, sizeM);
         checkSize ("Rn", mlcp.Rn, "NT", sizeN, true);
         checkSize ("bn", mlcp.Rn, "NT", sizeN, true);
         checkSize ("the", mlcp.the, "NT", sizeN, true);
      }
      if (sizeD != 0) {
         checkRowSize ("DT", mlcp.DT, sizeM);
         checkSize ("Rd", mlcp.Rd, "DT", sizeD, true);
         checkSize ("bd", mlcp.bd, "DT", sizeD, true);
         checkSize ("phi", mlcp.phi, "DT", sizeD, true);
         checkSize ("flim", mlcp.flim, "DT", sizeD, false);
      }
   }

   public static double checkSolve (
      Object M, int sizeM, SparseBlockMatrix GT, SparseBlockMatrix NT,
      VectorNd vel, VectorNd lam, VectorNd the, VectorNd bm, VectorNd bmCheck) {
      int numG = (GT != null ? GT.colSize() : 0);
      int numN = (NT != null ? NT.colSize() : 0);
      if (bmCheck == null) {
         bmCheck = new VectorNd (sizeM);
      }
      if (M instanceof SparseBlockMatrix) {
         ((SparseBlockMatrix)M).mul (bmCheck, vel, sizeM, sizeM);
      }
      else {
         double[] diag = ((VectorNd)M).getBuffer();
         for (int i = 0; i < sizeM; i++) {
            bmCheck.set (i, diag[i] * vel.get(i));
         }
      }
      if (numG > 0) {
         lam.negate();
         GT.mulAdd (bmCheck, lam, sizeM, numG);
         lam.negate();
      }
      if (numN > 0) {
         the.negate();
         NT.mulAdd (bmCheck, the, sizeM, numN);
         the.negate();
      }
      bmCheck.sub (bm);
      double err = bmCheck.infinityNorm();
      return err;
   }

   // Local epsilon-equality helper for the Stage C cuDSS tests below.
   // KKTSolverTest doesn't extend UnitTest, so we inline a small check.
   private static void requireClose (
      String label, VectorNd a, VectorNd b, double tol) {
      if (a.size() != b.size()) {
         throw new TestException (
            label + ": size mismatch " + a.size() + " vs " + b.size());
      }
      double scale = Math.max (1.0, (a.norm() + b.norm()) / 2);
      VectorNd diff = new VectorNd (a);
      diff.sub (b);
      double rel = diff.norm() / scale;
      if (rel > tol) {
         throw new TestException (
            label + ": relative diff " + rel + " > " + tol +
            "\n  a = " + a.toString ("%10.6f") +
            "\n  b = " + b.toString ("%10.6f"));
      }
   }

   // Stage C: equality-KKT regression test for cuDSS-backed KKTSolver.
   // Compares vel/lam against the PARDISO-backed solver on the same
   // randomly-generated SPD M + dense GT system used in test().
   public void testCuDssEquality() {
      if (!CuDssSolver.isAvailable()) {
         System.out.println (
            "KKTSolverTest.testCuDssEquality: cuDSS not available -- skipping");
         return;
      }
      RandomGenerator.setSeed (0x4567);

      MatrixNdBlock Mblk = new MatrixNdBlock (6, 6);
      MatrixNdBlock GTblk = new MatrixNdBlock (6, 3);
      GTblk.setRandom();
      Mblk.setRandom();
      Mblk.mulTranspose (Mblk);            // make M SPD
      SparseBlockMatrix M = new SparseBlockMatrix();
      M.addBlock (0, 0, Mblk);
      SparseBlockMatrix GT = new SparseBlockMatrix();
      GT.addBlock (0, 0, GTblk);

      VectorNd Rg = new VectorNd (3);
      VectorNd bm = new VectorNd (6);
      VectorNd bg = new VectorNd (3);
      bm.setRandom();
      Rg.setRandom();
      Rg.absolute();

      for (int typeM : new int[] { Matrix.SYMMETRIC, Matrix.INDEFINITE }) {
         VectorNd velP = new VectorNd (6), lamP = new VectorNd (3);
         VectorNd velC = new VectorNd (6), lamC = new VectorNd (3);

         KKTSolver pardiso = new KKTSolver (SparseSolverId.Pardiso);
         pardiso.analyze (M, 6, GT, Rg, typeM);
         pardiso.factor (M, 6, GT, Rg);
         Status sP = pardiso.solve (velP, lamP, bm, bg);
         pardiso.dispose();
         if (sP != Status.SOLVED) {
            throw new TestException (
               "PARDISO solve failed for typeM=" + typeM + ": " + sP);
         }

         KKTSolver cudss = new KKTSolver (SparseSolverId.CuDss);
         cudss.analyze (M, 6, GT, Rg, typeM);
         cudss.factor (M, 6, GT, Rg);
         Status sC = cudss.solve (velC, lamC, bm, bg);
         cudss.dispose();
         if (sC != Status.SOLVED) {
            throw new TestException (
               "cuDSS solve failed for typeM=" + typeM + ": " + sC);
         }

         requireClose ("vel (typeM=" + typeM + ")", velC, velP, 1e-8);
         requireClose ("lam (typeM=" + typeM + ")", lamC, lamP, 1e-8);
      }

      // Diagonal-M path: M passed as a VectorNd. The diagonal-M analyze
      // hardcodes Matrix.SYMMETRIC inside KKTSolver, so there's no
      // matrix-type sweep here.
      VectorNd Mdiag = new VectorNd (6);
      Mdiag.setRandom();
      Mdiag.absolute();
      {
         VectorNd velP = new VectorNd (6), lamP = new VectorNd (3);
         VectorNd velC = new VectorNd (6), lamC = new VectorNd (3);

         KKTSolver pardiso = new KKTSolver (SparseSolverId.Pardiso);
         pardiso.analyze (Mdiag, 6, GT, Rg);
         pardiso.factor (Mdiag, 6, GT, Rg);
         pardiso.solve (velP, lamP, bm, bg);
         pardiso.dispose();

         KKTSolver cudss = new KKTSolver (SparseSolverId.CuDss);
         cudss.analyze (Mdiag, 6, GT, Rg);
         cudss.factor (Mdiag, 6, GT, Rg);
         cudss.solve (velC, lamC, bm, bg);
         cudss.dispose();

         requireClose ("vel diag", velC, velP, 1e-8);
         requireClose ("lam diag", lamC, lamP, 1e-8);
      }
   }

   // Stage C: refactor with new M values must reuse the cuDSS analysis
   // (CUDSS_PHASE_REFACTORIZATION fast path) and still produce the right
   // answer.
   public void testCuDssRefactor() {
      if (!CuDssSolver.isAvailable()) {
         return;
      }
      RandomGenerator.setSeed (0xABCD);

      MatrixNdBlock Mblk = new MatrixNdBlock (6, 6);
      MatrixNdBlock GTblk = new MatrixNdBlock (6, 3);
      GTblk.setRandom();
      Mblk.setRandom();
      Mblk.mulTranspose (Mblk);
      SparseBlockMatrix M = new SparseBlockMatrix();
      M.addBlock (0, 0, Mblk);
      SparseBlockMatrix GT = new SparseBlockMatrix();
      GT.addBlock (0, 0, GTblk);

      // Deliberately leave Rg = 0 to exercise the indefinite-KKT path
      // where cuDSS LDL is less accurate than PARDISO without iterative
      // refinement. With CUDSS_CONFIG_IR_N_STEPS enabled in the bridge,
      // residual-correction passes should bring cuDSS to PARDISO-comparable
      // precision on this system.
      VectorNd Rg = new VectorNd (3);
      VectorNd bm = new VectorNd (6); bm.setRandom();
      VectorNd bg = new VectorNd (3);
      VectorNd vel = new VectorNd (6);
      VectorNd lam = new VectorNd (3);

      KKTSolver cudss = new KKTSolver (SparseSolverId.CuDss);
      cudss.analyze (M, 6, GT, Rg, Matrix.SYMMETRIC);

      // Three factor/solve cycles with mutated M values -- second and
      // third must take the refactorization path.
      for (int iter = 0; iter < 3; iter++) {
         if (iter > 0) {
            for (int i = 0; i < 6; i++) {
               Mblk.set (i, i, Mblk.get (i, i) + 0.01);
            }
         }
         cudss.factor (M, 6, GT, Rg);
         Status s = cudss.solve (vel, lam, bm, bg);
         if (s != Status.SOLVED) {
            throw new TestException (
               "cuDSS refactor iter " + iter + " failed: " + s);
         }
         // Cross-check against PARDISO at this iteration's M values.
         KKTSolver pardiso = new KKTSolver (SparseSolverId.Pardiso);
         pardiso.analyze (M, 6, GT, Rg, Matrix.SYMMETRIC);
         pardiso.factor (M, 6, GT, Rg);
         VectorNd velP = new VectorNd (6), lamP = new VectorNd (3);
         pardiso.solve (velP, lamP, bm, bg);
         pardiso.dispose();
         requireClose ("refactor iter " + iter + " vel", vel, velP, 1e-8);
         requireClose ("refactor iter " + iter + " lam", lam, lamP, 1e-8);
      }
      cudss.dispose();
   }

   public void testCuDssKktDeviceValues() {
      if (!CuDssSolver.isAvailable()) {
         return;
      }
      boolean oldEnable = KKTSolver.getKktDeviceValuesEnabled();
      KKTSolver.setKktDeviceValuesEnabled (true);
      try {
         RandomGenerator.setSeed (0x8912);
         MatrixNdBlock Mblk = new MatrixNdBlock (6, 6);
         MatrixNdBlock GTblk = new MatrixNdBlock (6, 3);
         GTblk.setRandom();
         Mblk.setRandom();
         Mblk.mulTranspose (Mblk);
         SparseBlockMatrix M = new SparseBlockMatrix();
         M.addBlock (0, 0, Mblk);
         SparseBlockMatrix GT = new SparseBlockMatrix();
         GT.addBlock (0, 0, GTblk);

         VectorNd Rg = new VectorNd (3);
         VectorNd bm = new VectorNd (6);
         VectorNd bg = new VectorNd (3);
         Rg.setRandom();
         Rg.absolute();
         bm.setRandom();
         bg.setRandom();

         KKTSolver pardiso = new KKTSolver (SparseSolverId.Pardiso);
         pardiso.analyze (M, 6, GT, Rg, Matrix.SYMMETRIC);
         pardiso.factor (M, 6, GT, Rg);
         VectorNd velP = new VectorNd (6);
         VectorNd lamP = new VectorNd (3);
         pardiso.solve (velP, lamP, bm, bg);
         pardiso.dispose();

         KKTSolver cudss = new KKTSolver (SparseSolverId.CuDss);
         cudss.analyze (M, 6, GT, Rg, Matrix.SYMMETRIC);
         cudss.factor (M, 6, GT, Rg);
         if (!cudss.lastFactorUsedDeviceValues()) {
            throw new TestException (
               "cuDSS KKT factor did not use device values");
         }
         VectorNd velC = new VectorNd (6);
         VectorNd lamC = new VectorNd (3);
         cudss.solve (velC, lamC, bm, bg);
         cudss.dispose();

         requireClose ("kkt device values vel", velC, velP, 1e-8);
         requireClose ("kkt device values lam", lamC, lamP, 1e-8);
      }
      finally {
         KKTSolver.setKktDeviceValuesEnabled (oldEnable);
      }
   }

   public static void main (String[] args) {
      KKTSolverTest tester = new KKTSolverTest();
      PardisoSolver.printThreadInfo = false;
      try {
         //tester.test();
         //tester.testFromFile ("blockCollide3.txt");
         tester.testFromFile ("MLCPtest.txt");
         tester.testKktMBlockSlotMap();
         tester.testCuDssEquality();
         tester.testCuDssRefactor();
         tester.testCuDssKktDeviceValues();
      }
      catch (Exception e) {
         e.printStackTrace();
         System.exit (1);
      }
      System.out.println ("\nPassed\n");
   }
}
