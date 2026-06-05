/**
 * Copyright (c) 2014, by the Authors: John E Lloyd (UBC)
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package maspack.matrix;

import java.util.Arrays;
import java.util.ArrayList;

import maspack.util.TestException;

/**
 * A version of SparseBlockMatrix that allows blocks to be accessed by number
 */
public class SparseNumberedBlockMatrix extends SparseBlockMatrix {

   /**
    * Maps numbered matrix block entries to their corresponding positions in a
    * CRS values array. Entries within each block are ordered in the same
    * row-major structural order used by {@link MatrixBlock#getBlockCRSValues}.
    */
   public static class CrsBlockSlotMap {
      private Partition myPart;
      private int myNumRows;
      private int myNumCols;
      private int myNumVals;
      private int[] myBlockOffs;
      private int[] mySlots;

      CrsBlockSlotMap (
         Partition part, int numRows, int numCols, int numVals,
         int[] blockOffs, int[] slots) {
         myPart = part;
         myNumRows = numRows;
         myNumCols = numCols;
         myNumVals = numVals;
         myBlockOffs = blockOffs;
         mySlots = slots;
      }

      public Partition getPartition() {
         return myPart;
      }

      public int rowSize() {
         return myNumRows;
      }

      public int colSize() {
         return myNumCols;
      }

      public int numVals() {
         return myNumVals;
      }

      public int numMappedBlockValues() {
         return mySlots.length;
      }

      public int getBlockSlotOffset (int blockNumber) {
         checkBlockNumber (blockNumber);
         return myBlockOffs[blockNumber];
      }

      public int numBlockSlots (int blockNumber) {
         checkBlockNumber (blockNumber);
         return myBlockOffs[blockNumber+1] - myBlockOffs[blockNumber];
      }

      public boolean hasBlockSlots (int blockNumber) {
         return (blockNumber >= 0 &&
                 blockNumber + 1 < myBlockOffs.length &&
                 myBlockOffs[blockNumber+1] > myBlockOffs[blockNumber]);
      }

      public int getBlockSlot (int blockNumber, int localSlotIdx) {
         checkBlockNumber (blockNumber);
         int off = myBlockOffs[blockNumber];
         int nextOff = myBlockOffs[blockNumber+1];
         if (localSlotIdx < 0 || off + localSlotIdx >= nextOff) {
            throw new IllegalArgumentException (
               "local slot index "+localSlotIdx+" out of range");
         }
         return mySlots[off + localSlotIdx];
      }

      public int getBlockValueSlot (MatrixBlock blk, int i, int j) {
         Partition blkPart = blockPartition (blk);
         if (blkPart == Partition.None || !isStoredEntry (blk, i, j, blkPart)) {
            return -1;
         }
         int localSlotIdx = localSlotIndex (blk, i, j, blkPart);
         return getBlockSlot (blk.getBlockNumber(), localSlotIdx);
      }

      public int[] getSlots() {
         return Arrays.copyOf (mySlots, mySlots.length);
      }

      private void checkBlockNumber (int blockNumber) {
         if (blockNumber < 0 || blockNumber + 1 >= myBlockOffs.length) {
            throw new IllegalArgumentException (
               "block number "+blockNumber+" out of range");
         }
      }

      private Partition blockPartition (MatrixBlock blk) {
         int bi = blk.getBlockRow();
         int bj = blk.getBlockCol();
         if (myPart == Partition.Full) {
            return Partition.Full;
         }
         else if (myPart == Partition.UpperTriangular) {
            if (bj == bi) {
               return Partition.UpperTriangular;
            }
            else if (bj > bi) {
               return Partition.Full;
            }
            else {
               return Partition.None;
            }
         }
         else {
            return Partition.None;
         }
      }
   }

   // code to implement the number map
                                               
   protected MatrixBlock[] myNumberMap;
   // next number to be assigned when all free numbers are used up
   protected int myMaxNumber; 
   protected int[] myFreeNumbers;
   protected int myNumFreeNumbers;
   protected int myInitialCapacity = -1;

   private static boolean isStoredEntry (
      MatrixBlock blk, int i, int j, Partition part) {
      if (i < 0 || i >= blk.rowSize() || j < 0 || j >= blk.colSize()) {
         throw new IllegalArgumentException (
            "block entry ("+i+","+j+") out of range");
      }
      if (part == Partition.UpperTriangular && j < i) {
         return false;
      }
      else if (part != Partition.Full && part != Partition.UpperTriangular) {
         return false;
      }
      return blk.valueIsNonZero (i, j);
   }

   private static int numStoredEntries (MatrixBlock blk, Partition part) {
      int num = 0;
      if (part == Partition.None) {
         return 0;
      }
      for (int i=0; i<blk.rowSize(); i++) {
         for (int j=0; j<blk.colSize(); j++) {
            if (isStoredEntry (blk, i, j, part)) {
               num++;
            }
         }
      }
      return num;
   }

   private static int localSlotIndex (
      MatrixBlock blk, int i, int j, Partition part) {
      int idx = 0;
      for (int ii=0; ii<blk.rowSize(); ii++) {
         for (int jj=0; jj<blk.colSize(); jj++) {
            if (isStoredEntry (blk, ii, jj, part)) {
               if (ii == i && jj == j) {
                  return idx;
               }
               idx++;
            }
         }
      }
      return -1;
   }

   private static Partition blockPartition (
      MatrixBlock blk, Partition part) {
      int bi = blk.getBlockRow();
      int bj = blk.getBlockCol();
      if (part == Partition.Full) {
         return Partition.Full;
      }
      else if (part == Partition.UpperTriangular) {
         if (bj == bi) {
            return Partition.UpperTriangular;
         }
         else if (bj > bi) {
            return Partition.Full;
         }
         else {
            return Partition.None;
         }
      }
      throw new UnsupportedOperationException (
         "Matrix partition "+part+" not supported");
   }

   private int allocNumber() {
      int num;
      if (myNumFreeNumbers > 0) {
         num = myFreeNumbers[--myNumFreeNumbers];
      }
      else {
         num = myMaxNumber;
      }
      if (num >= myMaxNumber) {
         myMaxNumber = num + 1;
      }
      if (num >= myNumberMap.length) {
         myNumberMap = Arrays.copyOf (myNumberMap, Math.max(2*num, num+1));
      }
      return num;
   }

   private void freeNumber (int num) {
      if (myNumFreeNumbers >= myFreeNumbers.length) {
         // grow the list of free numbers
         myFreeNumbers = Arrays.copyOf (
            myFreeNumbers, (3*myFreeNumbers.length)/2+1);
      }
      myFreeNumbers[myNumFreeNumbers++] = num;
      // if (num == myMaxNumber - 1) {
      //    int k = num - 1;
      //    while (k >= 0 && myNumberMap[k] == null) {
      //       k--;
      //    }
      //    myMaxNumber = k + 1;
      // }
   }

   public SparseNumberedBlockMatrix() {
      this (new int[0], -1);
   }

   public SparseNumberedBlockMatrix (int[] rowColSizes) {
      this (rowColSizes, rowColSizes, -1);
   }

   public SparseNumberedBlockMatrix (int[] rowSizes, int[] colSizes) {
      this (rowSizes, colSizes, -1);
   }

   public SparseNumberedBlockMatrix (int[] rowColSizes, int initialCapacity) {
      this (rowColSizes, rowColSizes, initialCapacity);
   }

   protected void initRowColSizes (int[] rowSizes, int[] colSizes) {
      super.initRowColSizes (rowSizes, colSizes);

      if (myInitialCapacity < 0) {
         int initialCapacity =
            Math.max(16, Math.max (myNumBlockRows, myNumBlockCols));
         myNumberMap = new MatrixBlock[initialCapacity];
      }
   }

   public SparseNumberedBlockMatrix (
      int[] rowSizes, int[] colSizes, int initialCapacity) {
      super (rowSizes, colSizes);
      initializeFreeList (initialCapacity);
   }

   void initializeFreeList (int initialCapacity) {
      myInitialCapacity = initialCapacity;
      if (initialCapacity < 0) {
         initialCapacity =
            Math.max(16, Math.max (myNumBlockRows, myNumBlockCols));
      }
      myNumberMap = new MatrixBlock[initialCapacity];
      myFreeNumbers = new int[0];
      myMaxNumber = 0;
      myNumFreeNumbers = 0;
   }

   void copyFreeList (SparseNumberedBlockMatrix S) {
      myNumFreeNumbers = S.myNumFreeNumbers;
      myInitialCapacity = S.myInitialCapacity;
      myMaxNumber = S.myMaxNumber;
      myFreeNumbers = Arrays.copyOf (S.myFreeNumbers, S.myNumFreeNumbers);
   }

   public int addBlock (int bi, int bj, MatrixBlock blk) {
      MatrixBlock oldBlk = doAddBlock (bi, bj, blk);
      int num;
      if (blk.getBlockNumber() != -1) {
         num = blk.getBlockNumber();
      }
      else if (oldBlk == null) {
         num = allocNumber();
      }
      else {
         // just reuse the old number
         num = oldBlk.getBlockNumber();
      }
      blk.setBlockNumber (num);
      myNumberMap[num] = blk;
      return num;
   }

   public boolean removeBlock (MatrixBlock oldBlk) {
      if (doRemoveBlock (oldBlk)) {
         disposeBlock (oldBlk);
         return true;
      }
      else {
         return false;
      }
   }

   protected void disposeBlock (MatrixBlock blk) {
      int num = blk.getBlockNumber();
      myNumberMap[num] = null;
      freeNumber (num);
      blk.setBlockNumber (-1);
   }

   private void clearNumberMap() {
      for (int i=0; i<myNumberMap.length; i++) {
         myNumberMap[i] = null;
      }
      myMaxNumber = 0;
      myNumFreeNumbers = 0;
   }

   public void removeAllBlocks() {
      super.removeAllBlocks();
      clearNumberMap();
   }

   public void removeAllRows() {
      super.removeAllRows();
      clearNumberMap();
   }

   public void removeAllCols() {
      super.removeAllCols();
      clearNumberMap();
   }

   public MatrixBlock getBlockByNumber (int num) {
      return myNumberMap[num];
   }

   /**
    * Builds a mapping from numbered matrix blocks to CRS value-array slots for
    * a principal block-aligned sub-matrix. The CRS slot numbering is 0-based,
    * matching the values array returned by {@link #getCRSValues}.
    *
    * @param part matrix partition to map
    * @param numRows number of rows delimiting the sub-matrix
    * @param numCols number of columns delimiting the sub-matrix
    * @return block-to-CRS slot map
    */
   public CrsBlockSlotMap createCrsBlockSlotMap (
      Partition part, int numRows, int numCols) {

      if (part != Partition.Full && part != Partition.UpperTriangular) {
         throw new UnsupportedOperationException (
            "Matrix partition "+part+" not supported");
      }
      int numBlkRows = getAlignedBlockRow (numRows);
      int numBlkCols = getAlignedBlockCol (numCols);
      if (numRows > rowSize() || numCols > colSize()) {
         throw new IllegalArgumentException (
            "submatrix exceeds "+getSize()+" matrix size");
      }
      if (numBlkRows == -1 || numBlkCols == -1) {
         throw new IllegalArgumentException (
            "submatrix is not block aligned");
      }

      int numVals = numNonZeroVals (part, numRows, numCols);
      int[] colIdxs = new int[numVals];
      int[] rowOffs = new int[numRows+1];
      getCRSIndices (colIdxs, rowOffs, part, numRows, numCols);
      for (int i=0; i<rowOffs.length; i++) {
         rowOffs[i]--;
      }

      int[] blockOffs = new int[myMaxNumber+1];
      for (int bi=0; bi<numBlkRows; bi++) {
         for (MatrixBlock blk=myRows[bi].myHead;
              blk != null && blk.getBlockCol() < numBlkCols;
              blk=blk.next()) {
            int num = blk.getBlockNumber();
            blockOffs[num+1] = numStoredEntries (
               blk, blockPartition (blk, part));
         }
      }
      for (int i=0; i<myMaxNumber; i++) {
         blockOffs[i+1] += blockOffs[i];
      }

      int[] nextBlockOffs = Arrays.copyOf (blockOffs, blockOffs.length);
      int[] nextRowOffs = Arrays.copyOf (rowOffs, rowOffs.length);
      int[] slots = new int[blockOffs[myMaxNumber]];

      for (int bi=0; bi<numBlkRows; bi++) {
         int rowBase = myRowOffsets[bi];
         for (MatrixBlock blk=myRows[bi].myHead;
              blk != null && blk.getBlockCol() < numBlkCols;
              blk=blk.next()) {
            Partition blkPart = blockPartition (blk, part);
            if (blkPart == Partition.None) {
               continue;
            }
            int num = blk.getBlockNumber();
            for (int i=0; i<blk.rowSize(); i++) {
               for (int j=0; j<blk.colSize(); j++) {
                  if (isStoredEntry (blk, i, j, blkPart)) {
                     slots[nextBlockOffs[num]++] = nextRowOffs[rowBase+i]++;
                  }
               }
            }
         }
      }

      return new CrsBlockSlotMap (
         part, numRows, numCols, numVals, blockOffs, slots);
   }

   public CrsBlockSlotMap createCrsBlockSlotMap (Partition part) {
      return createCrsBlockSlotMap (part, rowSize(), colSize());
   }

   String getBlockStr (MatrixBlock blk) {
      if (blk == null) {
         return "null";
      }
      else {
         return ("("+blk.getBlockRow()+","+blk.getBlockCol()+
                 "), number "+blk.getBlockNumber()+", hash "+blk.hashCode());
      }
   }

   public void checkConsistency() {
      super.checkConsistency();

      int numBlksChk = 0;
      for (int bi=0; bi<myNumBlockRows; bi++) {
         for (MatrixBlock blk=myRows[bi].myHead; blk!=null; blk = blk.next()) {
            if (myNumberMap[blk.getBlockNumber()] != blk) {
               throw new TestException (
                  "blk" + getBlockStr(blk) +
                  ": inconsistent map entry: " +
                  getBlockStr(myNumberMap[blk.getBlockNumber()]));
            }
            numBlksChk++;
         }
      }
      int numBlks = 0;
      int numFree = 0;
      for (int i=0; i<myNumberMap.length; i++) {
         MatrixBlock blk = myNumberMap[i];
         // see if i is a free number 
         boolean iIsFree = false;
         for (int k=0; k<myNumFreeNumbers; k++) {
            if (myFreeNumbers[k] == i) {
               iIsFree = true;
               break;
            }
         } 
         if (blk != null) {
            if (iIsFree) {
               throw new TestException (
                  "number map index "+i+" is free but contains a block");
            }
            if (blk.getBlockNumber() != i) {
               throw new TestException (
                  "inconsistent number map entry for blk "+blk.getBlockNumber());
            }
            numBlks++;
         }
         else {
            // see if i is a free number 
            if (i < myMaxNumber) {
               if (!iIsFree) {
                  throw new TestException (
                     "number "+i+" is not on the free list");
               }
               numFree++;
            }
            else {
               if (iIsFree) {
                  throw new TestException (
                     "number "+i+" is on the free list");
               }
            }
         }
      }
      if (numBlks != numBlksChk) {
         throw new TestException (
            "map has "+numBlks+" but matrix has only "+numBlksChk);
      }
      if (numBlks+numFree != myMaxNumber) {
         throw new TestException (
            "myMaxNumber="+myMaxNumber+", expecting "+(numBlks+numFree));
      }

   }

   public void set (SparseBlockMatrix S) {
      if (S instanceof SparseNumberedBlockMatrix) {
         SparseNumberedBlockMatrix SN = (SparseNumberedBlockMatrix)S;
         copyFreeList (SN);
         myNumberMap = new MatrixBlock[myNumberMap.length];
         super.set (SN);
      }
      else {
         initializeFreeList (-1);
         super.set (S);
      }     
   }

   /**
    * Creates a clone of this NumberedSparseBlockMatrix, along with clones of
    * all the associated MatrixBlocks.
    */
   public SparseNumberedBlockMatrix clone() {
      SparseNumberedBlockMatrix M = (SparseNumberedBlockMatrix)super.clone();

      M.myNumberMap = new MatrixBlock[myNumberMap.length];
      for (int bi=0; bi<myNumBlockRows; bi++) {
         for (MatrixBlock blk=M.myRows[bi].myHead; blk!=null; blk = blk.next()) {
            M.myNumberMap[blk.getBlockNumber()] = blk;
         }
      }
      M.myFreeNumbers = Arrays.copyOf (myFreeNumbers, myNumFreeNumbers);
      return M;
   }
}
