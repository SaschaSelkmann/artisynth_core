package artisynth.core.gpudebug;

import maspack.matrix.*;
import maspack.properties.PropertyMode;
import artisynth.core.mechmodels.*;
import artisynth.core.femmodels.*;
import artisynth.core.materials.*;

/**
 * Validation harness for the GPU FEM CRS assembly. For a given FemModel3d it
 * assembles the position-Jacobian (stiffness) CRS values two ways and diffs them
 * entrywise:
 *
 *   CPU: FemModel3d.addPosJacobianCrsValues (neighbor-based, the reference)
 *   GPU: FemModel3d.addPosJacobianCrsValueContributions -> the full set of
 *        CuDssSolver.add*DeviceValues kernels (mirroring MechSystemSolver's
 *        direct-CRS device dispatch) -> CuDssSolver.getDeviceValues readback
 *
 * Mismatches are printed with their (row,col) -> (nodeI.cI, nodeJ.cJ) mapping so
 * the offending blocks can be identified. Also prints which GPU contribution
 * categories were active, so you can tell which kernel path a model exercises.
 *
 * Unlike the solve-based equivalence tests in FemModel3dTest, this compares the
 * raw assembled CRS values BEFORE the factor, so a kernel/marshalling bug is
 * pinpointed to its entries instead of surfacing as a velocity diff. Exits
 * nonzero on any mismatch (and zero when cuDSS is unavailable), so it can run
 * in scripts/CI on GPU hosts. Run:
 *   java -cp "$CLASSPATH" artisynth.core.gpudebug.GpuCrsCompare
 */
public class GpuCrsCompare {

   static FemModel3d tetGrid (FemMaterial mat) {
      FemModel3d fem = FemFactory.createTetGrid (null, 1.0, 0.4, 0.4, 2, 1, 1);
      fem.setMaterial (mat);
      fem.setDensity (1000);
      // fix the x-min face so the stiffness has a few constrained nodes too
      for (FemNode3d node : fem.getNodes()) {
         if (node.getRestPosition().x <= -0.5 + 1e-6) {
            node.setDynamic (false);
         }
      }
      return fem;
   }

   static double[] cpuCrs (FemModel3d fem, MechSystem.GpuAssemblyContext ctx,
                           double s) {
      ctx.clearCrsValues();
      fem.addPosJacobianCrsValues (ctx, s);
      return ctx.getCrsValues().clone();
   }

   static double[] gpuCrs (
      MechSystem.GpuAssemblyContext ctx, int n, FemModel3d fem, double s,
      int[] counts) {

      ctx.clearCrsValues();
      fem.addPosJacobianCrsValueContributions (ctx, s);

      counts[0] = ctx.numCrsValueContributions();
      counts[1] = ctx.numScaledDiagonal3Contributions();
      counts[2] = ctx.numScaledBlock3Contributions();
      counts[3] = ctx.numMaterialStiffness3Contributions();
      counts[4] = ctx.numMaterialStiffness3ElementContributions();
      counts[5] = ctx.numLinearElasticStiffness3ElementContributions();
      counts[6] = ctx.numLinearElasticStiffness3ElementGeometryContributions();
      counts[7] = ctx.numDilationalStiffness3ElementContributions();

      maspack.solvers.CuDssSolver cudss = new maspack.solvers.CuDssSolver();
      int[] rowOffs = ctx.getZeroBasedCrsRowOffs();
      double[] dummy = new double[rowOffs[n]];
      cudss.analyze (
         dummy, ctx.getZeroBasedCrsColIdxs(), rowOffs, n, Matrix.INDEFINITE);
      cudss.clearDeviceValues();
      cudss.addDeviceValues (
         ctx.getCrsValueContributionSlots(), ctx.getCrsValueContributions(),
         ctx.numCrsValueContributions(), 1.0);
      cudss.addScaledDiagonal3DeviceValues (
         ctx.getScaledDiagonal3ContributionSlots(),
         ctx.getScaledDiagonal3Contributions(),
         ctx.numScaledDiagonal3Contributions(), 1.0);
      cudss.addScaledBlock3DeviceValues (
         ctx.getScaledBlock3ContributionSlots(),
         ctx.getScaledBlock3Contributions(),
         ctx.getScaledBlock3ContributionScales(),
         ctx.numScaledBlock3Contributions(), 1.0);
      cudss.addMaterialStiffness3DeviceValues (
         ctx.getMaterialStiffness3ContributionSlots(),
         ctx.getMaterialStiffness3Gis(), ctx.getMaterialStiffness3Gjs(),
         ctx.getMaterialStiffness3Ds(), ctx.getMaterialStiffness3Sigmas(),
         ctx.getMaterialStiffness3Dvs(),
         ctx.numMaterialStiffness3Contributions(), 1.0);
      cudss.addMaterialStiffness3ElementDeviceValues (
         ctx.getMaterialStiffness3ElementNodeCounts(),
         ctx.getMaterialStiffness3ElementNodeOffsets(),
         ctx.getMaterialStiffness3ElementPairOffsets(),
         ctx.getMaterialStiffness3ElementIpOffsets(),
         ctx.getMaterialStiffness3ElementGradOffsets(),
         ctx.getMaterialStiffness3ElementPairNodeIdxs(),
         ctx.getMaterialStiffness3ElementBlockSlots(),
         ctx.getMaterialStiffness3ElementNodeDims(),
         ctx.getMaterialStiffness3ElementNodeTransforms(),
         ctx.getMaterialStiffness3ElementGrads(),
         ctx.getMaterialStiffness3ElementDs(),
         ctx.getMaterialStiffness3ElementSigmas(),
         ctx.getMaterialStiffness3ElementDvs(),
         ctx.numMaterialStiffness3ElementContributions(), 1.0);
      cudss.addLinearElasticStiffness3ElementDeviceValues (
         ctx.getLinearElasticStiffness3ElementNodeCounts(),
         ctx.getLinearElasticStiffness3ElementPairOffsets(),
         ctx.getLinearElasticStiffness3ElementIpOffsets(),
         ctx.getLinearElasticStiffness3ElementGradOffsets(),
         ctx.getLinearElasticStiffness3ElementPairNodeIdxs(),
         ctx.getLinearElasticStiffness3ElementBlockSlots(),
         ctx.getLinearElasticStiffness3ElementParams(),
         ctx.getLinearElasticStiffness3ElementGrads(),
         ctx.getLinearElasticStiffness3ElementDvs(),
         ctx.numLinearElasticStiffness3ElementContributions(), 1.0);
      cudss.addLinearElasticStiffness3ElementGeometryDeviceValues (
         ctx.getLinearElasticStiffness3ElementGeometryNodeCounts(),
         ctx.getLinearElasticStiffness3ElementGeometryNodeOffsets(),
         ctx.getLinearElasticStiffness3ElementGeometryPairOffsets(),
         ctx.getLinearElasticStiffness3ElementGeometryIpOffsets(),
         ctx.getLinearElasticStiffness3ElementGeometryNaturalGradOffsets(),
         ctx.getLinearElasticStiffness3ElementGeometryPairNodeIdxs(),
         ctx.getLinearElasticStiffness3ElementGeometryBlockSlots(),
         ctx.getLinearElasticStiffness3ElementGeometryNodeDims(),
         ctx.getLinearElasticStiffness3ElementGeometryNodeTransforms(),
         ctx.getLinearElasticStiffness3ElementGeometryParams(),
         ctx.getLinearElasticStiffness3ElementGeometryNodePositions(),
         ctx.getLinearElasticStiffness3ElementGeometryNaturalGrads(),
         ctx.getLinearElasticStiffness3ElementGeometryIpWeights(),
         ctx.numLinearElasticStiffness3ElementGeometryContributions(), 1.0);
      cudss.addDilationalStiffness3ElementDeviceValues (
         ctx.getDilationalStiffness3ElementNodeCounts(),
         ctx.getDilationalStiffness3ElementNodeOffsets(),
         ctx.getDilationalStiffness3ElementPressureCounts(),
         ctx.getDilationalStiffness3ElementPairOffsets(),
         ctx.getDilationalStiffness3ElementConstraintOffsets(),
         ctx.getDilationalStiffness3ElementRinvOffsets(),
         ctx.getDilationalStiffness3ElementPairNodeIdxs(),
         ctx.getDilationalStiffness3ElementBlockSlots(),
         ctx.getDilationalStiffness3ElementNodeDims(),
         ctx.getDilationalStiffness3ElementNodeTransforms(),
         ctx.getDilationalStiffness3ElementConstraints(),
         ctx.getDilationalStiffness3ElementRinvs(),
         ctx.numDilationalStiffness3ElementContributions(), 1.0);
      double[] gpu = new double[rowOffs[n]];
      cudss.getDeviceValues (gpu);
      cudss.dispose();
      return gpu;
   }

   static boolean compare (String label, FemModel3d fem) {
      MechModel mech = new MechModel();
      mech.addModel (fem);
      SparseNumberedBlockMatrix M = new SparseNumberedBlockMatrix();
      mech.buildSolveMatrix (M);
      SparseNumberedBlockMatrix.CrsBlockSlotMap slotMap =
         M.createCrsBlockSlotMap (Matrix.Partition.Full);
      MechSystem.GpuAssemblyContext ctx =
         new MechSystem.GpuAssemblyContext (
            M, slotMap, mech.getStructureVersion());

      double s = 1.0;
      double[] cpu = cpuCrs (fem, ctx, s);
      int n = ctx.getZeroBasedCrsRowOffs().length - 1;
      int[] counts = new int[8];
      double[] gpu = gpuCrs (ctx, n, fem, s, counts);

      int[] rowOffs = ctx.getZeroBasedCrsRowOffs();
      int[] colIdxs = ctx.getZeroBasedCrsColIdxs();
      int[] rowOf = new int[cpu.length];
      for (int r=0; r<n; r++) {
         for (int k=rowOffs[r]; k<rowOffs[r+1]; k++) {
            rowOf[k] = r;
         }
      }
      double maxDiff = 0;
      int nMismatch = 0;
      for (int i=0; i<cpu.length; i++) {
         double d = Math.abs (cpu[i]-gpu[i]);
         maxDiff = Math.max (maxDiff, d);
         if (d > 1e-6*Math.max (1.0, Math.abs (cpu[i]))) {
            nMismatch++;
            if (nMismatch <= 12) {
               int row = rowOf[i], col = colIdxs[i];
               System.out.printf (
                  "  crs[%3d] row=%2d col=%2d (n%d.%d,n%d.%d) cpu=% .4e gpu=% .4e%n",
                  i, row, col, row/3, row%3, col/3, col%3, cpu[i], gpu[i]);
            }
         }
      }
      System.out.printf (
         "%-22s nnz=%d  contrib{crs=%d diag3=%d block3=%d mat3=%d matElem3=%d "+
         "linElem3=%d linGeom3=%d dil3=%d}%n",
         label, cpu.length, counts[0], counts[1], counts[2], counts[3],
         counts[4], counts[5], counts[6], counts[7]);
      System.out.printf (
         "  -> maxDiff=%.3e  mismatches=%d  %s%n%n",
         maxDiff, nMismatch, (nMismatch==0 ? "OK" : "<<< FAIL"));
      return nMismatch == 0;
   }

   public static void main (String[] args) {
      if (!maspack.solvers.CuDssSolver.isAvailable()) {
         System.out.println ("cuDSS unavailable; cannot run GPU CRS compare");
         return;
      }
      boolean ok = true;

      LinearMaterial lin = new LinearMaterial (50000, 0.33, /*corotated=*/false);
      lin.setCorotatedMode (PropertyMode.Explicit);
      ok &= compare ("Linear(non-corot)", tetGrid (lin));

      LinearMaterial linCo = new LinearMaterial (50000, 0.33, /*corotated=*/true);
      linCo.setCorotatedMode (PropertyMode.Explicit);
      ok &= compare ("Linear(corot)", tetGrid (linCo));

      ok &= compare ("MooneyRivlin(incomp)",
         tetGrid (new MooneyRivlinMaterial (1000, 200, 0, 0, 0, 5e6)));

      ok &= compare ("NeoHookean", tetGrid (new NeoHookeanMaterial (50000, 0.33)));

      if (!ok) {
         System.out.println ("GpuCrsCompare FAILED");
         System.exit (1);
      }
      System.out.println ("GpuCrsCompare passed");
   }
}
