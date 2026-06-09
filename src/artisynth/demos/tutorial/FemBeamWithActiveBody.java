package artisynth.demos.tutorial;

import java.awt.Color;

import artisynth.core.femmodels.FemFactory;
import artisynth.core.femmodels.FemModel3d;
import artisynth.core.femmodels.FemNode3d;
import artisynth.core.materials.LinearMaterial;
import artisynth.core.mechmodels.MechModel;
import artisynth.core.mechmodels.MechSystemSolver;
import artisynth.core.mechmodels.PointFrameAttachment;
import artisynth.core.mechmodels.RigidBody;
import artisynth.core.workspace.RootModel;
import maspack.matrix.Point3d;
import maspack.matrix.RigidTransform3d;
import maspack.properties.PropertyMode;
import maspack.render.RenderProps;
import maspack.solvers.SparseSolverId;

/**
 * Minimal model that isolates the FEM-coupled-to-ACTIVE-rigid-body case for the
 * GPU attachment-reduction work.
 *
 * <p>A non-corotated linear-elastic FEM beam is fixed at its x-min face. A
 * dynamic ("active") rigid box is placed at the x-max (free) end and attached to
 * the tip FEM nodes with {@link PointFrameAttachment}. Because the attachment
 * master (the box) is active, the implicit solve must reduce the slave-node
 * stiffness onto the box's 6 DOF (G^T K G), which currently forces a CPU
 * assembly fallback (a FEM node with an attachment has "indirect neighbours", so
 * {@code FemModel3d.hasIndirectGpuAssemblyContributions()} returns true and the
 * GPU linear-elastic kernel is gated off). This model is the validation vehicle
 * for moving that reduction onto the GPU.
 *
 * <p>Configured for the GPU device-assembly path: cuDSS solver, constrained
 * backward Euler, non-corotated {@link LinearMaterial} with explicit corotated
 * mode. No muscles, joints, contact, or wrapping, so the attachment reduction is
 * the ONLY thing keeping the assembly off the GPU.
 *
 * <pre>
 *   artisynth -noGui -model artisynth.demos.tutorial.FemBeamWithActiveBody \
 *             -playFor 0.05 -exitOnBreak
 * </pre>
 */
public class FemBeamWithActiveBody extends RootModel {

   public void build (String[] args) {
      MechModel mech = new MechModel ("mech");
      addModel (mech);

      // Non-corotated linear FEM beam (the GPU linear-elastic kernel config).
      FemModel3d fem = FemFactory.createTetGrid (null, 1.0, 0.4, 0.4, 8, 4, 4);
      LinearMaterial mat = new LinearMaterial (50000, 0.33, /*corotated=*/false);
      mat.setCorotatedMode (PropertyMode.Explicit);
      fem.setMaterial (mat);
      fem.setDensity (1000);
      fem.setStiffnessDamping (0.1);
      fem.setParticleDamping (0.5);
      fem.setName ("beam");
      mech.addModel (fem);
      RenderProps.setFaceColor (fem, new Color (0.71f, 0.71f, 0.85f));

      // Fix the x-min face.
      double xMin = Double.POSITIVE_INFINITY;
      double xMax = Double.NEGATIVE_INFINITY;
      for (FemNode3d n : fem.getNodes()) {
         xMin = Math.min (xMin, n.getRestPosition().x);
         xMax = Math.max (xMax, n.getRestPosition().x);
      }
      for (FemNode3d n : fem.getNodes()) {
         if (n.getRestPosition().x <= xMin + 1e-6) {
            n.setDynamic (false);
         }
      }

      // Active rigid box at the free (x-max) end, attached to the tip nodes.
      RigidBody box = RigidBody.createBox ("endBox", 0.1, 0.45, 0.45, 1000);
      box.setPose (new RigidTransform3d (xMax + 0.05, 0, 0));
      box.setDynamic (true);                  // ACTIVE master
      mech.addRigidBody (box);

      int nattached = 0;
      for (FemNode3d n : fem.getNodes()) {
         if (n.getRestPosition().x >= xMax - 1e-6) {
            mech.addAttachment (new PointFrameAttachment (box, n));
            RenderProps.setSphericalPoints (n, 0.012, Color.ORANGE);
            nattached++;
         }
      }
      RenderProps.setFaceColor (box, new Color (0.9f, 0.5f, 0.2f));
      System.out.println (
         "FemBeamWithActiveBody: attached "+nattached+" tip nodes to active box");

      // GPU device-assembly path.
      mech.setMatrixSolver (SparseSolverId.CuDss);
      mech.setIntegrator (MechSystemSolver.Integrator.ConstrainedBackwardEuler);
      mech.setGravity (0, 0, -9.8);
   }
}
