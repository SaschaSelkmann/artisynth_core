package artisynth.demos.gpu;

import java.awt.Color;
import java.io.IOException;

import artisynth.demos.tutorial.ToyMuscleArmFEM;
import artisynth.core.femmodels.FemModel3d;
import artisynth.core.femmodels.FemNode3d;
import artisynth.core.materials.LinearMaterial;
import artisynth.core.mechmodels.MechSystemSolver;
import artisynth.core.mechmodels.PointFrameAttachment;
import artisynth.core.mechmodels.RigidBody;
import maspack.matrix.Point3d;
import maspack.matrix.RigidTransform3d;
import maspack.properties.PropertyMode;
import maspack.render.RenderProps;
import maspack.solvers.SparseSolverId;
import maspack.util.PathFinder;

/**
 * GPU-assembly test variant of {@link ToyMuscleArmFEM}.
 *
 * <p>This subclass keeps the two-link FEM muscle arm but:
 * <ul>
 *   <li>switches the FEM material to a NON-corotated {@link LinearMaterial} with
 *       {@code CorotatedMode = Explicit} (the configuration the GPU linear-elastic
 *       kernels support — corotated is an inherited property that is otherwise
 *       reset to {@code true} when the FEM is attached);</li>
 *   <li>selects the cuDSS matrix solver and the constrained backward-Euler
 *       integrator (the default-on GPU device-assembly path);</li>
 *   <li>attaches a dynamic ("active") rigid-body END-EFFECTOR to the tip of
 *       {@code link1Fem} via {@link PointFrameAttachment}. This is the moving
 *       rigid-body-coupled-to-FEM case whose Jacobian reduction (G^T K G onto the
 *       active master DOFs) is the target of the GPU attachment-reduction work.</li>
 * </ul>
 *
 * <p>Launch headless, e.g.:
 * <pre>
 *   artisynth -noGui -model artisynth.demos.gpu.ToyMuscleArmFEMGpu \
 *             -playFor 0.05 -exitOnBreak
 * </pre>
 * Add {@code -Dartisynth.gpuAssembly.status=true} to see per-solve routing.
 */
public class ToyMuscleArmFEMGpu extends ToyMuscleArmFEM {

   // End-effector parameters.
   protected double endEffWidth = 0.12;
   protected double endEffDensity = 1000.0;
   // tip of link1 is near z = 1.2 (link1 centered at z = 0.9, length 0.6)
   protected double endEffZ = 1.26;
   // attach link1 FEM nodes whose z is within this distance of the tip face
   protected double tipAttachZTol = 0.03;

   public ToyMuscleArmFEMGpu() {
      // ToyMuscleArmFEM resolves its mesh data directory from the runtime
      // class (PathFinder.getSourceRelativePath(this, "data/")); since this
      // subclass lives in a different package, point geodir back at the base
      // class's data directory (artisynth/demos/tutorial/data).
      geodir = PathFinder.getSourceRelativePath (ToyMuscleArmFEM.class, "data/");
   }

   @Override
   public void build (String[] args) throws IOException {
      super.build (args);

      // Reconfigure the FEM material for the GPU linear-elastic kernel path:
      // non-corotated LinearMaterial with an explicit corotated mode so the flag
      // is not reset to true when the FEM is (re)attached to the model.
      retargetFemMaterial (myLink0Fem, link0YoungsModulus, link0PoissonsRatio);
      retargetFemMaterial (myLink1Fem, link1YoungsModulus, link1PoissonsRatio);

      // Route through the GPU device-assembly path.
      myMech.setMatrixSolver (SparseSolverId.CuDss);
      myMech.setIntegrator (
         MechSystemSolver.Integrator.ConstrainedBackwardEuler);

      // Add a dynamic rigid-body end-effector and attach it to the link1 tip.
      addEndEffector();
   }

   protected void retargetFemMaterial (
      FemModel3d fem, double youngsModulus, double poissonsRatio) {
      LinearMaterial mat =
         new LinearMaterial (youngsModulus, poissonsRatio, /*corotated=*/false);
      fem.setMaterial (mat);
      // Pin corotated=false AFTER the material is hosted by the FEM, so the
      // explicit mode survives property inheritance from the model hierarchy
      // (setting the mode before setMaterial gets reset to Inherited on attach).
      mat.setCorotated (false);
      mat.setCorotatedMode (PropertyMode.Explicit);
   }

   protected void addEndEffector() {
      RigidBody endEff = RigidBody.createBox (
         "endEffector",
         endEffWidth, endEffWidth, endEffWidth, endEffDensity);
      endEff.setPose (new RigidTransform3d (0.0, 0.0, endEffZ + endEffWidth/2));
      // active (dynamic) so it is a true moving master, not a fixed boundary
      endEff.setDynamic (true);
      myMech.addRigidBody (endEff);

      int nattached = 0;
      double zMax = Double.NEGATIVE_INFINITY;
      for (FemNode3d node : myLink1Fem.getNodes()) {
         zMax = Math.max (zMax, node.getPosition().z);
      }
      for (FemNode3d node : myLink1Fem.getNodes()) {
         Point3d p = node.getPosition();
         if (zMax - p.z <= tipAttachZTol) {
            myMech.addAttachment (new PointFrameAttachment (endEff, node));
            RenderProps.setSphericalPoints (node, 0.008, Color.ORANGE);
            nattached++;
         }
      }
      if (nattached == 0) {
         throw new IllegalArgumentException (
            "end-effector: no link1 tip nodes found to attach");
      }
      RenderProps.setFaceColor (endEff, new Color (0.9f, 0.5f, 0.2f));
      System.out.println (
         "ToyMuscleArmFEMGpu: end-effector attached to "+nattached+
         " link1 tip nodes");
   }
}
