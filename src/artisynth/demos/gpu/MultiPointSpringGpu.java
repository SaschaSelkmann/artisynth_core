package artisynth.demos.gpu;

import artisynth.core.mechmodels.MechModel;
import artisynth.core.mechmodels.MechSystemSolver.Integrator;
import artisynth.core.mechmodels.MultiPointSpring;
import artisynth.core.mechmodels.Particle;
import artisynth.core.mechmodels.RigidBody;
import artisynth.core.workspace.RootModel;
import maspack.matrix.AxisAngle;
import maspack.matrix.RigidTransform3d;
import maspack.matrix.Vector3d;
import maspack.solvers.SparseSolverId;

/**
 * Minimal GPU-assembly test for {@link MultiPointSpring}: a chain of FREE
 * particles strung on a single multipoint spring, with one fixed anchor, under
 * gravity. Using free particles (not frame markers on a body) keeps the only
 * GPU contributions the particle masses (diag3) plus the multipoint-spring
 * Jacobian, so it isolates the spring's GPU CRS hooks
 * (assemble{Pos,Vel}JacobianCrsValue*). With {@code wrap=true} the strand also
 * wraps an active rigid cylinder, exercising the 6x6 wrappable-frame blocks.
 *
 * <p>cuDSS + ConstrainedBackwardEuler. Launch headless, e.g.:
 * <pre>
 *   artisynth -noGui -model artisynth.demos.gpu.MultiPointSpringGpu \
 *             -playFor 0.05 -exitOnBreak
 * </pre>
 */
public class MultiPointSpringGpu extends RootModel {

   protected boolean wrap = false;

   public void build (String[] args) {
      for (String a : args) {
         if (a.equals ("-wrap")) {
            wrap = true;
         }
      }
      MechModel mech = new MechModel ("mech");
      mech.setGravity (0, 0, -9.8);
      mech.setMatrixSolver (SparseSolverId.CuDss);
      mech.setIntegrator (Integrator.ConstrainedBackwardEuler);

      Particle p0 = new Particle (1.0, 0.0, 0.0, 0.0);
      p0.setDynamic (false);          // fixed anchor
      mech.addParticle (p0);
      Particle p1 = new Particle (1.0, 0.2, 0.0, 0.0);
      mech.addParticle (p1);
      Particle p2 = new Particle (1.0, 0.8, 0.0, 0.0);
      mech.addParticle (p2);
      Particle p3 = new Particle (1.0, 1.0, 0.0, 0.0);
      mech.addParticle (p3);

      MultiPointSpring spring = new MultiPointSpring (100.0, 1.0, 0.0);
      spring.addPoint (p0);
      spring.addPoint (p1);
      if (wrap) {
         // mark the p1->p2 segment wrappable BEFORE closing it with p2
         spring.setSegmentWrappable (50);
      }
      spring.addPoint (p2);
      spring.addPoint (p3);

      if (wrap) {
         // an ACTIVE rigid cylinder placed across the straight p1->p2 segment so
         // the strand must wrap over it; its 6x6 frame block is a real active DOF,
         // so the wrappable coupling stays in the device M block. Axis along y
         // (rotate the default z-axis cylinder 90 deg about x).
         RigidBody cyl = RigidBody.createCylinder (
            "cyl", 0.25, 0.6, 50.0, 24);
         cyl.setPose (new RigidTransform3d (
            new Vector3d (0.5, 0.0, -0.2),
            new AxisAngle (1, 0, 0, Math.PI/2)));
         mech.addRigidBody (cyl);
         spring.addWrappable (cyl);
         spring.updateWrapSegments();
         System.out.println (
            "MultiPointSpringGpu: wrappableSegs="+spring.hasWrappableSegments()+
            " activeLength="+spring.getActiveLength()+" (straight=1.0)");
      }

      mech.addMultiPointSpring (spring);
      addModel (mech);
   }
}
