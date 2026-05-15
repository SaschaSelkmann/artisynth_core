/**
 * Copyright (c) 2026, by the Authors: ArtiSynth Team
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package artisynth.demos.fem;

import java.awt.Color;
import java.util.LinkedList;

import artisynth.core.femmodels.FemFactory;
import artisynth.core.femmodels.FemModel.SurfaceRender;
import artisynth.core.femmodels.FemModel3d;
import artisynth.core.femmodels.FemNode3d;
import artisynth.core.gui.ControlPanel;
import artisynth.core.mechmodels.HingeJoint;
import artisynth.core.mechmodels.MechModel;
import artisynth.core.mechmodels.MechSystemSolver.Integrator;
import artisynth.core.mechmodels.RigidBody;
import artisynth.core.workspace.RootModel;
import maspack.geometry.MeshFactory;
import maspack.matrix.RigidTransform3d;
import maspack.render.RenderProps;
import maspack.render.Renderer;
import maspack.spatialmotion.SpatialInertia;

/**
 * A finer-meshed variant of {@link ArticulatedFem} intended as a benchmark
 * for the constrained-FE solve path. Several deformable FEM modules are
 * connected by hinge joints to rigid-body link blocks; each FEM module is
 * attached to its neighbors via {@code attachPoint} (bilateral
 * constraints). The integrator is forced to
 * {@link Integrator#ConstrainedBackwardEuler} so every step exercises
 * {@code KKTSolver}.
 *
 * <p>Mesh size is configurable via {@code -nlinks}, {@code -nelemsx},
 * {@code -nelemsz}. Defaults are sized so the assembled global matrix is
 * large enough to make the GPU win meaningfully: 4 FEM links, each a
 * 24x10x10 tet grid, gives ~12000 FEM nodes plus 8 rigid bodies plus
 * 4 hinge constraints. The resulting KKT system runs through
 * {@code KKTSolver} with bilateral constraints from the FEM attachments
 * and the joint constraints.
 */
public class ArticulatedFemBig extends RootModel {

   protected MechModel myMechMod;

   private static final double DENSITY = 1000;

   private double myBoxLength = 0.1;
   private double myBoxHeight = 0.3;
   private double myFemLength = 0.6;
   private double myFemHeight = 0.2;

   private int parseInt (String[] args, int i, int defaultValue) {
      if (i + 1 >= args.length) return defaultValue;
      try { return Integer.parseInt (args[i + 1]); }
      catch (NumberFormatException e) { return defaultValue; }
   }

   private RigidBody makeBox() {
      double mass = myBoxLength * myBoxHeight * myBoxHeight * DENSITY;
      RigidBody box = new RigidBody();
      box.setInertia (SpatialInertia.createBoxInertia (
         mass, myBoxLength, myBoxHeight, myBoxHeight));
      box.setMesh (MeshFactory.createBox (
         myBoxLength, myBoxHeight, myBoxHeight), null);
      return box;
   }

   @Override
   public void build (String[] args) {
      int nlinks  = 4;
      int nelemsx = 24;
      int nelemsz = 10;
      for (int i = 0; i < args.length; i++) {
         if      (args[i].equals ("-nlinks"))  nlinks  = parseInt (args, i, nlinks);
         else if (args[i].equals ("-nelemsx")) nelemsx = parseInt (args, i, nelemsx);
         else if (args[i].equals ("-nelemsz")) nelemsz = parseInt (args, i, nelemsz);
      }

      double linkLength = myFemLength + 2 * myBoxLength;
      myMechMod = new MechModel ("mech");

      RigidTransform3d X = new RigidTransform3d();
      RigidBody lastBox = null;
      int totalNodes = 0;

      for (int i = 0; i < nlinks; i++) {
         double linkCenter = linkLength * (-nlinks / 2.0 + i + 0.5);

         LinkedList<FemNode3d> leftNodes  = new LinkedList<FemNode3d>();
         LinkedList<FemNode3d> rightNodes = new LinkedList<FemNode3d>();

         FemModel3d femMod = FemFactory.createTetGrid (
            null, myFemLength, myFemHeight, myFemHeight,
            nelemsx, nelemsz, nelemsz);
         femMod.setDensity (DENSITY);
         femMod.setLinearMaterial (200000, 0.4, true);
         femMod.setGravity (0, 0, -9.8);
         femMod.setStiffnessDamping (0.002);
         femMod.setSurfaceRendering (SurfaceRender.None);

         double eps = 1e-6;
         for (FemNode3d n : femMod.getNodes()) {
            double x = n.getPosition().x;
            if      (x <= -myFemLength / 2 + eps) leftNodes.add (n);
            else if (x >=  myFemLength / 2 - eps) rightNodes.add (n);
         }
         totalNodes += femMod.numNodes();

         X.p.set (linkCenter, 0, 0);
         femMod.transformGeometry (X);
         myMechMod.addModel (femMod);

         RigidBody leftBox = makeBox();
         X.p.set (linkCenter - (myBoxLength + myFemLength) / 2, 0, 0);
         leftBox.setPose (X);
         myMechMod.addRigidBody (leftBox);

         RigidBody rightBox = makeBox();
         X.p.set (linkCenter + (myBoxLength + myFemLength) / 2, 0, 0);
         rightBox.setPose (X);
         myMechMod.addRigidBody (rightBox);

         for (FemNode3d n : leftNodes)  myMechMod.attachPoint (n, leftBox);
         for (FemNode3d n : rightNodes) myMechMod.attachPoint (n, rightBox);

         RigidTransform3d TCA = new RigidTransform3d();
         RigidTransform3d TCW = new RigidTransform3d();
         TCA.p.set (-myBoxLength / 2, 0, myBoxHeight / 2);
         TCA.R.mulAxisAngle (1, 0, 0, Math.PI / 2);
         TCW.mul (leftBox.getPose(), TCA);
         HingeJoint joint;
         if (lastBox == null) {
            joint = new HingeJoint (leftBox, TCW);
         }
         else {
            joint = new HingeJoint (leftBox, lastBox, TCW);
         }
         RenderProps.setFaceColor (joint, new Color (0.15f, 0.15f, 1f));
         joint.setShaftLength (0.5);
         joint.setShaftRadius (0.01);
         myMechMod.addBodyConnector (joint);

         lastBox = rightBox;
      }
      lastBox.setDynamic (false);

      // ConstrainedBackwardEuler routes every step through KKTSolver,
      // which is the cuDSS-accelerated path as of KKT Stages C/D.
      myMechMod.setIntegrator (Integrator.ConstrainedBackwardEuler);
      myMechMod.setProfiling (true);  // emit avgSolveTime per step
      addModel (myMechMod);
      // Enable MechSystemSolver's KKT-breakdown profiling. Triggered by
      // CUDSS_BRIDGE_TIMING env var (same toggle the bridge reads), so
      // a single "profile this run" gesture turns on both layers.
      String dbg = System.getenv ("CUDSS_BRIDGE_TIMING");
      if (dbg != null && !dbg.isEmpty() && !dbg.equals ("0")) {
         myMechMod.getSolver().profileKKTSolveTime = true;
      }

      System.out.println (
         "ArticulatedFemBig: nlinks=" + nlinks
         + " nelemsx=" + nelemsx + " nelemsz=" + nelemsz
         + " totalFemNodes=" + totalNodes
         + " integrator=" + myMechMod.getIntegrator()
         + " matrixSolver=" + myMechMod.getMatrixSolver());

      addControlPanel (myMechMod);
   }

   protected void addControlPanel (MechModel mech) {
      myControlPanel = new ControlPanel ("options", "");
      myControlPanel.addWidget (mech, "integrator");
      myControlPanel.addWidget (mech, "matrixSolver");
      myControlPanel.addWidget (mech, "maxStepSize");
      addControlPanel (myControlPanel);
   }

   protected ControlPanel myControlPanel;

   @Override
   public String getAbout() {
      return "Articulated FEM benchmark: FEM links connected by hinge "
         + "joints with rigid-body link blocks. Exercises bilateral "
         + "constraints + joints through KKTSolver.";
   }
}
