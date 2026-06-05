package artisynth.demos.tutorial;

import java.awt.Color;
import java.io.IOException;

import artisynth.core.femmodels.FemFactory;
import artisynth.core.femmodels.FemMarker;
import artisynth.core.femmodels.FemModel3d;
import artisynth.core.femmodels.FemNode3d;
import artisynth.core.gui.ControlPanel;
import artisynth.core.materials.LinearMaterial;
import artisynth.core.materials.SimpleAxialMuscle;
import artisynth.core.mechmodels.HingeJoint;
import artisynth.core.mechmodels.MechModel;
import artisynth.core.mechmodels.MultiPointMuscle;
import artisynth.core.mechmodels.Muscle;
import artisynth.core.mechmodels.Particle;
import artisynth.core.mechmodels.Point;
import artisynth.core.mechmodels.PointFrameAttachment;
import artisynth.core.mechmodels.RigidBody;
import artisynth.core.mechmodels.RigidCylinder;
import artisynth.core.mechmodels.Wrappable;
import artisynth.core.workspace.RootModel;
import maspack.geometry.PolygonalMesh;
import maspack.matrix.Point3d;
import maspack.matrix.RigidTransform3d;
import maspack.matrix.Vector3d;
import maspack.render.RenderProps;
import maspack.render.Renderer;
import maspack.util.PathFinder;

/**
 * A toy two-link mechanical arm controlled by two sets of opposing muscles.
 *
 * This version replaces the two rigid links with FEM beam models.
 *
 * Joint concept:
 *
 *    base -- hinge -- rigid cylinder -- FEM link 0
 *
 * and
 *
 *    FEM link 0 -- rigid cylinder -- hinge -- rigid cylinder -- FEM link 1
 *
 * The joint cylinders are attached to FEM nodes using PointFrameAttachment.
 * The selected FEM nodes are located near an imaginary cylindrical surface
 * around the hinge axis. This mimics a pin/hole-like coupling region without
 * requiring a true geometric hole in the FEM mesh.
 *
 * Muscle concept:
 *
 *    muscle -- FemMarker -- FEM mesh
 *
 * Muscles are attached directly to FemMarkers, following ArtiSynth's
 * FemBeamWithMuscle example.
 */
public class ToyMuscleArmFEM extends RootModel {

   // Directory in which to locate mesh data.
   protected String geodir = PathFinder.getSourceRelativePath (this, "data/");

   // ----------------------------------------------------------------------
   // Global mechanical parameters
   // ----------------------------------------------------------------------

   protected double muscleStiffness = 1000.0;
   protected double muscleDamping = 0.0;

   /*
    * Global inertial damping of the mechanical system.
    */
   protected double mechInertialDamping = 1.0;

   /*
    * FEM damping parameters.
    *
    * Particle damping damps nodal velocities directly.
    * Stiffness damping damps deformation-related oscillations.
    */
   protected double link0ParticleDamping = 2.0;
   protected double link0StiffnessDamping = 0.002;

   protected double link1ParticleDamping = 2.0;
   protected double link1StiffnessDamping = 0.002;

   // ----------------------------------------------------------------------
   // FEM link parameters
   // ----------------------------------------------------------------------

   /*
    * The original model is arranged along the z-axis:
    *
    * link0: approximately from z = 0.0 to z = 0.6
    * link1: approximately from z = 0.6 to z = 1.2
    *
    * Therefore, lengthZ is the longitudinal link dimension.
    */
   protected double link0WidthX = 0.20;
   protected double link0WidthY = 0.20;
   protected double link0LengthZ = 0.60;

   /*
    * Finer mesh than the first test model. A cylindrical node selection only
    * makes sense if enough nodes exist near the intended cylindrical surface.
    */
   protected int link0ElemX = 8;
   protected int link0ElemY = 6;
   protected int link0ElemZ = 24;

   protected double link0Density = 1000.0;
   protected double link0YoungsModulus = 5.0e8;
   protected double link0PoissonsRatio = 0.33;

   protected double link1WidthX = 0.10;
   protected double link1WidthY = 0.15;
   protected double link1LengthZ = 0.60;

   /*
    * Link 1 is narrower, so the cylindrical attachment radius must be smaller
    * than half the smallest cross-sectional dimension.
    */
   protected int link1ElemX = 8;
   protected int link1ElemY = 6;
   protected int link1ElemZ = 24;

   protected double link1Density = 1000.0;
   protected double link1YoungsModulus = 5.0e8;
   protected double link1PoissonsRatio = 0.33;

   // ----------------------------------------------------------------------
   // Joint and wrapping parameters
   // ----------------------------------------------------------------------

   /*
    * The joint cylinder radius must fit inside the narrower link.
    *
    * Since link1WidthX = 0.10, half width is 0.05. Therefore, a radius of
    * 0.035 is safer than 0.055 for a pin/hole-like coupling region.
    */
   protected double jointCylinderRadius = 0.035;
   protected double jointCylinderLength = 0.22;
   protected int jointCylinderSegments = 32;

   /*
    * The imaginary cylindrical attachment surface has the same radius as the
    * joint cylinder by default.
    *
    * radialTolerance controls how far away from the intended cylinder surface
    * a node may be and still be attached. A larger value attaches more nodes
    * and stiffens the coupling region. A smaller value is more selective but
    * may attach too few nodes.
    */
   protected double jointAttachmentCylinderRadius = jointCylinderRadius;
   protected double jointAttachmentRadialTolerance = 0.018;

   /*
    * Since the cylinder axis is the global y-axis, the cylinder length is
    * evaluated in y-direction.
    */
   protected double jointAttachmentHalfLengthY = jointCylinderLength / 2.0;

   /*
    * Optional minimum number of attached nodes. This is a simple sanity check.
    */
   protected int minJointAttachedNodes = 4;

   /*
    * The wrap body remains a rigid cylinder as in the original model.
    * It is attached to the link1 joint cylinder instead of directly to the FEM,
    * because otherwise some FEM nodes would be attached twice.
    */
   protected double wrapCylinderRadius = 0.12;
   protected double wrapCylinderLength = 0.25;
   protected int wrapCylinderSegments = 32;

   /*
    * FEM marker rendering radius.
    */
   protected double femMarkerRadius = 0.02;
   protected double tipMarkerRadius = 0.02;

   /*
    * Connector density should usually not be zero in dynamic simulations.
    */
   protected double connectorDensity = 1000.0;

   // ----------------------------------------------------------------------
   // Model components
   // ----------------------------------------------------------------------

   protected MechModel myMech;

   protected FemModel3d myLink0Fem;
   protected FemModel3d myLink1Fem;

   protected RigidBody myLink0Joint0Body;
   protected RigidBody myLink0Joint1Body;
   protected RigidBody myLink1Joint1Body;

   protected HingeJoint myHinge0;
   protected HingeJoint myHinge1;

   protected FemMarker myTipMkr;

   /**
    * Creates a rectangular hexahedral FEM beam.
    *
    * A simple isotropic linear elastic material is used initially.
    *
    * Other possible material models in ArtiSynth include, for example,
    * NeoHookeanMaterial, MooneyRivlinMaterial, StVenantKirchoffMaterial,
    * OgdenMaterial, or anisotropic/transversely isotropic materials.
    *
    * The correct material model depends on whether the link should behave like
    * a technical elastic structure, soft biological tissue, tendon-like tissue,
    * or another deformable component.
    */
   protected FemModel3d createFemLink (
      String name,
      double widthX,
      double widthY,
      double lengthZ,
      int elemX,
      int elemY,
      int elemZ,
      double density,
      double youngsModulus,
      double poissonsRatio,
      double particleDamping,
      double stiffnessDamping,
      double zCenter) {

      FemModel3d fem = new FemModel3d (name);

      FemFactory.createHexGrid (
         fem,
         widthX,
         widthY,
         lengthZ,
         elemX,
         elemY,
         elemZ);

      fem.setDensity (density);
      fem.setMaterial (new LinearMaterial (youngsModulus, poissonsRatio));

      fem.setParticleDamping (particleDamping);
      fem.setStiffnessDamping (stiffnessDamping);

      translateFemNodes (fem, 0.0, 0.0, zCenter);

      myMech.addModel (fem);

      RenderProps.setFaceColor (fem, new Color (0.71f, 0.71f, 0.85f));
      RenderProps.setLineColor (fem, Color.DARK_GRAY);
      RenderProps.setLineWidth (fem, 1);

      return fem;
   }

   /**
    * Translates all nodes of an FEM. This is only used during construction.
    */
   protected void translateFemNodes (
      FemModel3d fem, double dx, double dy, double dz) {

      for (FemNode3d node : fem.getNodes()) {
         Point3d p = new Point3d (node.getPosition());
         p.x += dx;
         p.y += dy;
         p.z += dz;
         node.setPosition (p);
      }
   }

   /**
    * Creates a FEM marker and adds it directly to the FEM model.
    *
    * FemMarkers are suitable direct attachment points for axial springs and
    * muscles.
    */
   protected FemMarker createFemMarker (
      FemModel3d fem, double x, double y, double z) {

      FemMarker mkr = new FemMarker (null, x, y, z);
      fem.addMarker (mkr);

      RenderProps.setSphericalPoints (mkr, femMarkerRadius, Color.BLUE);

      return mkr;
   }

   /**
    * Creates a fixed particle in world coordinates.
    *
    * This is used for muscle attachment points on the fixed base side.
    */
   protected Particle createFixedParticle (
      String name, double x, double y, double z) {

      Particle p = new Particle (name, 0.0, x, y, z);
      p.setDynamic (false);
      myMech.addParticle (p);

      RenderProps.setSphericalPoints (p, femMarkerRadius, Color.BLUE);

      return p;
   }

   /**
    * Attaches FEM nodes near an imaginary cylindrical surface to a rigid body.
    *
    * The cylinder axis is parallel to the global y-axis. Therefore, the radial
    * distance is computed in the x-z plane:
    *
    *    r = sqrt((x - xCenter)^2 + (z - zCenter)^2)
    *
    * Nodes are attached if they are close to the desired cylinder radius and
    * lie within the cylinder half-length in y-direction.
    *
    * This does not create a real geometric hole. It only creates a pin-like
    * coupling region in the current rectangular FEM mesh.
    */
   protected void attachFemNodesNearCylinderSurface (
      RigidBody body,
      FemModel3d fem,
      double xCenter,
      double yCenter,
      double zCenter,
      double cylinderRadius,
      double radialTolerance,
      double halfLengthY) {

      int numAttached = 0;

      for (FemNode3d node : fem.getNodes()) {
         Point3d p = node.getPosition();

         double dx = p.x - xCenter;
         double dz = p.z - zCenter;
         double radialDistance = Math.sqrt (dx * dx + dz * dz);

         boolean nearCylinderSurface =
            Math.abs (radialDistance - cylinderRadius) <= radialTolerance;

         boolean insideCylinderLength =
            Math.abs (p.y - yCenter) <= halfLengthY;

         if (nearCylinderSurface && insideCylinderLength) {
            myMech.addAttachment (new PointFrameAttachment (body, node));
            RenderProps.setSphericalPoints (node, 0.007, Color.GREEN);
            numAttached++;
         }
      }

      if (numAttached < minJointAttachedNodes) {
         System.out.println (
            "WARNING: only " + numAttached +
            " FEM nodes were attached to rigid body '" + body.getName() +
            "'. Consider increasing mesh resolution or radialTolerance.");
      }

      if (numAttached == 0) {
         throw new IllegalArgumentException (
            "No FEM nodes were attached to rigid body '" + body.getName() +
            "'. Increase radialTolerance, refine the mesh, or check the " +
            "joint position and cylinder radius.");
      }
   }

   /**
    * Creates a rigid joint cylinder and attaches FEM nodes near an imaginary
    * cylindrical surface to it.
    *
    * The cylinder axis is aligned with the hinge axis, i.e. the global y-axis.
    */
   protected RigidBody createJointCylinder (
      String name,
      FemModel3d fem,
      double x,
      double y,
      double z) {

      RigidBody cyl = RigidBody.createCylinder (
         name,
         jointCylinderRadius,
         jointCylinderLength,
         connectorDensity,
         jointCylinderSegments);

      /*
       * RigidBody.createCylinder() creates the cylinder along its local z-axis.
       * The hinge axis is the global y-axis. Rotate the cylinder by -90 degrees
       * about the x-axis so that the cylinder axis is parallel to the hinge axis.
       */
      cyl.setPose (
         new RigidTransform3d (
            x,
            y,
            z,
            0.0,
            0.0,
            -Math.PI / 2.0));

      myMech.addRigidBody (cyl);

      attachFemNodesNearCylinderSurface (
         cyl,
         fem,
         x,
         y,
         z,
         jointAttachmentCylinderRadius,
         jointAttachmentRadialTolerance,
         jointAttachmentHalfLengthY);

      RenderProps.setFaceColor (cyl, new Color (0.45f, 0.45f, 0.55f));

      return cyl;
   }

   /**
    * Creates a point-to-point muscle between two points.
    *
    * The points may be FemMarkers, Particles, or other Point subclasses.
    */
   protected Muscle addMuscle (
      Point p0,
      Point p1,
      double maxForce) {

      Muscle muscle = new Muscle (null, 0.0);
      muscle.setMaterial (
         new SimpleAxialMuscle (
            muscleStiffness,
            muscleDamping,
            maxForce));

      muscle.setPoints (p0, p1);
      myMech.addAxialSpring (muscle);

      RenderProps.setLineStyle (muscle, Renderer.LineStyle.SPINDLE);
      RenderProps.setLineColor (muscle, Color.RED);
      RenderProps.setLineRadius (muscle, 0.02);

      return muscle;
   }

   /**
    * Creates a wrapped muscle between two points and wraps it around a wrappable
    * body.
    */
   protected MultiPointMuscle addWrappedMuscle (
      Point p0,
      Point p1,
      Wrappable wrapBody,
      double maxForce) {

      MultiPointMuscle muscle = new MultiPointMuscle();
      muscle.setMaterial (
         new SimpleAxialMuscle (
            muscleStiffness,
            muscleDamping,
            maxForce));

      muscle.addPoint (p0);
      muscle.setSegmentWrappable (50);
      muscle.addPoint (p1);
      muscle.addWrappable (wrapBody);
      muscle.updateWrapSegments();

      myMech.addMultiPointSpring (muscle);

      RenderProps.setLineStyle (muscle, Renderer.LineStyle.SPINDLE);
      RenderProps.setLineColor (muscle, Color.RED);
      RenderProps.setLineRadius (muscle, 0.02);

      return muscle;
   }

   /**
    * Adds a hinge joint between two rigid bodies.
    */
   protected HingeJoint addHingeJoint (
      RigidBody body0,
      RigidBody body1,
      double x,
      double y,
      double z,
      double minDeg,
      double maxDeg) {

      HingeJoint hinge = new HingeJoint (
         body0,
         body1,
         new Point3d (x, y, z),
         new Vector3d (0.0, 1.0, 0.0));

      myMech.addBodyConnector (hinge);
      hinge.setThetaRange (minDeg, maxDeg);

      hinge.setShaftLength (0.4);
      RenderProps.setFaceColor (hinge, Color.BLUE);

      return hinge;
   }

   public void build (String[] args) throws IOException {

      myMech = new MechModel ("mech");
      myMech.setInertialDamping (mechInertialDamping);
      addModel (myMech);

      // -------------------------------------------------------------------
      // Fixed rigid base
      // -------------------------------------------------------------------

      PolygonalMesh mesh = new PolygonalMesh (geodir + "bracketedBase.obj");
      RigidBody base = RigidBody.createFromMesh (
         "base",
         mesh,
         1000.0,
         1.0);

      base.setDynamic (false);
      myMech.addRigidBody (base);

      // -------------------------------------------------------------------
      // FEM links
      // -------------------------------------------------------------------

      myLink0Fem = createFemLink (
         "link0Fem",
         link0WidthX,
         link0WidthY,
         link0LengthZ,
         link0ElemX,
         link0ElemY,
         link0ElemZ,
         link0Density,
         link0YoungsModulus,
         link0PoissonsRatio,
         link0ParticleDamping,
         link0StiffnessDamping,
         0.3);

      myLink1Fem = createFemLink (
         "link1Fem",
         link1WidthX,
         link1WidthY,
         link1LengthZ,
         link1ElemX,
         link1ElemY,
         link1ElemZ,
         link1Density,
         link1YoungsModulus,
         link1PoissonsRatio,
         link1ParticleDamping,
         link1StiffnessDamping,
         0.9);

      // -------------------------------------------------------------------
      // Joint connector cylinders
      // -------------------------------------------------------------------

      /*
       * Base joint:
       *
       *    base -- hinge -- link0Joint0Body -- link0Fem
       *
       * The FEM nodes are selected near an imaginary cylindrical surface around
       * the hinge axis.
       */
      myLink0Joint0Body = createJointCylinder (
         "link0Joint0Body",
         myLink0Fem,
         0.0,
         0.0,
         0.0);

      /*
       * Inter-link joint:
       *
       *    link0Fem -- link0Joint1Body -- hinge -- link1Joint1Body -- link1Fem
       */
      myLink0Joint1Body = createJointCylinder (
         "link0Joint1Body",
         myLink0Fem,
         0.0,
         0.0,
         0.6);

      myLink1Joint1Body = createJointCylinder (
         "link1Joint1Body",
         myLink1Fem,
         0.0,
         0.0,
         0.6);

      myHinge0 = addHingeJoint (
         myLink0Joint0Body,
         base,
         0.0,
         0.0,
         0.0,
         -70.0,
         70.0);

      myHinge1 = addHingeJoint (
         myLink1Joint1Body,
         myLink0Joint1Body,
         0.0,
         0.0,
         0.6,
         -120.0,
         120.0);

      // -------------------------------------------------------------------
      // Wrapping body
      // -------------------------------------------------------------------

      RigidCylinder wrapCylinder = new RigidCylinder (
         "wrapSurface",
         wrapCylinderRadius,
         wrapCylinderLength,
         connectorDensity,
         wrapCylinderSegments);

      wrapCylinder.setPose (
         new RigidTransform3d (
            0.0,
            0.0,
            0.6,
            0.0,
            0.0,
            Math.PI / 2.0));

      myMech.addRigidBody (wrapCylinder);

      /*
       * Do not attach the wrap cylinder directly to link1Fem here.
       *
       * The wrap cylinder is located at the same position as the link1 joint
       * cylinder. Attaching both rigid bodies to nearby FEM nodes would attach
       * the same FEM nodes twice, which is not allowed in ArtiSynth.
       *
       * Instead, attach the wrap cylinder to the link1 joint cylinder. The
       * joint cylinder is already attached to link1Fem and distributes loads
       * into the FEM mesh.
       */
      myMech.attachFrame (wrapCylinder, myLink1Joint1Body);

      RenderProps.setFaceColor (
         wrapCylinder,
         new Color (0.75f, 0.61f, 0.75f));

      // -------------------------------------------------------------------
      // Muscle attachment points
      // -------------------------------------------------------------------

      /*
       * Muscle endpoints on FEM links are FemMarkers.
       */
      FemMarker l0LeftMarker = createFemMarker (
         myLink0Fem,
         -0.1,
         0.0,
         0.4);

      FemMarker l0RightMarker = createFemMarker (
         myLink0Fem,
         0.1,
         0.0,
         0.4);

      FemMarker l1LeftMarker = createFemMarker (
         myLink1Fem,
         -0.05,
         0.0,
         1.2);

      FemMarker l1RightMarker = createFemMarker (
         myLink1Fem,
         0.05,
         0.0,
         1.2);

      FemMarker l0LeftWrapMarker = createFemMarker (
         myLink0Fem,
         -0.1,
         0.0,
         0.5);

      FemMarker l0RightWrapMarker = createFemMarker (
         myLink0Fem,
         0.1,
         0.0,
         0.5);

      /*
       * Fixed muscle endpoints on the base side are represented as fixed
       * particles in world coordinates.
       */
      Particle baseLeftPoint = createFixedParticle (
         "baseLeftPoint",
         -0.48,
         0.0,
         -0.15);

      Particle baseRightPoint = createFixedParticle (
         "baseRightPoint",
         0.48,
         0.0,
         -0.15);

      // -------------------------------------------------------------------
      // Muscles
      // -------------------------------------------------------------------

      Muscle muscleL0 = addMuscle (
         l0LeftMarker,
         baseLeftPoint,
         1000.0);

      Muscle muscleR0 = addMuscle (
         l0RightMarker,
         baseRightPoint,
         1000.0);

      MultiPointMuscle muscleL1 = addWrappedMuscle (
         l1LeftMarker,
         l0LeftWrapMarker,
         wrapCylinder,
         500.0);

      MultiPointMuscle muscleR1 = addWrappedMuscle (
         l1RightMarker,
         l0RightWrapMarker,
         wrapCylinder,
         500.0);

      // -------------------------------------------------------------------
      // Tip marker
      // -------------------------------------------------------------------

      myTipMkr = createFemMarker (
         myLink1Fem,
         0.0,
         0.0,
         1.25);

      RenderProps.setSphericalPoints (myTipMkr, tipMarkerRadius, Color.GREEN);

      // -------------------------------------------------------------------
      // Initial configuration
      // -------------------------------------------------------------------

      /*
       * If strong initial distortions occur, first test the model with both
       * angles set to zero. With FEM links, setting joint angles after creating
       * straight FEM beams can impose an initial elastic deformation.
       */
      myHinge0.setTheta (0.0);
      myHinge1.setTheta (0.0);

      myMech.updateWrapSegments();

      muscleL0.setRestLength (muscleL0.getLength());
      muscleR0.setRestLength (muscleR0.getLength());
      muscleL1.setRestLength (muscleL1.getLength());
      muscleR1.setRestLength (muscleR1.getLength());

      // -------------------------------------------------------------------
      // Control panel
      // -------------------------------------------------------------------

      ControlPanel panel = new ControlPanel();
      panel.addWidget ("excitation L0", muscleL0, "excitation");
      panel.addWidget ("excitation R0", muscleR0, "excitation");
      panel.addWidget ("excitation L1", muscleL1, "excitation");
      panel.addWidget ("excitation R1", muscleR1, "excitation");
      addControlPanel (panel);

      // -------------------------------------------------------------------
      // General rendering
      // -------------------------------------------------------------------

      RenderProps.setPointColor (myTipMkr, Color.GREEN);
   }
}