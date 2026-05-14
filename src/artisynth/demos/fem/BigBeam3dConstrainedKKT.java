/**
 * Copyright (c) 2026, by the Authors: ArtiSynth Team
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package artisynth.demos.fem;

import artisynth.core.mechmodels.MechSystemSolver;

/**
 * BigBeam3d variant that uses {@link
 * MechSystemSolver.Integrator#ConstrainedBackwardEuler} instead of the
 * default Trapezoidal integrator. This routes every time-step through
 * {@code MechSystemSolver.constrainedBackwardEuler} and so exercises
 * {@code KKTSolver}, which is the path the Stage C generalization wired
 * up for cuDSS.
 *
 * <p>The default mesh and material from {@link BigBeam3d} give a
 * 33.8k-node hex beam (~101k mechanical DOF) plus zero bilateral
 * constraints (only node-level setDynamic(false) for the fixed end).
 * That makes this an "equality-only" KKT benchmark: useful for measuring
 * the cuDSS-backed KKT path against PARDISO without any contact / LCP
 * complications. For a model with actual bilateral constraints
 * (joint-coupled FEM, FEM attached to a rigid body), extend this further.
 */
public class BigBeam3dConstrainedKKT extends BigBeam3d {

   @Override
   public void build (String[] args) {
      super.build (args);
      myMechMod.setIntegrator (
         MechSystemSolver.Integrator.ConstrainedBackwardEuler);
      System.out.println (
         "BigBeam3dConstrainedKKT: integrator="
         + myMechMod.getIntegrator() + ", matrixSolver="
         + myMechMod.getMatrixSolver());
   }
}
