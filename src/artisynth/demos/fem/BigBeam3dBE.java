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
 * MechSystemSolver.Integrator#BackwardEuler} instead of the default
 * trapezoidal/constrained integrator. This forces the time-step to go
 * through the regular direct solve path
 * ({@code MechSystemSolver.backwardEuler}), which is the path the
 * optional cuDSS backend accelerates.
 *
 * <p>Used as the FE benchmark target for comparing PARDISO and cuDSS
 * direct solve performance. There is no contact in this model, so
 * KKT/Murty solvers are not exercised and {@code myDirectSolver} alone
 * carries the cost of every step.
 */
public class BigBeam3dBE extends BigBeam3d {

   @Override
   public void build (String[] args) {
      super.build (args);
      myMechMod.setIntegrator (MechSystemSolver.Integrator.BackwardEuler);
      System.out.println (
         "BigBeam3dBE: integrator forced to "
         + myMechMod.getIntegrator() + ", matrixSolver="
         + myMechMod.getMatrixSolver());
   }
}
