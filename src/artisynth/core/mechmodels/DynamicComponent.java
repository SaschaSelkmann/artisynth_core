/**
 * Copyright (c) 2014, by the Authors: John E Lloyd (UBC)
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package artisynth.core.mechmodels;

import artisynth.core.modelbase.ModelComponent;
import artisynth.core.modelbase.StructureChangeEvent;
import artisynth.core.modelbase.TransformableGeometry;
import maspack.util.DataBuffer;
import maspack.matrix.Matrix;
import maspack.matrix.MatrixBlock;
import maspack.matrix.SparseBlockMatrix;
import maspack.matrix.SparseNumberedBlockMatrix;
import maspack.matrix.VectorNd;
import maspack.matrix.Vector3d;

import java.util.*;

public interface DynamicComponent
   extends DynamicAgent, ModelComponent, ForceEffector, TransformableGeometry {
   
   public void addAttachmentRequest (AttachingComponent ac);

   public boolean removeAttachmentRequest (AttachingComponent ac);

   /**
    * GPU FEM element-evaluation support: if this component's internal elastic
    * force can be computed on the device as a stiffness SpMV K*u, writes its
    * rest displacement u (current - rest position) into {@code buf} starting at
    * {@code idx} and returns {@code idx} advanced past it. Returns -1 if this
    * component is not GPU-elastic eligible (the caller then treats the whole
    * system as ineligible and keeps the host force path). Default: -1.
    */
   default int getElasticGpuRestDisplacement (double[] buf, int idx) {
      return -1;
   }
}
