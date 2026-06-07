# GPU FEM Assembly — Findings & Development Notes

Working notes for continuing the GPU-offload of ArtiSynth FEM assembly.
Companion to `plan_gpu_assembly.md` (whose **Section 0** holds the authoritative
status summary). This file collects the concrete, reproducible findings and the
mechanics needed to keep working — things that are not obvious from the code or
git history.

Last updated: 2026-06-06. Branch: `sascha/cudss-0.8-stabilization`.

---

## 1. Environment

- CUDA 12.8 (`/usr/local/cuda/bin/nvcc`), GPU **NVIDIA GTX 1650**.
- cuDSS **0.8.0** (bumped from 0.7.1). Native lib loaded as `CuDssJNI.0.8.0`
  (`lib/Linux64/libCuDssJNI.so.0.8.0`); the old `0.7.1` lib still sits in `lib/`.
- Java build: `make` → exit 0 (javac, `--release 8`). Java 17 host JDK.
- `cudss/` (untracked) = NVIDIA CUDALibrarySamples reference checkout. Not repo code.
- Branch is **not** the originally planned `feature/gpu-fem-assembly`; 33 commits
  ahead of `master`.

## 2. Real architecture (differs from the original plan in `plan_gpu_assembly.md` §3)

There is **no** `artisynth.core.gpu` package and **no** `gpuFemAssembly.cu`.
Everything was integrated into existing classes + the cuDSS bridge.

Java side:
- `MechSystem.GpuAssemblyContext` — carries the CRS slot map and per-contribution
  descriptor arrays (the data the GPU kernels consume).
- `FemModel3d` — capability predicates + contribution emission:
  `isGpuLinearElasticMaterial`, `materialSupportsGpuMaterialStiffness3`,
  `canAssembleLinearElasticStiffness3CrsValueContributions`,
  `addLinearElasticStiffness3CrsValueContributions`, dilational/material variants.
- `MechSystemSolver` — routing, global toggles, profiling, **CPU-vs-GPU
  verification** (`checkGpuAssemblyCrsValues`), direct-CRS solve plumbing.
- `MechSystemBase`, `MechModel` — assembly hooks (`assembleGpuPosJacobianCrsValues`,
  `assembleGpuVelJacobianCrsValues`).
- `maspack.matrix.SparseNumberedBlockMatrix` — CRS slot map support.
- `maspack.solvers.CuDssSolver` — device-values API: `clearDeviceValues`,
  `addMaterialStiffness3DeviceValues`, `factorDeviceValues`, `solve`.

Native side — all kernels in `src/maspack/solvers/lib/cudssBridgeKernels.cu`:
`axmy`, `extractDiag`, `zeroValues`, `scatterAddValues`, `addScaledDiagonal3`,
`addScaledBlock3`, `addMaterialStiffness3` (+ per-element variant),
`addLinearElasticStiffness3Element`, `addLinearElasticStiffness3ElementGeometry`,
`addDilationalStiffness3Element`. JNI in `CuDssJNI.cc`, bridge in `cudssBridge.cc`.

## 3. Toggles (all global, not per-model)

The original plan's per-model `FemModel3d.setGpuAssembly(boolean)` was **not**
implemented. Instead: global static flags in `MechSystemSolver` + persisted
prefs in `SimulationSettings`/`SimulationPrefs`, plus JVM system properties:

| System property | Effect |
|---|---|
| `artisynth.gpuAssembly.status` | print per-solve routing/status diagnostics |
| `artisynth.gpuAssembly.profile` | print per-phase timing + *why a kernel path is disabled* |
| `artisynth.gpuAssembly.directCrs` | enable the direct device-CRS solve path |
| `artisynth.gpuAssembly.verifyCrs` | assemble on GPU **and** CPU, compare values, throw on mismatch |
| `artisynth.gpuAssembly.enabled` | master enable flag |
| `artisynth.gpuAssembly.requireFull` | require all contributions to be GPU-assembled |
| `artisynth.gpuAssembly.kktDeviceValues` | KKT M block device-values path |

GPU assembly is auto-enabled when the **cuDSS** matrix solver is selected.

## 4. The built-in CPU-vs-GPU equivalence check (key for this work)

`MechSystemSolver.checkGpuAssemblyCrsValues(phase, context)`:
- `vals` = GPU-assembled CRS values (`context.getCrsValues()`).
- `chk`  = CPU reference via `mySolveMatrix.getCRSValues(...)` (Maspack block→CRS).
- element-wise compare, tolerance `max(1e-12, 1e-12 * maxAbs(chk))`,
  throws `InternalErrorException` on the first slot exceeding tolerance.

Enable with `-Dartisynth.gpuAssembly.verifyCrs=true`. This is exactly the
numerical-equivalence gate the plan (guarantee **b**) calls for — it already
exists; the missing piece is exercising it on a config where the kernels fire.

## 5. **Critical finding: GPU assembly kernels are gated off for the common case**

In `FemModel3dTest.testBackwardEulerDirectCrsSolve` (default **corotated**
`LinearMaterial`, element soft-incompressibility), the device-CRS solve engages
but the status line reports:

```
matrixSolver=CuDss directCrs=true deviceCrs=true directCrsStatus=deviceValues
valueSource=hostMatrixValuesToDeviceCrs
contributions={generic=0 diag3=16 block3=88 material3=0 materialElem3=0/0/0
               linearElem3=0/0/0 linearGeomElem3=0/0/0 dilationElem3=0/0/0}
```

So FEM stiffness is still **CPU-assembled** and only copied into the device CRS
buffer. The specialized kernels never run. Confirmed with
`-Dartisynth.gpuAssembly.profile=true`:

```
linearElem3 disabled: corotated=true
material3   disabled: softIncomp=ELEMENT
```

Gating conditions for the linear-elastic kernel
(`canAssembleLinearElasticStiffness3CrsValueContributions` /
`isGpuLinearElasticMaterial`):
- material must be `LinearMaterial` with **`corotated == false`** ← the default
  `LinearMaterial` is corotated, so demos/tests miss it unless they opt out;
- no Young's-modulus field, no material state;
- no shell elements;
- no augmenting/auxiliary materials on any element;
- no "indirect" neighbor contributions (`hasIndirectGpuAssemblyContributions()`).

The `material3` batch additionally requires soft-incompressibility off (the
default is element-level, which disables it).

**Implication:** G3/G5/G6 kernels are implemented and unit-tested at the
device-call level (`testMaterialStiffness3ContextAssembly` passes), but **no
end-to-end run yet drives a realistic FemModel through them with non-zero
contribution counters and asserts equivalence.** That is the next deliverable.

## 6. `BigBeam3d` is the intended end-to-end vehicle

On this branch `src/artisynth/demos/fem/BigBeam3d.java` was changed from
`MooneyRivlinMaterial` to:

```java
LinearMaterial mat = new LinearMaterial (500000, 0.33, /*corotated=*/false);
mat.setCorotatedMode (PropertyMode.Explicit);
```

i.e. deliberately **non-corotated linear**, hex elements — exactly the config
that satisfies the GPU linear-elastic gate. So BigBeam3d is the behavioral/
performance demo for the GPU path. (Note: "run BigBeam3d" = behavioral check;
the strict numerical check is `verifyCrs` above. `BigBeam3d.advance()` also has
a commented-out `SolveMatrixTest.testStiffness(myMechMod, 1e-8)` — a
finite-difference stiffness self-check, not a CPU-vs-GPU comparison.)

## 7. How to run things (reproducible)

Always `source setup.bash` first. Unit-test mains need explicit
`-cp "$CLASSPATH"`; some need their source dir as cwd.

```bash
# Unit tests
cd src/maspack/solvers && java -cp "$CLASSPATH" maspack.solvers.KKTSolverTest      # needs MLCPtest.txt (this dir)
java -cp "$CLASSPATH" maspack.solvers.CuDssSolverTest
java -cp "$CLASSPATH" maspack.matrix.SparseNumberedBlockMatrixTest
cd src/artisynth/core/femmodels && java -cp "$CLASSPATH" artisynth.core.femmodels.FemModel3dTest
#   real cuDSS device step is gated: add -Dartisynth.gpuAssembly.directCrs=true

# Passing JVM system properties to the `artisynth` launcher:
#   the bin/artisynth script hard-sets JAVA_OPTS, so use JAVA_TOOL_OPTIONS instead
export JAVA_TOOL_OPTIONS="-Dartisynth.gpuAssembly.status=true \
  -Dartisynth.gpuAssembly.directCrs=true -Dartisynth.gpuAssembly.verifyCrs=true"

# Model build args (e.g. -nx/-ny) must be wrapped in literal [ ] brackets:
artisynth -noGui -matrixSolver CuDss \
  -model artisynth.demos.fem.BigBeam3d '[' -nx 4 -ny 2 ']' -playFor 0.003
```

Native rebuild (if kernels change): `cd src/maspack/solvers/lib && make cudss`
(needs nvcc + libcudss at `CUDSS_INC`/`CUDSS_LIB`, defaults under
`/usr/.../libcudss/12`).

## 8. Verification results (2026-06-06, GTX 1650, cuDSS 0.8.0)

All passing: `SparseNumberedBlockMatrixTest`, `KKTSolverTest` (incl. cuDSS
device-values / refactor / equality), `CuDssSolverTest`, `FemModel3dTest`
(incl. `testFemNeighborCrsAssembly`, `testMaterialStiffness3ContextAssembly`),
`testBackwardEulerDirectCrsSolve` (real cuDSS `BackwardEuler` step — but kernels
not fired, see §5).

### 8a. Why `BigBeam3d` does NOT exercise the GPU kernels (2026-06-06/07)

Running `BigBeam3d` (the demo that was deliberately switched to non-corotated
`LinearMaterial` for this work) under cuDSS does **not** fire the kernels:

- Default integrator is `ConstrainedBackwardEuler` → status
  `directCrsStatus=integratorConstrainedBackwardEuler`. The direct-CRS / GPU
  assembly path is gated to the **plain `BackwardEuler`** integrator only
  (`MechSystemSolver.java:1227`: `if (myIntegrator != Integrator.BackwardEuler)`).
  Virtually all FEM demos use the constrained integrator, so they never engage.
- Even forced to `BackwardEuler`, `BigBeam3d` attaches FEM nodes to rigid
  blocks. Attachments create **indirect neighbors**, so
  `hasIndirectGpuAssemblyContributions()` is true and the linear-elastic gate
  is closed; additionally the velocity Jacobian for the rigid/attachment DOFs is
  not GPU-provided ("GPU assembly requested for backwardEuler velocity Jacobian
  but this MechSystem does not provide it; using CPU assembly") and the whole
  step falls back to CPU.

**Full set of conditions for the GPU linear-elastic path to actually run:**
plain `BackwardEuler` integrator; cuDSS solver; `-Dartisynth.gpuAssembly.directCrs=true`;
a pure FEM model with **no rigid bodies and no attachments** (fixing nodes via
`setDynamic(false)` is fine); non-corotated `LinearMaterial` with
`setCorotatedMode(Explicit)`; no shells; no augmenting/auxiliary materials.

### 8c. ROOT CAUSE + FIX (2026-06-07): lower block triangle was never assembled

Found via a single-tet readback comparison (`CuDssSolver.getDeviceValues`, a new
native readback added for this; see §12). For one non-corotated linear tet, the
GPU-assembled CRS values matched the CPU reference EXACTLY for every diagonal
block and every **upper** off-diagonal block (`bj >= bi`), but every **lower**
off-diagonal block (block row > block col, e.g. (n1,n0), (n2,n1), (n3,n2)) was
**zero on the GPU**. The kernel math (B-matrix, Voigt D, spatial gradient, dv,
scale) was therefore correct all along — entries it filled were bit-exact.

Why it mattered: the direct-CRS path builds the slot map with
`Matrix.Partition.Full` and analyzes the device matrix as `Matrix.INDEFINITE`,
which maps to cuDSS **MT_GENERAL** (full view, `Matrix.INDEFINITE == 0x0`, so the
`SYMMETRIC` bit is clear). cuDSS thus reads BOTH triangles, but the GPU
contribution builders only emitted the upper one.

The bug: `FemModel3d.addLinearElasticStiffness3CrsValueContributions` (and the
sibling `addMaterialStiffness3CrsValueContributions`) enumerated node pairs with
`if (!mySolveMatrixSymmetricP || bj >= bi)` — upper block triangle only. Since
`myNbrs` is a full nnodes×nnodes neighbor matrix and each ordered pair (i,j) maps
to the distinct CRS block at (bi,bj), the fix is to emit **every** pair with
`bj != -1`. The kernel already computes the correct K_ij = Bi^T D Bj·dv for any
(i,j), so no kernel change was needed.

Fix verified:
- single-tet CRS readback: `maxDiff = 0.0` (was 1.2e4).
- `testBackwardEulerLinearElasticEquivalence`: kernel diff `5.6e-17` (was 0.038),
  control `3.5e-17` → test now GREEN.
- full suite (FemModel3dTest default + directCrs, CuDssSolverTest, KKTSolverTest)
  passes.

Scope note: the same fix was applied to the `material3` (nonlinear-material)
contribution builder, which had the identical upper-only enumeration. It is
verified only by shared logic + the full suite, NOT by a dedicated nonlinear GPU
equivalence test (none exists yet — the nonlinear device path is hard to reach;
see §5/§8a gating). The `dilational` and generic `block3`/`diag3` builders were
NOT changed; if they feed the same general device matrix with off-diagonal
coupling they likely need the same treatment — audit before enabling those.

### 8b. (historical) the GPU linear-elastic geometry kernel was numerically WRONG

A new equivalence test (`FemModel3dTest.testBackwardEulerLinearElasticEquivalence`,
gated on `-Dartisynth.gpuAssembly.directCrs=true`) builds a pure, attachment-free
non-corotated linear tet beam and steps it once with cuDSS (GPU assembly) and
once with Pardiso (CPU), comparing the active velocity state. It is
self-validating via a control:

- **Control** (corotated → cuDSS host-assembles values, no kernel) vs Pardiso:
  `max diff = 3.5e-17` → bit-identical. cuDSS solve + harness are fair.
- **Kernel** (non-corotated → GPU `linearGeomElem3` kernel fires, status shows
  `valueSource=hostReferenceGeometryToGpuKernels linearGeomElem3=160/160/1600`)
  vs Pardiso: **`max diff = 0.038`, ref = 0.049 → ~78% relative error.**

Because the control is exact, the discrepancy is **a real bug in the GPU
linear-elastic geometry stiffness kernel** (`addLinearElasticStiffness3ElementGeometry`
in `cudssBridgeKernels.cu` and/or the descriptor marshalling in
`FemModel3d.addLinearElasticStiffness3CrsValueContributions` /
`CuDssSolver.addLinearElasticStiffness3ElementGeometryDeviceValues`), not the
solver. The kernels were never validated end-to-end before this, which is how
the bug went unnoticed.

The routing test (`testLinearElasticStiffness3Routing`, runs always, no GPU
needed) passes: it confirms non-corotated emits geometry descriptors
(`numLinearElasticStiffness3ElementGeometryContributions() > 0`) and corotated
emits none.

**Status: FIXED (2026-06-07) — see §8c for the root cause and fix.** The
equivalence test is now GREEN (kernel diff ~5.6e-17).

## 9. Next steps (ordered)

1. ~~Fix the GPU linear-elastic geometry stiffness kernel.~~ **DONE (§8c).**
   Root cause was the upper-only pair enumeration in the marshalling, not the
   kernel. Equivalence test is green.
2. **Audit the other contribution builders for the same upper-only bug.** The
   `material3` builder got the same fix (unverified by a dedicated test). The
   `dilational`, generic `block3`, and `diag3` builders were left unchanged —
   confirm whether they also need full (lower-triangle) emission for the
   general device matrix, and add a nonlinear-material GPU equivalence test to
   cover `material3`. (Alternative design worth considering: analyze the device
   matrix as cuDSS SYMMETRIC/upper instead of GENERAL, so upper-only assembly is
   correct by construction and cheaper — but verify the whole device matrix is
   always symmetric in this path first.)
3. **Strategic decision:** the actual common case is **corotated** linear
   material AND the **constrained** integrator. The current GPU path supports
   neither (corotated gate closed; only plain BackwardEuler). Decide whether to
   (a) extend GPU assembly to corotated linear, and (b) wire the constrained
   (KKT) integrator's FEM stiffness to GPU assembly — without these two, the
   GPU path covers almost no real models.
4. Only after 1-3: nonlinear material kernels (G5) and more element types (G6).
5. Capture a real per-phase profile on `ArticulatedFemBig` to confirm the ≥3×
   `updateStressAndStiffness` speedup target before broadening coverage.

## 11. Tests added (2026-06-07)

In `src/artisynth/core/femmodels/FemModel3dTest.java`:

- `testLinearElasticStiffness3Routing()` — always runs (no GPU). Asserts a
  non-corotated `LinearMaterial` emits GPU linear-elastic **geometry**
  descriptors and a corotated one emits none. Guards the kernel routing/gate.
  NOTE: the path increments the *geometry* counter
  (`numLinearElasticStiffness3ElementGeometryContributions()`), not the plain
  `numLinearElasticStiffness3ElementContributions()` — an easy trap.
- `testBackwardEulerLinearElasticEquivalence()` — gated on
  `-Dartisynth.gpuAssembly.directCrs=true` + cuDSS. cuDSS-vs-Pardiso single-step
  velocity comparison with a corotated control. GREEN after the §8c fix. Run it:
  `cd src/artisynth/core/femmodels && java -cp "$CLASSPATH" \
   -Dartisynth.gpuAssembly.directCrs=true artisynth.core.femmodels.FemModel3dTest`

## 12. New native readback: CuDssSolver.getDeviceValues (2026-06-07)

Added `CuDssSolver.getDeviceValues(double[] vals)` (→ `CuDssBridge::getDeviceValues`
→ `cudaMemcpy` D2H of `myValsD`). Copies the device-side CRS values back to host.
Added purely as a debug/verification aid (it is what localized §8c by enabling a
direct entrywise CPU-vs-GPU CRS comparison for a single tet). It is also the
natural building block for a real value-level verifyCrs of the GPU-kernel path
(today's `checkGpuAssemblyCrsValues` only compares CPU context values, see §4).

Native rebuild reminder: `make cudss` needs `JAVA_HOME` set for `jni.h`, e.g.
`export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64`. Adding a native method
requires regenerating `maspack_solvers_CuDssSolver.h` (the Makefile does this;
it is a checked-in generated file).

Reusable harness `src/artisynth/core/gpudebug/GpuCrsCompare.java` (kept): for a
given FemModel3d it assembles the position-Jacobian CRS the CPU way and the GPU
way (full device dispatch + `getDeviceValues`) and diffs entrywise, printing
which contribution categories were active. Used to verify the §8c fix across
materials (see §13). Recreate single-element variants from this if needed.

## 13. Real GPU-vs-CPU device verifyCrs safety net (2026-06-07)

Built the in-step device verification the §8c bug motivated. Previously
`checkGpuAssemblyCrsValues` only compared CPU context values vs the CPU block
matrix (CPU-vs-CPU) and SKIPPED the kernel paths (incomplete), so it could not
have caught §8c. The new check actually reads the device buffer back.

`MechSystemSolver.verifyGpuDeviceCrsValues(cudss, context, h, phase)`:
- reads the GPU-assembled device CRS values back with `CuDssSolver.getDeviceValues`,
- assembles a CPU reference into the SAME context CRS value array (identical slot
  ordering) via `assembleGpuVelJacobianCrsValues(-h)` +
  `assembleGpuPosJacobianCrsValues(-h*h)` + `addActiveMassMatrixCrsValues`,
- compares entrywise, tol = max(1e-9, 1e-9·max|cpu|), throws
  `InternalErrorException` on mismatch (naming the offending CRS slot).

Wiring: `verifyGpuAssemblyCrs` no longer DISABLES the device path (removed the
`!verifyGpuAssemblyCrs` guard on `tryDirectCrsOnly`). Now the check runs once per
backwardEuler step when BOTH flags are set:
`-Dartisynth.gpuAssembly.directCrs=true -Dartisynth.gpuAssembly.verifyCrs=true`.
Semantics: "run the directCrs device assembly AND verify it against CPU each
step." verifyCrs alone (no directCrs) keeps the old CPU-context behaviour. Cost:
assembles a CPU reference every step (defeats the perf win) — intended as a
dev/CI correctness gate, not for production runs.

Verified both ways:
- positive: passes on the fixed code (FemModel3dTest with both flags → Passed).
- negative: temporarily scaling the geometry device contribution by 1.05 made it
  throw `GPU device CRS verification failed for backwardEuler: max error ... at
  CRS value 1237 (gpu=45.62, cpu=44.25)`, then restored to 1.0.

This now guards the whole device assembly (marshalling + kernels) for ANY future
material/element kernel: enable the two flags in a GPU CI run and a §8c-class bug
fails immediately instead of silently corrupting the solve.

## 14. KKT-path verify + second bug (rest vs current positions) (2026-06-07)

Extended the device verifyCrs net to the constrained (ConstrainedBackwardEuler /
KKT) integrator path. The KKT solve factors its M block from the same
contribution descriptors as backwardEuler but scatters them through the KKT
M-block slot map (`getKktMBlockSlotMap`). `MechSystemSolver.verifyKktMDeviceCrsValues`
re-assembles those contributions standalone on a scratch cuDSS solver (M-block
CRS structure), reads them back, and compares to a CPU reference in the same
ordering — catching both a §8c-class marshalling bug and an M-block slot-map
error, without touching the live KKT solve. Runs per kktFactor step when
`-Dartisynth.gpuAssembly.kktDeviceValues` + `-Dartisynth.gpuAssembly.verifyCrs`
are set. The 8-call device dispatch was extracted to a shared helper
`addStiffnessDeviceValues` (used by the backwardEuler path and both verifiers).

**Second real bug found by this (multi-step):** the linear-elastic GEOMETRY
kernel marshalling built `elemNodePositions` from `getLocalPosition()` (the
CURRENT, deformed positions). Non-corotated linear stiffness is evaluated at the
REST configuration (it is constant), so the device stiffness DRIFTED as the mesh
deformed — perfect at step 1 (rest==current), diverging thereafter (the KKT
verify showed mismatches growing 0 -> 954 -> 1021, maxErr 1e-2 over steps). Fix:
use `getRestPosition()` (FemModel3d.addLinearElasticStiffness3CrsValueContributions).
After the fix the KKT verify holds at maxErr ~2e-14 across all steps. This is why
the single-step `testBackwardEulerLinearElasticEquivalence` missed it — a proper
multi-step assembly guard belongs in the CI verify task.

**Third real bug, FIXED:** a multi-step cuDSS-device vs Pardiso backward-euler
behavioural comparison diverged (~11% after 6 steps) even though
`verifyGpuDeviceCrsValues` passed every step (the assembled MATRIX is correct) —
so the divergence was in the directCrs RHS, not the stiffness. Root cause: the
ELEMENT-kernel contribution builders (linear-elastic geometry, material3,
dilational) fill only their device descriptor arrays, NOT the context's CPU CRS
value array (`myCrsValues`) — only the generic/block3/diag3 builders mirror to
`myCrsValues`. So `directCrsVelValues = directCrsContext.getCrsValues().clone()`,
captured after the velocity-Jacobian *Contributions* assembly, was missing the
stiffness-damping (`beta*K`) part, and the `mulAddCrsValues(myB, myU, ...)` J*v
term was incomplete. Harmless at step 1 (v=0), wrong once velocity builds up
(~beta = stiffnessDamping ≈ 11%). Fix: assemble the COMPLETE velocity Jacobian on
the host via the neighbor-based `assembleGpuVelJacobianCrsValues` for the J*v
term (the device assembly still uses the descriptors). After the fix the
multi-step behavioural comparison matches to ~1.1e-15, and the linear equivalence
test now steps 6 times. (Perf note: this adds a CPU velocity-Jacobian CRS scatter
to the directCrs path; computing J*v on the device later would remove it.)

Gotcha discovered: `corotated` is an **inherited** property
(`LinearMaterialBase`, default `PropertyMode.Inherited`); constructing
`new LinearMaterial(E, nu, false)` is not enough — the value is reset when the
material is attached to the model. Must call
`mat.setCorotatedMode(PropertyMode.Explicit)` (this is why `BigBeam3d` does it).

## 10. Open questions / risks

- Atomic-add determinism in the reduction (plan §6): not yet stressed because
  kernels haven't run end-to-end on a large model.
- `valueSource=hostMatrixValuesToDeviceCrs` vs. true GPU assembly: confirm the
  routing actually switches to GPU-sourced values once the gate is satisfied.
- Corotated default means most existing demos will silently stay on CPU — the
  GPU path's real-world coverage is currently near zero until step 3 above.
