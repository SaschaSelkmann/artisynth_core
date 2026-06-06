# Plan: GPU FEM Assembly for ArtiSynth (with cuDSS Backend)

> **NOTE (2026-06-06):** Sections 1–10 below are the *original* plan and are
> kept for historical context. The implementation that actually landed
> deviates from that plan in architecture, toggle design, and cuDSS version.
> Read **Section 0 (Implementation status)** first — it is the authoritative
> description of where the code is today. Where Section 0 and Sections 3/9/10
> disagree, Section 0 wins.

## 0. Implementation status (as of 2026-06-06)

### Environment / build

- Branch: `sascha/cudss-0.8-stabilization` (33 commits ahead of `master`),
  **not** the originally proposed `feature/gpu-fem-assembly` off
  `feature/cudss-contact`.
- cuDSS bumped from **0.7.1 → 0.8.0** (original plan targeted 0.7.1). The
  branch name signals that the current priority is *stabilizing the 0.8
  bridge*, not adding new material/element kernels.
- Toolchain present and working: CUDA 12.8 (nvcc), GPU GTX 1650,
  `lib/Linux64/libCuDssJNI.so.0.8.0` (loaded as `CuDssJNI.0.8.0`).
- Java compiles clean (`make` → exit 0, javac with `--release 8`).
- Untracked `cudss/` is the NVIDIA CUDALibrarySamples reference checkout,
  not part of the repo.

### Architecture actually implemented (differs from Section 3)

There is **no** `artisynth.core.gpu` package and **no** `gpuFemAssembly.cu`.
The work was integrated into existing classes and into the cuDSS bridge:

| Original plan component        | What actually exists |
|---|---|
| `artisynth.core.gpu.*` package | `MechSystem.GpuAssemblyContext` + assembly hooks in `FemModel3d`, `MechSystemSolver`, `MechSystemBase`, `MechModel` |
| `GpuFemAssembler` / `GpuFemState` | `GpuAssemblyContext` carries CRS slot maps + per-contribution descriptor arrays |
| `GpuFemKernels` registry       | capability predicates in `FemModel3d` (`isGpuLinearElasticMaterial`, `materialSupportsGpuMaterialStiffness3`, `canAssembleLinearElasticStiffness3CrsValueContributions`, …) with per-step CPU fallback |
| `gpuFemAssembly.{h,cu}`         | kernels added to `src/maspack/solvers/lib/cudssBridgeKernels.cu` |
| `getValuesDevicePointer()` on bridge | `CuDssSolver` device-values API: `clearDeviceValues`, `addMaterialStiffness3DeviceValues`, `factorDeviceValues`, plus scatter/diag/block kernels |
| per-model `FemModel3d.setGpuAssembly(boolean)` | **global** static flags in `MechSystemSolver` + `SimulationSettings`/`SimulationPrefs` + system properties (`artisynth.gpuAssembly.status`, `.profile`, `.kktDeviceValues`, `.directCrs`), auto-enabled when the cuDSS matrix solver is selected |

CUDA kernels present in `cudssBridgeKernels.cu`: `axmy`, `extractDiag`,
`zeroValues`, `scatterAddValues`, `addScaledDiagonal3`, `addScaledBlock3`,
`addMaterialStiffness3` (+ per-element variant), `addLinearElasticStiffness3Element`,
`addLinearElasticStiffness3ElementGeometry`, `addDilationalStiffness3Element`.

### Stage status

| Stage | Status | Notes |
|---|---|---|
| G1 Profiling | **done** | `GpuAssemblyProfiling` flag; per-phase timing in `updateStressAndStiffness`, `addPosJacobian`, `addVelJacobian`, `MechSystemSolver` |
| G2 Capability + toggle | **done (redesigned)** | global toggles instead of per-model; capability predicates + step-by-step CPU fallback |
| G3 Tet+Linear PoC | **partial** | kernels written + unit-tested at the device-call level; not yet exercised end-to-end (see gating below) |
| G4 Direct GPU CSR write | **done & validated** | `deviceCrs` path engages in `testBackwardEulerDirectCrsSolve`; KKT M device-values path passes `KKTSolverTest` |
| G5 Material coverage | **partial** | generic `addMaterialStiffness3` (CPU-computed tangent D per IP) + linear-elastic kernel; no nonlinear stress kernels (MooneyRivlin/NeoHookean) yet |
| G6 Element coverage | **partial** | linear-elastic element + geometry kernels; material-stiffness batch is element-agnostic over IP contributions |
| G7 Damping/mass | **partial** | active mass blocks routed through GPU contribution batches |
| G8 Constraint Jacobians | not started | (plan defers indefinitely) |
| G9 Perf hardening | not started | |

### Verification run (2026-06-06, GTX 1650, cuDSS 0.8.0)

All passing: `SparseNumberedBlockMatrixTest`, `KKTSolverTest` (incl. cuDSS
device-values / refactor / equality), `CuDssSolverTest`, `FemModel3dTest`
(incl. `testFemNeighborCrsAssembly`, `testMaterialStiffness3ContextAssembly`),
and `testBackwardEulerDirectCrsSolve` (real cuDSS `BackwardEuler` step).

### Key open issue — GPU assembly kernels do not fire in realistic configs

In `testBackwardEulerDirectCrsSolve` the device-CRS solve *does* engage
(`directCrs=true deviceCrs=true directCrsStatus=deviceValues`), **but** the
status line reports `valueSource=hostMatrixValuesToDeviceCrs` and the kernel
contribution counters are all zero
(`material3=0 linearElem3=0/0/0 linearGeomElem3=0/0/0 dilationElem3=0/0/0`);
only generic `diag3` / `block3` scatter is used. The FEM stiffness is still
assembled on the CPU and merely copied into the device CRS buffer.

Cause (confirmed via `-Dartisynth.gpuAssembly.profile=true`): the specialized
kernels are gated off for the common case.

- `linearElem3 disabled: corotated=true` — the GPU linear-elastic kernel
  requires `!mat.isCorotated()`, but the **default `LinearMaterial` is
  corotated**, so demos/tests that don't explicitly pass `corotated=false`
  never hit it.
- `material3 disabled: softIncomp=ELEMENT` — element-level soft
  incompressibility (the default) also disables the material3 batch.
- Additional gates: no shell elements, no augmenting/auxiliary materials,
  no Young's-modulus field, no indirect neighbor contributions.

**Implication:** G3/G5/G6 are implemented at the kernel + data-structure
level and unit-tested in isolation, but there is no end-to-end test that
(a) drives a realistic `FemModel3d` through the linear-elastic / dilational
kernels with non-zero contribution counters, and (b) asserts the resulting
stiffness/forces match the CPU path within tolerance. Closing that gap is
the natural next step before extending material/element coverage.

### Recommended next steps

1. Build an equivalence test that satisfies the gate (non-corotated
   `LinearMaterial`, soft-incompressibility off), confirm
   `linearElem3`/`linearGeomElem3` counters become non-zero, and Frobenius-
   compare GPU-assembled vs CPU-assembled stiffness to ~1e-10 (the G3/G7
   acceptance criterion the plan asks for but that isn't yet met end-to-end).
2. Decide whether corotated `LinearMaterial` — the actual common case — is
   in scope for a GPU kernel; if yes it becomes the real G3 target.
3. Only then proceed to nonlinear materials (G5) and more element types (G6).
4. Capture a real per-phase profile on `ArticulatedFemBig` to confirm the
   assembly speedup target (≥3×) before broadening coverage.

---

This document plans the next major optimization round after the cuDSS solver
work (`plan_fe.md`, `plan_kkt.md` / `plan_kkt_revised.md`, branches
`feature/cudss-fe-backend`, `feature/cudss-kkt-equality`,
`feature/cudss-contact`).

The motivation is concrete and data-driven. From the profiling we landed in
the cuDSS work (`CUDSS_BRIDGE_TIMING=1` + `profileKKTSolveTime`):

```
ArticulatedFemBig (33458 DOF, ConstrainedBackwardEuler, step 2):
  Force / state setup (FEM stress + stiffness + force assembly)  ~185 ms  65%
  KKT 'build matrix' (Maspack block -> CRS assembly)              ~65 ms  23%
  cuDSS factor + solve                                            ~26 ms   9%
  H2D / D2H transfers                                              ~1 ms  <1%
  Step total                                                     ~287 ms 100%
```

cuDSS is finishing its share in 9% of the step. The other **88% is upstream
CPU work** that the cuDSS path cannot affect. To make GPU acceleration win
in absolute step time (not just in the solver phase), we have to move that
work onto the GPU too. This document plans that effort.

The single hard constraint: **no ArtiSynth functionality may be lost**.
Every existing demo, every custom material a user wrote, every element
type, must continue to work after this plan lands.

## 1. Scope

In scope:

- GPU-resident FEM internal-force and tangent-stiffness assembly for the
  **built-in** element types (Tet, Hex, Pyramid, Wedge, and their quadratic
  variants) and the **built-in** material families (LinearMaterial,
  MooneyRivlin, NeoHookean, StVenantKirchoff, and the handful of others
  that are widely used in demos).
- Direct write of assembled CSR values into the GPU-side buffer cuDSS
  already owns, eliminating the Maspack-side block-to-CRS conversion for
  the FEM portion of the global matrix.
- A capability-detection layer so each `(element, material)` pair routes
  to GPU when the kernel is available and to the existing CPU path
  otherwise. Mixed assembly is supported.
- Opt-in flag at the `FemModel3d` level (`setGpuAssembly(boolean)`), off
  by default.
- Numerical-equivalence tests against the CPU path for every supported
  combination.

Explicitly out of scope:

- Custom user-defined `FemMaterial` / `FemElement3d` subclasses — these
  always stay on the CPU path.
- Shell and membrane elements (different integration; defer to a later
  round if there's demand).
- BSpline3dElement (specialized; defer).
- Embedded mesh + skinning forces (not on the FEM stiffness path).
- Rigid body inertia, joint constraints, contact forces — already cheap
  on CPU and not the bottleneck.
- Position / velocity integration itself — the few-MB state vector
  update is fast on CPU.
- Multi-GPU.
- Switching ArtiSynth's primary state to live on the GPU (kept on Java
  side; mirrored to GPU per step).

The Maspack-side block-to-CRS work for *non-FEM* portions of the global
matrix (rigid bodies, joints, attachments) stays on CPU. The cuDSS bridge
writes FEM-assembled CSR values into the right slots of the global CSR
buffer; everything else is CPU-assembled and copied as today.

## 2. Functionality-preservation strategy

Four guarantees:

**(a) Behavioral.** Every demo and model under `artisynth.demos.*` and
`artisynth.models.*` runs to completion and produces identical (within
double-precision rounding) behavior, whether or not GPU assembly is
enabled. Verified by: a deterministic-state-comparison test that steps
selected demos N times with both paths and asserts position equality
within a configurable epsilon. Failures gate the commit.

**(b) Numerical.** GPU and CPU paths must produce bit-identical (or
epsilon-identical) global stiffness matrices and internal force vectors
on every supported `(element, material)` combination. Verified by: a
matrix-level unit test that calls `addPosJacobian` / `updateForces` once
with each path and Frobenius-compares the results. Tolerance: 1e-10 of
the matrix's Frobenius norm. Failures gate the per-material commit.

**(c) API.** No public method signatures change. `FemModel3d.updateForces`,
`addPosJacobian`, `getStiffnessMatrix`, etc. retain exact current
semantics. GPU acceleration is invisible to model authors and to
existing user code. The only new public API is the opt-in toggle:
`FemModel3d.setGpuAssembly(boolean)`.

**(d) Extensibility.** Users can still subclass `FemMaterial` and
`FemElement3d` with arbitrary Java code. Any model containing a
non-built-in material or element type automatically falls back to the
CPU path for that piece. Verified by: a unit test that registers a
trivial custom-material subclass and confirms it still works under
`setGpuAssembly(true)`.

A central capability-detection method makes the routing decision:

```java
boolean canAssembleOnGpu (FemElement3dBase e, FemMaterial mat) {
    return GpuFemKernels.kernelFor(e.getClass(), mat.getClass()) != null;
}
```

`GpuFemKernels.kernelFor()` returns a function pointer to the right
GPU kernel pair (stress + tangent) or `null` if the combination isn't
supported. Adding a new material is a matter of writing the kernel
and registering it; nothing else changes.

## 3. Architecture sketch

```
FemModel3d.updateStressAndStiffness()        [CPU orchestration]
  ├── if (gpuEnabled && allElementsGpuSupported())
  │      GpuFemAssembler.assemble (this)     [single GPU dispatch]
  │        ├── push node positions H2D        [persistent buffer; small]
  │        ├── launch per-element-type kernels [stress + force + K_e]
  │        ├── atomic-add into global GPU CSR values buffer
  │        └── (skip H2D of vals at solve time; already there)
  │
  └── else: existing per-element CPU loop      [unchanged]
        for (e : myElements)
            computeStressAndStiffness (e, ...)
            -> writes into FemNodeNeighbor.myStiffness
        Maspack block-to-CRS at solve time     [unchanged]
```

The crucial integration point: the GPU CSR values buffer is **the same
one cuDSS factor/solve already uses** — `myValsD` in `CuDssBridge`. After
GPU assembly fills it, we no longer need the per-step
`cudaMemcpyAsync(myValsD, vals, ...)` that currently happens in
`factor()`. Saves ~3 ms / step on top of the assembly itself.

New top-level Java components:

```
artisynth.core.gpu.FemAssemblyConfig       // global opt-in flags
artisynth.core.gpu.GpuFemAssembler          // per-FemModel3d, owns
                                            //   GPU element data
artisynth.core.gpu.GpuFemKernels            // material/element kernel
                                            //   registry
artisynth.core.gpu.GpuFemState              // node positions, rest
                                            //   positions on GPU
```

New native:

```
src/maspack/solvers/lib/gpuFemAssembly.{h,cu}    // GPU kernels and
                                                  // dispatch
src/maspack/solvers/lib/gpuFemAssemblyJNI.cc     // JNI bridge
```

The cuDSS native code (`cudssBridge`) gains a method to expose its
`myValsD` device pointer to `GpuFemAssembler`, so the assembler can write
directly into cuDSS's buffer instead of going through host.

## 4. Staged delivery

Each stage is independently testable, independently committable, and
preserves all existing functionality.

### Stage G1: Profiling + decision baseline

Capture the actual per-phase CPU breakdown so the rest of the plan can
be ranked by impact rather than guesswork.

Deliverables:

- New `FemModel3d.setProfilingDetailed(boolean)` that emits per-phase
  timing inside `updateStressAndStiffness` (element loop, per-material
  category if cheap to identify, post-loop incompressibility,
  symmetrization).
- Run the breakdown on `BigBeam3dConstrainedKKT` and `ArticulatedFemBig`
  and document where the 185 ms goes.
- No production code paths change.

Outcome: a numeric ranking of which materials / element types are worth
GPU-accelerating first. If a single material dominates (likely
LinearMaterial or MooneyRivlin in our demos), that's stage G3.

### Stage G2: Capability layer + opt-in toggle

Land the routing infrastructure without any GPU code yet.

Deliverables:

- `FemModel3d.setGpuAssembly(boolean)`, default false. New
  `gpuAssembly` model property listed in the GUI.
- `GpuFemKernels.kernelFor(elementType, materialType)`. Returns null
  for all combinations initially.
- `FemAssemblyConfig.isGpuAvailable()` — true when cuDSS is loaded
  AND a `GpuFemAssembler` can be constructed.
- `updateStressAndStiffness` adds the routing check but the GPU branch
  is never taken yet (no kernels registered).

Outcome: a no-op refactor that the next stages plug into.

Acceptance: all existing tests pass; FE benchmarks unchanged.

### Stage G3: First-material proof of concept — Tet + LinearMaterial

The smallest end-to-end vertical slice. LinearMaterial is the simplest
constitutive law (constant tangent, linear stress) and Tet elements
have one integration point.

Deliverables:

- `GpuFemState` mirrors node positions + rest positions to a GPU
  buffer (one mirror per FemModel3d). Updated each step via
  cudaMemcpyAsync.
- `gpuFemAssembly.cu`: kernel `tet_linear_stress_stiffness`. One
  thread per Tet element. Computes element deformation gradient,
  applies linear stress, integrates K_e, writes per-node-neighbor
  contributions into a per-element shared buffer.
- A reduction step that atomic-adds each element's contributions
  into the global CSR values buffer (whose layout matches what
  Maspack would have produced).
- The CSR layout for FemModel3d is computed once during analyze and
  cached as a per-element block-position table on the GPU. Computing
  this table is the slot that previously took place inside
  Maspack's `getBlockCRSValues`.

Validation (numerical):

- A new `GpuFemAssemblyTest` builds a small Tet+LinearMaterial model,
  computes the global stiffness matrix and internal force vector
  under both CPU and GPU paths, and Frobenius-compares to 1e-12.

Validation (behavior):

- Run a Tet+LinearMaterial demo (e.g. `BigBeam3d` with TetGrid +
  LinearMaterial) for N steps both ways; assert per-node positions
  equal to 1e-10 each step.

Performance check:

- Time `updateStressAndStiffness` under both paths on ArticulatedFemBig
  with all-LinearMaterial elements. Expect 5-20x speedup on this
  phase. If less than 2x, escalate to design review before adding
  more materials.

Acceptance: numerical + behavior tests pass; existing demos unchanged
under `setGpuAssembly(false)` (default).

### Stage G4: Direct GPU CSR write

Skip the device-to-host of FEM values; let the assembler write
directly into cuDSS's `myValsD`.

Deliverables:

- `CuDssBridge` exposes a `getValuesDevicePointer()` to
  `GpuFemAssembler`, with a checked contract that the layouts match.
- The assembler's reduction step writes to that pointer instead of
  a separate buffer.
- The per-step `cudaMemcpyAsync(myValsD, vals_host, ...)` in
  `CuDssBridge::factor` is bypassed when assembly was on GPU.

Validation: same as G3, plus a check that the value buffer cuDSS sees
matches the CPU-assembled values byte-for-byte (subject to atomic-add
nondeterminism, which is tolerable since we control kernel
determinism at the warp level).

Performance check: factor/solve trace should show H2D dropping to
~0 ms. Total step-time win on ArticulatedFemBig: target 65% of the
65 ms 'build matrix' gone (the FEM portion of it).

### Stage G5: Material coverage

Add the next-most-used materials one at a time. Each is a separate
commit with its own numerical test and demo regression.

Order by frequency in `artisynth.demos.*` (G1 determines this; tentative):

- G5a: MooneyRivlin (nonlinear; very common in demos)
- G5b: NeoHookeanMaterial
- G5c: StVenantKirchoffMaterial
- G5d: LinearMaterial variants (transversely isotropic, etc.)

Each commit:
- Adds one or two kernels (stress + tangent) to `gpuFemAssembly.cu`.
- Registers them in `GpuFemKernels`.
- Adds a unit test that compares CPU vs GPU on a synthetic mesh with
  that material.
- Verifies one demo that uses the material runs identically.

Materials that are rarely used (e.g. ViscoelasticMaterial,
ScaledFemMaterial wrappers around custom inner materials) stay on
CPU forever. The capability layer makes that automatic.

### Stage G6: Element type coverage

Same per-element approach.

Order by frequency:

- G6a: HexElement (8 IP, very common)
- G6b: WedgeElement (6 IP)
- G6c: PyramidElement (5 IP)
- G6d: QuadtetElement (4 IP)
- G6e: QuadhexElement (27 IP, heavy)

The kernel for each element is a separate function that gets called
from a per-element-type dispatch loop. The material kernels written
in G5 are reused.

### Stage G7: Damping + mass

Damping: Rayleigh = alpha*M + beta*K, both element-local. Mass
matrix: usually static or rarely-updated, so the gain is small unless
mass-proportional remodeling is in play. Modest win; do if the
profile in G1 says it matters.

### Stage G8: Constraint Jacobians

`G` (bilateral) and `N` (unilateral) constraints come from joints,
attachments, contact. They are typically much smaller than `K` (a few
thousand entries vs millions). Probably not worth GPU-fying — defer
indefinitely unless data says otherwise.

### Stage G9: Performance hardening (optional)

If G3-G7 land speedups but a real workload still has CPU hot spots,
investigate:

- CUDA Graph capture for the per-step assembly + factor + solve
  pipeline (the option we deferred from the cuDSS speedup round).
- Pinned host memory for the position mirror (we showed earlier this
  is marginal; reconsider only after assembly is on GPU).
- Async overlap: position update + GPU assembly + GPU factor
  pipelined across steps.
- Mixed precision (FP32 for kernels, FP64 for cuDSS factor).
- Multiple CUDA streams if FEM models can be assembled independently.

## 5. Estimated effort

Honest sizing, on an experienced single developer:

| Stage | Effort | Notes |
|---|---|---|
| G1 | 1 day | Mostly instrumentation |
| G2 | 2 days | Pure refactor + scaffolding |
| G3 | 2-3 weeks | First kernel + reduction + tests + perf review |
| G4 | 3-5 days | Direct CSR write; subtle layout work |
| G5 | 1 week per material | 4 materials = 4 weeks |
| G6 | 1 week per element type | 5 types = 5 weeks |
| G7 | 1-2 weeks | If pursued |
| G8 | skip |  |
| G9 | open-ended | Pure perf, no API change |

Total minimum scope (G1-G6 only, no damping/Jacobian/perf-hardening):
~12-14 weeks of focused work. A more realistic calendar estimate with
review cycles, debugging on unusual configurations, and CI-on-GPU
setup: 4-6 months.

This is fundamentally a research-project-sized undertaking, not a
sprint. It should be planned as such.

## 6. Risk register

Numerical:

- **Floating-point determinism of atomic adds.** Atomic add order is
  nondeterministic in CUDA, so two GPU runs of the same model may
  produce bit-different stiffness values, which propagate through the
  factorization. Mitigation: deterministic ordering at the per-element
  reduction (sort contributions, sum in fixed order), accepting a
  small perf hit, OR document the nondeterminism and rely on it being
  smaller than ArtiSynth's existing FP-rounding noise. Decision
  deferred to G3.

- **Numerical-equivalence regressions during material additions.**
  G5/G6 each add code; each must pass the equivalence test before
  the commit lands. Discipline.

Functionality:

- **Missed extension paths.** Some FEM extension points (custom
  integration rules, augmenting materials, muscle materials with
  unusual force directions) are easy to overlook. Mitigation: every
  G5/G6 commit runs the demo regression. If a demo regresses, the
  capability layer adds an automatic fallback for that combination.

- **Mid-simulation switching.** A model might transiently introduce
  a non-GPU element (e.g. user code activates a custom material at
  step 100). Mitigation: capability check runs once per
  updateStressAndStiffness, not once at init. The fallback is
  step-by-step.

- **Embedded muscle / skinning forces.** Some demos add forces
  outside the FEM element loop. These remain CPU-assembled and just
  add into the same global force vector. No interaction with the
  GPU CSR.

Performance:

- **Small models will be slower.** Below ~10k DOF the GPU pipeline
  overhead dominates the FEM assembly cost. Mitigation:
  `setGpuAssembly` default false; users opt in for big models.
  Auto-tune in G2 to pick a sensible threshold? Stretch.

- **Memory pressure.** A GPU with 8 GB free can hold a few million
  FEM elements. Models with embedded surfaces and many materials may
  hit memory walls. Mitigation: track GPU allocation; fail gracefully
  (refuse to enable GPU assembly with a clear message).

Maintenance:

- **CUDA version drift.** cuDSS and cuSPARSE APIs evolve. Mitigation:
  pin the toolchain (we're on CUDA 12.8 + cuDSS 0.7.1 today); update
  with care.

- **GPU CI.** ArtiSynth's existing CI is CPU-only. Numerical-
  equivalence tests need a GPU runner. Mitigation: tests run locally;
  CI flag skips them when no GPU; manual gating on releases. Or set
  up a self-hosted GPU runner.

## 7. Go / no-go criteria

Proceed past G3 only if:

- LinearMaterial+Tet GPU path produces 1e-12 equal stiffness vs CPU.
- ArticulatedFemBig with all-LinearMaterial+Tet shows >= 3x speedup
  in `updateStressAndStiffness` under GPU vs CPU.
- All existing demos still pass.

Proceed past G4 only if:

- The direct CSR write produces a global value buffer byte-identical
  to what Maspack would assemble (or within atomic-nondeterminism
  bounds).
- BigBeam3dBE + cuDSS step time drops by at least 30% vs Stage 6
  numbers from `feature/cudss-fe-backend`.

Stop or rescope (treat as research, not production) if:

- Numerical equivalence is provably impossible (e.g. atomic-add
  nondeterminism breaks downstream tests).
- Memory footprint of GPU FEM state exceeds typical user GPUs (>16 GB).
- A representative real-user model requires custom materials that we
  cannot reasonably GPU-port and fallbacks dominate the timeline.

## 8. Relationship to the cuDSS branches

This plan extends the cuDSS work but does not modify it. Specifically:

- `feature/cudss-fe-backend`, `feature/cudss-kkt-equality`,
  `feature/cudss-contact` remain the supported delivery channels for
  cuDSS-accelerated SOLVING.
- This plan adds a new feature line for cuDSS-accelerated ASSEMBLY,
  on top of those branches.
- Suggested branch name: `feature/gpu-fem-assembly`, off
  `feature/cudss-contact`.

The opt-in toggles are independent:

```
-matrixSolver CuDss               (solver)             [already shipped]
+ FemModel.setGpuAssembly(true)   (assembly)           [this plan]
```

A model can run cuDSS without GPU assembly (today's behavior) or
with both. Running GPU assembly without cuDSS is technically
possible (the assembled CSR could be copied back to host for PARDISO)
but isn't useful — defer it.

## 9. What a first concrete deliverable would look like

After roughly the first 3 weeks (G1+G2+G3) the visible change is:

```bash
artisynth -matrixSolver CuDss -model artisynth.demos.fem.BigBeam3dBE
# (existing behavior, no change)

artisynth -matrixSolver CuDss -model artisynth.demos.fem.BigBeam3dBE \
   -script enable-gpu-assembly.py
# (gpu assembly enabled for the LinearMaterial+Tet elements,
#  resulting in measurably faster update of stress and stiffness)
```

`enable-gpu-assembly.py` is a one-line Jython script that calls
`mech.findComponent("models/fem").setGpuAssembly(true)`.

All other demos run unchanged. Custom-material models fall back to
CPU automatically. No public API beyond the toggle.

## 10. Decision points for the user

Before starting G1, several decisions need confirmation:

1. **Branch off** `feature/cudss-contact` (this is the latest cuDSS
   branch) and start `feature/gpu-fem-assembly`.
2. **First material** to target after profiling G1: usually
   LinearMaterial because it's simplest. If your real workloads use
   MooneyRivlin or other, start there instead.
3. **Atomic-add determinism**: accept nondeterminism (faster) or
   force deterministic reductions (slower). Decision can be deferred
   until G3 shows actual numbers.
4. **GPU CI**: are you OK running numerical-equivalence tests only
   locally for now, or do we want a GPU runner set up?
5. **Custom materials in your own workflow**: do you have any? If
   yes, name them so we can either include them in the GPU coverage
   or confirm the fallback path is the right answer.
