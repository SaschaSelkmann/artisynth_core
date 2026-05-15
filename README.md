ArtiSynth: Combined MultiBody/FEM Simulation
--------------------------------------------
[<img src="https://www.artisynth.org/uploads/Main/ArtiSynthSplashSmall.png">](https://www.artisynth.org)

This is the core distribution of ArtiSynth, a 3D mechanical modeling
system that supports the combined simulation of multibody and finite
element (FEM) models, together with contact and constraints.  It is
targeted predominantly at biomechanical and biomedical applications,
but can also be used for general purpose mechanical simulation. It is
freely available under a two-clause BSD-style open source license.

The system is implemented in Java, and provides a rich set of modeling
components, including particles, rigid bodies, finite elements with
both linear and nonlinear materials, point-to-point muscles, and
various bilateral and unilateral constraints including contact. A
graphical interface allows interactive component navigation, model
editing, and simulation control.

Full details are provided on the ArtiSynth website at
[www.artisynth.org](https://www.artisynth.org).

Installation instructions, including what to do after
you clone from Github, are available at
[www.artisynth.org/installGuides](https://www.artisynth.org/installGuides).

Other documentation on how to use the system and build models is
available at [www.artisynth.org/doc](https://www.artisynth.org/doc).

List of publications in which ArtiSynth was used: [www.artisynth.org/Main/Publications](https://www.artisynth.org/Main/Publications)

--------------------------------------------------------------------

### Running on WSL2 (Windows Subsystem for Linux)

ArtiSynth can run with a full GUI inside WSL2 using WSLg, which is
included in Windows 11 and Windows 10 builds 22000+.

**Prerequisites**

- WSLg enabled (check: `echo $DISPLAY` should print `:0` or similar)
- Java 17+ installed in WSL2 (e.g. `sudo apt install openjdk-17-jdk`)
- `gcc` installed for the one-time shim build (`sudo apt install gcc`)

**Steps**

```bash
# 1. From the artisynth_core directory, set up the environment:
source setup.bash

# 2. Launch ArtiSynth:
artisynth
```

`bin/artisynth` automatically detects WSL2 and applies a workaround for
a Mesa GLX crash (`SIGSEGV` in `glXQueryDrawable`) that otherwise
prevents JOGL from initialising.  On first run it compiles a small
native shim (`support/linux/wsl2_glx_fix.c`) into
`lib/Linux64/libwsl2_glx_fix.so` and loads it via `LD_PRELOAD`.
Subsequent runs reuse the compiled shim.

Rendering uses Mesa's software renderer (llvmpipe, OpenGL 4.5), so no
GPU pass-through is required.  Performance is adequate for interactive
use; complex FEM models may run slower than on a native Linux desktop.

--------------------------------------------------------------------

### Optional: GPU-accelerated solver via NVIDIA cuDSS

ArtiSynth ships with an optional cuDSS-based sparse direct solver
backend that accelerates the linear-system solves at the core of every
implicit integrator (BackwardEuler, ConstrainedBackwardEuler,
FullBackwardEuler, Trapezoidal, static), plus the KKT, contact, and
friction paths via `KKTSolver` and `MurtyMechSolver`. PARDISO remains
the default; cuDSS is opt-in.

The backend includes a preconditioned BiCGStab "hybrid solve" path
(equivalent to PARDISO's CGS-with-stale-factor) so the per-step
factorization can be skipped on stable matrices.

**When cuDSS wins:** large FE-heavy models (~50k+ DOF), constrained FE
with bilateral attachments and joints. For these workloads cuDSS
typically runs 1.2-1.7x faster than PARDISO 8-thread on a single
modern GPU. Validated against PARDISO at 1e-8 relative residual on
equality KKT systems.

**When cuDSS loses:** small models (a few hundred to a few thousand
DOF). For small contact/friction demos, GPU launch overhead dominates
and PARDISO can be 2-20x faster. The bottleneck for these models is
typically CPU-side FEM assembly, which is *not* what the solver
accelerates.

#### Prerequisites

- NVIDIA GPU with compute capability >= 7.0 (Volta or newer).
- CUDA 12 toolkit (`nvcc`, `cuda_runtime.h`).
- NVIDIA cuDSS 0.7.x for CUDA 12 (`libcudss.so`, `cudss.h`).
- cuSPARSE and cuBLAS (ship with the CUDA toolkit).
- `gcc` / `g++` to build the JNI bridge.
- Same Java toolchain as the regular ArtiSynth build.

#### Installing CUDA and cuDSS on Ubuntu / Debian / WSL2

Add NVIDIA's APT repository:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

(Substitute `ubuntu2204` etc. for your distribution as needed.)

Install the CUDA toolkit and cuDSS for CUDA 12:

```bash
sudo apt install cuda-toolkit-12-8 libcudss0-dev-cuda-12
```

On WSL2 the CUDA installation uses the Windows-side GPU via NVIDIA's
WSL driver, so no additional driver is needed inside WSL itself. The
Windows host must have a recent enough NVIDIA driver (R555+ for CUDA
12.8).

Verify the install:

```bash
nvcc --version                        # should report CUDA 12.x
ls /usr/include/libcudss/12/cudss.h   # should exist
nvidia-smi                            # should list your GPU
```

The build links explicitly against the CUDA-12 variant of cuDSS at
`/usr/lib/x86_64-linux-gnu/libcudss/12/`, regardless of what
`/etc/alternatives/libcudss.so` points to (which may default to
CUDA-13 on multi-version installs).

#### Building the cuDSS JNI bridge

The regular `make` build does NOT compile any cuDSS code, so machines
without CUDA continue to build normally. The bridge is opt-in:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
cd src/maspack/solvers/lib
make cudss
```

This produces `lib/Linux64/libCuDssJNI.so.0.7.1`. Successful build
links against `libcudss.so.0`, `libcudart.so.12`, `libcusparse.so.12`,
and `libcublas.so.12` (verify with `ldd lib/Linux64/libCuDssJNI.so.0.7.1`).

Environment variable overrides for non-default installation paths:

| Variable | Default | Meaning |
|---|---|---|
| `CUDA_HOME` | `/usr/local/cuda` | CUDA toolkit root |
| `CUDSS_INC` | `/usr/include/libcudss/12` | cuDSS headers |
| `CUDSS_LIB` | `/usr/lib/x86_64-linux-gnu/libcudss/12` | cuDSS shared libs |

#### Running ArtiSynth with cuDSS

Once `lib/Linux64/libCuDssJNI.so.0.7.1` exists, select cuDSS via the
`-matrixSolver` CLI option:

```bash
source setup.bash
artisynth -matrixSolver CuDss -model artisynth.demos.fem.BigBeam3dConstrainedKKT
```

`CuDss` is also selectable from the GUI: open the MechModel's property
panel and change `matrixSolver` to `CuDss`. Models with implicit
friction (`-useImplicitFriction`) also pick up cuDSS automatically.

If the native library is missing or the GPU is unavailable, ArtiSynth
prints a clear message and falls back to PARDISO without crashing:

```
Matrix solver CuDss requested but cuDSS native library is unavailable;
staying on Pardiso
```

#### Benchmark demos

These models are included specifically as cuDSS benchmarks:

| Demo | Integrator | What it exercises |
|---|---|---|
| `artisynth.demos.fem.BigBeam3dBE` | `BackwardEuler` (unconstrained) | Regular FE direct solve through `myDirectSolver` |
| `artisynth.demos.fem.BigBeam3dConstrainedKKT` | `ConstrainedBackwardEuler` | KKT path with bilateral constraints |
| `artisynth.demos.fem.ArticulatedFemBig` | `ConstrainedBackwardEuler` | KKT + rigid bodies + hinge joints + FEM attachments |

Each accepts `-nx`, `-ny` (or `-nelemsx`, `-nelemsz`) for mesh
refinement; defaults are sized to make the GPU advantage visible.

#### Diagnostics: profiling per-phase timing

To see where each step's time goes (transfer / factor / solve / BiCGStab):

```bash
CUDSS_BRIDGE_TIMING=1 artisynth -matrixSolver CuDss \
   -model artisynth.demos.fem.ArticulatedFemBig
```

The bridge emits stderr lines like:

```
[cudss-timing] factor n=33458 nnz=617447: H2D vals=0.65ms FACTOR=47.55ms total=48.20ms
[cudss-timing] solve n=33458: H2D b=0.14ms SOLVE=11.55ms D2H x=0.09ms total=11.75ms
[cudss-timing] bicgstab n=33458 iters=2: total=12.40ms (6.20ms/iter)
```

The same toggle also enables ArtiSynth's `profileKKTSolveTime` in the
provided benchmark demos, giving a complete CPU-vs-GPU breakdown.
Programmatic toggle from Java: `CuDssSolver.setTimingEnabled(true)`.

#### Supported features

| Path | Backend | Notes |
|---|---|---|
| Regular implicit solve (`backwardEuler`) | cuDSS | BiCGStab hybrid available |
| Constrained KKT (`constrainedBackwardEuler`, `Trapezoidal`, `fullBackwardEuler`) | cuDSS | via generalized `KKTSolver` |
| Static analysis (`StaticIncrementalStep`, `StaticLineSearch`) | cuDSS | via `KKTSolver` |
| Position-correction projection | cuDSS | via `KKTSolver` |
| Rigid-body contact projection | cuDSS | via `RigidBodySolver` |
| Unilateral contact (LCP via `KKTSolver.buildLCP`) | cuDSS | multi-RHS path |
| Active-set contact pivoting (`MurtyMechSolver`) | cuDSS | analyze/factor/solve dispatched generically |
| Friction (explicit and implicit) | cuDSS | Murty path |
| Explicit integrators (RK4, ForwardEuler, SymplecticEuler) | n/a | no linear solve to accelerate; matrix solver unused |

#### Known limitations

- `UmfpackSolver` is not supported as a Murty backend (rejected at
  construction). PARDISO and cuDSS only.
- Custom user-defined `FemMaterial` subclasses run on CPU for assembly;
  the solver is independent of material choice.
- Linux only; cuDSS has no macOS support and the Windows toolchain
  hasn't been validated. `CuDssSolver.isAvailable()` returns false
  cleanly on unsupported platforms.

--------------------------------------------------------------------

### Files in the top directory:

<dl>

<dt>.classpath</dt>
<dd>Classpath information used by the Eclipse IDE</dd>

<dt>.git*</dt>
<dd>Repository and configuration information used by the Git SCM</dd>

<dt>.project</dt>
<dd>Project information used by the Eclipse IDE</dd>

<dt>ArtiSynth.launch</dt>
<dd>Default "launch" configuration for used by the Eclipse IDE</dd>

<dt>ArtiSynth_launch</dt>
<dd>Backup copy of the default version of ArtiSynth.launch</dd>

<dt>.settings/</dt>
<dd>Settings information used by the Eclipse IDE</dd>

<dt>DISTRO_EXCLUDE</dt>
<dd>Files that should be excluded from the current precompiled distribution</dd>

<dt>EXTCLASSPATH.sample</dt>
<dd>Example EXTCLASSPATH file. An EXTCLASSPATH files indicates to the
ArtiSynth Launcher additional classpaths that should be used when
searching for models.</dd>

<dt>LICENSE</dt>
<dd>Licensing and terms of use</dd>

<dt>Makefile</dt>
<dd>Makefile for compiling and doing certain maintenance operations in
a shell environment</dd>

<dt>Makefile.base</dt>
<dd>Base definitions for Makefile and Makefile in subdirectories</dd>

<dt>README.md</dt>
<dd>This file</dd>

<dt>CODE_OF_CONDUCT.md</dt>
<dd>Project code of conduct</dd>

<dt>VERSION</dt>
<dd>Current distribution version</dd>

<dt>bin/</dt>
<dd>Stand-alone programs, mostly implemented as scripts.
The program 'artisynth' starts up the ArtiSynth system.</dd>

<dt>classes/</dt>
<dd>Root directory for compiled classes</dd>

<dt>demoModels.txt</dt>
<dd>List of "banner" demo models available when ArtiSynth starts up</dd>

<dt>doc/</dt>
<dd>System documenation</dd>

<dt>eclipseSettings.zip</dt>
<dd>Default project settings for the Eclipse IDE (obsolete since these
are now checked directly into the repository)</dd>

<dt>lib/</dt>
<dd>Java libraries, plus architecture-specific libraries for native
code support, mostly involving linear solvers, Java OpenGL (JOGL), and
collision detection</dd>

<dt>mainModels.txt</dt>
<dd>List of "banner" anatomical models in the ArtiSynth Models package
(which must be installed)</dd>

<dt>matlab/</dt>
<dd>Matlab scripts for running ArtiSynth from matlab</dd>

<dt>modelMenu.xml</dt>
<dd>Default configuraion for the model menu</dd>
    
<dt>scriptMenu.xml</dt>
<dd>Default configuraion for the model menu</dd>
    
<dt>scripts/</dt>
<dd>Jython scripts for basic testing, etc. Some of these may assume
the installation of additional projects.</dd>

<dt>setup.bash</dt>
<dd>Example ArtiSynth setup script for bash</dd>

<dt>setup.csh</dt>
<dd>Example ArtiSynth setup script for chs/tcsh</dd>

<dt>src/</dt>
<dd>ArtiSynth source code</dd>

<dt>support/</dt>
<dd>Configuration information for external IDEs and support software,
including default settings for the Eclipse IDE</dd>

<dt>tmp/</dt>
<dd>Temp directory</dd>

</dl>
