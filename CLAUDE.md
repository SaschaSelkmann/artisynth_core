# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

---

## Build Commands

```bash
# Source environment variables first (sets ARTISYNTH_HOME, CLASSPATH, PATH)
source setup.bash

# Compile all Java sources → classes/
make

# Equivalent via Gradle
./gradlew compileJava

# Compile a single package (run from within its src/ subtree)
cd src && make build

# Download/update native libraries (lib/) — runs automatically on first build
./bin/updateArtisynthLibs

# Generate Javadoc
make javadocs

# Clean compiled output
make klean
```

## Running ArtiSynth

```bash
source setup.bash           # must be done once per shell session

artisynth                                                    # GUI
artisynth -model artisynth.demos.tutorial.TutorialDemo       # load specific model
artisynth -noGui -model artisynth.demos.mech.RigidBodyDemo   # headless
artisynth -help                                              # all options
```

On WSL2 the launch script automatically preloads a Mesa GLX shim (`lib/Linux64/libwsl2_glx_fix.so`) to prevent a SIGSEGV; this is built once from `support/linux/wsl2_glx_fix.c`.

## Running Tests

```bash
# Unit tests in the current directory
make test

# Recursively run all unit tests
make TEST

# Regression test suite (Jython-based, headless)
./bin/regression.sh
```

Unit-test programs are declared via `JAVA_TEST_PROGRAMS` in each package's `Makefile`. Regression scripts live in `scripts/` (e.g. `basicRegressionTest.py`, `femRegressionTest.py`, `inverseRegressionTest.py`) and are driven by Jython inside ArtiSynth.

## Repository Layout

```
artisynth_core/
├── src/
│   ├── artisynth/
│   │   ├── core/          # simulation framework (see below)
│   │   └── demos/         # runnable example models
│   └── maspack/           # math, geometry, rendering, utilities
├── classes/               # compiled output (git-ignored)
├── lib/                   # third-party jars + platform natives
├── bin/                   # shell scripts (artisynth, compile, regression…)
├── scripts/               # Jython regression test scripts
├── doc/                   # Javadoc generation
└── support/               # IDE configs, WSL2 GLX shim source
```

## Architecture

### Two top-level libraries

| Package root | Role |
|---|---|
| `artisynth.core.*` | Simulation framework, GUI, model API |
| `maspack.*` | Math, geometry, rendering, properties, solvers |

Keep their boundaries clear: `maspack` is a standalone utility layer; `artisynth.core` builds on top of it.

### Key sub-packages in `artisynth.core`

- **driver** — application entry point (`Launcher`, `Main`), scheduler, model loading
- **modelbase** — component hierarchy base classes; every model element is a `ModelComponent` with a parent/child tree
- **mechmodels** — rigid bodies, points, frames, joints, constraints, collision
- **femmodels** — finite element models, nodes, elements, material properties
- **materials** — constitutive material definitions (FEM and contact)
- **inverse** — inverse kinematics/dynamics solver
- **probes** — time-series input/output probes attached to model properties
- **gui** — swing GUI, control panels, timeline
- **workspace** — document/workspace management

### Key sub-packages in `maspack`

- **matrix** — linear algebra (`Matrix3d`, `VectorNd`, …)
- **geometry** — meshes, shapes, spatial transforms
- **spatialmotion** — rigid-body spatial algebra
- **render** — OpenGL rendering pipeline (JOGL-based)
- **properties** — reflection-based property system powering GUI editing
- **numerics / solvers** — linear and nonlinear solvers, IPOPT wrapper

### Core design patterns

- **Component hierarchy** — every model object is a `ModelComponent`; models are trees navigable by path string (e.g. `"models/0/bodies/jaw"`).
- **Property system** (`maspack.properties`) — components expose named, typed properties that the GUI can discover and edit at runtime without code changes.
- **Probes** — attach `InputProbe`/`OutputProbe` to any property to drive or record it over simulated time.
- **Jython scripting** — models and tests can be created or controlled via Python scripts running inside the embedded Jython interpreter.
- **Custom class loader** — `Launcher` uses a `URLClassLoader` so plugins on `EXTCLASSPATH` are isolated and can be added without modifying the core classpath.

### Extension / plugin model

Set `EXTCLASSPATH` (colon-separated) before launching to add external jars. The `artisynth-rl` sibling directory is an example plugin that follows this convention.
