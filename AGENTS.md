# ArtiSynth Development Guidelines

These instructions guide agentic work on ArtiSynth documentation, examples,
and Java API changes. Keep changes scoped to the requested documentation or
workflow surface unless the user explicitly asks for code edits.

## Project Context

- ArtiSynth is a Java-based biomechanical simulation platform. Public APIs are
  documented through Javadoc at <https://www.artisynth.org/doc/javadocs/>.
- Treat `artisynth.core.*`, `artisynth.models.*`, and `maspack.*` packages as
  the main user-facing API surface. Prefer public Javadoc-visible classes and
  methods when writing examples or documentation.
- Do not invent package names or APIs. Confirm names against the local source
  tree or the published Javadoc before documenting them.
- Keep ArtiSynth and Maspack boundaries clear: `artisynth.core.*` contains the
  simulation framework and model components; `maspack.*` contains supporting
  math, geometry, rendering, properties, utilities, and solver infrastructure.

## API And Documentation Rules

- Breaking changes require a deprecation period first. Do not remove or rename
  public classes, methods, fields, or model entry points without documenting the
  migration path in a prior release.
- Prefer existing ArtiSynth naming patterns over imported conventions from
  other projects. Preserve Java-style `CamelCase` classes, `camelCase` methods,
  and package naming already used in the surrounding code.
- Keep examples and docs on public APIs. Avoid directing users to internal
  helpers unless they are already part of the documented public workflow.
- Use Javadoc links for API references when possible, e.g. `{@link ClassName}`,
  `{@link #methodName(...)}`, or fully qualified links when needed to avoid
  ambiguity.
- Document physical quantities with SI units where the meaning is user-facing:
  positions `[m]`, velocities `[m/s]`, angles `[rad]`, forces `[N]`, moments
  `[N m]`, stiffness or material constants with their conventional SI units.
- When documenting model examples, name the relevant package and launch path
  clearly. Prefer existing demo/model structure under `artisynth.demos.*` and
  `artisynth.models.*`.
- Avoid adding required dependencies. Strongly prefer Java standard library,
  ArtiSynth, and Maspack facilities already present in the project.

## Workflow

- Create a feature branch before committing. Do not commit directly to `main`.
  Use `<username>/feature-desc` unless the user requests another branch name.
- Keep documentation-only work documentation-only. Do not reformat or modify
  Java code while updating agent guidelines, contribution text, or release
  workflow documents.
- Use concise, imperative commit messages, e.g. `Adapt agent guidelines`.
  Keep the subject around 50 characters and wrap the body at 72 characters.
- If a documentation change describes user-facing behavior, verify it against
  the published Javadoc or local source before committing.
- If a change affects public release notes, update the appropriate changelog or
  release-note section with user-facing wording and migration guidance when
  behavior changes.
- Preserve existing copyright years. Use the file creation year for new SPDX
  copyright lines; do not create year ranges when modifying files.

## Build And Test Notes

- Use the repository's existing build commands. Do not introduce a new build
  system for agentic workflow changes.
- Prefer targeted checks for documentation edits: Markdown rendering, link
  validity, and references to real packages/classes.
- For Java changes, use the project's existing compile and test workflow before
  committing. If the local repository is not available or the command is
  unknown, report that explicitly instead of guessing.
- Regression tests should demonstrate the bug or behavior being changed before
  the fix when code changes are part of the task.

## Pull Requests

- If opening a pull request, follow the repository template when one exists.
- PR descriptions should explain what changed and why, identify documentation
  only changes explicitly, and link related issues.
- For deprecated, changed, or removed public API, include migration guidance,
  for example: "Deprecate `OldClass` in favor of `NewClass`."

## Examples

- Follow existing ArtiSynth demo conventions instead of creating a new example
  structure.
- Register or document examples where the repository already lists runnable
  demos or model entry points.
- Include enough validation detail for examples that simulate physical systems:
  expected stability, important units, and the model or demo class to launch.
