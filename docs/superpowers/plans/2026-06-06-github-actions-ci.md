# Plan: GitHub Actions CI

Status: implemented (`.github/workflows/ci.yml`).

## Goal

Run the existing headless test suite (`test/run_tests.sh`) automatically on every PR/push to
`main`, on Linux, under both supported Godot versions (4.5 + 4.6), with the slow native build and
Python deps cached.

## Requirements (from the maintainer)

- Platform: **Linux** only.
- Godot: **4.5 + 4.6** (matrix).
- "Decide what to run" → run the **full `test/run_tests.sh`** (it does not train; it uses committed
  golden models).
- **Cache** the Python venvs and the ncnn build.

## Build model (how the repo builds locally — see docs/dev/building.md)

- `godot-cpp/` and `thirdparty/` are **gitignored** (fetched/built, not vendored).
- `godot-cpp`: clone `-b 4.5`, `scons target=template_debug` + `template_release`.
- `thirdparty/ncnn`: clone (pinned tag `20260526`), CMake static lib (`NCNN_BUILD_TOOLS=OFF`) →
  `thirdparty/ncnn/build/install/lib/libncnn.a` (the path `SConstruct` links against).
- INT8 CLI tools: a second CMake build via `scripts/build_ncnn_tools.sh` (`NCNN_BUILD_TOOLS=ON`) →
  `thirdparty/ncnn/tools-bin/` (idempotent: skips if the three binaries exist). `run_tests.sh`
  invokes it, so the tools must be present at test time.
- Extension: `scons platform=linux target=template_debug` + `template_release` → `bin/`.
- Python: `.venv` (3.14, `requirements-convert.txt`: pnnx/torch) + `.venv-train`
  (3.13, `requirements-train.txt`: godot-rl/onnxruntime/ncnn/onnxscript). `.venv-sf` is optional;
  the SF smoke auto-skips without it.

## Key decisions

1. **Build the extension once.** A GDExtension built against godot-cpp 4.5 is forward-compatible, so
   the same `bin/` runs under both Godot 4.5 and 4.6. → `build` job builds; `test` matrix consumes
   the artifact on both engines. Avoids rebuilding the native stack per Godot version.
2. **Cache the heavy native builds**, not just outputs: cache `godot-cpp/` and `thirdparty/ncnn/`
   keyed on their pinned refs + `CACHE_VERSION`. On a hit the clone+CMake steps are skipped entirely.
   The `build` job runs `build_ncnn_tools.sh` so `tools-bin/` is baked into the ncnn cache; the
   `test` job restores that cache (read-only) so its `build_ncnn_tools.sh` short-circuits.
3. **Cache the venvs** keyed on `hashFiles(requirements-*.txt)` + python versions + `CACHE_VERSION`
   (torch is large; this is the dominant install cost).
4. **Skip the SF smoke** in CI: don't install `.venv-sf`. It auto-skips; a real in-CI socket
   training run would be slow and flaky. Local `train_sf.sh` + the unit tests already cover it.
5. **No xvfb**: Godot `--headless` uses the dummy display driver; no X server needed.

## Pins / knobs

- `GODOT_CPP_BRANCH=4.5`, `NCNN_TAG=20260526`, `CACHE_VERSION=v1` (env block).
- Godot binaries: `4.5.2-stable` / `4.6.3-stable` (latest patches at authoring; bump in the matrix).
- Bump `CACHE_VERSION` to invalidate all caches and force a cold rebuild.

## Follow-ups / not in scope

- macOS / Windows / arm64 matrix legs (deploy targets — release packaging, not CI gating yet).
- A periodic (cron) job exercising the SF smoke and/or a short real training run.
- Publishing `bin/` for Asset Library releases (backlog item 25) can reuse the `build` job.
