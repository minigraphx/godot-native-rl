# Documentation Restructure — Audience Separation

**Status:** Approved (2026-06-04)
**Topic:** Split documentation by reader, slim the README to a game-developer landing page,
add a clean training-setup path, and keep the AI-agent instruction files small.

## Problem

The current docs do not separate audiences:

- **`README.md` is 746 lines and ~80% repo-contributor content** — cloning `godot-cpp`, building
  ncnn as a static lib, SCons invocations, universal/multi-arch builds, and pnnx conversion
  internals. A *game developer* who downloaded the package and wants to run an example or train a
  policy has to scroll past all of it; the game-facing material (Godot Usage, Examples, Training
  Bridge) is buried at the tail.
- **A fresh GitHub download has no working extension.** `bin/` is gitignored and untracked, so
  `NcnnRunner` does not exist until the user builds the C++ themselves. The docs do not make the
  "you must build first (for now)" reality clear to a game developer, nor do they leave a clean
  slot for a future prebuilt binary.
- **No reproducible training setup.** Training scripts assume two hand-built venvs (`.venv` for
  pnnx/torch, `.venv-train` for godot-rl) with no `requirements*.txt` or setup script, so
  "train your own AI" is undocumented for someone starting from nothing.
- **`CLAUDE.md` has grown large** (full "Current state" inventory + a long gotchas list) even
  though it is always loaded into agent context, and there is no vendor-neutral agent entry file.

## Goals

1. A game developer who downloaded the package can, from a single slim `README.md` and a
   `docs/guide/` set, **run the example scenes** and **train their own AI** end-to-end.
2. Repo contributors have a clear, separate home (`CONTRIBUTING.md` → `docs/dev/`) for
   build-from-source, architecture, and the long-form "why".
3. AI-agent instruction files (`CLAUDE.md`, new `AGENTS.md`) are **small and clean**, pointing into
   `docs/dev/` for depth — no information lost, just relocated.
4. No content is silently dropped; every move updates the cross-references that point at it.

## Non-goals

- Shipping a prebuilt binary / GitHub release is **out of scope** here (tracked as a follow-up
  issue). Docs reference the prebuilt path as "coming" and keep build-from-source as the working
  primary path for now.
- No rewrite of the example game logic, training scripts, or the conversion pipeline — only their
  documentation and a thin setup wrapper.
- No docs-site generator (mkdocs etc.); plain Markdown in the repo.

## Audiences & entry points

| Reader | Entry file | Gets |
|---|---|---|
| **Game developer** | `README.md` (slim) → `docs/guide/` | install, run examples, train, convert, deploy, sensors, build-your-own-agent |
| **Repo contributor** | `CONTRIBUTING.md` → `docs/dev/` | build from source, architecture, the "why", gotchas, backlog/specs |
| **AI agents** | `CLAUDE.md` (trimmed) + `AGENTS.md` (new, thin) | small always-loaded index pointing into `docs/dev/` |

## Target file tree

```
README.md              ← slim, game-dev landing (~150 lines)
CONTRIBUTING.md        ← contributor entry, points to docs/dev/
AGENTS.md              ← thin, vendor-neutral, points to CLAUDE.md
CLAUDE.md              ← trimmed index (depth moved to docs/dev/)
requirements-train.txt ← godot-rl, onnxruntime, ncnn, onnxscript (the .venv-train set)
requirements-convert.txt ← pnnx, torch (the .venv set)
scripts/setup_training.sh ← creates venvs, pip-installs from the requirements files
docs/
  guide/               ← GAME DEVELOPER
    getting-started.md   (install: prebuilt section marked "coming / see Releases";
                          build-from-source link is the working path today; enable plugin)
    running-examples.md  (chase / rover / hide-and-seek — copy-paste run; folds in the
                          existing chase tutorial)
    training.md          (install Python → setup_training.sh → train_chase.sh → convert → deploy)
    deploying.md         (NcnnRunner usage, INT8, VecNormalize, platform targets)
    sensors.md           (raycast / relative-position / camera / grid reference)
    building-your-agent.md (controller + sensors + reward + Sync in your own scene)
  dev/                 ← CONTRIBUTOR
    building.md          (godot-cpp + ncnn + SCons + universal/multi-arch — moved from README)
    DEVELOPMENT.md       (architecture / why — stays)
    gotchas.md           (the "learned the hard way" list — moved out of CLAUDE.md)
  examples/            ← existing per-example material (chase tutorial folded into guide/)
  superpowers/         ← specs/plans (unchanged, contributor-facing)
  BACKLOG.md, godot-rl-gap-analysis-*.md, ncnn_vs_onnx.md, pytorch_mlp_test_model.md
                       ← stay at docs/ root (reference / contributor-facing)
```

## Component details

### README.md (746 → ~150 lines)

**Keeps:** one-paragraph what/why + the "moat" hook; the `ncnn_vs_onnx.md` decision-guide callout
near the top; a "Quick start — run an example" block; a "Train your own AI" pointer into
`docs/guide/training.md`; a short feature/sensor bullet list; links into `docs/guide/` and a single
line pointing contributors to `CONTRIBUTING.md`.

**Moves out** to `docs/dev/building.md`: `godot-cpp` checkout, ncnn static-lib build, SCons
invocations, universal/multi-architecture builds (macOS/Linux/Windows), and the manual pnnx
conversion internals (the one-command `export_to_ncnn.py` summary stays game-facing in
`docs/guide/deploying.md`).

### docs/guide/ (game developer)

- **getting-started.md** — prerequisites (Godot 4.6+); install: a **prebuilt** section marked
  "coming — see Releases" plus the **working** build-from-source path (link to `docs/dev/building.md`,
  condensed "what you'll run" summary); enable the plugin (Project Settings → Plugins).
- **running-examples.md** — run chase / rover / hide-and-seek with the shipped models, copy-paste
  commands; absorbs `docs/examples/chase_the_target_tutorial.md` (or links to it if kept).
- **training.md** — install Python 3.13 → `scripts/setup_training.sh` → `scripts/train_chase.sh`
  (and rover/hide-seek/cleanrl variants) → convert (`export_to_ncnn.py`) → deploy. References the
  requirements files and the sleep/`caffeinate` gotcha.
- **deploying.md** — `NcnnRunner` usage from GDScript, the controllers, INT8 quantization,
  VecNormalize obs-stats replay, platform targets (web/console/mobile/edge — the moat).
- **sensors.md** — reference for RaycastSensor2D/3D (incl. `class_sensor`),
  RelativePositionSensor2D/3D, CameraSensor, GridSensor2D/3D; how auto-discovery works.
- **building-your-agent.md** — wire a controller + sensors + reward + `Sync` into your own scene
  (the "train your own AI in your own game" path).

### docs/dev/ (contributor)

- **building.md** — the full build-from-source content moved out of README (working primary path
  today).
- **DEVELOPMENT.md** — stays; remains the deep architecture/why reference.
- **gotchas.md** — the long "learned the hard way" list moved out of `CLAUDE.md`; `CLAUDE.md` keeps
  only the few daily-biting ones and links here.

### CLAUDE.md trim + AGENTS.md

- **CLAUDE.md** keeps: "What this is", **Key commands** (terse), the handful of critical gotchas,
  and **pointers** to `docs/dev/gotchas.md`, `docs/dev/DEVELOPMENT.md`, `docs/dev/building.md`. The
  long "Current state" inventory and the full gotchas list relocate to `docs/dev/`. Target: roughly
  half current size, **no information lost** (relocated + linked).
- **AGENTS.md** — ~15 lines, vendor-neutral: "this project's agent instructions live in `CLAUDE.md`;
  deep dev reference in `docs/dev/`."

### Training setup tooling

- **requirements-train.txt** — `godot-rl`, `onnxruntime`, `ncnn`, `onnxscript` (the `.venv-train`
  set, Python 3.13).
- **requirements-convert.txt** — `pnnx`, `torch` (the `.venv` set).
- **scripts/setup_training.sh** — **plain venvs primary** (conda documented as the alternative in
  `training.md`): creates both venvs, pip-installs from the two requirements files, idempotent
  (skips already-created venvs), prints the next command. Honors the existing two-venv split and
  Python-version constraints from `CLAUDE.md` (3.14 for convert, 3.13 for train).

## Migration hygiene

- Moves are **relocations, not deletions** — content is preserved.
- Update every cross-reference to a moved path: `README.md`, `CLAUDE.md`, `docs/DEVELOPMENT.md`
  (→ `docs/dev/DEVELOPMENT.md`), `docs/godot-rl-gap-analysis-*.md`, and any `test/` or `scripts/`
  comments that point at moved docs.
- Where a public path changes (e.g. `docs/DEVELOPMENT.md` → `docs/dev/DEVELOPMENT.md`), leave a
  short "moved" note in the style of the existing README library-move callout.
- Per `CLAUDE.md` convention: docs must match the change in the **same** change — README, CLAUDE.md,
  and the gap analysis are updated together; no stale paths left behind.

## Dependencies / follow-ups

- **Prebuilt binary release** (GitHub release or move binaries into the addon) is a separate
  follow-up issue. The prebuilt install section in `getting-started.md` references it as "coming"
  until then. (Relates to backlog item 25 — Asset Library release.)

## Testing / verification

- `./test/run_tests.sh` must stay green (docs-only + new requirements/setup files should not affect
  it; confirm the script's doc-path references, if any, still resolve).
- `scripts/setup_training.sh` is verified by a clean run that produces both venvs and lets
  `scripts/train_chase.sh` start (connect to trainer) — or, if a full run is too heavy for CI, by a
  dry-run/`--check` mode that validates the requirements install resolves.
- Link-check the moved paths (no dangling relative links in README / CLAUDE.md / docs).

## Open questions

None blocking. Prebuilt-binary release is deferred by decision (see Dependencies).
