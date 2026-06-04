# Documentation Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the repo's documentation so a game developer lands on a slim README → `docs/guide/`, contributors get `CONTRIBUTING.md` → `docs/dev/`, AI-agent files (`CLAUDE.md`, new `AGENTS.md`) stay small, and training has a reproducible setup path.

**Architecture:** Mostly content relocation + re-framing of existing Markdown, plus three new build artifacts (`requirements-train.txt`, `requirements-convert.txt`, `scripts/setup_training.sh`). No example/training/conversion code changes. Every move updates the cross-references that point at it; nothing is dropped, only relocated.

**Tech Stack:** Markdown, Bash (POSIX `sh`-compatible where practical), Python venv tooling. Spec: `docs/superpowers/specs/2026-06-04-documentation-restructure-design.md`.

---

## File structure (what each file owns)

**New (game-dev, `docs/guide/`):** `getting-started.md` (install + enable plugin), `running-examples.md` (run shipped models), `training.md` (setup → train → convert → deploy), `deploying.md` (NcnnRunner/INT8/VecNormalize/platforms), `sensors.md` (sensor reference), `building-your-agent.md` (wire your own scene).

**New (contributor, `docs/dev/`):** `building.md` (build-from-source moved from README), `gotchas.md` (long list moved from CLAUDE.md). `DEVELOPMENT.md` moves here from `docs/`.

**New (root):** `CONTRIBUTING.md`, `AGENTS.md`, `requirements-train.txt`, `requirements-convert.txt`, `scripts/setup_training.sh`, `test/python/test_setup_training.py`.

**Slimmed/trimmed:** `README.md` (~746→~150 lines), `CLAUDE.md` (~245→~120 lines).

**Conventions:** GDScript/docs use the repo's existing Markdown style. Commit after every task. Run `git clean -f -- '*.gd.uid'` is irrelevant here (no Godot import passes). Keep relative links repo-root-relative as the existing docs do.

---

## Task 1: Scaffold dirs and move DEVELOPMENT.md into docs/dev/

**Files:**
- Create: `docs/guide/.gitkeep`, `docs/dev/.gitkeep`
- Move: `docs/DEVELOPMENT.md` → `docs/dev/DEVELOPMENT.md`
- Modify: `CLAUDE.md:184` (path reference)

- [ ] **Step 1: Create the two audience folders**

```bash
cd "/Users/andreas/Lokale Dokumente/godot-dev/godot-native-rl"
mkdir -p docs/guide docs/dev
touch docs/guide/.gitkeep docs/dev/.gitkeep
```

- [ ] **Step 2: Move DEVELOPMENT.md with history**

```bash
git mv docs/DEVELOPMENT.md docs/dev/DEVELOPMENT.md
```

- [ ] **Step 3: Update the CLAUDE.md reference to the new path**

In `CLAUDE.md`, find the line (around 184):
`` `docs/DEVELOPMENT.md`. Keep CLAUDE.md terse (it's always-loaded); put new deep-dives there. ``
Replace `docs/DEVELOPMENT.md` with `docs/dev/DEVELOPMENT.md`.

- [ ] **Step 4: Verify no dangling references to the old path remain**

Run: `grep -rn "docs/DEVELOPMENT.md" --include='*.md' --include='*.sh' --include='*.py' . | grep -v 'docs/superpowers/'`
Expected: no output (the only non-spec reference was CLAUDE.md:184, now updated). Note the bare `DEVELOPMENT.md` mention at CLAUDE.md:24 ("see DEVELOPMENT.md \"deploy contract\"") is prose, not a link — leave it or qualify as `docs/dev/DEVELOPMENT.md` for clarity.

- [ ] **Step 5: Commit**

```bash
git add -A docs CLAUDE.md
git commit -m "docs: scaffold docs/guide + docs/dev, move DEVELOPMENT.md"
```

---

## Task 2: Extract build-from-source into docs/dev/building.md

**Files:**
- Create: `docs/dev/building.md`
- Modify: `README.md` (sections will be removed in Task 5; here we only copy them out)

**Source sections in current `README.md` (by heading):** `## Prerequisites` (37), `## Platform Setup` (45), `## Project Setup` (76: clone deps, build godot-cpp, build ncnn), `## Build The GDExtension` (130) incl. `### Enable the plugin` (149), `## Universal / Multi-Architecture Builds` (325), and the manual conversion internals under `## Convert ONNX To ncnn` → `### 1) Install pnnx` (200), `### 2) Convert model` (206), `### 4) Verify the conversion` (291), `### The "fast path"` (308). Keep the one-command `export_to_ncnn.py` summary OUT of here (it goes to the game-dev `deploying.md` in Task 8).

- [ ] **Step 1: Create docs/dev/building.md and move the build content verbatim**

Create `docs/dev/building.md` starting with this header, then paste the source sections listed above (preserve their code blocks exactly):

```markdown
# Building From Source

> **Contributor / from-source build.** Game developers should start at
> [docs/guide/getting-started.md](../guide/getting-started.md) — a prebuilt extension is the
> intended happy path (see Releases, coming). Until that ships, this from-source build is the
> working way to get a `NcnnRunner` for your platform.

## Prerequisites
... (move from README "## Prerequisites")

## Platform Setup (macOS / Linux / Windows)
... (move from README "## Platform Setup")

## Project Setup
... (move "### 1) Clone dependencies", "### 2) Build godot-cpp bindings", "### 3) Build ncnn as static library")

## Build The GDExtension
... (move "## Build The GDExtension" + "### Enable the plugin in Godot")

## Universal / Multi-Architecture Builds
... (move the whole section)

## Manual ONNX → ncnn conversion (internals)
> The one-command path (`scripts/export_to_ncnn.py`) is documented for game developers in
> [docs/guide/deploying.md](../guide/deploying.md). This section is the manual pnnx breakdown.
... (move "### 1) Install pnnx", "### 2) Convert model", "### 4) Verify the conversion", "### The \"fast path\"")
```

- [ ] **Step 2: Verify building.md is self-contained**

Run: `grep -n '^#' docs/dev/building.md`
Expected: the headings above, in order. Run `grep -c '```' docs/dev/building.md` and confirm an even number (all code fences closed).

- [ ] **Step 3: Commit**

```bash
git add docs/dev/building.md
git commit -m "docs: extract build-from-source into docs/dev/building.md"
```

---

## Task 3: Move the long gotchas list into docs/dev/gotchas.md

**Files:**
- Create: `docs/dev/gotchas.md`
- (CLAUDE.md is trimmed in Task 6 — here we only copy the content out)

**Source:** the `## Operational gotchas (learned the hard way)` bullet list in `CLAUDE.md`.

- [ ] **Step 1: Create docs/dev/gotchas.md with the full list**

Create `docs/dev/gotchas.md`:

```markdown
# Operational Gotchas (learned the hard way)

> Long-form companion to `CLAUDE.md`. CLAUDE.md keeps only the few daily-biting items and links
> here for the rest. Contributor-facing.

... (paste the entire "## Operational gotchas" bullet list from CLAUDE.md verbatim)
```

- [ ] **Step 2: Verify all bullets carried over**

Run: `grep -c '^- \*\*' docs/dev/gotchas.md`
Expected: matches the count from `grep -c '^- \*\*' CLAUDE.md` taken within the gotchas section before trimming (record the number; Task 6 removes them from CLAUDE.md). Sanity target: ≥ 10 bullets.

- [ ] **Step 3: Commit**

```bash
git add docs/dev/gotchas.md
git commit -m "docs: move operational gotchas into docs/dev/gotchas.md"
```

---

## Task 4: Add requirements files

**Files:**
- Create: `requirements-train.txt`, `requirements-convert.txt`

Per `CLAUDE.md` two-venv reality: convert venv = pnnx + torch (Python 3.14); train venv = godot-rl, onnxruntime, ncnn, onnxscript (Python 3.13).

- [ ] **Step 1: Create requirements-train.txt**

```
# Training + verification env (.venv-train, Python 3.13 — torch wheels don't exist for 3.14).
# Used by scripts/train_*.sh and the parity/verify scripts.
godot-rl
onnxruntime
ncnn
onnxscript
```

- [ ] **Step 2: Create requirements-convert.txt**

```
# Conversion env (.venv, Python 3.14). Used by scripts/export_to_ncnn.py / pnnx.
pnnx
torch
```

- [ ] **Step 3: Commit**

```bash
git add requirements-train.txt requirements-convert.txt
git commit -m "build: add requirements-train.txt and requirements-convert.txt"
```

---

## Task 5: setup_training.sh — failing test first

**Files:**
- Create: `test/python/test_setup_training.py`
- Create (in next task): `scripts/setup_training.sh`

The script is exercised by a stdlib `unittest` that checks structure/idempotence without doing a real (heavy) pip install — it runs the script in a `--check` dry-run mode that validates Python interpreters and requirements files resolve, and prints the next command.

- [ ] **Step 1: Write the failing test**

Create `test/python/test_setup_training.py`:

```python
import os
import subprocess
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "setup_training.sh")


class TestSetupTraining(unittest.TestCase):
    def test_script_exists_and_is_executable(self):
        self.assertTrue(os.path.isfile(SCRIPT), "scripts/setup_training.sh missing")
        self.assertTrue(os.access(SCRIPT, os.X_OK), "setup_training.sh not executable")

    def test_check_mode_runs_and_names_next_step(self):
        # --check must not create venvs or pip-install; it validates and prints guidance.
        result = subprocess.run(
            [SCRIPT, "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout + result.stderr
        self.assertIn("requirements-train.txt", out)
        self.assertIn("requirements-convert.txt", out)
        self.assertIn("train_chase.sh", out)  # points at the next command
        # --check is non-destructive: it must not have created the venvs.
        self.assertFalse(os.path.isdir(os.path.join(REPO_ROOT, ".venv-train")))
        self.assertFalse(os.path.isdir(os.path.join(REPO_ROOT, ".venv")))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/homebrew/bin/python3.14 -m unittest test.python.test_setup_training -v`
Expected: FAIL — `setup_training.sh missing` (script not created yet).

- [ ] **Step 3: Commit the failing test**

```bash
git add test/python/test_setup_training.py
git commit -m "test: setup_training.sh structure + --check dry-run (failing)"
```

---

## Task 6: setup_training.sh — implementation

**Files:**
- Create: `scripts/setup_training.sh`

Plain venvs are primary (conda documented as the alternative in `training.md`, Task 9). `--check` validates without mutating. Real run creates `.venv` (convert) and `.venv-train` (train), idempotent (skips existing), installs from the requirements files, prints next step.

- [ ] **Step 1: Write the script**

Create `scripts/setup_training.sh`:

```bash
#!/usr/bin/env bash
# Create the two Python venvs for training + conversion and install their deps.
# Plain venvs are the primary path; conda is documented as an alternative in
# docs/guide/training.md. Idempotent: existing venvs are reused.
#
#   ./scripts/setup_training.sh           # create + install
#   ./scripts/setup_training.sh --check   # validate only, no venv creation, no install
#
# Overrides: PYTHON_TRAIN (default python3.13), PYTHON_CONVERT (default python3.14).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_TRAIN="${PYTHON_TRAIN:-python3.13}"
PYTHON_CONVERT="${PYTHON_CONVERT:-python3.14}"
REQ_TRAIN="requirements-train.txt"
REQ_CONVERT="requirements-convert.txt"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "Training stack setup"
echo "  train venv:   .venv-train  (interpreter: $PYTHON_TRAIN, deps: $REQ_TRAIN)"
echo "  convert venv: .venv        (interpreter: $PYTHON_CONVERT, deps: $REQ_CONVERT)"

for f in "$REQ_TRAIN" "$REQ_CONVERT"; do
	if [ ! -f "$f" ]; then
		echo "ERROR: missing $f" >&2
		exit 1
	fi
done

if [ "$CHECK_ONLY" -eq 1 ]; then
	echo "--check: requirements files present."
	command -v "$PYTHON_TRAIN" >/dev/null 2>&1 || echo "NOTE: $PYTHON_TRAIN not on PATH (needed for .venv-train; override with PYTHON_TRAIN=)."
	command -v "$PYTHON_CONVERT" >/dev/null 2>&1 || echo "NOTE: $PYTHON_CONVERT not on PATH (needed for .venv; override with PYTHON_CONVERT=)."
	echo "Next: ./scripts/setup_training.sh   then   ./scripts/train_chase.sh"
	exit 0
fi

create_venv() {
	# $1 = interpreter, $2 = venv dir, $3 = requirements file
	if [ -d "$2" ]; then
		echo "  $2 already exists — reusing."
	else
		command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not found (override with the matching PYTHON_ env var)." >&2; exit 1; }
		echo "  creating $2 with $1 ..."
		"$1" -m venv "$2"
	fi
	"$2/bin/python" -m pip install --upgrade pip
	"$2/bin/python" -m pip install -r "$3"
}

create_venv "$PYTHON_TRAIN" ".venv-train" "$REQ_TRAIN"
create_venv "$PYTHON_CONVERT" ".venv" "$REQ_CONVERT"

echo "Done. Next: ./scripts/train_chase.sh   (see docs/guide/training.md)"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x scripts/setup_training.sh
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `/opt/homebrew/bin/python3.14 -m unittest test.python.test_setup_training -v`
Expected: PASS (both tests). `--check` exits 0, prints the requirements filenames and `train_chase.sh`, and creates no venvs.

- [ ] **Step 4: Confirm .gitignore already covers the venvs**

Run: `grep -nE '^\.venv' .gitignore`
Expected: `.venv` and `.venv-train` are ignored (CLAUDE.md says both are gitignored). If missing, add them.

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_training.sh
git commit -m "build: add scripts/setup_training.sh (plain-venv training setup)"
```

---

## Task 7: Game-dev guide — getting-started + running-examples

**Files:**
- Create: `docs/guide/getting-started.md`, `docs/guide/running-examples.md`

**Sources to reuse:** README `## Examples` (651) for the run commands; README `### Enable the plugin in Godot` (149) for the plugin step; README headless run block (`#### Headless`, 481).

- [ ] **Step 1: Create docs/guide/getting-started.md**

```markdown
# Getting Started

For **game developers** who downloaded this package and want to run an example or train a policy.

## 1. Prerequisites
- **Godot 4.6+**.
- The **`NcnnRunner` GDExtension** for your platform.

## 2. Get the extension
**Prebuilt (recommended, coming):** download the extension for your platform from the project
**Releases** and drop `bin/` + `ncnn_runner.gdextension` into the project root. *(Prebuilt binaries
are not published yet — see the build path below until they are.)*

**Build from source (works today):** follow
[docs/dev/building.md](../dev/building.md). You'll clone `godot-cpp`, build ncnn as a static lib,
and run SCons. This produces `bin/<platform>/...`.

## 3. Enable the plugin
... (move the "Enable the plugin in Godot" step from README: Project Settings → Plugins → enable
"Godot Native RL")

## 4. Next
- Run a shipped example: [running-examples.md](running-examples.md)
- Train your own AI: [training.md](training.md)
- Build an agent in your own scene: [building-your-agent.md](building-your-agent.md)
```

- [ ] **Step 2: Create docs/guide/running-examples.md**

Move the per-example run instructions from README `## Examples` (chase / rover / hide-and-seek) here, framed for a downloaded package. Each example: what it is (1 line), the command to run it with its shipped model, what you should see. Fold in the from-scratch chase tutorial by linking it:

```markdown
# Running the Example Scenes

These ship with pre-trained ncnn models — run them with no Python setup.

## Chase the Target (2D)
... (move from README "### Chase The Target (2D)")
> Want to build this from scratch? See the
> [chase tutorial](../examples/chase_the_target_tutorial.md).

## 3D Raycast Rover
... (move from README "### 3D Raycast Rover", incl. the parallel-training note as a pointer to training.md)

## Hide & Seek (2D self-play)
... (move from README "### Hide & Seek", link to examples/hide_and_seek/README.md)
```

- [ ] **Step 3: Verify links resolve**

Run: `for f in docs/guide/getting-started.md docs/guide/running-examples.md; do echo "== $f =="; grep -oE '\]\([^)]+\)' "$f"; done`
Manually confirm each relative target exists (e.g. `../dev/building.md`, `../examples/chase_the_target_tutorial.md`).

- [ ] **Step 4: Commit**

```bash
git add docs/guide/getting-started.md docs/guide/running-examples.md
git commit -m "docs: add game-dev getting-started + running-examples guides"
```

---

## Task 8: Game-dev guide — training + deploying

**Files:**
- Create: `docs/guide/training.md`, `docs/guide/deploying.md`

**Sources:** README `### Running training` (477) + `#### Headless` (481) command list; CLAUDE.md "Key commands" for convert/deploy one-liners; README `## Convert ONNX To ncnn` → `### One command (recommended)` (159) for the game-facing convert summary.

- [ ] **Step 1: Create docs/guide/training.md**

```markdown
# Training Your Own AI

You train with the standard `godot-rl` Python stack and deploy with native ncnn.

## 1. Python setup (once)
Install **Python 3.13** (training) and **Python 3.14** (conversion), then:

```bash
./scripts/setup_training.sh
```

This creates `.venv-train` (godot-rl + verify deps) and `.venv` (pnnx + torch) from
`requirements-train.txt` / `requirements-convert.txt`. Re-running is safe (existing venvs are
reused). Override interpreters with `PYTHON_TRAIN=` / `PYTHON_CONVERT=`. Validate without
installing: `./scripts/setup_training.sh --check`.

**conda alternative:** create two envs and `pip install -r` the same files; see
[../dev/DEVELOPMENT.md](../dev/DEVELOPMENT.md) for the two-env rationale.

## 2. Train
... (move the headless training command list from README "#### Headless": train_chase.sh,
train_rover.sh, parallel, hide_seek, cleanrl — with the TIMESTEPS/SCENE notes from CLAUDE.md)

> **macOS:** wrap long runs in `caffeinate -is` — sleep kills the training socket
> (see [../dev/gotchas.md](../dev/gotchas.md)).

## 3. Convert + deploy
After training produces a checkpoint/ONNX, convert to ncnn and deploy — see
[deploying.md](deploying.md).
```

- [ ] **Step 2: Create docs/guide/deploying.md**

```markdown
# Deploying (native ncnn inference)

## Convert your trained model
... (move README "### One command (recommended)": `.venv-train/bin/python scripts/export_to_ncnn.py
models/model.onnx` summary; mention --via, TorchScript path, link manual internals to
[../dev/building.md](../dev/building.md#manual-onnx--ncnn-conversion-internals))

## Use it in Godot
... (the NcnnRunner / NcnnAIController2D|3D usage — move the runtime-usage parts of README
"## Godot Usage" that are about *running* a model, not building)

## INT8 quantization (mobile/edge)
... (move from README "### INT8 quantization", game-facing summary; link build_ncnn_tools.sh)

## VecNormalize obs stats
... (controller `obs_norm_stats_path` + export_vecnormalize.py, from CLAUDE.md Key commands)

## Platform targets (the moat)
Web/WASM, console, mobile, edge — ncnn is statically linked, no .NET/runtime. (1–2 paragraphs
from README "## What This Repository Provides" + CLAUDE.md "The moat".)
```

- [ ] **Step 3: Verify links resolve**

Run: `for f in docs/guide/training.md docs/guide/deploying.md; do echo "== $f =="; grep -oE '\]\([^)]+\)' "$f"; done`
Confirm each target exists. The `#manual-onnx--ncnn-conversion-internals` anchor must match the heading created in Task 2 (`## Manual ONNX → ncnn conversion (internals)`); adjust the slug if GitHub differs.

- [ ] **Step 4: Commit**

```bash
git add docs/guide/training.md docs/guide/deploying.md
git commit -m "docs: add game-dev training + deploying guides"
```

---

## Task 9: Game-dev guide — sensors + building-your-agent

**Files:**
- Create: `docs/guide/sensors.md`, `docs/guide/building-your-agent.md`

**Sources:** README `## Sensors` (582), README `### Wire-Up In Scene` (528), README `### Agent Contract` (465).

- [ ] **Step 1: Create docs/guide/sensors.md**

Move the README `## Sensors` reference here verbatim (RaycastSensor2D/3D incl. `class_sensor`,
RelativePositionSensor2D/3D, CameraSensor, GridSensor2D/3D), plus a short intro on auto-discovery
(`collect_sensors()` duck-typed tree order, from CLAUDE.md). Header:

```markdown
# Sensors

Flat sensors extend `ISensor2D`/`ISensor3D` and are auto-discovered (tree order) by the
controller's `collect_sensors()`. Add a sensor node under your agent and it joins the observation.
... (move README "## Sensors")
```

- [ ] **Step 2: Create docs/guide/building-your-agent.md**

```markdown
# Building an Agent in Your Own Scene

Wire a controller + sensors + reward + Sync into your game.

## Agent contract
... (move README "### Agent Contract")

## Wire-up
... (move README "### Wire-Up In Scene": Sync node, controller export props, policy_name)

## Reward
... (RewardBuilder/RewardAdapter/terms — short overview + link to existing reward docs/specs)

## Sensors
See [sensors.md](sensors.md).

## Train it
See [training.md](training.md).
```

- [ ] **Step 3: Verify links + commit**

Run: `for f in docs/guide/sensors.md docs/guide/building-your-agent.md; do grep -oE '\]\([^)]+\)' "$f"; done` and confirm targets exist.

```bash
git add docs/guide/sensors.md docs/guide/building-your-agent.md
git commit -m "docs: add game-dev sensors + building-your-agent guides"
```

---

## Task 10: Slim the README to a game-dev landing page

**Files:**
- Modify: `README.md` (rewrite, ~746 → ~150 lines)

All build/convert-internals/sensor/example detail now lives in `docs/dev/` and `docs/guide/`. README becomes a router. Confirm every section being removed has a home (Task 2 building.md, Tasks 7–9 guides) before deleting.

- [ ] **Step 1: Rewrite README.md**

Replace the entire file with:

```markdown
# Godot Native RL (ncnn GDExtension)

Reinforcement learning for **Godot 4.6+** with **native ncnn inference** — statically linked C++,
no C#/.NET, no external runtime. Train with the standard `godot-rl` Python stack; deploy native on
web/WASM, console, mobile, desktop, and edge.

> **ncnn vs ONNX Runtime?** Honest decision guide:
> [docs/ncnn_vs_onnx.md](docs/ncnn_vs_onnx.md).

## Quick start (game developers)

1. **Install** — get the extension and enable the plugin:
   [docs/guide/getting-started.md](docs/guide/getting-started.md).
2. **Run an example** — pre-trained models, no Python needed:
   [docs/guide/running-examples.md](docs/guide/running-examples.md).
3. **Train your own AI** — `./scripts/setup_training.sh` then train → convert → deploy:
   [docs/guide/training.md](docs/guide/training.md).

## Guides
- [Getting started](docs/guide/getting-started.md) — install + enable the plugin
- [Running the examples](docs/guide/running-examples.md) — chase / rover / hide & seek
- [Training your own AI](docs/guide/training.md) — setup, train, the parallel-training fast path
- [Deploying](docs/guide/deploying.md) — NcnnRunner, INT8, VecNormalize, platform targets
- [Sensors](docs/guide/sensors.md) — raycast, relative-position, camera, grid
- [Building an agent in your scene](docs/guide/building-your-agent.md)

## What you get
- `NcnnRunner` C++ node: `load_model`, `run_inference`, `run_inference_image`,
  `run_discrete_action`.
- `NcnnAIController2D` / `NcnnAIController3D` + auto-discovered sensors + a Signal→Reward builder.
- godot_rl v0.8.2-compatible training bridge (`NcnnSync`) incl. multi-policy + parallel arenas.
- Convert (`scripts/export_to_ncnn.py`) and INT8 quantize for deployment.

## The moat
ncnn statically linked enables web/WASM and console deployment (ONNX/.NET can't), game-side INT8
quantization, async inference, and Godot-native ideas (Signal→Reward, NavMesh sensor) — none
replicable by a Python-server or managed-runtime framework.

## Contributing / building from source
Building the GDExtension, architecture, and dev notes:
[CONTRIBUTING.md](CONTRIBUTING.md) → [docs/dev/](docs/dev/).

## License
... (keep the existing license note from README "## Notes" / footer)
```

- [ ] **Step 2: Verify README length and links**

Run: `wc -l README.md` — expected ~150 (well under 250).
Run: `grep -oE '\]\([^)]+\)' README.md` and confirm every target exists (`docs/guide/*`, `docs/ncnn_vs_onnx.md`, `CONTRIBUTING.md`, `docs/dev/`).

- [ ] **Step 3: Confirm no game-facing content was lost**

Run: `grep -rn "run_inference_image\|class_sensor\|caffeinate\|export_to_ncnn" docs/guide docs/dev`
Expected: each appears somewhere under `docs/` (it moved, not vanished).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: slim README to a game-dev landing page"
```

---

## Task 11: Trim CLAUDE.md and add AGENTS.md

**Files:**
- Modify: `CLAUDE.md` (~245 → ~120 lines)
- Create: `AGENTS.md`

CLAUDE.md keeps "What this is", **Key commands** (terse), a handful of critical gotchas, and pointers. The full gotchas list (now in `docs/dev/gotchas.md`, Task 3) is replaced by a pointer. The long "Current state" inventory is condensed to a few lines + a pointer to `docs/dev/DEVELOPMENT.md`.

- [ ] **Step 1: Replace the gotchas section in CLAUDE.md with a pointer**

Delete the full `## Operational gotchas (learned the hard way)` bullet list. Replace with:

```markdown
## Operational gotchas

Full list (learned the hard way): **[docs/dev/gotchas.md](docs/dev/gotchas.md)**. The few that bite
daily:
- **`class_name` is unreliable headless** — prefer path-based `extends "res://addons/..."`.
- **Two venvs** — `.venv` (3.14, pnnx+torch) convert; `.venv-train` (3.13, godot-rl) train. Create
  both with `./scripts/setup_training.sh`.
- **macOS: never sleep during training** — wrap in `caffeinate -is`.
- **Rebuild the extension on a fresh clone** — `bin/` is gitignored.
```

- [ ] **Step 2: Condense the "Current state" inventory**

Replace the long `## Current state (working, on main)` block with a ~6-line summary + pointer:

```markdown
## Current state (working, on `main`)

Full train → convert → deploy loop works end-to-end (headless CI tests). Reusable library in
`addons/godot_native_rl/` (`sync.gd`/`NcnnSync`, `controllers/`, `reward/`, `sensors/`,
`training/`, `net/`); C++ GDExtension at repo root (`src/ncnn_runner.{h,cpp}`). Examples:
`chase_the_target` (2D), `rover_3d` (3D), `hide_and_seek` (2D self-play). Wire protocol is
godot_rl v0.8.2-compatible. **Architecture + data flow + deploy contract:
[docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md).**
```

Keep the `## Key commands`, `## Conventions`, `## Roadmap & backlog`, and `## The moat` sections as-is (terse already).

- [ ] **Step 3: Verify CLAUDE.md shrank and lost nothing**

Run: `wc -l CLAUDE.md` — expected ~120 (down from 245).
Run: `grep -n "docs/dev/gotchas.md\|docs/dev/DEVELOPMENT.md\|setup_training.sh" CLAUDE.md`
Expected: all three pointers present.

- [ ] **Step 4: Create AGENTS.md**

```markdown
# Agent Instructions

This project's agent guidance lives in **[CLAUDE.md](CLAUDE.md)** — read it first (what the project
is, key commands, conventions, the daily gotchas).

Deeper references for any AI or human contributor:
- **[docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md)** — architecture, data flow, deploy contract.
- **[docs/dev/gotchas.md](docs/dev/gotchas.md)** — the full "learned the hard way" list.
- **[docs/dev/building.md](docs/dev/building.md)** — build the GDExtension from source.

Game-developer docs (running examples, training, deploying) live under
**[docs/guide/](docs/guide/)**.

Keep CLAUDE.md terse — it is always loaded into context. Put new deep-dives in `docs/dev/`.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md AGENTS.md
git commit -m "docs: trim CLAUDE.md, add vendor-neutral AGENTS.md"
```

---

## Task 12: Add CONTRIBUTING.md (contributor entry point)

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Create CONTRIBUTING.md**

```markdown
# Contributing

Entry point for **repo contributors**. (Game developers: see
[README.md](README.md) → [docs/guide/](docs/guide/).)

## Build from source
[docs/dev/building.md](docs/dev/building.md) — godot-cpp, ncnn static lib, SCons, multi-arch.

## Architecture & internals
[docs/dev/DEVELOPMENT.md](docs/dev/DEVELOPMENT.md) — data flow, inference-backend boundary, the
algorithm-agnostic deploy contract.

## Gotchas
[docs/dev/gotchas.md](docs/dev/gotchas.md) — read before debugging headless/training/convert issues.

## Tests
Run `./test/run_tests.sh` — must be green before merge (headless GDScript unit tests + Python
protocol/helper tests + inference/golden regressions).

## Workflow & roadmap
Superpowers workflow (brainstorm → spec → plan → TDD) — see
[CLAUDE.md](CLAUDE.md) "Conventions". Open work: GitHub issues (`backlog` label) + `docs/BACKLOG.md`.

## Docs hygiene
Update README, CLAUDE.md, the relevant `docs/guide/` or `docs/dev/` page, and the gap analysis in
the **same** change. Stale paths/commands count as a bug.
```

- [ ] **Step 2: Verify links + commit**

Run: `grep -oE '\]\([^)]+\)' CONTRIBUTING.md` and confirm each target exists.

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md contributor entry point"
```

---

## Task 13: Final cross-reference sweep + tests green

**Files:**
- Modify: any file with a now-stale link (sweep result)

- [ ] **Step 1: Sweep for stale references to moved/removed README sections and old paths**

Run:
```bash
grep -rn "chase_the_target_tutorial" --include='*.md' . | grep -v 'docs/superpowers/'
grep -rn "docs/DEVELOPMENT.md" --include='*.md' --include='*.sh' --include='*.py' . | grep -v 'docs/superpowers/'
grep -rn "README.md#" --include='*.md' .
```
Fix any hit that points at a section no longer in README (e.g. the chase tutorial's
`[top-level README](../../README.md)` build link → repoint to `../dev/building.md`).

- [ ] **Step 2: Update the chase tutorial's prereq links to the new homes**

In `docs/examples/chase_the_target_tutorial.md` §1 Prerequisites, repoint the build link from the
README to `../dev/building.md`, and replace the manual `pip install` venv snippet with a pointer to
`../guide/training.md` + `scripts/setup_training.sh` (keep it consistent with the new setup path).

- [ ] **Step 3: Verify the doc tree matches the spec**

Run: `find docs/guide docs/dev -name '*.md' | sort`
Expected exactly: `docs/dev/DEVELOPMENT.md`, `docs/dev/building.md`, `docs/dev/gotchas.md`,
`docs/guide/building-your-agent.md`, `docs/guide/deploying.md`, `docs/guide/getting-started.md`,
`docs/guide/running-examples.md`, `docs/guide/sensors.md`, `docs/guide/training.md`.

- [ ] **Step 4: Link-check all top-level + guide + dev docs**

Run:
```bash
for f in README.md CONTRIBUTING.md AGENTS.md CLAUDE.md docs/guide/*.md docs/dev/*.md; do
  grep -oE '\]\(([^)]+)\)' "$f" | sed -E 's/\]\(//;s/\)//' | while read -r link; do
    case "$link" in
      http*|\#*) continue ;;
    esac
    target="$(dirname "$f")/${link%%#*}"
    [ -e "$target" ] || echo "DANGLING: $f -> $link"
  done
done
```
Expected: no `DANGLING:` lines.

- [ ] **Step 5: Run the full test suite**

Run: `./test/run_tests.sh`
Expected: green (includes the new `test/python/test_setup_training.py`, auto-discovered under
`test/python/`). Docs moves must not affect GDScript/protocol tests.

- [ ] **Step 6: Commit any sweep fixes**

```bash
git add -A
git commit -m "docs: fix cross-references after restructure; tests green"
```

---

## Task 14: Update the gap analysis / backlog note

**Files:**
- Modify: `docs/godot-rl-gap-analysis-2026-06-02.md` (only if it references moved paths or the DX/docs track)
- Modify: `docs/BACKLOG.md` (if a docs/DX item is listed)

- [ ] **Step 1: Check for references to update**

Run: `grep -n "README\|DEVELOPMENT.md\|docs/guide\|docs/dev\|getting started\|onboarding" docs/godot-rl-gap-analysis-2026-06-02.md docs/BACKLOG.md`
Update any path that moved; if the DX/Distribution track lists a docs/onboarding item, note this restructure against it.

- [ ] **Step 2: Commit (if anything changed)**

```bash
git add docs/godot-rl-gap-analysis-2026-06-02.md docs/BACKLOG.md
git commit -m "docs: reflect documentation restructure in gap analysis/backlog"
```

---

## Self-review notes (already applied)

- **Spec coverage:** audiences/entry points (Tasks 7–12), file tree (Tasks 1–12), README slim
  (Task 10), guide set (7–9), dev set (2–3, Task 1), CLAUDE trim + AGENTS (11), CONTRIBUTING (12),
  requirements + setup script (4–6), migration hygiene (13–14), prebuilt-as-coming (Task 7). All
  spec sections map to a task.
- **No silent drops:** Task 10 Step 3 and Task 13 grep-verify moved content still exists under
  `docs/`.
- **Type/name consistency:** `setup_training.sh` `--check` contract is identical in the test (Task
  5) and implementation (Task 6); requirements filenames identical across Tasks 4/6/9; the
  `#manual-onnx--ncnn-conversion-internals` anchor defined in Task 2 is referenced in Task 8.
- **Prebuilt binary release** stays out of scope (spec "Dependencies / follow-ups"); docs reference
  it as coming only.
