# Ray/RLlib Backend (new API stack) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth training backend — stock Ray/RLlib PPO on the **new API stack** (RLModule + EnvRunner) over the godot_rl wire protocol — that trains the chase example and exports the trained actor through the standard TorchScript → `export_to_ncnn.py` path, with a green (auto-skipping) end-to-end smoke in `run_tests.sh` **plus a committed golden-inference fixture captured from the real training run** (Task 8b — scope amended 2026-06-09, see the spec's amendment note). Ecosystem-interop item.

**Architecture:** Isolate `ray[rllib]` in a dedicated `.venv-rllib` (current ray pins `gymnasium==1.2.2` exactly; `godot-rl==0.8.2` declares `<=1.0.0` — they cannot co-resolve, so godot-rl is installed `--no-deps` on top). The stock `RayVectorGodotEnv` is old-API-stack only, so a thin custom **gymnasium adapter** (`GodotRLlibEnv`) wraps godot_rl's env glue instead. A trainer (`scripts/train_rllib.py`) drives new-stack `PPOConfig` with `num_env_runners=0` (single socket, CleanRL-simple orchestration). An exporter (`scripts/export_rllib_to_torchscript.py`) extracts the actor from the checkpointed RLModule, traces it to TorchScript + `.pt.shape.json` sidecar. An orchestrator (`scripts/train_rllib.sh`) chains train → export → `export_to_ncnn.py` convert+parity.

**Tech Stack:** Python 3.13, `ray[rllib]==2.55.*`, `godot-rl==0.8.2` (`--no-deps`), gymnasium 1.2.2, torch pinned to match `.venv-train` (cross-venv `.pt` contract), pnnx/ncnn, Godot 4.5+, stdlib `unittest`, bash.

**Spec:** `docs/superpowers/specs/2026-06-09-rllib-backend-design.md` (GitHub issue **#110**; the closing PR carries `Closes #110`).

**Prerequisites:** built GDExtension (`addons/godot_native_rl/bin/`), a Godot 4.5+ binary (`GODOT=`), `.venv-train` + `.venv` present (`./scripts/setup_training.sh`). Task 2 (the live compat gate) cannot run without them — **execute this plan on a dev machine that already has them** (it was authored in a cloud container that doesn't).

---

## File Structure

- **Create** `requirements-rllib.txt` — RLlib venv deps (`ray[rllib]`, torch pin, godot-rl runtime deps).
- **Modify** `scripts/setup_training.sh` — add the fourth `.venv-rllib` create + the `--no-deps` godot-rl step + `--check` listing.
- **Modify** `test/python/test_setup_training.py` — assert `requirements-rllib.txt` is named by `--check`.
- **Create** `scripts/train_rllib.py` — pure helpers (`parse_args`, `ppo_config_overrides`, `nest_action`) + `GodotRLlibEnv` adapter + `main()` driving new-stack PPO.
- **Create** `test/python/test_train_rllib.py` — unit tests for the pure helpers (no ray import).
- **Create** `scripts/export_rllib_to_torchscript.py` — pure helpers (`latest_checkpoint`, `actor_logit_layout`) + `main()` (RLlib checkpoint → traced `.pt` + sidecar).
- **Create** `test/python/test_export_rllib_to_torchscript.py` — unit tests for the pure helpers (no ray import).
- **Create** `scripts/train_rllib.sh` — orchestrator (train → export → ncnn).
- **Modify** `test/run_tests.sh` — add the guarded end-to-end RLlib smoke step.
- **Create** `models/chase_rllib_policy.ncnn.{param,bin}` — committed fp32 fixture from the real run (Task 8b).
- **Create** `test/unit/test_chase_rllib_golden_inference.gd` — golden-inference regression (Task 8b).
- **Modify (docs)** `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`. (**No** `docs/BACKLOG.md` change — #110 is a GitHub-only item.)

**Convention reminders (from CLAUDE.md):** Python 4-space indent; heavy imports (`ray`/`torch`/`godot_rl`/`numpy`/`gymnasium`) **lazy inside `main()`**/class methods so pure helpers stay import-light and testable; tests are stdlib `unittest` auto-discovered by `run_tests.sh`. Do NOT push to `main`.

**Branch:** implementation continues on **`claude/eloquent-cerf-fb6073`** (the branch carrying this spec + plan) — draft **PR #115** grows into the feature PR (SF-style design+implementation in one). Retitle it `feat: Ray/RLlib training backend (#110)` and add `Closes #110` to the body when marking it ready.

---

## Task 1: `.venv-rllib` dependency isolation

**Files:**
- Create: `requirements-rllib.txt`
- Modify: `scripts/setup_training.sh`
- Test: `test/python/test_setup_training.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestSetupTraining` in `test/python/test_setup_training.py` (after the SF method):

```python
    def test_check_mode_names_rllib_requirements(self):
        # The RLlib backend lives in a fourth venv (.venv-rllib); --check must name its requirements file.
        result = subprocess.run(
            [SCRIPT, "--check"], cwd=REPO_ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout + result.stderr
        self.assertIn("requirements-rllib.txt", out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_setup_training -v`
Expected: FAIL — `requirements-rllib.txt` not in output.

- [ ] **Step 3: Create `requirements-rllib.txt`**

First confirm the torch pin in `.venv-train` (`.venv-train/bin/pip show torch`), then:

```
# Ray/RLlib training backend (new API stack) — isolated venv (.venv-rllib).
# ray[rllib] pins gymnasium==1.2.2 EXACTLY; godot-rl==0.8.2 declares gymnasium<=1.0.0 (+ SB3),
# so they cannot co-resolve: godot-rl is installed --no-deps by setup_training.sh on top of this.
# Its actual runtime use (spaces.{Discrete,Box,Dict,Tuple}, 5-tuple step) is gym-1.2-compatible.
# torch is pinned to match .venv-train so the traced .pt loads there for the ncnn parity check.
# See docs/superpowers/specs/2026-06-09-rllib-backend-design.md
ray[rllib]==2.55.*
torch==<MATCH .venv-train — fill in from pip show>
numpy          # godot_rl runtime dep (godot-rl itself installed --no-deps)
```

- [ ] **Step 4: Wire `.venv-rllib` into `scripts/setup_training.sh`**

Mirror the `.venv-sf` wiring exactly (vars `PYTHON_RLLIB="${PYTHON_RLLIB:-python3.13}"` / `REQ_RLLIB="requirements-rllib.txt"`, banner line, requirements-presence loop, `--check` interpreter note, `create_venv "$PYTHON_RLLIB" ".venv-rllib" "$REQ_RLLIB"`). Then add the **post-install `--no-deps` step** immediately after that `create_venv` call:

```bash
# godot-rl declares gymnasium<=1.0.0 (+ SB3 pins) which conflicts with ray[rllib]'s
# gymnasium==1.2.2; its runtime use is compatible, so install it without deps.
if [ -x ".venv-rllib/bin/pip" ]; then
	".venv-rllib/bin/pip" install --no-deps "godot-rl==0.8.2"
fi
```

(Idempotent: pip no-ops when already satisfied. If `create_venv` skips a healthy venv, this still runs cheaply.)

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_setup_training -v` → PASS.

- [ ] **Step 6: Create the real `.venv-rllib` (one-time, heavy — ray is the largest dep in the repo)**

Run: `./scripts/setup_training.sh`
Expected: existing venvs reported "reusing"; `.venv-rllib` created with `gymnasium-1.2.2`, then godot-rl installed `--no-deps`. Sanity:

```bash
.venv-rllib/bin/python -c "
import gymnasium, ray
from godot_rl.core.godot_env import GodotEnv
print('gymnasium', gymnasium.__version__, '| ray', ray.__version__, '| GodotEnv import OK')"
```

Expected: `gymnasium 1.2.2 | ray 2.55.x | GodotEnv import OK` — **the import half of the spec's §2 assumption.** If `GodotEnv`'s import chain pulls a missing dep (e.g. SB3 via `godot_rl/__init__`), add the *minimal* missing runtime deps to `requirements-rllib.txt` (never SB3's gymnasium pin) and re-run.

- [ ] **Step 7: Commit**

```bash
git add requirements-rllib.txt scripts/setup_training.sh test/python/test_setup_training.py
git commit -m "feat: add .venv-rllib isolated venv for the RLlib backend (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Live compat gate — GodotEnv under gymnasium 1.2.2 (spec step 0)

**The go/no-go gate.** Proves godot_rl's env glue actually runs under gym 1.2.2 against a live Godot before any backend code is written. No repo files change in this task.

- [ ] **Step 1: Round-trip a few steps against the chase scene**

Terminal A (trainer side):

```bash
.venv-rllib/bin/python - <<'PY'
from godot_rl.core.godot_env import GodotEnv
env = GodotEnv(env_path=None, port=11008, show_window=False, seed=0, action_repeat=8, speedup=8)
obs, info = env.reset()
print("obs space:", env.observation_space, "| action space:", env.action_space, "| n:", env.num_envs)
for i in range(5):
    obs, reward, term, trunc, info = env.step([[0]], order_ij=True)
    print("step", i, "reward", reward, "term", term, "trunc", trunc)
env.close()
print("COMPAT GATE: OK")
PY
```

Terminal B (within ~30 s): `$GODOT --headless --path . res://examples/chase_the_target/chase_the_target_train.tscn speedup=8 action_repeat=8`

Expected: handshake + 5 stepped transitions + `COMPAT GATE: OK`. **Record the printed obs/action spaces verbatim** — Task 3's adapter shape assumptions (flat 5-dim "obs" Box; `{"move": Discrete(5)}`) come from here, and the action nesting `[[0]]` is confirmed or corrected here too.

- [ ] **Step 2: If it fails for real API reasons** (gymnasium signature/behavior change, not a missing pin): **STOP — trigger the spec §8 fallback** (pin ray ~2.40 / gymnasium 1.0.0, old API stack + stock `RayVectorGodotEnv`). That is a plan rewrite, not an improvisation; surface it to the user before proceeding.

---

## Task 3: `train_rllib.py` pure helpers + `GodotRLlibEnv` adapter

**Files:**
- Create: `scripts/train_rllib.py`
- Test: `test/python/test_train_rllib.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_train_rllib.py` (same shape as `test_train_sf.py`: `sys.path.insert` the `scripts/` dir; **importing the module must not import ray/torch/gymnasium**):

```python
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_rllib as tr  # noqa: E402


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        cfg = tr.parse_args([])
        self.assertEqual(cfg.base_port, 11008)
        self.assertEqual(cfg.experiment, "chase_rllib")
        self.assertEqual(cfg.train_dir, "logs/rllib")
        self.assertEqual(cfg.speedup, 8)
        self.assertEqual(cfg.action_repeat, 8)

    def test_overrides(self):
        cfg = tr.parse_args(["--timesteps", "2000", "--base_port", "12000"])
        self.assertEqual(cfg.timesteps, 2000)
        self.assertEqual(cfg.base_port, 12000)


class TestPpoConfigOverrides(unittest.TestCase):
    def test_single_socket_and_parity_safe(self):
        cfg = tr.parse_args(["--timesteps", "2000"])
        o = tr.ppo_config_overrides(cfg)
        # num_env_runners=0 => rollouts on the driver: exactly one env, one socket.
        self.assertEqual(o["num_env_runners"], 0)
        # No obs normalization anywhere (ncnn parity: the exported actor must be a plain MLP).
        self.assertFalse(o.get("normalize_obs", False))
        self.assertEqual(o["framework"], "torch")


class TestNestAction(unittest.TestCase):
    def test_scalar_to_godot_structure(self):
        # GodotEnv.step wants one list per agent, one entry per action key: Discrete scalar -> [[a]].
        self.assertEqual(tr.nest_action(3), [[3]])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_train_rllib -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_rllib'`.

- [ ] **Step 3: Write the pure helpers + adapter skeleton**

Create `scripts/train_rllib.py`: module docstring (fourth backend, interop framing, isolated venv, spec pointer), then:

- `RLlibConfig(NamedTuple)`: `timesteps, base_port, speedup, action_repeat, seed, experiment, train_dir` (defaults per the spec §4.5: `chase_rllib`, `logs/rllib`, 8/8, port 11008; default timesteps modest, e.g. 200_000).
- `parse_args(argv) -> RLlibConfig` — argparse, `allow_abbrev=False`.
- `nest_action(a: int) -> list[list[int]]` — `[[int(a)]]` (structure confirmed in Task 2).
- `ppo_config_overrides(cfg) -> dict` — plain dict of the new-stack knobs: `{"framework": "torch", "num_env_runners": 0, "normalize_obs": False, "seed": cfg.seed, "train_batch_size": ..., "lr": ...}` (hyperparams modest; this is interop proof, not a leaderboard — borrow magnitudes from `train_cleanrl.py`).
- `class GodotRLlibEnv` — **defined inside a factory function** (`make_godot_env_cls()` or lazily in `main()`) so module import stays gymnasium-free, OR defined at module top with imports inside methods; pick whichever keeps the Step-1 test green. Surface (per spec §4.3 + Task 2 findings): `gymnasium.Env` subclass; `__init__(self, config=None)` reads `base_port/speedup/action_repeat/seed` from the RLlib `env_config` dict and constructs the underlying godot_rl env (`GodotEnv` directly, or delegate to `CleanRLGodotEnv` if its single-env squeeze is less code — implementer's choice, spec allows both); `observation_space` = the flat 5-dim `Box`; `action_space` = `Discrete(5)`; `reset(seed=, options=) -> (obs, info)`; `step(a)` re-nests via `nest_action`, returns the gymnasium 5-tuple with scalars (squeeze the single-agent batch dim); `close()` shuts the socket so Godot exits.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_train_rllib -v` → PASS (in `.venv-train`, which has **no ray** — this is what enforces the lazy-import discipline).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_rllib.py test/python/test_train_rllib.py
git commit -m "feat: train_rllib.py pure helpers + GodotRLlibEnv gymnasium adapter (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: `train_rllib.py` main() (new-stack PPO driver)

**Files:**
- Modify: `scripts/train_rllib.py`

Needs a live Godot + ray, so not unit-tested; exercised by the smoke (Task 7).

- [ ] **Step 1: Introspect the installed ray 2.55 new-stack API**

```bash
.venv-rllib/bin/python - <<'PY'
import inspect
from ray.rllib.algorithms.ppo import PPOConfig
c = PPOConfig()
print("api_stack:", inspect.signature(c.api_stack))
print("env_runners:", [p for p in inspect.signature(c.env_runners).parameters][:8])
print("rl_module:", [p for p in inspect.signature(c.rl_module).parameters][:6])
PY
```

Record: whether the new stack is already the default (expected in 2.55), the exact `env_runners`/`training` kwarg names, and how to size the MLP (`rl_module(model_config=...)` / `DefaultModelConfig`). **Trust the introspection over this plan's recipe.**

- [ ] **Step 2: Implement `main()`**

Lazy heavy imports. Shape:

```
cfg = parse_args(argv)
ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=2)
config = (PPOConfig()
    .api_stack(enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True)  # explicit even if default
    .environment(GodotRLlibEnv, env_config={...from cfg...})
    .env_runners(num_env_runners=0)
    .framework("torch")
    .training(...modest hyperparams from ppo_config_overrides...))
algo = config.build_algo()          # or .build() — per Step 1 introspection
while sampled < cfg.timesteps:      # read the lifetime env-steps counter from result; print progress
    result = algo.train()
ckpt = algo.save(<abs path under cfg.train_dir/cfg.experiment>)
print("checkpoint:", ckpt)          # the exporter consumes this layout
algo.stop(); ray.shutdown()         # closes the env -> Godot exits
return 0
```

The env-steps counter key (`env_runners/num_env_steps_sampled_lifetime` or similar) is version-named — find it in the first `result` dict and fail loud if absent rather than looping forever. Also print the resolved checkpoint path's **RLModule subdirectory** once (walk the saved dir) — Task 5's `latest_checkpoint` helper encodes it.

- [ ] **Step 3: Verify it imports and parses (no training)**

Run: `.venv-rllib/bin/python scripts/train_rllib.py --help` → argparse help, exit 0.
Run: `.venv-train/bin/python -m unittest test.python.test_train_rllib -v` → still PASS (lazy imports didn't leak).

- [ ] **Step 4: Commit**

```bash
git add scripts/train_rllib.py
git commit -m "feat: train_rllib.py main() — new-API-stack PPO over the godot_rl wire (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: `export_rllib_to_torchscript.py` pure helpers

**Files:**
- Create: `scripts/export_rllib_to_torchscript.py`
- Test: `test/python/test_export_rllib_to_torchscript.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_export_rllib_to_torchscript.py` covering:
- `actor_logit_layout([5]) == (5, [5])`, `actor_logit_layout([3, 2, 4]) == (9, [3, 2, 4])`, empty/non-positive raise `ValueError` (same contract as the SF exporter's helper);
- `latest_checkpoint(tmpdir, experiment)`: create two fake checkpoint dirs with the marker file/layout recorded in Task 4 Step 2 under `tmpdir/experiment/`, different mtimes → returns the newest; raises `FileNotFoundError` when none exist.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_export_rllib_to_torchscript -v` → `ModuleNotFoundError`.

- [ ] **Step 3: Write the pure helpers**

Module docstring (RLlib checkpoint → traced TorchScript actor + `.pt.shape.json` sidecar; raw-logits deploy contract; **pinned to the requirements-rllib ray version, fails loud on unexpected module structure**; spec pointer). Implement `actor_logit_layout` (copy the proven SF implementation) and `latest_checkpoint` (mtime-sorted, explicit error). Stdlib imports only at module top.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_export_rllib_to_torchscript -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_rllib_to_torchscript.py test/python/test_export_rllib_to_torchscript.py
git commit -m "feat: export_rllib_to_torchscript.py pure helpers (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: `export_rllib_to_torchscript.py` main() (RLModule → traced `.pt` + sidecar)

**Files:**
- Modify: `scripts/export_rllib_to_torchscript.py`

**The riskiest task** (spec §8: RLModule internals are version-coupled). Introspect first; trust the live structure over this recipe. Only the output contract is fixed: `forward(obs: Tensor[B,5]) -> Tensor[B,5]` raw logits, saved as `.pt` + `{"inputshape": "[1,5]"}` sidecar.

- [ ] **Step 1: Introspect the checkpointed RLModule (needs a Task-4 smoke-run checkpoint; if none yet, run Task 7 Step 1 first with tiny TIMESTEPS and circle back)**

```bash
.venv-rllib/bin/python - <<'PY'
from ray.rllib.core.rl_module.rl_module import RLModule
m = RLModule.from_checkpoint("<rl_module dir printed by train_rllib.py>")
print(type(m).__name__)
print([n for n, _ in m.named_children()])        # expect: encoder, pi, vf
import torch
out = m.forward_inference({"obs": torch.zeros(1, 5)})
print({k: getattr(v, "shape", v) for k, v in out.items()})  # expect action_dist_inputs [1,5]
PY
```

- [ ] **Step 2: Implement `parse_args` + `main()`**

`parse_args`: `--train_dir` (default `logs/rllib`), `--experiment` (default `chase_rllib`), `--checkpoint` (explicit path override, skips `latest_checkpoint`), `--obs_dim` (default 5), `--nvec` (default `[5]`), `--out` (default `models/chase_rllib_policy.pt`).

`main()` (lazy imports: torch, ray RLModule):
1. resolve checkpoint (`--checkpoint` or `latest_checkpoint`), load the RLModule;
2. extract the actor path per Step-1 introspection — preferred: a thin `nn.Module` wrapper holding the actor-branch encoder + `pi` head whose `forward(obs)` returns logits; acceptable fallback: wrap `forward_inference` itself if the dict-in/dict-out tracing is clean;
3. **fail loud** (clear `RuntimeError` naming the ray pin) if expected children/keys are missing — never trace garbage;
4. sanity: wrapper logits `==` `forward_inference`'s `action_dist_inputs` on random obs (tight atol), and shape `(1, total_logits)` via `actor_logit_layout`;
5. `torch.jit.trace` with `torch.zeros(1, obs_dim)`; save `--out`; write the `.pt.shape.json` sidecar `{"inputshape": f"[1,{obs_dim}]"}` (derived, not hardcoded) — same sidecar contract as `scripts/export_torchscript.py`.

- [ ] **Step 3: Verify imports + helper tests still pass**

Run: `.venv-rllib/bin/python scripts/export_rllib_to_torchscript.py --help` → help, exit 0.
Run: `.venv-train/bin/python -m unittest test.python.test_export_rllib_to_torchscript -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_rllib_to_torchscript.py
git commit -m "feat: export_rllib_to_torchscript.py main() — RLModule actor to traced .pt (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: `train_rllib.sh` orchestrator + end-to-end run

**Files:**
- Create: `scripts/train_rllib.sh`

- [ ] **Step 1: Write `scripts/train_rllib.sh`**

Mirror `train_cleanrl.sh` (single socket — no SF log-watcher), plus the two export steps:

```bash
#!/usr/bin/env bash
# Orchestrates Ray/RLlib (new API stack) PPO training over the godot_rl wire protocol, then
# exports the trained actor to ncnn:
#   1. start the RLlib trainer in .venv-rllib (env opens server on BASE_PORT, blocks for Godot)
#   2. launch the headless Godot chase training scene (connects on BASE_PORT)
#   3. wait for the trainer; kill Godot (trap cleans up stray ray workers)
#   4. export the RLlib checkpoint -> TorchScript .pt + sidecar (.venv-rllib)
#   5. convert .pt -> ncnn + parity check (export_to_ncnn.py, .venv-train -> .venv/bin/pnnx)
# Fourth backend alongside SB3, CleanRL and SampleFactory. Ecosystem interop: see #110.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY_RLLIB="${PY_RLLIB:-.venv-rllib/bin/python}"
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-200000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
BASE_PORT="${BASE_PORT:-11008}"
EXPERIMENT="${EXPERIMENT:-chase_rllib}"
TRAIN_DIR="${TRAIN_DIR:-logs/rllib}"
OUTDIR="${OUTDIR:-models}"
SCENE="${SCENE:-res://examples/chase_the_target/chase_the_target_train.tscn}"

PT_PATH="$OUTDIR/chase_rllib_policy.pt"
# ... trainer & + sleep 5 + Godot client on port=$BASE_PORT & + wait/kill/rc plumbing
#     (copy train_cleanrl.sh verbatim, including the trap EXIT cleanup; add
#      `pkill -f "ray::" || true` to the trap so stray ray workers die with the script)
# ... "$PY_RLLIB" scripts/export_rllib_to_torchscript.py --train_dir "$TRAIN_DIR" \
#         --experiment "$EXPERIMENT" --out "$PT_PATH"
# ... "$PY_TRAIN" scripts/export_to_ncnn.py "$PT_PATH" --outdir "$OUTDIR"
```

`chmod +x scripts/train_rllib.sh`.

- [ ] **Step 2: Tiny end-to-end run (the integration moment — expect iteration here)**

Run: `TIMESTEPS=4000 TRAIN_DIR=$(mktemp -d)/logs OUTDIR=$(mktemp -d)/models ./scripts/train_rllib.sh`
Expected: trainer binds 11008 → Godot connects → a few PPO iterations → checkpoint → `.pt` + sidecar → `export_to_ncnn.py` converts via pnnx and prints a **passing parity check** → `chase_rllib_policy.ncnn.{param,bin}` in the temp OUTDIR. Debug order when it fails: handshake (Task 2 wiring) → step-counter key (Task 4) → RLModule extraction (Task 6) → cross-venv `.pt` load (torch pins, Task 1 Step 3).

- [ ] **Step 3: Commit**

```bash
git add scripts/train_rllib.sh
git commit -m "feat: train_rllib.sh orchestrator (train -> export -> ncnn) (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Guarded RLlib smoke in `run_tests.sh`

**Files:**
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Add the guarded smoke step**

Insert next to the SF smoke block (mirror its structure exactly):

```bash
echo "== RLlib backend smoke (skipped if .venv-rllib absent) =="
if [ -x .venv-rllib/bin/python ]; then
	RLLIB_TMP="$(mktemp -d)"
	TIMESTEPS="${RLLIB_SMOKE_TIMESTEPS:-4000}" \
	TRAIN_DIR="$RLLIB_TMP/logs" OUTDIR="$RLLIB_TMP/models" EXPERIMENT="chase_rllib_smoke" \
		./scripts/train_rllib.sh
	test -f "$RLLIB_TMP/models/chase_rllib_policy.ncnn.param" || { echo "FAIL: RLlib ncnn .param not produced" >&2; rm -rf "$RLLIB_TMP"; exit 1; }
	test -f "$RLLIB_TMP/models/chase_rllib_policy.ncnn.bin"   || { echo "FAIL: RLlib ncnn .bin not produced" >&2; rm -rf "$RLLIB_TMP"; exit 1; }
	rm -rf "$RLLIB_TMP"
	echo "RLlib smoke OK."
else
	echo "SKIP: .venv-rllib not present (run scripts/setup_training.sh to enable the RLlib smoke)."
fi
```

(`export_to_ncnn.py` fails non-zero on parity mismatch, so file-existence asserts suffice. CI auto-skips — no `.venv-rllib` there, same as SF.)

- [ ] **Step 2: Run the full suite**

Run: `./test/run_tests.sh`
Expected: all existing steps green + `RLlib smoke OK.` (or the SKIP line on machines without the venv) + `All tests passed.`

- [ ] **Step 3: Commit**

```bash
git add test/run_tests.sh
git commit -m "test: end-to-end RLlib smoke in run_tests.sh (auto-skips w/o .venv-rllib) (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8b: Golden-inference fixture + regression (scope amendment 2026-06-09)

**Files:**
- Create: `models/chase_rllib_policy.ncnn.param`, `models/chase_rllib_policy.ncnn.bin` (committed fixture)
- Create: `test/unit/test_chase_rllib_golden_inference.gd`

The real run's byproduct becomes CI's permanent, **ray-free** regression (the test needs only the
ncnn runner — no venv), mirroring `test_chase_sf_golden_inference.gd` / the CleanRL twin.

- [ ] **Step 1: Real (non-smoke) training run into `models/`**

Run: `./scripts/train_rllib.sh` (defaults: `TIMESTEPS=200000`, `OUTDIR=models`). On macOS wrap in
`caffeinate -is`. Expected: parity check passes; `models/chase_rllib_policy.ncnn.{param,bin}`
written. Optionally sanity-watch a deploy episode before blessing the model.

- [ ] **Step 2: Generate the golden argmaxes**

Run the **5 shared GOLDEN observations** (copy them verbatim from
`test/unit/test_chase_sf_golden_inference.gd`) through the new fixture and record each argmax —
e.g. via `.venv-train/bin/python` with the `ncnn` package, or a throwaway headless GD script using
`NcnnRunner`. The expected actions will differ from the SF/CleanRL goldens (different trained
weights); only the obs vectors are shared.

- [ ] **Step 3: Write the failing test**

Create `test/unit/test_chase_rllib_golden_inference.gd` mirroring the SF golden test exactly
(same harness, same structure, this model's path + argmaxes). Put placeholder argmaxes first to
see it fail, then fill in Step 2's values.

- [ ] **Step 4: Run it, then the full suite**

Run: `$GODOT --headless --path . --script res://test/unit/test_chase_rllib_golden_inference.gd` → PASS.
Run: `./test/run_tests.sh` → all green (the new test is auto-discovered by the unit-test loop).

- [ ] **Step 5: Commit (fixture + test together)**

```bash
git add models/chase_rllib_policy.ncnn.param models/chase_rllib_policy.ncnn.bin \
    test/unit/test_chase_rllib_golden_inference.gd
git commit -m "test: committed RLlib golden-inference fixture + regression (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Docs (same-change, per repo convention)

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`

- [ ] **Step 1: `CLAUDE.md` — key command** (after the SampleFactory bullet):

```markdown
- **Train (chase, RLlib backend):** `./scripts/train_rllib.sh` — stock Ray/RLlib PPO (new API
  stack) over the godot_rl wire protocol via a thin custom gymnasium adapter (the stock
  `RayVectorGodotEnv` is old-API-stack only). Runs in the isolated **`.venv-rllib`** (ray pins
  `gymnasium==1.2.2`, godot-rl installed `--no-deps`); single socket (`num_env_runners=0`);
  exports the RLModule actor → TorchScript → `export_to_ncnn.py`. Ecosystem interop (#110), not a
  replacement for the custom trainers. `TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`/`BASE_PORT`/
  `EXPERIMENT`/`TRAIN_DIR`/`OUTDIR`/`SCENE` overrides.
```

- [ ] **Step 2: `CLAUDE.md` — venv gotcha three → four** (add `.venv-rllib` (3.13, ray[rllib] — pins `gymnasium==1.2.2`, godot-rl `--no-deps`, so isolated) to the bullet; keep it terse). Also append a `GitHub #110 (RLlib backend …)` entry to the roadmap "Done:" list (the GitHub-#NN convention used for #74/#81).

- [ ] **Step 3: `README.md` — add RLlib to the training-backends list** with the interop framing (match surrounding formatting; lead with "stock RLlib works against an unmodified env").

- [ ] **Step 4: Gap analysis — flip the row.** In `docs/godot-rl-gap-analysis-2026-06-02.md`, change the `RayVectorGodotEnv (RLlib)` row from `**Gap** (#110)` to done, with the one-line caveat: new-stack path uses a custom adapter; the stock wrapper is old-API-stack only.

- [ ] **Step 5: Verify referenced paths exist**

Run: `for f in scripts/train_rllib.sh scripts/train_rllib.py scripts/export_rllib_to_torchscript.py requirements-rllib.txt; do test -e "$f" && echo "ok $f" || echo "MISSING $f"; done` → four `ok` lines.

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: RLlib backend — README/CLAUDE/gap-analysis (#110)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Final verification

- [ ] `./test/run_tests.sh` — all green including `RLlib smoke OK.`
- [ ] `.venv-train/bin/pip show gymnasium | grep Version` → unchanged (the 1.2.2 lives only in `.venv-rllib`).
- [ ] Push and mark the PR ready; body carries `Closes #110` + spec/plan links.

---

## Notes / known risks (carried from the spec §8)

- **Task 2 is the go/no-go gate.** If godot_rl truly breaks under gymnasium 1.2.2, the fallback (ray ~2.40 + old stack + stock `RayVectorGodotEnv`) replaces Tasks 3–6 — stop and re-plan, don't improvise.
- **Introspect-first beats the recipe** (Tasks 4 Step 1, 6 Step 1): RLlib's new-stack API surface and checkpoint layout are version-coupled; the only fixed contracts are the gymnasium env surface, the raw-logits trace, and the sidecar format.
- **Cross-venv `.pt`:** trace in `.venv-rllib`, parity-load in `.venv-train` — the torch pins must match (Task 1 Step 3). SF already proves this contract.
- **Ray process hygiene:** `num_env_runners=0` + `include_dashboard=False` + the trap-EXIT `pkill` keep CI/macOS clean; don't raise worker counts in this PR (multi-runner port orchestration is an explicit follow-up).
- **Golden fixture (Task 8b):** commit the exact fp32 `.ncnn.{param,bin}` the real run produced — never a re-trained or re-exported variant — and derive the goldens from *that* file. (This supersedes the spec's original §9 "no fixture" line; see the spec's 2026-06-09 amendment note.)
