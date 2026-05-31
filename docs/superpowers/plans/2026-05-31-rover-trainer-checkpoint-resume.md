# Rover Trainer Checkpoint/Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `scripts/train_rover.py` checkpoint periodically and auto-resume from the latest checkpoint, so a shutdown-interrupted training run continues on re-run instead of restarting from zero.

**Architecture:** Two import-light pure helpers (`latest_checkpoint`, `remaining_timesteps`) at module top (unit-tested with stdlib `unittest`); heavy imports (torch/SB3/godot_rl) move inside `main()`. `main()` uses SB3 `CheckpointCallback` and, unless `--fresh`, loads the latest checkpoint and continues toward the target timesteps.

**Tech Stack:** Python 3.13 (`.venv-train`), Stable-Baselines3 PPO, godot-rl; stdlib `unittest` + `tempfile`/`re`/`pathlib` for tests (no torch import needed).

**Spec:** `docs/superpowers/specs/2026-05-31-rover-trainer-checkpoint-resume-design.md`

**Conventions:**
- Python uses 4-space indentation. Tests live in `test/python/`, run via `./test/run_tests.sh`'s
  `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'` (auto-discovered — no
  runner change needed). Tests add `scripts/` to `sys.path` and `import <module>` (see
  `test/python/test_export_to_ncnn.py`).
- Verify branch is `feat/rover-checkpoint-resume` before each commit. Never commit on main.
- Run the full suite with `./test/run_tests.sh` (must end `All tests passed.`).

---

## File structure

- **Modify** `scripts/train_rover.py` — add 2 pure helpers at top, move heavy imports into `main()`, add checkpoint/resume CLI args + logic.
- **Create** `test/python/test_train_rover.py` — stdlib `unittest` for the 2 helpers.
- **Modify** `scripts/train_rover.sh` — `CHECKPOINT_FREQ` + `FRESH` passthrough.
- **Modify** `.gitignore` — ignore `models/rover_checkpoints/` + `models/rover_policy.*`.

---

## Task 1: Pure helpers + import-light refactor (TDD)

**Files:**
- Modify: `scripts/train_rover.py`
- Test: `test/python/test_train_rover.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_train_rover.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_rover as tr  # noqa: E402


class TestLatestCheckpoint(unittest.TestCase):
    def test_missing_dir_returns_none(self):
        self.assertIsNone(tr.latest_checkpoint("/no/such/dir/anywhere"))

    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(tr.latest_checkpoint(d))

    def test_picks_highest_step_count(self):
        with tempfile.TemporaryDirectory() as d:
            for name in (
                "rover_ckpt_5000_steps.zip",
                "rover_ckpt_50000_steps.zip",
                "rover_ckpt_25000_steps.zip",
                "unrelated.txt",
            ):
                (Path(d) / name).touch()
            self.assertEqual(
                tr.latest_checkpoint(d),
                str(Path(d) / "rover_ckpt_50000_steps.zip"),
            )

    def test_ignores_non_matching_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.zip").touch()
            (Path(d) / "rover_ckpt_notanumber_steps.zip").touch()
            self.assertIsNone(tr.latest_checkpoint(d))


class TestRemainingTimesteps(unittest.TestCase):
    def test_difference(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 125_000), 275_000)

    def test_done_equals_total(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 400_000), 0)

    def test_overshoot_clamps_to_zero(self):
        self.assertEqual(tr.remaining_timesteps(400_000, 450_000), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_train_rover -v`
Expected: FAIL — `AttributeError: module 'train_rover' has no attribute 'latest_checkpoint'` (the helpers don't exist yet). (If importing `train_rover` itself errors because the heavy top-level imports pull in something unavailable, that also counts as red — Step 3 fixes it by moving them into `main`.)

- [ ] **Step 3: Refactor imports + add the helpers**

In `scripts/train_rover.py`, **remove** these four module-top import lines:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
```

So the module-top imports become just:

```python
import argparse
import pathlib
import re
```

Add the helpers immediately after those imports (before `def main()`):

```python
_CKPT_RE = re.compile(r"^rover_ckpt_(\d+)_steps\.zip$")


def latest_checkpoint(checkpoint_dir: str):
    """Path to the checkpoint with the highest step count in checkpoint_dir, or None.

    Matches SB3 CheckpointCallback's `rover_ckpt_<N>_steps.zip` naming; tolerates a
    missing/empty directory and ignores non-matching filenames.
    """
    d = pathlib.Path(checkpoint_dir)
    if not d.is_dir():
        return None
    best = None
    best_steps = -1
    for f in d.iterdir():
        m = _CKPT_RE.match(f.name)
        if m is not None and int(m.group(1)) > best_steps:
            best_steps = int(m.group(1))
            best = str(f)
    return best


def remaining_timesteps(total: int, done: int) -> int:
    """Timesteps left to reach `total` given `done` already trained (never negative)."""
    return max(0, total - done)
```

Then add the four removed imports as the **first lines inside `main()`** (so the module stays
import-light for the tests):

```python
def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

    parser = argparse.ArgumentParser(allow_abbrev=False)
    # ... rest of main() unchanged for now ...
```

(Leave the rest of `main()` exactly as-is in this task — checkpoint wiring is Task 2.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_train_rover -v`
Expected: PASS — 7 tests OK.

- [ ] **Step 5: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 6: Commit**

```bash
git add scripts/train_rover.py test/python/test_train_rover.py
git commit -m "feat: rover trainer checkpoint helpers (latest_checkpoint, remaining_timesteps) + import-light refactor"
```

---

## Task 2: Checkpoint/resume wiring in `main()`

**Files:**
- Modify: `scripts/train_rover.py`

- [ ] **Step 1: Replace `main()` with the checkpoint/resume version**

Replace the entire `def main() -> None:` body with this (keeps the existing hyperparameters,
adds the three new args + the resume logic). The module-top imports/helpers from Task 1 stay as they are.

```python
def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model_path", type=str, default="models/rover_policy.zip")
    parser.add_argument("--onnx_export_path", type=str, default="models/rover_policy.onnx")
    parser.add_argument("--checkpoint_freq", type=int, default=25_000)
    parser.add_argument("--checkpoint_dir", type=str, default="models/rover_checkpoints")
    parser.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    args = parser.parse_args()

    # env_path=None => in-editor training: opens the server and waits for a Godot client.
    env = StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    # Periodic checkpoints so an interrupted run (e.g. shutdown) can resume.
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.checkpoint_dir,
        name_prefix="rover_ckpt",
    )

    ckpt = None if args.fresh else latest_checkpoint(args.checkpoint_dir)
    if ckpt is not None:
        model = PPO.load(ckpt, env=env)
        steps = remaining_timesteps(args.timesteps, model.num_timesteps)
        print("Resuming from %s at %d steps; %d remaining" % (ckpt, model.num_timesteps, steps))
        if steps > 0:
            model.learn(steps, reset_num_timesteps=False, callback=checkpoint_cb)
    else:
        print("Starting fresh (%d timesteps)" % args.timesteps)
        # Note: do NOT pass seed= to PPO — StableBaselinesGodotEnv.seed() raises
        # NotImplementedError. The env seed is set via the env constructor above.
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=256,
            batch_size=64,
            tensorboard_log="logs/sb3",
        )
        model.learn(args.timesteps, callback=checkpoint_cb)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    onnx_path = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
    export_model_as_onnx(model, str(onnx_path))
    print("Exported ONNX to:", onnx_path)

    env.close()
```

- [ ] **Step 2: Verify the module imports and the CLI exposes the new args**

Run: `.venv-train/bin/python scripts/train_rover.py --help`
Expected: usage text lists `--checkpoint_freq`, `--checkpoint_dir`, and `--fresh` (argparse builds the
parser before any env/training starts, so `--help` returns immediately with exit 0).

- [ ] **Step 3: Confirm the helpers still import cleanly (no heavy deps at module load)**

Run: `.venv-train/bin/python -m unittest test.python.test_train_rover -v`
Expected: PASS — 7 tests OK (the `main()` rewrite didn't touch the module-top helpers).

- [ ] **Step 4: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_rover.py
git commit -m "feat: rover trainer auto-resumes from latest checkpoint (CheckpointCallback + --fresh)"
```

---

## Task 3: Shell passthrough + gitignore

**Files:**
- Modify: `scripts/train_rover.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Add `CHECKPOINT_FREQ` + `FRESH` to `scripts/train_rover.sh`**

In `scripts/train_rover.sh`, after the line `ACTION_REPEAT="${ACTION_REPEAT:-8}"`, add:

```bash
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-25000}"
FRESH_FLAG=""
if [ -n "${FRESH:-}" ]; then
	FRESH_FLAG="--fresh"
fi
```

Then change the trainer launch line from:

```bash
"$PY" scripts/train_rover.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
```

to:

```bash
"$PY" scripts/train_rover.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" --checkpoint_freq "$CHECKPOINT_FREQ" $FRESH_FLAG &
```

(Note: `scripts/train_rover.sh` uses TAB indentation like the other shell scripts — match it. `$FRESH_FLAG` is intentionally unquoted so an empty value expands to nothing.)

- [ ] **Step 2: Syntax-check the shell script**

Run: `bash -n scripts/train_rover.sh`
Expected: no output, exit 0 (valid syntax).

- [ ] **Step 3: Add gitignore rules**

Append to `.gitignore`:

```
models/rover_checkpoints/
models/rover_policy.*
```

- [ ] **Step 4: Verify the ignore rules take effect**

Run: `mkdir -p models/rover_checkpoints && touch models/rover_checkpoints/rover_ckpt_1000_steps.zip models/rover_policy.zip && git status --porcelain | grep -E 'rover_checkpoints|rover_policy' || echo "IGNORED (good)"; rm -rf models/rover_checkpoints models/rover_policy.zip`
Expected: `IGNORED (good)` — neither path shows as untracked.

- [ ] **Step 5: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 6: Commit**

```bash
git add scripts/train_rover.sh .gitignore
git commit -m "feat: train_rover.sh CHECKPOINT_FREQ/FRESH passthrough; gitignore rover checkpoints + artifacts"
```

---

## Self-review notes (author)

- **Spec coverage:** pure helpers + import-light refactor + helper tests (T1); CLI args
  `--checkpoint_freq`/`--checkpoint_dir`/`--fresh`, `CheckpointCallback`, auto-resume with
  `remaining==0` skip + final save/export (T2); `.sh` `CHECKPOINT_FREQ`/`FRESH` passthrough and
  `.gitignore` rules (T3). Auto-resume + `--fresh`, logging, and "rover only" scope all covered.
- **`remaining==0` edge:** handled in T2 (`if steps > 0: learn(...)`; the final save/export run
  regardless, so a re-run after completion still emits the deployable artifacts).
- **Type/name consistency:** `latest_checkpoint(checkpoint_dir)` and `remaining_timesteps(total,
  done)` signatures + the `rover_ckpt_<N>_steps.zip` naming (regex `^rover_ckpt_(\d+)_steps\.zip$`
  vs `name_prefix="rover_ckpt"`) match across helper, test, and `CheckpointCallback`. Arg names
  (`--checkpoint_freq`, `--checkpoint_dir`, `--fresh`) are consistent between `main()` and the `.sh`.
- **Placeholders:** none — every step has concrete code/commands.
- **No real training run** is part of this plan (consistent with the spec).
```
