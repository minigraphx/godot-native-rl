# Rover Trainer Checkpoint/Resume — Design

**Date:** 2026-05-31
**Relates to:** backlog item 6 (3D rover example) — makes the deferred training run shutdown-survivable
**Status:** Approved (brainstorm → spec)

## Problem & motivation

`scripts/train_rover.py` only persists the model at the very end of `learn()` (final `.zip` + ONNX
export). A 400k-timestep PPO run takes ~45–90 min; if the machine is shut down (or the run is
interrupted) mid-training, all progress is lost and the next run starts from zero. We want the run
to be safely interruptible: periodic checkpoints + automatic resume on re-run.

(Background: a full shutdown kills both local processes — the SB3 trainer and the headless Godot
env — and there is no in-RAM survival. Checkpoint-to-disk + resume-from-disk is the only robust fix
for local training.)

## Decisions (from brainstorming)

1. **Auto-resume:** re-running `train_rover.sh` automatically loads the latest checkpoint and
   continues toward the target timesteps. A `--fresh` flag (and `FRESH=1` env on the `.sh`) forces a
   clean start. The trainer logs clearly whether it resumed (and from which file / how many steps
   remain) or started fresh.
2. **Checkpointing:** SB3 `CheckpointCallback`, default every `25_000` timesteps, into
   `models/rover_checkpoints/` (gitignored).
3. **Pure helpers + tests:** the resume *decision logic* is extracted into two import-light pure
   functions, unit-tested with stdlib `unittest` under `.venv-train` (no torch/SB3 import needed),
   matching the `export_to_ncnn.py` testing pattern. No real training run is part of this work.
4. **Scope:** rover trainer only; `scripts/train_chase.py` is untouched.

## Architecture

### `scripts/train_rover.py`

New CLI args (added to the existing parser):
- `--checkpoint_freq` (int, default `25_000`) — save a checkpoint every this many timesteps.
- `--checkpoint_dir` (str, default `models/rover_checkpoints`) — where checkpoints are written.
- `--fresh` (flag, default off) — ignore any existing checkpoint and start from scratch.

Import-light **pure helpers** (defined at module top, no heavy imports needed to call them):

```python
def latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the path to the checkpoint with the highest step count in checkpoint_dir,
    or None if the directory is missing/empty or contains no rover_ckpt_<N>_steps.zip."""

def remaining_timesteps(total: int, done: int) -> int:
    """Timesteps left to train toward `total` given `done` already completed (never negative)."""
    return max(0, total - done)
```

`latest_checkpoint` parses the integer `<N>` from filenames matching `rover_ckpt_<N>_steps.zip`
(SB3 `CheckpointCallback`'s naming with `name_prefix="rover_ckpt"`) and returns the max; it tolerates
a missing directory and ignores non-matching filenames.

**`main()` flow** (SB3/torch imported lazily inside `main`, as today):
1. Parse args; build the `StableBaselinesGodotEnv` + `VecMonitor` (unchanged).
2. Construct `CheckpointCallback(save_freq=args.checkpoint_freq, save_path=args.checkpoint_dir,
   name_prefix="rover_ckpt")`.
3. Resume decision:
   - `ckpt = None if args.fresh else latest_checkpoint(args.checkpoint_dir)`
   - If `ckpt`: `model = PPO.load(ckpt, env=env)`; `steps = remaining_timesteps(args.timesteps,
     model.num_timesteps)`; print `"Resuming from <ckpt> at <num_timesteps> steps; <steps> remaining"`;
     `model.learn(steps, reset_num_timesteps=False, callback=ckpt_cb)`.
   - Else: `model = PPO("MultiInputPolicy", env, ...)` (unchanged hyperparameters); print
     `"Starting fresh (<timesteps> timesteps)"`; `model.learn(args.timesteps, callback=ckpt_cb)`.
4. Final `model.save(save_model_path)` + `export_model_as_onnx(...)` + `env.close()` — unchanged.

Edge cases:
- `remaining == 0` (target already reached by the checkpoint): skip `learn`, go straight to the
  final save/export so a re-run still produces the deployable artifacts.
- `CheckpointCallback` creates `checkpoint_dir` if absent; `latest_checkpoint` returns `None` for a
  missing dir, so the first run starts fresh cleanly.

### `scripts/train_rover.sh`

- Add `CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-25000}"` and forward `--checkpoint_freq "$CHECKPOINT_FREQ"`.
- Add a `FRESH` passthrough: if `FRESH` is set (e.g. `FRESH=1`), append `--fresh` to the python call.
- Everything else (port wait, Godot launch, wait/kill) unchanged.

### `.gitignore`

Add:
```
models/rover_checkpoints/
models/rover_policy.*
```
(Checkpoints are transient; the trained rover artifacts mirror the existing `models/chase_policy.*`
ignore rule. The eventual *deployable* ncnn model is copied into `examples/rover_3d/models/` in the
final training step, separate from these.)

## Testing strategy

New `test/python/test_train_rover.py` (stdlib `unittest`, runs under `.venv-train` via the existing
`run_tests.sh` discovery — imports only `train_rover`'s pure helpers, never torch/SB3):

- `latest_checkpoint`: empty/missing dir → `None`; single checkpoint → that path; multiple → the
  highest step count (e.g. `rover_ckpt_50000_steps.zip` over `rover_ckpt_5000_steps.zip`, not
  lexicographic); ignores unrelated files. Uses `tempfile` dirs with touched files.
- `remaining_timesteps`: `total > done` → difference; `done == total` → `0`; `done > total` → `0`.

The two helpers must be importable without torch/SB3 installed (heavy imports stay inside `main`),
so the test imports `scripts/train_rover.py` as a module and calls the helpers directly. Full
`./test/run_tests.sh` stays green.

## Scope boundaries / non-goals

- No real PPO training run is executed here (that remains item 6's deferred final step; it will now
  benefit from checkpoint/resume).
- `train_chase.py` is unchanged (could get the same treatment later if wanted).
- No distributed/remote training; this only makes *local* runs interruptible.

## Follow-ups to record on completion

1. Item 6 final step: run `TIMESTEPS=400000 ./scripts/train_rover.sh` (now resumable), convert to
   ncnn, ship the model, add the trained-rover golden regression.
2. Optionally port checkpoint/resume to `train_chase.py` for symmetry.
