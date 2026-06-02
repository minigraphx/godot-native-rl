# CleanRL Backend Implementation Plan

> **For agentic workers:** TDD — write the failing helper tests first (RED), implement the
> single-file PPO (GREEN), then the orchestrator. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `scripts/train_cleanrl.py` (single-file CleanRL PPO over godot_rl's `CleanRLGodotEnv`)
+ `scripts/train_cleanrl.sh`, training the chase example and exporting ONNX consumable by
`scripts/export_to_ncnn.py`. Pure helpers unit-tested with stdlib `unittest`; heavy imports lazy in
`main()`.

**Spec:** `docs/superpowers/specs/2026-06-02-cleanrl-backend-design.md`

**Conventions:**
- Python 4-space indent. Tests in `test/python/`, add `scripts/` to `sys.path`, `import train_cleanrl`.
- Heavy imports (`torch`, `numpy`, `gymnasium`, `godot_rl`) inside `main()` / inside the helper that
  needs them — module top must import cleanly with none of them present.
- Do **NOT** pass `seed=` to anything that calls `env.seed()` — seed via the env constructor only.
- Run only the python suite here:
  `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'`.

---

## File structure

- **Create** `scripts/train_cleanrl.py` — single-file PPO + pure helpers.
- **Create** `scripts/train_cleanrl.sh` — orchestrator (mirror `train_chase.sh`), `chmod +x`.
- **Create** `test/python/test_train_cleanrl.py` — stdlib `unittest` for the pure helpers.

(Do NOT edit README, CLAUDE.md, BACKLOG.md, run_tests.sh, or existing `train_*` scripts.)

---

## Task 1: Pure helpers + failing tests (RED → GREEN)

- [ ] **Step 1 (RED):** Write `test/python/test_train_cleanrl.py` covering `compute_gae`,
  `discrete_action_dims`, `num_updates`, `parse_args`, `obs_dim`, `act_layout` (and a
  `skipUnless(torch)` `layer_init` check). Hand-compute GAE expectations. Run it → fails with
  `ModuleNotFoundError`/`AttributeError`.

- [ ] **Step 2 (GREEN):** Create `scripts/train_cleanrl.py` with the module-top pure helpers and
  `PPOConfig`. Keep `torch`/`numpy`/`gymnasium`/`godot_rl` imports lazy. `compute_gae` uses plain
  numpy (import numpy inside the function or accept arrays — but it must not import at module top).
  - Decision: `compute_gae` takes numpy arrays and imports `numpy` **inside** the function, so the
    module top stays import-free. Tests import numpy themselves to build inputs.

- [ ] **Step 3:** Run `.venv-train/bin/python -m unittest test.python.test_train_cleanrl -v` → all
  green.

---

## Task 2: PPO loop + ONNX export in `main()`

- [ ] **Step 1:** Implement `Agent` (nn.Module: shared MLP → actor logits head of `sum(nvec)` +
  critic scalar), the per-dim `Categorical` action helper, the rollout/learn loop, save `.pt`,
  and `export_actor_as_onnx` (godot_rl `obs`/`state_ins`→`output`/`state_outs` naming, opset 17).
  All heavy imports inside `main()` / `export_actor_as_onnx`.

- [ ] **Step 2:** Smoke: `.venv-train/bin/python scripts/train_cleanrl.py --help` → exit 0, lists all
  args, no Godot/torch error. (No real training.)

- [ ] **Step 3:** Re-run the helper tests → still green (the `main()` additions don't import heavy
  deps at module load).

---

## Task 3: Orchestrator `scripts/train_cleanrl.sh`

- [ ] **Step 1:** Create it mirroring `train_chase.sh` (TAB indent, `set -euo pipefail`, GODOT/PY/
  TIMESTEPS/SPEEDUP/ACTION_REPEAT env overrides, scene
  `res://examples/chase_the_target/chase_the_target_train.tscn`, launches trainer →
  `train_cleanrl.py`). `chmod +x scripts/train_cleanrl.sh`.

- [ ] **Step 2:** `bash -n scripts/train_cleanrl.sh` → exit 0.

---

## Task 4: Full python suite

- [ ] **Step 1:** `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'` →
  all green. (Do NOT run the full `./test/run_tests.sh` — the parent runs the full suite centrally.)

---

## Deferred follow-ups (not in this plan)

- Ship a trained CleanRL chase model (`models/chase_cleanrl_policy.{onnx,ncnn.param,ncnn.bin}`) +
  a golden ncnn regression, like the SB3 chase/rover examples. Needs a real ~30-min training run.
- README "Key commands" / CLAUDE.md note for the new trainer; BACKLOG item 17 → done. (Reported to
  the user; not edited here per task constraints.)
