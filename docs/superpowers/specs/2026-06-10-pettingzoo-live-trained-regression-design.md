# PettingZoo interop: live-trained two-policy ncnn regression (#118)

**Date:** 2026-06-10
**Issue:** [#118](https://github.com/minigraphx/godot-native-rl/issues/118) — follow-up to #111 (PR #117)
**Status:** approved design

## Problem

#111 shipped the `GodotParallelEnv` PettingZoo `ParallelEnv` adapter and
`scripts/train_pettingzoo.py` multi-policy PPO, proven deterministically via PettingZoo's
`parallel_api_test`. The **live full training run** through this path was deferred. Until it
happens, nothing pins the PettingZoo path's deploy-side behavior with real trained weights —
the "deterministic now, live-trained run as a follow-up" pattern (same as #74 for SAC and
#79 for SampleFactory) is half-finished.

## Goal

Run a full Hide & Seek seeker+hider multi-policy job through `scripts/train_pettingzoo.sh`,
commit the resulting two-policy ncnn fixtures, and pin them with the same two-layer regression
the custom-PPO multi-policy example (#26) has: deterministic golden inference + a behavioral
LOS floor.

## Non-goals

- RLlib multi-policy *via* `GodotParallelEnv` (the canonical upstream PettingZoo usage) —
  deferred to its own GitHub issue, filed as part of this change.
- Any changes to the trainer, the adapter, or the Hide & Seek env itself.
- Committing TorchScript intermediates (`models/*.pt` stays gitignored).

## Design

### 1. Training run (produces the artifacts; not itself committed)

`caffeinate -is ./scripts/train_pettingzoo.sh` on the dev Mac mini, stock defaults:

- `TIMESTEPS=800000` (same budget the existing custom-PPO multi-policy models used —
  apples-to-apples for the LOS comparison), `NUM_STEPS=256`, `SPEEDUP=8`, `ACTION_REPEAT=8`
- Scene: `res://examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn`
  (8 tiled worlds, `--multi-policy` identity gate)

The trainer exports one actor per `policy_name` to TorchScript
(`models/pettingzoo_{seeker,hider}.pt` + shape sidecars) and runs
`scripts/export_to_ncnn.py --via torchscript` on each, which parity-verifies
(50/50 argmax match, atol 1e-2) at conversion time and fails loud otherwise.

### 2. Committed fixtures

- `models/pettingzoo_seeker.ncnn.{param,bin}`
- `models/pettingzoo_hider.ncnn.{param,bin}`

Exactly the names from the issue, beside the other backend fixtures
(`models/chase_{cleanrl,sf,rllib}_policy.ncnn.*`).

### 3. Golden-inference unit test

`test/unit/test_pettingzoo_golden_inference.gd`, cloned from
`test/unit/test_hide_seek_multipolicy_golden_inference.gd`:

- The same 5 fixed 15-dim observations; index 14 (role flag) forced to 1.0 for the seeker
  probe and 0.0 for the hider probe.
- Expected argmaxes captured from the real ncnn deploy path
  (`NcnnRunner.run_discrete_action`) against the new fixtures; a throwaway capture run prints
  them during implementation.
- Auto-discovered by the `test/unit/test_*.gd` loop in `run_tests.sh` — no harness edit for
  this layer.

### 4. Behavioral LOS regression

`test/integration/trained_pettingzoo_eval.tscn` (regression-only scene, so it lives in
`test/integration/` beside `trained_ball_chase_scene.tscn`, not in `examples/`):

- Instances `res://examples/hide_and_seek/hide_seek_world.tscn`; both agents
  `control_mode = 3` (trained ncnn) with `model_param_path`/`model_bin_path` pointing at the
  pettingzoo fixtures.
- Reuses `test/integration/trained_hide_seek_multipolicy_checker.gd` **unchanged**:
  `frames_to_run = 3000`, `min_los_fraction = 0.08`, `rng_seed = 1` (deterministic seeded
  spawns; policies are argmax-deterministic).
- One new line in `test/run_tests.sh` next to the existing multipolicy eval (~line 71).

**Acceptance gate before committing fixtures:** the trained seeker must clear the floor with
healthy margin — the existing models observe ~20–44% LOS. If the new seeker lands below ~15%,
re-run training rather than lowering the floor.

### Alternatives considered

- **Parameterize the existing eval scene** (injectable model paths): new plumbing for a
  ~30-line scene saving — YAGNI.
- **Python-side behavioral check via the PettingZoo env:** needs a live socket + venv in the
  test suite, and doesn't exercise the actual ncnn deploy path, which is the point.

## Error handling / risks

- 0-update budgets exit loud (#119) — non-issue at 800k.
- pnnx parity failure aborts the export before any fixture exists.
- Behavioral flakiness pre-empted by the seeded deterministic run (already solved by the
  existing checker).
- Multi-hour macOS run wrapped in `caffeinate -is` (sleep gotcha).

## Docs + hygiene (same change)

- `CLAUDE.md`: mark GitHub #118 done in the backlog list; note the committed fixtures in the
  train_pettingzoo bullet.
- `docs/godot-rl-gap-analysis-2026-06-02.md`: update the PettingZoo interop row (live-trained
  now proven).
- File a new GitHub issue for the deferred RLlib-via-`GodotParallelEnv` sibling.
- PR closes #118. `docs/BACKLOG.md` untouched (#118 was never a listed item).

## Testing summary

| Layer | Artifact | What it catches |
|---|---|---|
| Conversion parity | `export_to_ncnn.py` (at export time) | TorchScript↔ncnn drift |
| Golden inference | `test/unit/test_pettingzoo_golden_inference.gd` | conversion/runtime regressions, fixture swaps |
| Behavioral floor | `test/integration/trained_pettingzoo_eval.tscn` + existing checker | "did the live run actually learn" |
