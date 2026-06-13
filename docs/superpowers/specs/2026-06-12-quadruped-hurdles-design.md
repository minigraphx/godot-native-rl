# Quadruped Hurdles — #60 M2 (run + jump hurdles)

**Date:** 2026-06-12 · **Issue:** #60 milestone 2 · **Status:** approved (autonomous batch run —
design decided + documented per the standing mandate)

## Goal

The M1 quadruped learns to clear hurdles on its way to the finish: forward **RaycastSensor3D**
perception, a **clear-the-hurdle bonus**, and a game-side **curriculum** (flat → low hurdles →
race spacing). Jump timing emerges inside the existing 8-motor continuous action space — no new
mechanics.

## Decisions (and why)

- **Same example dir** (`examples/quadruped_walk/`): M2 is the same creature + track, one more
  scene family. A separate dir would duplicate the builder/math/game.
- **Hurdles = code-built StaticBody3D boxes on collision layer 2** (`hurdle_track.gd`), so a
  closeness-only `RaycastSensor3D` with `collision_mask = 2` reads *hurdle proximity* and never
  confuses ground/walls. Layout is a pure function (count, spacing, start_z → z list) for unit
  tests; the node rebuilds on `apply_curriculum(params)`.
- **Sensor placement:** the rig is code-built at runtime, so the sensor node can't live under the
  torso in the scene. It's a child of the world; the agent snaps `sensor.global_position` to the
  torso (+ fixed forward orientation, level with the ground) before each read. Level rays keep the
  obs stable while the torso pitches.
  6 rays (3 wide × 2 high), hfov 40°, vfov 25°, length 6 m → OBS 29 + 6 = **35**.
- **Clear detection:** `HurdleTrack.count_newly_passed(torso_z)` — monotonic index over the sorted
  hurdle z's; each newly passed hurdle pays `clear_bonus` (default 1.0). Falling *on* a hurdle ends
  the episode via the existing fall terminal, so "cleared" ≡ passed it upright. Pure-testable,
  reset with the episode.
- **Reward:** M1 v3 locomotion reward unchanged + clear bonus. The new agent also fixes the
  terminal-reward ordering wart (#207): no `zero_reward()` after the fall penalty — the sync must
  read it.
- **Curriculum** (`quadruped_hurdles_curriculum.json`, applied by the existing
  `CurriculumController` pointed at the HurdleTrack):
  - stage 0 `flat`: 0 hurdles (recover M1 walking under the new obs space),
  - stage 1 `low`: 4 hurdles, height 0.15, spacing 8,
  - stage 2 `race`: 6 hurdles, height 0.3, spacing 6.
  Promotion thresholds picked from M1's observed reward scale; tuned at run time if promotions
  stall (printed + surfaced via `curriculum_stage` info).
- **Trainer:** reuse `train_quadruped.{sh,py}`; the `.sh` gains an `OUT=` passthrough mapping to
  `--save_model_path/--pt_export_path` (the `.py` already has the flags). M2 runs with
  `OUT=models/quadruped_hurdles SCENE=res://examples/quadruped_walk/quadruped_hurdles_train_parallel.tscn`.

## Components

| File | Role |
|---|---|
| `hurdle_track.gd` | pure layout + StaticBody3D builder + curriculum apply + passed-counter |
| `quadruped_hurdles_agent.gd` | extends QuadrupedAgent: +raycast obs, +clear bonus, fixed terminal ordering |
| `quadruped_hurdles_world.tscn` | world + HurdleTrack + sensor + agent |
| `quadruped_hurdles_train{,_parallel}.tscn` | Sync(+CurriculumController) training scenes |
| `quadruped_hurdles_curriculum.json` | 3 stages |
| `test/unit/test_quadruped_hurdles.gd` | layout math, passed-counter, obs size (injected cast), curriculum apply |

Deploy scene + trained net + behavioral/golden regressions land with the training run (same
pattern as M1: behavioral = clears ≥ K hurdles / reaches distance D; golden = fixed obs → outputs).

## Error handling

- HurdleTrack with `hurdle_count = 0` builds nothing and `count_newly_passed` returns 0 (stage 0).
- Agent without a sensor degrades to zero-filled ray slots + one warning (obs size stays 35 — the
  wire contract must not drift with scene wiring).

## Testing

Headless unit tests for every pure helper; the existing quadruped smoke pattern extended to the
hurdles world (random motors, no crash, obs size 35). Trained regressions post-run.
