# 3DBall Example (Unity Parity) — Design (#47)

**Status:** design / decisions made autonomously per working agreement
**Date:** 2026-06-12
**Issue:** [#47](https://github.com/minigraphx/godot-native-rl/issues/47) (`area:parity`,
`priority:4`, `needs-training-run`)

## Goal

Re-create Unity ML-Agents' **3DBall** natively: a platform tilts (2 continuous actions) to keep a
ball balanced; train in Godot → ncnn deploy, shipping a trained net + regressions. The canonical
continuous-control "hello world" — parity proof, not a Unity-asset import.

## Decisions (documented)

1. **Mechanics (faithful to Unity 3DBall):**
   - Platform: `AnimatableBody3D` box, script-rotated. *Why AnimatableBody:* kinematic platforms
     impart contact motion to rigid bodies correctly (StaticBody3D doesn't when moved;
     RigidBody3D would fight direct rotation control). Tilt clamped to ±0.35 rad (~20°).
   - Ball: `RigidBody3D` sphere spawned slightly above the platform center with a small random
     initial offset (seeded RNG) so episodes differ.
   - **Actions (2, continuous, clamped game-side like fly_by):** tilt-rate around X and Z;
     applied per physics frame as `rotation += rate * tilt_speed * delta`.
   - **Obs (8 floats, Unity-matching):** platform rotation.x, rotation.z, ball position relative
     to the platform center (3), ball linear velocity (3). All values naturally small — no
     normalization needed (Unity ships it unnormalized too).
   - **Reward:** +0.1 per step while the ball is alive (Unity's exact scheme); on fall
     (ball.y below the platform by a margin, or |relative x/z| beyond the platform half-extent +
     margin) −1 and terminate (set `done`, mirroring the quadruped's fall-termination lesson —
     the core's timeout alone never fires when episodes end early).
2. **Layout** (issue-specified dir): `examples/3dball/` — `ball_balance_game.gd` (platform/ball
   runtime + pure helpers), `ball_balance_agent.gd` (controller contract),
   `ball_balance_world.tscn` (tiling unit), `ball_balance_train.tscn`,
   `ball_balance_train_parallel.tscn` (8 worlds via `ParallelArena`, spacing 50 — small arena),
   `ball_balance.tscn` (watchable deploy: camera/light + inference), `models/` (committed net).
3. **Training:** `scripts/train_ball_balance.{py,sh}` — fly_by PPO pattern (TorchScript →
   `export_to_ncnn.py`, `--atol 0.05` per the quadruped lesson), default `TIMESTEPS=500000`
   (3DBall converges fast), parallel scene default. Jolt backend (project-wide).
4. **Regressions:** golden-inference (continuous 8-out... 2-out: two action means; rtol 1e-2 +
   atol 1e-2 cross-platform tolerance per the quadruped lesson) + behavioral: in the deploy scene
   under ncnn inference, the ball must survive ≥ N frames (e.g. 1500 of 1800) — a balanced ball
   survives indefinitely, an untrained one falls in ~100; generous margin for Jolt cross-platform
   variance. Both in `run_tests.sh`.
5. **Docs:** README example bullet, CLAUDE.md key command, gap-analysis parity row, BACKLOG (not
   listed — GitHub-only issue), `Closes #47`.

## Testing

Unit: pure helpers (obs assembly, fall detection, tilt clamping). Headless physics smoke (random
actions, obs finite, ball eventually falls and resets — proves termination path). Golden +
behavioral post-training.

## Non-goals

Visual polish beyond a watchable scene; Unity model import; curriculum/normalization (not needed).
