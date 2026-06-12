# GridWorld Example (Unity Parity) — Design (#48)

**Status:** design / decisions made autonomously per working agreement
**Date:** 2026-06-12
**Issue:** [#48](https://github.com/minigraphx/godot-native-rl/issues/48) (`area:parity`,
`priority:4`, `needs-training-run`)

## Goal

Re-create Unity ML-Agents' **GridWorld** natively: navigate a grid to the goal while avoiding
pits, discrete 4-direction movement; train → ncnn deploy with a committed net + regressions.
The discrete-action Unity-parity companion to 3DBall (#47).

## The issue's deferred decision, resolved

**GridWorld over PushBlock** — the issue's criterion is "whichever better exercises a sensor we
want a worked example for": `RaycastSensor3D` already stars in rover; **`GridSensor2D` (item 11)
shipped with no worked example**. GridWorld is its natural showcase.

## Decisions

1. **Mechanics (Unity GridWorld essence):** a `cell_size`-quantized 2D field (default 8×8 cells)
   with 1 goal (`Area2D`, layer GOAL) and N=3 pits (`Area2D`, layer PIT), positions seeded-random
   per episode (never overlapping, never on the agent spawn). Agent = `Node2D` moved one cell per
   decision, clamped to the field. **Actions: 5 discrete** (stay + 4 directions — matching chase's
   action shape so the deploy path is byte-familiar). **Terminal:** goal (+1) or pit (−1), plus
   the controller's `reset_after` cap; done signaled on terminal (established lesson).
   Step penalty −0.01 (find the goal fast).
2. **Obs = GridSensor2D showcase + goal vector:** a `GridSensor2D` centered on the agent —
   5×5 cells, 2 layers (pits, goal) = 50 floats — plus the normalized goal-relative vector (2)
   = **52 floats**. The grid sensor reads the real `Area2D`s via its query path (the whole point
   of the showcase).
3. **Layout:** `examples/gridworld/` — `gridworld_game.gd` (grid state, spawning, movement,
   terminal detection; pure helpers for cell math), `gridworld_agent.gd`,
   `gridworld_world.tscn` (tiling unit), `gridworld_train.tscn` + `_train_parallel.tscn`
   (`ParallelArena2D`, 8 worlds), `gridworld.tscn` (watchable deploy with the game's
   `_draw` visualizer, chase-style), `models/`.
4. **Training:** `scripts/train_gridworld.{py,sh}` — chase PPO pattern, `TIMESTEPS=300000`
   default (discrete navigation converges fast), ONNX export path (discrete MLP — the
   chase-standard route) → `export_to_ncnn.py`.
5. **Regressions:** golden-inference (discrete argmax over fixed obs — rover pattern, exact) +
   behavioral: deploy scene reaches the goal ≥ K times in N frames (rover pattern). Both in
   `run_tests.sh`.
6. **Docs:** README, CLAUDE.md, gap-analysis parity row, `Closes #48`.

## Testing

Unit: cell math, spawn non-overlap, terminal detection, obs assembly size (52). Integration:
headless smoke with a scripted walk reaching goal + pit (both terminals exercised). Golden +
behavioral post-training.

## Non-goals

Unity's visual-observation variant (our #35 CNN example covers visual obs separately);
PushBlock (can be a later example if wanted — not required by the issue).
