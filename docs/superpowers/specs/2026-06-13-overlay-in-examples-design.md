# PolicyDebugOverlay in Example Play Scenes — Design

**Date:** 2026-06-13
**Issue:** #231 — Add PolicyDebugOverlay (F3 dev overlay) to all example play scenes
**Follow-up:** #232 — NcnnCrowdController should emit `inference_step`
**Context:** part of the v0.3.x examples hardening push (related: #224, #226, #229)
**Status:** Approved (design)

## Problem

The Policy Debugger overlay (`PolicyDebugOverlay`, issue #49) is a drop-in node that shows live
obs / action probabilities / identity for any running ncnn agent, toggled with F3. Today only
`chase_the_target_debug.tscn` carries it. A developer running any other example has no in-scene way
to inspect what the trained policy is doing.

## Decisions (with user)

1. **Role: developer tool, not a viewer explainer.** Keep the overlay's existing defaults — hidden
   until F3, and freed in release/exported builds (`debug_build_only=true`). It is present for
   developers; it does not alter what a casual viewer or an exported/web demo sees.
2. **Scope: standalone play scenes only.** Not training/world sub-scenes (no inference there), not
   the existing chase debug scene (already has it), not regression/special scenes.
3. **Crowd: skip + follow-up.** `chase_crowd` uses `NcnnCrowdController`, which does not emit
   `inference_step`; excluded here and tracked in #232.

## Key facts (verified)

- `PolicyDebugOverlay` is a `CanvasLayer` that, with `controllers` empty, **auto-discovers** every
  node emitting `inference_step` under the scene root. Defaults: `toggle_key=F3`,
  `start_visible=false`, `debug_build_only=true`, `bar_width=8`.
- `inference_step(debug: Dictionary)` is declared/emitted by `NcnnAIController2D` and
  `NcnnAIController3D` (via `NcnnControllerCore`). Every standard example agent extends one of these,
  so the overlay works with **zero script changes**.
- `NcnnCrowdController` does **not** emit `inference_step` → crowd is out of scope (→ #232).
- `quadruped_race.tscn` uses a **single static `Agent` node** whose model `sequential_race.gd`
  hot-swaps between generations via `swap_model()` — so the overlay's `_ready`-time auto-discovery
  covers it; no dynamically-spawned agents to miss.
- Reference wiring already exists in `chase_the_target_debug.tscn` (it additionally sets
  `start_visible=true` + `debug_build_only=false`; the play-scene additions use the defaults instead).

## Design

### Mechanism

Into each target `.tscn`, add:
- one `ext_resource` of type `Script` pointing at
  `res://addons/godot_native_rl/debug/policy_debug_overlay.gd` (new unused `id`), and
- one node `[node name="PolicyDebugOverlay" type="CanvasLayer" parent="."]` with `script =
  ExtResource("<id>")` and **no property overrides** (defaults give the agreed dev-tool behavior).

`load_steps` in the scene header is incremented by one. No GDScript, no addon changes.

### Target scenes (13)

- `examples/chase_the_target/chase_the_target.tscn`
- `examples/rover_3d/rover_3d.tscn`
- `examples/ball_chase/ball_chase.tscn`
- `examples/fly_by/fly_by.tscn`
- `examples/quadruped_walk/quadruped_walk_track.tscn`
- `examples/quadruped_walk/quadruped_hurdles_track.tscn`
- `examples/quadruped_walk/quadruped_race.tscn`
- `examples/quadruped_walk/hexapod_walk_track.tscn`
- `examples/hide_and_seek/hide_and_seek.tscn`
- `examples/hide_and_seek/hide_and_seek_multipolicy.tscn`
- `examples/gridworld/gridworld.tscn`
- `examples/3dball/ball_balance.tscn`
- `examples/visual_chase/visual_chase.tscn`

### Excluded (with reason)

- `chase_the_target_debug.tscn` — already has the overlay.
- `*_train*.tscn`, `*_world.tscn` — no inference happening.
- `hide_and_seek_multipolicy_eval.tscn`, `hide_and_seek_selfplay_{hider,seeker}.tscn` —
  regression/special scenes, not the primary demo.
- `chase_the_target/chase_crowd.tscn` — `NcnnCrowdController` gap (→ #232).
- `coop_collect/*` — no play scene exists (#228).

### Testing

A single headless structure test (`test/unit/test_overlay_in_examples.gd`, `extends SceneTree`,
using `test/harness.gd`) that, for each of the 13 target scene paths:
- loads the scene as a `PackedScene` and instantiates it **without entering the tree** (no model
  load, no inference), and
- asserts exactly one child node whose script is
  `res://addons/godot_native_rl/debug/policy_debug_overlay.gd` is present.

This catches a missed scene or a broken/duplicate wire cheaply. Wired into `test/run_tests.sh` via
the existing `test/unit/test_*.gd` glob (no explicit wiring needed).

### Error handling

The overlay already handles its own edge cases: it warns and skips non-emitting controllers, frees
itself in release builds, and no-ops with an empty tracked list. Nothing new to add.

## Out of scope

- Making the overlay visible by default or in exported builds (it stays a dev tool).
- Per-example `get_debug_status()` context lines (nice-to-have; not required for a dev tool).
- Crowd-controller `inference_step` emission (#232).
- Any change to `PolicyDebugOverlay` itself.

## Docs to update in the same change

- A short note in the example/running docs that **F3** opens the Policy Debugger in any example
  (`docs/guide/running-examples.md` and/or the Policy Debugger section of `README.md`).
- Close #231 via the PR (`Closes #231`).
