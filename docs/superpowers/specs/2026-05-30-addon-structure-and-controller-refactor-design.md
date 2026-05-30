# Addon Structure + `NcnnAIController` Base Refactor — Design

**Date:** 2026-05-30
**Backlog item:** 5 (`docs/BACKLOG.md`)
**Roadmap reference:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md` §4 Phase 1A
**Status:** Approved (brainstorm → spec)

## Problem & motivation

The reusable library (controller, training bridge, reward module, sensors) currently lives as
loose files at the repo root and under `reward/` / `sensors/`. To support Asset Library
installation and a clean 2D/3D controller story, it needs to live under a proper
`addons/godot_native_rl/` layout with a `plugin.cfg`, and the controller needs to be split so
2D and 3D agents share one implementation instead of duplicating it.

This item delivers the **structural prerequisite**: the addon folder + `plugin.cfg`, the moved
GDScript library, and a controller refactor (shared core + thin `Node2D`/`Node3D` wrappers).
It explicitly does **not** package the compiled GDExtension for distribution — that
(binary-per-platform packaging + submission) is a separate release item.

## Decisions (from brainstorming)

1. **Move scope:** GDScript library only — `sync.gd`, the controllers, `reward/`, `sensors/`
   move into `addons/godot_native_rl/`. The compiled GDExtension (`ncnn_runner.gdextension`,
   `bin/`, `src/`, `SConstruct`) **stays at the repo root, untouched.** Root demo files
   (`NcnnAgent.gd`, `node_2d.gd`, `main.tscn`) are out of scope and stay put.
2. **Controller sharing:** a shared `RefCounted` core (`ncnn_controller_core.gd`) holds the
   node-agnostic episode/reward state machine; `NcnnAIController2D extends Node2D` and a new
   thin `NcnnAIController3D extends Node3D` delegate to it. (GDScript single-inheritance
   forbids one base extending both `Node2D` and `Node3D`, so composition, not inheritance, is
   the sharing mechanism.)
3. **3D scope:** create the thin `NcnnAIController3D` wrapper now (with a headless smoke test)
   to prove the core is genuinely node-agnostic. The 3D **example + training + 3D-specific
   sensors** stay in item 6.
4. **Backward compatibility:** clean move + README migration note. Preserve every `class_name`
   so `extends`/typed references keep resolving; update all in-repo path-based references; keep
   the full suite green. External users who `preload` old `res://` paths must update them.
5. **Asset Library publish:** deferred to a new backlog item (move the `.gdextension` +
   prebuilt all-platform binaries into the addon, repoint the manifest + `SConstruct` target,
   submit). This refactor is designed so that becomes a localized change.

## Architecture

### Target layout

```
addons/godot_native_rl/
  plugin.cfg                       # Asset-Library metadata (name, description, version, script=)
  plugin.gd                        # minimal @tool EditorPlugin (no-op _enter_tree/_exit_tree)
  sync.gd                          # class_name NcnnSync  (moved from res://sync.gd)
  controllers/
    ncnn_controller_core.gd        # NEW — RefCounted shared core (state machine + reward + obs-space)
    ncnn_ai_controller_2d.gd       # class_name NcnnAIController2D extends Node2D (moved + refactored)
    ncnn_ai_controller_3d.gd       # NEW — class_name NcnnAIController3D extends Node3D (thin)
  reward/                          # moved from res://reward/ (internal preloads repathed)
    reward.gd, reward_adapter.gd, reward_builder.gd
    terms/ reward_term.gd, step_penalty_term.gd, alive_bonus_term.gd, event_bonus_term.gd, progress_shaping_term.gd
  sensors/                         # moved from res://sensors/ (internal preloads repathed)
    raycast_math.gd, raycast_sensor_2d.gd, raycast_sensor_3d.gd
```

Stays at root (untouched): `ncnn_runner.gdextension`, `bin/`, `src/`, `SConstruct`,
`NcnnAgent.gd`, `node_2d.gd`, `main.tscn`, `examples/`, `scripts/`, `models/`, `test/`.

### Controller decomposition

**`ncnn_controller_core.gd` (RefCounted)** — node-agnostic, unit-testable. Owns the shared
state and logic:

- State: `done: bool`, `reward: float`, `n_steps: int`, `needs_reset: bool`,
  `heuristic: String`, `reward_source` (nullable), `reset_after: int`.
- `step() -> void` — `n_steps += 1`; if `n_steps > reset_after`: `needs_reset = true` and
  `done = true` (the current `_physics_process` base behavior, godot_rl convention).
- `reset() -> void` — `n_steps = 0`, `needs_reset = false`.
- `reset_if_done() -> void` — `if done: reset()`.
- `zero_reward() -> void`, `set_done_false() -> void`, `get_done() -> bool`,
  `set_heuristic(h: String) -> void`.
- `accumulate(adapters: Array, ctx) -> void` — `if reward_source != null: reward +=
  reward_source.evaluate(ctx)`; `for a in adapters: reward += a.drain()`. (`ctx` is the
  wrapper node, matching today's `reward_source.evaluate(self)`.)
- `static obs_space_from_obs(obs: Dictionary) -> Dictionary` — `{"obs": {"size":
  [obs["obs"].size()], "space": "box"}}`.

**`NcnnAIController2D extends Node2D`** (refactored, same `class_name`, same public API) and
the new **`NcnnAIController3D extends Node3D`** (thin, mirrors 2D):

- Keep **natively on the wrapper**: the `ControlModes` enum and `@export var control_mode`
  (NcnnSync reads *and writes* `agent.control_mode` and `agent.ControlModes`), plus `@export`
  `reset_after`, `model_param_path`, `model_bin_path`, `input_blob_name`, `output_blob_name`.
- Hold `var _core := NcnnControllerCore.new()` and the `NcnnRunner` child.
- **Node/subclass-bound glue stays in the wrapper** (≈20 lines, identical in 2D and 3D — the
  one accepted duplication): `_ready()` (add to group `AGENT`, collect reward adapters, set
  `_core.reset_after`, set up runner if `NCNN_INFERENCE`), `_setup_ncnn_runner()`,
  `set_ncnn_runner_for_test()`, `infer_and_act()` (calls `get_obs()`/`get_action_space()`/
  `set_action()` — user overrides), `collect_reward_adapters()`, `_physics_process()` (calls
  `_core.step()`).
- **Forwarding properties** expose core state under the existing names so subclasses
  (`ChaseAgent`) and tests are unchanged:
  ```gdscript
  var done: bool:
      get: return _core.done
      set(v): _core.done = v
  var reward: float:
      get: return _core.reward
      set(v): _core.reward = v
  var n_steps: int: get/set -> _core.n_steps
  var needs_reset: bool: get/set -> _core.needs_reset      # NcnnSync writes this directly
  var heuristic: String: get/set -> _core.heuristic
  var reward_source: get/set -> _core.reward_source
  ```
- Method delegation: `reset()`, `reset_if_done()`, `zero_reward()`, `set_done_false()`,
  `get_done()`, `set_heuristic()` → `_core`; `accumulate_reward()` → `_core.accumulate(
  _reward_adapters, self)`; `get_obs_space()` → `NcnnControllerCore.obs_space_from_obs(get_obs())`.
- Abstract contract (`get_obs`, `get_reward`, `get_action_space`, `set_action`) stays as
  `assert(false, …)` stubs overridden by the concrete agent.

### Data flow (unchanged externally)

`NcnnSync` ↔ agent contract is byte-for-byte the same: `get_obs_space`, `get_obs`,
`get_reward`, `set_action`, `get_done`, `set_done_false`, `zero_reward`, `set_heuristic`,
`infer_and_act`, `control_mode`/`ControlModes`, `needs_reset`. Only the *implementation*
location changes (wrapper → core).

## Backward compatibility & path migration

`class_name`s preserved (`NcnnAIController2D`, `NcnnSync`, `RewardBuilder`, `RewardTerm`,
`RaycastSensor2D/3D`, `RaycastMath`, …) → all `extends ClassName` / typed refs keep resolving
regardless of file location. Only **path-based** references change:

- **Internal preloads/extends** (repath `res://reward|sensors` → `res://addons/godot_native_rl/…`):
  controller's `preload(reward_adapter.gd)`; `reward_builder.gd` (5 preloads); the 3 reward
  terms' `extends "res://reward/terms/reward_term.gd"`; both raycast sensors' `RaycastMath`
  preload; `examples/chase_the_target/chase_agent.gd`'s `reward_builder.gd` preload.
- **Tests** (repath preloads): `test_reward_*.gd` (6 files), `test_controller_reward_accumulation.gd`,
  `test_sync_inference.gd`, `test_sync_messages.gd`, `test_raycast_*.gd` (3),
  `test_chase_reward_parity.gd`.
- **`.tscn` scenes** (only `res://sync.gd` → `res://addons/godot_native_rl/sync.gd`): 5 files —
  `examples/chase_the_target/chase_the_target.tscn`, `…_train.tscn`,
  `test/integration/protocol_test_scene.tscn`, `inference_smoke_scene.tscn`,
  `trained_chase_scene.tscn`. (These `ext_resource` lines use `path=` only, no `uid=`; the
  moving scripts have no `.uid` companions — so updating the path string is sufficient.)
- `test/unit/stub_agent.gd` and `test/integration/protocol_stub_agent.gd` use
  `extends NcnnAIController2D` (class_name) → **no change needed.**

README gets a short "Library moved under `addons/godot_native_rl/`" migration note: `class_name`
usage is unchanged; external code that preloaded old `res://` paths must repath.

## `plugin.cfg` + `plugin.gd`

```ini
[plugin]
name="Godot Native RL"
description="GDExtension RL framework: native ncnn inference + godot_rl_agents-compatible training bridge, reward authoring, and sensors."
author="minigraphx"
version="0.1.0"
script="plugin.gd"
```

```gdscript
@tool
extends EditorPlugin
func _enter_tree() -> void:
	pass
func _exit_tree() -> void:
	pass
```

A no-op EditorPlugin (required because `plugin.cfg` mandates `script=`). The GDExtension and
`class_name`s auto-register independently of the plugin being enabled; the plugin entry just
makes it a recognized, toggleable, Asset-Library-shaped addon.

## Testing strategy

TDD where new logic is introduced; headless harness (`test/harness.gd`), `preload` refs.

- **New `test/unit/test_controller_core.gd`** (pure, no Node): `step()` increments and crosses
  `reset_after` → `done`/`needs_reset`; `reset()` zeroes; `reset_if_done()`; `zero_reward()`;
  `set_done_false()`/`get_done()`; `set_heuristic()`; `accumulate()` with a stub `reward_source`
  (returns a fixed value, records `ctx`) + stub adapters (return `drain()` values);
  `obs_space_from_obs()` shape.
- **New `test/unit/test_controller_3d.gd`**: a stub agent `extends NcnnAIController3D` overriding
  the contract; assert `reset_after`→`done` via `_core`, `get_obs_space()` shape, and that the
  forwarding properties (`reward`, `needs_reset`, `done`) read/write through to the core.
- **Update** existing `test_controller.gd`, `test_controller_inference.gd`,
  `test_controller_reward_accumulation.gd` and all moved-path preloads. These passing again is
  the backward-compat proof for the 2D wrapper API.
- **Full gate:** `./test/run_tests.sh` green, including the unchanged **trained-chase
  inference** and **golden inference regression** (scenes load the moved `sync.gd` via repathed
  `ext_resource`; the controller resolves via `class_name`).

## Scope boundaries / explicit deferrals (→ backlog)

- **Asset Library *publish*** (move `.gdextension` + prebuilt per-platform binaries into the
  addon, repoint manifest + `SConstruct` target, fill metadata, submit) → **new backlog item**.
- **3D example + training run + 3D-specific sensors** → item 6.
- Root demo files (`NcnnAgent.gd` / `NcnnAgentHelper`, `node_2d.gd`, `main.tscn`) not moved.
- No change to the GDExtension, `bin/`, `src/`, or `SConstruct` in this item.

## Follow-ups to record on completion

1. New backlog item: "Asset Library release" (extension packaging + multi-platform binaries +
   submission).
2. Item 6 consumes `NcnnAIController3D` for the navigate-to-target example.
3. Optionally fold sensor auto-discovery (`collect_sensors()`, deferred from item 3) into the
   refactored controller core in a later pass.
