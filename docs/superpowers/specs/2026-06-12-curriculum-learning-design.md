# Curriculum Learning — Design (#28)

**Status:** design / approved for planning
**Date:** 2026-06-12
**Issue:** [#28](https://github.com/minigraphx/godot-native-rl/issues/28) (`area:training`, `priority:3`, backlog item 52)

## Goal

Progressive difficulty via environment-parameter staging: the environment starts easy and gets
harder as the policy earns it. Game-side by default (works with **every** training backend
unchanged), with an optional trainer-driven override message for custom training loops. Demonstrated
with a **real trained run** on `chase_the_target` whose stage promotions are performance-gated.

This also unblocks #60 M2 (quadruped hurdles: flat → low hurdles → race spacing is exactly a
3-stage curriculum).

## Why game-side first

The trainer→Godot wire handles only `reset` / `call` / `action`; stock SB3/SampleFactory/RLlib
trainers cannot send custom messages without forks. A controller that tracks episode outcomes
in-engine and promotes stages itself works with all backends, mirrors the repo's game-side
precedent (`RunningNormSensor`), and reports its stage through the **existing** per-agent `info`
field — zero protocol change for the default path.

## Components

### 1. `addons/godot_native_rl/training/curriculum.gd` — pure logic (RefCounted)

- **Stages:** ordered `Array` of
  `{ "name": String, "params": Dictionary, "promote": { "metric": "mean_reward"|"success_rate",
  "threshold": float, "window": int, "min_episodes": int } }`.
  The final stage needs no `promote` block (nothing to promote to).
- **API:**
  - `set_stages(stages: Array)` — validates shape, fails loud on malformed stages.
  - `record_episode(reward: float, success: bool)` — pushes into a rolling window
    (size = current stage's `window`).
  - `should_promote() -> bool` — true iff episodes-in-window ≥ `min_episodes` AND the chosen
    metric over the window ≥ `threshold`. Always false on the final stage.
  - `advance() -> bool` — moves to the next stage, clears the window; false if already final.
  - `current_params() -> Dictionary`, `stage_index() -> int`, `stage_name() -> String`,
    `is_final() -> bool`, `stage_count() -> int`.
  - `set_stage(i: int)` — direct jump (used by the trainer override); clears the window.
- No scene/node dependencies; fully unit-testable (promotion edge cases: empty window,
  min_episodes gate, success-rate vs mean-reward metric, final-stage terminal behavior,
  out-of-range `set_stage`).

### 2. `addons/godot_native_rl/training/curriculum_controller.gd` — thin Node wrapper

- **Config:** `stages_json_path: String` export (JSON file: `{"stages": [...]}`), or
  `set_stages(stages)` programmatically (takes precedence; JSON is convenience).
- **Inputs:** public `record_episode(reward: float, success: bool)` — called by the agent in its
  episode-reset branch (one line), or wired from a game signal by the user.
- **Outputs:**
  - On promotion, defers param application to the **next** `record_episode`'s reset boundary —
    params never change mid-episode. Application = calling `apply_curriculum(params: Dictionary)`
    on the node at `game_path` (method name overridable via `apply_method` export). Missing
    method ⇒ loud `push_error` once.
  - `signal stage_changed(index: int, name: String, params: Dictionary)`.
  - Adds itself to group `CURRICULUM` in `_ready` (discovery by `NcnnSync` and tests).
  - Prints promotions: `Curriculum: promoted to stage N "name" after E episodes (metric=X)`.
- **Trainer-override mode:** `set_external_control(true)` disables auto-promotion (explicit
  trainer control wins); set by `NcnnSync` when a `curriculum` message arrives.
- **Stage visibility to the trainer:** the demo agent includes `{"curriculum_stage": N}` in its
  `get_info()`; documented as the pattern (no protocol change).

### 3. Trainer override — one additive wire message

- `NcnnSync._handle_message` gains a `"curriculum"` case:
  - `{"type": "curriculum", "stage": N}` → `controller.set_external_control(true)`;
    `controller.jump_to_stage(N)` (validated; out-of-range ⇒ `push_warning`, ignored).
  - `{"type": "curriculum", "params": {...}}` → bypasses stages entirely: applies the given
    params at the next reset boundary (external control implied).
  - Controller discovery: first node in the `CURRICULUM` group; absent ⇒ `push_warning`, message
    dropped (never crashes a trainer that speaks the extension against a scene without curriculum).
- Backward compatible: stock trainers never emit the message; the handshake is unchanged.
- **Python helper:** `scripts/curriculum_client.py` — `send_curriculum_stage(conn, stage)` /
  `send_curriculum_params(conn, params)` over the existing JSON-message framing; usable from the
  custom single-file trainers (CleanRL-style, multipolicy). Stdlib-only, unit-tested for framing.

### 4. Demo: chase_the_target 3-stage curriculum + real trained run

- `examples/chase_the_target/chase_curriculum.json` — 3 stages raising difficulty
  (target speed and/or spawn distance; exact values tuned during the run).
- `chase_the_target_train.tscn` variant `chase_the_target_train_curriculum.tscn`:
  adds `CurriculumController` (+ JSON path) — the base train scene stays untouched.
- `ChaseGame.apply_curriculum(params)` — applies `target_speed` / `spawn_distance` (whatever the
  final param set is) at reset; `ChaseAgent` (curriculum variant or the existing agent guarded by
  a null check) calls `record_episode(episode_reward, caught_target)` in its reset branch and
  reports `curriculum_stage` via `get_info()`.
- **Trained demonstration (the issue deliverable):** one real SB3 run via `train_chase.sh` with
  `SCENE=` pointing at the curriculum scene. Evidence: Godot-side promotion log lines + stage
  values in the trainer's info stream, captured in the PR description. Stage thresholds are tuned
  so promotion demonstrably gates on performance (if chase converges too fast, thresholds rise).
  The existing committed chase model + golden fixtures are **not** replaced — curriculum changes
  training dynamics, not the deploy contract.

## Testing

- **Unit (`test/unit/test_curriculum.gd`):** promotion logic edge cases (window/min_episodes
  gates, both metrics, final stage, set_stage bounds, malformed stages fail loud).
- **Unit (`test/unit/test_curriculum_controller.gd`):** JSON loading, deferred param application
  at reset boundary, `stage_changed` emission, external-control disabling auto-promotion,
  missing-`apply_curriculum` loud error.
- **Integration (headless):** scene with controller + stub game; fake a stream of episode results;
  assert stage advancement + applied params. Plus the wire path: extend the existing protocol stub
  test to send a `curriculum` message and assert the controller jumped (mirrors
  `test/integration/run_protocol_test.py` patterns).
- **Python (`test/python/test_curriculum_client.py`):** message framing matches NcnnSync's
  expectations (length-prefixed JSON, same as the protocol test uses).
- All wired into `run_tests.sh` (unit auto-discovered; integration registered).

## Docs on landing

README (feature bullet + demo command), CLAUDE.md (key command: curriculum training variant),
gap-analysis (Unity ML-Agents curriculum parity row), BACKLOG item 52 checkbox, `Closes #28`.

## Non-goals (v1)

- Per-agent curricula (one controller = one global stage; parallel-arena worlds share the stage).
- Auto-regression to easier stages on performance collapse (promote-only; revisit if needed).
- Curriculum for the quadruped/hurdles (that's #60 M2, consuming this feature).
