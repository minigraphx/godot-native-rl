# Design: Frame-stacking + Running-normalization sensor wrappers (#17, #18)

**Date:** 2026-06-05
**Issues:** [#17](https://github.com/minigraphx/godot-native-rl/issues/17) (Observation History Buffer),
[#18](https://github.com/minigraphx/godot-native-rl/issues/18) (Running Normalization Sensor)
**Deferred follow-up:** [#70](https://github.com/minigraphx/godot-native-rl/issues/70) (explicit per-step advance hook)
**Roadmap:** novel-addons spec §3 B1/B2; godot_rl parity (`area:parity`, `priority:2`)

## Summary

Two new **dimension-agnostic** sensor wrappers under `addons/godot_native_rl/sensors/`, each
wrapping a single inner flat-float sensor and transforming its observation:

- **`ObsHistoryBuffer`** (#17) — keeps a sliding window of the last N observations and emits them
  concatenated. Memory without RNNs; the feed-forward analogue of the blocked recurrent item (22).
- **`RunningNormSensor`** (#18) — tracks rolling mean/variance (Welford) and normalizes its inner
  sensor's output online, so no Python `VecNormalize` is needed at deploy. Game-side complement to
  item 24 (which replays SB3 stats).

Both conform to the duck-typed sensor interface (`get_observation()` / `obs_size()`) and are
auto-discovered by `NcnnControllerCore.collect_sensors`.

## Component 1 — Shared discovery change: `collect_sensors` leaf semantics

`NcnnControllerCore._gather_sensor_obs` currently appends an obs-producing node's observation **and**
recurses into its subtree. New rule: **an obs-producing node is a discovery leaf** — once a child
matches `has_method("get_observation") && has_method("obs_size")`, append its obs and **do not
recurse into its children**.

```gdscript
static func _gather_sensor_obs(node: Node, out: Array) -> void:
	for child in node.get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			out.append_array(child.get_observation())
			continue  # leaf: a node that emits obs owns its subtree (e.g. wrappers)
		_gather_sensor_obs(child, out)
```

This lets a wrapper own its inner sensor as a child without the inner being counted twice. No
existing sensor nests sensors under another sensor, so nothing breaks today. A regression test
(`test/unit/test_collect_sensors_leaf.gd`) locks this exact semantic.

Nodes that have `get_observation()` but **not** `obs_size()` (e.g. `CameraSensor`, which returns a
hex string under its own obs key) still don't match the condition and are handled as before.

## Component 2 — Inner-sensor resolution (shared)

Each wrapper has **exactly one** inner obs-producing **child**, resolved duck-typed as the first
child with both `get_observation()` and `obs_size()`.

- Zero matching children, or more than one → `push_error(...)` and `obs_size()` returns `0`
  (fail loud, per coding-style error-handling rules).
- Because the inner is a *child* and the wrapper is a discovery leaf (Component 1), the inner is
  reached **only** through the wrapper — never double-counted.
- **Nesting works for free:** `ObsHistoryBuffer → RunningNormSensor → RaycastSensor3D` composes,
  since each wrapper calls its child's `get_observation()` / `obs_size()`.

## Component 3 — `ObsHistoryBuffer` (#17)

**Files:**
- `addons/godot_native_rl/sensors/frame_ring.gd` — pure `RefCounted` ring buffer (unit-testable in
  isolation).
- `addons/godot_native_rl/sensors/obs_history_buffer.gd` — thin `Node` wrapper.

**`FrameRing` (pure helper):**
- Constructed with `frame_size: int` and `length: int` (N). Backing storage initialized to zeros.
- `push(frame: Array)` — overwrites the oldest slot with `frame`.
- `flat() -> Array` — returns the N frames concatenated **oldest-first, newest-last**, zero-filled
  for slots not yet written.
- `clear()` — re-zero all slots.

**`ObsHistoryBuffer` (Node wrapper):**
- `@export var history_length := 4` (N).
- `obs_size()` → `history_length * inner.obs_size()` (stable from frame 1; depends only on the pure
  `inner.obs_size()`).
- `get_observation()` → `ring.push(inner.get_observation())`, then `return ring.flat()`.
- `reset()` → `ring.clear()` (per-episode frame-stack reset; see Component 5).

## Component 4 — `RunningNormSensor` (#18)

**Files:**
- `addons/godot_native_rl/sensors/running_stats.gd` — pure `RefCounted` Welford accumulator
  (`count: int`, `mean: Array`, `M2: Array`); `var = M2 / count`.
- `addons/godot_native_rl/sensors/running_norm_sensor.gd` — thin `Node` wrapper.

**`RunningStats` (pure helper):**
- `update(x: Array)` — Welford online update of `count`/`mean`/`M2`, element-wise over the vector.
- `variance() -> Array` — `M2[i] / count` (returns zeros while `count == 0`).
- `to_dict()` / `from_dict(d)` — serialize/deserialize `{count, mean, M2}` for the sidecar.

**`RunningNormSensor` (Node wrapper):**
- `@export var update_stats := true` — set **false** to freeze at deploy.
- `@export var epsilon := 1e-8` — matches SB3 `VecNormalize`.
- `@export var clip_obs := 10.0` — matches SB3 `VecNormalize`.
- `@export var stats_path := ""` — if non-empty and the file exists, load on `_ready()`.
- `obs_size()` → `inner.obs_size()` (1:1).
- `get_observation()`:
  1. `var x := inner.get_observation()`
  2. if `update_stats`: `stats.update(x)`
  3. return element-wise `clamp((x[i] - mean[i]) / sqrt(var[i] + epsilon), -clip_obs, +clip_obs)`
     (update-then-normalize, matching SB3 `VecNormalize`).
- `save_stats(path: String)` — write `stats.to_dict()` as JSON. Documented usage: call at the end of
  training; deploy sets `stats_path` to that file and `update_stats = false`.
- **No `reset()`** — running stats persist across episodes, so the wrapper is simply absent from the
  duck-typed reset propagation.

## Component 5 — Reset propagation hook (train/deploy symmetry)

Stateful sensors need an episode-reset signal. Add **duck-typed reset propagation** in the agent
**controller node** (`NcnnAIController2D` / `NcnnAIController3D`), whose `reset()` already delegates
to `_core.reset()` and which can resolve its own sensors via `collect_sensors(self)`. This path runs
in **both** the training loop and the deploy loop. After the existing reset logic, walk the
discovered sensors and call `reset()` on any that define it:

```gdscript
# In NcnnAIController2D/3D.reset(), after _core.reset():
for sensor in NcnnControllerCore.collect_sensors_nodes(self):
	if sensor.has_method("reset"):
		sensor.reset()
```

(`collect_sensors` today returns concatenated *observations*; reset needs the *nodes*. The plan adds
a sibling `collect_sensors_nodes(root)` that returns the discovered leaf nodes — or refactors the
existing walk to share one traversal. Implementation detail for the plan.)

This zeroes the history ring at episode start in **both** paths.

**Why this is the chosen (lighter) approach — and its accepted caveats:**

`get_obs()` is called once at handshake to measure the obs space (`sync.gd:118`), separate from the
per-step path (`sync.gd:368`); deploy calls it once per step with no handshake probe. Because the
trainer always sends a `reset` before the first real step, the stray handshake-probe frame is wiped
from the history ring before the first real observation — so training's first-step window matches
deploy's exactly. The running stats absorb one harmless extra sample at handshake (negligible vs.
thousands). The non-idempotency of `get_observation()` is documented.

The heavier alternative — a pure `get_observation()` plus an explicit per-step `advance()` plumbed
through `sync.gd` and `NcnnControllerCore` — is deferred to **#70**, to be picked up only if these
caveats actually bite.

## Component 6 — Dimension-agnostic exception

Every other sensor is split `_2d`/`_3d` because it reads geometry and must be a `Node2D`/`Node3D`.
These two wrappers only touch the inner sensor's flat float array — they never read geometry — so
they are single dimension-agnostic `Node`s. The inner child (a `RaycastSensor2D`, `RaycastSensor3D`,
etc.) carries the geometry. A short note in the sensors README/dir documents this intentional
exception to the `_2d`/`_3d` convention.

## Testing

| Test file | Verifies |
|-----------|---------|
| `test/unit/test_frame_ring.gd` | push / oldest-eviction / `flat()` order / zero-fill / `clear()` |
| `test/unit/test_running_stats.gd` | Welford `update`/`variance` vs. a naive reference; `to_dict`/`from_dict` round-trip |
| `test/unit/test_obs_history_buffer.gd` | `obs_size()`, zero-fill warm-up, window order, `reset()` zeroing — with a fake inner sensor |
| `test/unit/test_running_norm_sensor.gd` | update vs. freeze, clip bounds, epsilon, sidecar save→load — fake inner sensor |
| `test/unit/test_collect_sensors_leaf.gd` | leaf semantics: wrapper's inner child is **not** double-counted; nesting works |

Pure helpers (`FrameRing`, `RunningStats`) are tested in isolation. Wrappers use a fake inner sensor
(a small stub exposing `get_observation`/`obs_size`) so no physics world is needed. All wired into
`test/run_tests.sh`; full suite must print `All tests passed.` before merge.

## Out of scope

- Explicit per-step `advance()` hook (deferred — #70).
- Frame-stacking during `RECORD_EXPERT_DEMOS` (advances per physics frame in that niche mode; #70).
- `_2d`/`_3d` variants (intentionally dimension-agnostic — Component 6).

## Docs to update on merge (per CLAUDE.md push checklist)

- README sensors section: add the two wrappers + the dimension-agnostic note.
- `CLAUDE.md`: add #17/#18 to the Done list.
- `docs/BACKLOG.md`: flip items 46 (#17) and 47 (#18).
- Closing PR: `Closes #17`, `Closes #18`.
