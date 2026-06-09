# In-Editor Policy Debugger — Design

- **Date:** 2026-06-09
- **Issue:** [#23](https://github.com/minigraphx/godot-native-rl/issues/23) — Backlog item 49
- **Labels:** `area:novel`, `priority:3`
- **Status:** Design approved; ready for implementation plan

## 1. Purpose

During native ncnn inference at **deploy time**, overlay live agent internals in the running
Godot viewport so a developer can see, at a glance, *what the agent sees and what it wants*:

- the **observation vector** fed to the network,
- the **action distribution** (softmax probabilities for discrete keys, raw + tanh-squashed
  values for continuous keys) with the chosen action highlighted,
- **which policy / model generation** is loaded, and
- optional **game-specific status** (e.g. distance-to-goal, episode reward).

Pure GDScript + ncnn, **zero Python**, and trivially drop-in for an end-user game developer.
Works when running a scene from the editor (F6) and in debug/exported builds (including web),
which doubles as shareable "watch the AI think" demo material. It is **not** a training tool and
needs **non-`--headless` verification** (the overlay only renders with a viewport).

## 2. Non-goals (YAGNI)

Explicitly out of scope for v1; candidate follow-ups only:

- A true in-editor **dock panel** fed over the remote-debugger channel (`EditorDebuggerPlugin`).
- **Per-sensor grouping** of the obs vector (label slices by sensor node).
- **History / time-series graphing** of obs, logits, or reward.
- Adding `get_debug_status()` to the controller **base class** — it stays purely duck-typed.

## 3. Architecture

Three pieces. The controller's public API grows by **exactly one signal**; everything else is
new, optional, and lives in a new `addons/godot_native_rl/debug/` folder.

```
NcnnControllerCore.choose_and_apply_action()      (existing tap point)
  collect obs → normalize → run_inference → output(logits)
        │
        ├── [NEW] build immutable `debug` Dictionary
        │         emit  agent.inference_step(debug)        (if agent has the signal)
        │
        └── decode_actions → set_action                    (existing)

PolicyDebugOverlay (CanvasLayer node)
  - auto-discovers / or is pointed at controllers
  - connects to each controller's `inference_step`
  - stores the latest payload per controller
  - reads identity (policy/model) off the controller directly
  - polls optional `get_debug_status()` each redraw
  - renders via the pure helper

PolicyDebug (pure static helper, no Node)
  - turns a payload + identity + status into display lines
  - softmax/bar/segmentation/chosen-marker math  ← headless-testable core
```

### 3.1 Signal contract (immutable payload, built fresh each inference)

Declared on the controller node base classes so every example inherits it for free:

```gdscript
# ncnn_ai_controller_2d.gd & ncnn_ai_controller_3d.gd
signal inference_step(debug: Dictionary)
```

Payload shape:

```gdscript
debug = {
    "agent_name":    String,                 # agent.name
    "obs":           PackedFloat32Array,     # normalized vector fed to the net; [] on image path
    "obs_image":     Dictionary,             # { "w":int, "h":int, "c":int } or {} when not image-obs
    "logits":        PackedFloat32Array,     # raw network output, pre-decode
    "action_space":  Dictionary,             # { key: { "size":int, "action_type":String, "squash"?:bool } }
    "action":        Dictionary,             # decoded { key: int | Array[float] }
    "deterministic": bool,                   # core.deterministic_inference
}
```

**Raw-payload principle:** softmax probabilities are **not** sent over the signal. The helper
computes them from `logits` + `action_space` (reusing `InferenceMath.softmax`). This keeps the
payload raw and keeps the probability/segmentation math in a pure, unit-testable function.

**Cost when unused:** core gates emission with `if agent.has_signal("inference_step")`. Emission
to zero connections is free in Godot; the only added cost is constructing one small Dictionary at
**decision cadence** (not per physics frame), which is negligible. Existing examples that inherit
the signal but attach no overlay pay effectively nothing.

The payload is built in **all three** inference branches of `choose_and_apply_action`
(float-vector, recurrent multi-IO, image). On the recurrent path `logits` is the policy output
blob; on the image path `obs` is `[]` and `obs_image` carries the dimensions.

### 3.2 Identity header (read directly off the controller — no payload change)

The overlay already holds each controller reference, so it reads static identity once on connect:

- `policy_name` (multi-policy routing),
- model file **basename** from `model_param_path` (answers "which generation/checkpoint"),
- `deterministic_inference` / `inference_seed`.

This directly serves the #60 ragdoll-race story (different lanes load different checkpoint
"generations") and multi-policy scenes.

### 3.3 Optional game-specific status (duck-typed hook)

A controller **may** implement:

```gdscript
func get_debug_status() -> Dictionary:
    return { "dist_to_target": 0.34, "episode_reward": 12.5, "step": 87 }
```

The overlay polls it each redraw (`if controller.has_method("get_debug_status")`) and renders the
returned `{label: value}` pairs as a STATUS section. Absent → no STATUS section. The **core
inference path never calls this** and stays entirely game-agnostic; only the overlay touches it.

v1 wires it into the **chase** controller (`dist_to_target` + `episode_reward`) as the worked
example. Other examples opt in later.

## 4. Components / files

| Action | File | Purpose |
|--------|------|---------|
| modify | `addons/godot_native_rl/controllers/ncnn_controller_core.gd` | build + emit `debug` payload in the 3 inference branches |
| modify | `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` | declare `signal inference_step(debug: Dictionary)` |
| modify | `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` | declare `signal inference_step(debug: Dictionary)` |
| new | `addons/godot_native_rl/debug/policy_debug.gd` | **pure** static helpers: header line, status rows, obs rows + bars, action rows (softmax/raw/tanh), chosen-action marker, segmentation |
| new | `addons/godot_native_rl/debug/policy_debug_overlay.gd` | `CanvasLayer` node: discovery, toggle, debug-build gate, per-agent latest payload, render via helper |
| modify | `examples/chase_the_target/chase_agent.gd` | add optional `get_debug_status()` (distance + accumulated reward) |
| new | `examples/chase_the_target/chase_the_target_debug.tscn` | demo scene wiring the overlay (keeps golden-regression scenes untouched) |
| new | `test/unit/test_policy_debug.gd` | headless unit tests for the pure helper |
| modify | `README.md`, `CLAUDE.md`, `docs/BACKLOG.md` (item 49), `docs/guide/` | document + flip backlog checkbox; `Closes #23` |

### 4.1 `PolicyDebug` (pure helper) — responsibilities

A `RefCounted` with only static functions; no engine/viewport dependency so it is fully
headless-testable. Indicative surface (final names settled in the plan):

- `header_line(identity: Dictionary) -> String`
- `status_rows(status: Dictionary) -> PackedStringArray`
- `obs_rows(obs: PackedFloat32Array, bar_width: int) -> PackedStringArray`
- `action_rows(logits: PackedFloat32Array, action_space: Dictionary, action: Dictionary, deterministic: bool) -> PackedStringArray`
  - discrete key → `InferenceMath.softmax` over the segment, percent + bar, `‹chosen` marker on
    the index in the decoded `action[key]` (correct for both deterministic argmax and stochastic
    sampling, since the payload carries the actually-chosen action);
  - continuous key → raw value and `tanh`-squashed value (when `squash`), with bars;
  - handles **multi-key** action spaces by walking segments in `action_space` order.
- `bar(value: float, width: int) -> String` — magnitude bar (normalized/clamped).
- `render_lines(debug, identity, status, bar_width) -> PackedStringArray` — top-level composer
  used by the overlay.

### 4.2 `PolicyDebugOverlay` (node) — responsibilities

`CanvasLayer` with a single monospace text target (e.g. `Label`/`RichTextLabel`). Exported config:

```gdscript
@export var controllers: Array[NodePath] = []     # empty = auto-discover all
@export var toggle_key: Key = KEY_F3
@export var start_visible: bool = false
@export var debug_build_only: bool = true          # auto-disable in release exports
@export var bar_width: int = 8
```

Behavior:

- `_ready()`: if `debug_build_only and not OS.is_debug_build()` → `queue_free()` (vanishes in
  release). Resolve `controllers`; if empty, **auto-discover** every node in the scene tree that
  `has_signal("inference_step")`. Connect to each; cache identity. Set initial `visible`.
- On `inference_step(debug)`: store as the latest payload for that agent (keyed by instance).
- `_process`/`_unhandled_input`: toggle `visible` on `toggle_key`; when visible, rebuild text from
  the latest payloads + polled `get_debug_status()` via the helper.

### 4.3 End-user integration flow

Drop **one node** into the scene → run → press the toggle key. No agent edits, no signal wiring,
no production cleanup (auto-hidden in release). Scope to specific agents only by filling
`controllers`.

## 5. Data flow

```
decision step
  → collect obs (existing)
  → normalize (existing)
  → run_inference / multi / image → output (logits)
  → [NEW] build `debug`, emit agent.inference_step(debug)
  → decode_actions → set_action (existing)

overlay.inference_step(debug)  → store latest per agent
overlay._process (visible)     → poll get_debug_status(); render_lines(...) → text
```

## 6. Error handling / edge cases

- **NodePath null or not a controller** (no `inference_step` signal) → `push_warning`, skip it.
- **Auto-discovery finds nothing** → overlay renders an empty/"no agents" state; no crash.
- **logits / action_space size mismatch** (sum of segment sizes ≠ logits length) → render the
  segments that fit and flag the mismatch line; never index out of range.
- **Image-obs path** → `obs == []`; show `obs_image` dims, skip the obs-vector section.
- **Continuous-only / multi-key** action spaces → handled by walking `action_space` order.
- **`get_debug_status()` returns non-Dictionary / throws** → guarded; STATUS section skipped.
- **Release export with `debug_build_only`** → node frees itself in `_ready`; zero footprint.

## 7. Testing

### 7.1 Headless unit tests (`test/unit/test_policy_debug.gd`, via `test/harness.gd`)

Pure `PolicyDebug` helper — no viewport needed:

- softmax row probabilities sum to ~1.0; percentages and bar widths correct;
- chosen-action marker equals argmax for a known logit vector (deterministic);
- continuous key formats raw and `tanh`-squashed values; `squash=false` shows raw only;
- multi-key action space segments are split and labeled in order;
- header line formats policy + model basename + det flag;
- status rows render provided pairs; **missing hook → no STATUS rows**;
- edge cases: empty obs (image path) shows image dims; logits/space mismatch flagged, no crash.

### 7.2 Non-headless manual verification (the issue's explicit requirement)

Run `examples/chase_the_target/chase_the_target_debug.tscn` from the editor (F6):

- confirm live obs + action probabilities update and track the agent;
- confirm the header shows policy/model, STATUS shows distance + reward;
- confirm the toggle key shows/hides the overlay;
- capture a screenshot for the PR/docs.

### 7.3 Regression safety

No golden-model or existing-regression scene changes — the overlay lands only in the **new**
`_debug.tscn`. The signal addition is inert for agents without an attached overlay, so existing
headless suites are unaffected.

## 8. Relationships

- Tap point and APIs verified against `ncnn_controller_core.gd`, `action_decode.gd`,
  `inference_math.gd`, `i_sensor_2d/3d.gd`, and `src/ncnn_runner.h`.
- Reuses `InferenceMath.softmax` / `argmax` (no duplicate math).
- Identity header supports the #60 ragdoll-race "early vs late generation" showcase and
  multi-policy routing (#20 / #73).
- Stochastic sampling display aligns with existing `deterministic_inference` / `inference_seed`.
