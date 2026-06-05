# History-Buffer + Running-Norm Sensor Wrappers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two dimension-agnostic sensor wrappers — `ObsHistoryBuffer` (frame-stacking, #17) and `RunningNormSensor` (online VecNormalize-parity normalization, #18) — that wrap a single inner flat-float sensor and are auto-discovered by `collect_sensors`.

**Architecture:** Each wrapper is a plain `Node` holding exactly one obs-producing **child** (the inner sensor). `collect_sensors` is changed so an obs-producing node is a discovery **leaf** (it owns its subtree → no double-count). State (history ring / running stats) advances inside `get_observation()`; an episode-`reset()` is propagated to sensors from the controller node. Pure helpers (`FrameRing`, `RunningStats`) hold the logic and are unit-tested in isolation; the Node wrappers are thin.

**Tech Stack:** GDScript (TAB indentation), Godot 4.5+, dependency-free headless test harness (`test/harness.gd`, tests `extends SceneTree`, run via `godot --headless --path . --script res://test/...`).

**Spec:** `docs/superpowers/specs/2026-06-05-history-and-running-norm-sensors-design.md`

---

## Conventions for every task

- GDScript uses **TAB** indentation (not spaces). The code blocks below use tabs; preserve them.
- New sensor files extend their base **by path**, never by bare `class_name` (headless class-cache is unreliable — see CLAUDE.md). These two wrappers extend plain `Node` and are duck-typed, so they need no base-path extends.
- Run a single GDScript test with:
  `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_<foo>.gd`
  (Substitute your Godot binary; `GODOT=` env var is what `run_tests.sh` honors. A test passes when its last line is `Results: N passed, 0 failed` and it exits 0.)
- Tests are auto-discovered: `run_tests.sh` runs every `test/unit/test_*.gd`. No registration needed — just create the file with that name.
- Commit after each green task.

---

## Task 1: `FrameRing` pure ring-buffer helper (#17 core)

**Files:**
- Create: `addons/godot_native_rl/sensors/frame_ring.gd`
- Test: `test/unit/test_frame_ring.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_frame_ring.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const FrameRing = preload("res://addons/godot_native_rl/sensors/frame_ring.gd")

func _initialize() -> void:
	var h := Harness.new()

	# frame_size 2, length 3 -> flat() is 6 zeros before any push.
	var r := FrameRing.new(2, 3)
	h.assert_eq(r.flat(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "fresh ring -> all zeros")

	# One push -> newest is last, older slots still zero.
	r.push([1.0, 2.0])
	h.assert_eq(r.flat(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], "one push -> newest last, zero-filled front")

	# Fill exactly to length, oldest-first newest-last.
	r.push([3.0, 4.0])
	r.push([5.0, 6.0])
	h.assert_eq(r.flat(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "filled -> oldest-first order")

	# Overflow evicts the oldest.
	r.push([7.0, 8.0])
	h.assert_eq(r.flat(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "overflow evicts oldest")

	# clear() re-zeros.
	r.clear()
	h.assert_eq(r.flat(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "clear -> all zeros")

	# A push of the wrong frame size is rejected (no mutation, error path).
	r.push([1.0, 2.0])
	r.push([99.0])  # wrong size -> ignored
	h.assert_eq(r.flat(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], "wrong-size push ignored")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `GODOT=/opt/homebrew/bin/godot-mono; "$GODOT" --headless --path . --script res://test/unit/test_frame_ring.gd`
Expected: FAIL — cannot load `frame_ring.gd` (file does not exist) / parse error.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/frame_ring.gd`:

```gdscript
extends RefCounted

# Pure fixed-size ring of float frames for ObsHistoryBuffer (frame-stacking, #17).
# Holds `length` frames of `frame_size` floats each. Newest frame is emitted LAST by flat();
# slots not yet written read as zeros. No scene-tree / geometry dependencies — unit-testable.

var _frame_size: int
var _length: int
var _frames: Array = []  # Array of Array[float], length == _length, oldest at index 0

func _init(frame_size: int, length: int) -> void:
	_frame_size = max(0, frame_size)
	_length = max(0, length)
	clear()

func clear() -> void:
	_frames = []
	for i in _length:
		_frames.append(_zero_frame())

func push(frame: Array) -> void:
	if frame.size() != _frame_size:
		push_error("FrameRing.push: frame size %d != expected %d; ignored." % [frame.size(), _frame_size])
		return
	_frames.pop_front()              # drop oldest
	_frames.append(frame.duplicate())  # newest at the end (immutable copy)

func flat() -> Array:
	var out: Array = []
	for f in _frames:
		out.append_array(f)
	return out

func _zero_frame() -> Array:
	var z: Array = []
	for i in _frame_size:
		z.append(0.0)
	return z
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_frame_ring.gd`
Expected: PASS — `Results: 6 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/frame_ring.gd test/unit/test_frame_ring.gd
git commit -m "feat: FrameRing pure ring-buffer helper for frame-stacking (#17)"
```

---

## Task 2: `ObsHistoryBuffer` Node wrapper (#17)

**Files:**
- Create: `addons/godot_native_rl/sensors/obs_history_buffer.gd`
- Test: `test/unit/test_obs_history_buffer.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_obs_history_buffer.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ObsHistoryBuffer = preload("res://addons/godot_native_rl/sensors/obs_history_buffer.gd")

# Fake inner sensor: returns whatever obs we set; obs_size fixed at 2.
class FakeInner extends Node:
	var _obs: Array = [0.0, 0.0]
	func set_obs(o: Array) -> void:
		_obs = o
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return 2

func _initialize() -> void:
	var h := Harness.new()

	var wrap = ObsHistoryBuffer.new()
	wrap.history_length = 3
	var inner := FakeInner.new()
	wrap.add_child(inner)

	# obs_size is stable from frame 1: N * inner.obs_size().
	h.assert_eq(wrap.obs_size(), 6, "obs_size = 3 * 2")

	# First read: window zero-filled except newest frame.
	inner.set_obs([1.0, 2.0])
	h.assert_eq(wrap.get_observation(), [0.0, 0.0, 0.0, 0.0, 1.0, 2.0], "warm-up zero-fill, newest last")

	# Second read: older frame slides forward.
	inner.set_obs([3.0, 4.0])
	h.assert_eq(wrap.get_observation(), [0.0, 0.0, 1.0, 2.0, 3.0, 4.0], "second frame")

	# Fill + evict.
	inner.set_obs([5.0, 6.0])
	h.assert_eq(wrap.get_observation(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "filled")
	inner.set_obs([7.0, 8.0])
	h.assert_eq(wrap.get_observation(), [3.0, 4.0, 5.0, 6.0, 7.0, 8.0], "evict oldest")

	# reset() re-zeros the window.
	wrap.reset()
	inner.set_obs([9.0, 9.0])
	h.assert_eq(wrap.get_observation(), [0.0, 0.0, 0.0, 0.0, 9.0, 9.0], "reset clears window")

	wrap.free()

	# No inner child -> obs_size 0, empty obs (fail-loud error printed, no crash).
	var lonely = ObsHistoryBuffer.new()
	h.assert_eq(lonely.obs_size(), 0, "no inner -> obs_size 0")
	h.assert_eq(lonely.get_observation(), [], "no inner -> empty obs")
	lonely.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_obs_history_buffer.gd`
Expected: FAIL — cannot load `obs_history_buffer.gd`.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/obs_history_buffer.gd`:

```gdscript
extends Node

# Frame-stacking sensor wrapper (#17). Wraps exactly ONE inner flat-float sensor (a child with
# get_observation()/obs_size()) and emits the last `history_length` observations concatenated,
# oldest-first newest-last, zero-filled before the window is full.
#
# Dimension-agnostic ON PURPOSE: it only touches the inner sensor's flat float Array, never
# geometry, so there is no _2d/_3d split (unlike the geometry sensors). The inner child carries
# the dimensionality.
#
# Discovery: collect_sensors treats any obs-producing node as a leaf, so this wrapper is collected
# (not its inner child) — no double-count. get_observation() advances the ring; reset() clears it.

const FrameRing = preload("res://addons/godot_native_rl/sensors/frame_ring.gd")

## Number of past observations to stack (window length N).
@export var history_length: int = 4

var _ring  # FrameRing, lazily built once the inner sensor's size is known.
var _warned_no_inner := false

func obs_size() -> int:
	var inner := _find_inner()
	if inner == null:
		return 0
	return history_length * inner.obs_size()

func get_observation() -> Array:
	var inner := _find_inner()
	if inner == null:
		return []
	_ensure_ring(inner.obs_size())
	_ring.push(inner.get_observation())
	return _ring.flat()

func reset() -> void:
	if _ring != null:
		_ring.clear()

func _ensure_ring(frame_size: int) -> void:
	if _ring == null:
		_ring = FrameRing.new(frame_size, history_length)

func _find_inner() -> Node:
	var found: Node = null
	for child in get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			if found != null:
				_warn_inner("ObsHistoryBuffer: more than one inner sensor child; expected exactly one.")
				return null
			found = child
	if found == null:
		_warn_inner("ObsHistoryBuffer: no inner sensor child (need a child with get_observation()/obs_size()).")
	return found

func _warn_inner(msg: String) -> void:
	if not _warned_no_inner:
		push_error(msg)
		_warned_no_inner = true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_obs_history_buffer.gd`
Expected: PASS — `Results: 8 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/obs_history_buffer.gd test/unit/test_obs_history_buffer.gd
git commit -m "feat: ObsHistoryBuffer frame-stacking sensor wrapper (#17)"
```

---

## Task 3: `RunningStats` pure Welford helper (#18 core)

**Files:**
- Create: `addons/godot_native_rl/sensors/running_stats.gd`
- Test: `test/unit/test_running_stats.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_running_stats.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RunningStats = preload("res://addons/godot_native_rl/sensors/running_stats.gd")

func _approx(a: Array, b: Array, eps: float) -> bool:
	if a.size() != b.size():
		return false
	for i in a.size():
		if absf(float(a[i]) - float(b[i])) > eps:
			return false
	return true

func _initialize() -> void:
	var h := Harness.new()

	var s := RunningStats.new()

	# Samples for two features: [1,10], [2,20], [3,30].
	s.update([1.0, 10.0])
	s.update([2.0, 20.0])
	s.update([3.0, 30.0])

	h.assert_eq(s.count, 3, "count == 3")
	h.assert_true(_approx(s.mean, [2.0, 20.0], 1e-9), "mean == [2,20]")
	# Population variance: feature0 = 2/3, feature1 = 200/3.
	h.assert_true(_approx(s.variance(), [2.0 / 3.0, 200.0 / 3.0], 1e-9), "variance matches naive reference")

	# Round-trip through dict.
	var d := s.to_dict()
	var s2 := RunningStats.new()
	s2.from_dict(d)
	h.assert_eq(s2.count, 3, "round-trip count")
	h.assert_true(_approx(s2.mean, [2.0, 20.0], 1e-9), "round-trip mean")
	h.assert_true(_approx(s2.variance(), [2.0 / 3.0, 200.0 / 3.0], 1e-9), "round-trip variance")

	# Zero-count variance is all zeros (no div-by-zero).
	var empty := RunningStats.new()
	h.assert_eq(empty.variance(), [], "empty -> empty variance")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_running_stats.gd`
Expected: FAIL — cannot load `running_stats.gd`.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/running_stats.gd`:

```gdscript
extends RefCounted

# Pure online mean/variance accumulator (Welford) for RunningNormSensor (#18). Element-wise over a
# fixed-width float vector. var = M2 / count (population variance, matching SB3 VecNormalize's
# RunningMeanStd). No scene-tree dependency — unit-testable. Serializes to {count, mean, M2}.

var count: int = 0
var mean: Array = []  # Array[float], one per feature
var M2: Array = []    # Array[float], sum of squared deviations, one per feature

func update(x: Array) -> void:
	if count == 0:
		_init_dims(x.size())
	if x.size() != mean.size():
		push_error("RunningStats.update: vector size %d != established %d; ignored." % [x.size(), mean.size()])
		return
	count += 1
	for i in x.size():
		var xi := float(x[i])
		var delta := xi - mean[i]
		mean[i] += delta / count
		M2[i] += delta * (xi - mean[i])

func variance() -> Array:
	var out: Array = []
	if count == 0:
		return out
	for i in M2.size():
		out.append(M2[i] / count)
	return out

func to_dict() -> Dictionary:
	return {"count": count, "mean": mean.duplicate(), "M2": M2.duplicate()}

func from_dict(d: Dictionary) -> void:
	count = int(d.get("count", 0))
	mean = (d.get("mean", []) as Array).duplicate()
	M2 = (d.get("M2", []) as Array).duplicate()

func _init_dims(n: int) -> void:
	mean = []
	M2 = []
	for i in n:
		mean.append(0.0)
		M2.append(0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_running_stats.gd`
Expected: PASS — `Results: 7 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/running_stats.gd test/unit/test_running_stats.gd
git commit -m "feat: RunningStats pure Welford accumulator for running-norm (#18)"
```

---

## Task 4: `RunningNormSensor` Node wrapper (#18)

**Files:**
- Create: `addons/godot_native_rl/sensors/running_norm_sensor.gd`
- Test: `test/unit/test_running_norm_sensor.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_running_norm_sensor.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RunningNormSensor = preload("res://addons/godot_native_rl/sensors/running_norm_sensor.gd")

class FakeInner extends Node:
	var _obs: Array = [0.0]
	func set_obs(o: Array) -> void:
		_obs = o
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return 1

func _initialize() -> void:
	var h := Harness.new()

	var wrap = RunningNormSensor.new()
	wrap.clip_obs = 10.0
	var inner := FakeInner.new()
	wrap.add_child(inner)

	h.assert_eq(wrap.obs_size(), 1, "obs_size passthrough == 1")

	# First sample: after update count=1, mean=x, var=0 -> (x-mean)/sqrt(eps) == 0.
	inner.set_obs([5.0])
	var o1: Array = wrap.get_observation()
	h.assert_true(absf(o1[0]) < 1e-3, "first sample normalizes to ~0")

	# Feed more samples; output stays clipped within [-clip, clip].
	for v in [10.0, -10.0, 100.0, -100.0]:
		inner.set_obs([v])
		var o: Array = wrap.get_observation()
		h.assert_true(o[0] <= 10.0 + 1e-6 and o[0] >= -10.0 - 1e-6, "clipped within bounds (%s)" % v)

	# Freeze: stats stop updating. Capture count, then read again with update_stats=false.
	var frozen_count: int = wrap.stats_count()
	wrap.update_stats = false
	inner.set_obs([42.0])
	wrap.get_observation()
	h.assert_eq(wrap.stats_count(), frozen_count, "frozen -> count unchanged")

	# Sidecar save -> load round-trip reproduces normalization.
	var path := "user://test_running_norm_stats.json"
	wrap.save_stats(path)
	var wrap2 = RunningNormSensor.new()
	wrap2.stats_path = path
	wrap2.update_stats = false
	var inner2 := FakeInner.new()
	wrap2.add_child(inner2)
	wrap2._ready()  # trigger the load explicitly in the headless test
	inner.set_obs([7.0])
	inner2.set_obs([7.0])
	wrap.update_stats = false
	var a: Array = wrap.get_observation()
	var b: Array = wrap2.get_observation()
	h.assert_true(absf(a[0] - b[0]) < 1e-5, "loaded stats reproduce normalization")

	wrap.free()
	wrap2.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_running_norm_sensor.gd`
Expected: FAIL — cannot load `running_norm_sensor.gd`.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/running_norm_sensor.gd`:

```gdscript
extends Node

# Online observation-normalization sensor wrapper (#18). Wraps exactly ONE inner flat-float sensor
# (a child with get_observation()/obs_size()) and normalizes its output with a running mean/variance
# (Welford), matching SB3 VecNormalize: (x - mean) / sqrt(var + epsilon), clipped to +/- clip_obs.
#
# update-then-normalize each step (matches VecNormalize). Set update_stats=false to FREEZE at deploy.
# Persist with save_stats(path); set stats_path to load frozen stats on _ready(). Stats persist
# ACROSS episodes (no reset() — deliberately absent from sensor reset propagation).
#
# Dimension-agnostic ON PURPOSE (only touches the inner flat float Array): no _2d/_3d split.

const RunningStats = preload("res://addons/godot_native_rl/sensors/running_stats.gd")

## When true, the running stats are updated each step (training). Set false to freeze (deploy).
@export var update_stats: bool = true
## Numerical floor, matches SB3 VecNormalize default.
@export var epsilon: float = 1e-8
## Normalized values are clipped to [-clip_obs, +clip_obs], matches SB3 VecNormalize default.
@export var clip_obs: float = 10.0
## Optional path to a stats sidecar JSON; if it exists, loaded on _ready().
@export var stats_path: String = ""

var _stats  # RunningStats
var _warned_no_inner := false

func _ready() -> void:
	_stats = RunningStats.new()
	if stats_path != "" and FileAccess.file_exists(stats_path):
		var f := FileAccess.open(stats_path, FileAccess.READ)
		if f != null:
			var parsed = JSON.parse_string(f.get_as_text())
			if typeof(parsed) == TYPE_DICTIONARY:
				_stats.from_dict(parsed)
			else:
				push_error("RunningNormSensor: stats_path %s is not a JSON object." % stats_path)

func obs_size() -> int:
	var inner := _find_inner()
	return 0 if inner == null else inner.obs_size()

func get_observation() -> Array:
	var inner := _find_inner()
	if inner == null:
		return []
	_ensure_stats()
	var x: Array = inner.get_observation()
	if update_stats:
		_stats.update(x)
	return _normalize(x)

func save_stats(path: String) -> void:
	_ensure_stats()
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		push_error("RunningNormSensor.save_stats: cannot open %s for write." % path)
		return
	f.store_string(JSON.stringify(_stats.to_dict()))

func stats_count() -> int:
	_ensure_stats()
	return _stats.count

func _normalize(x: Array) -> Array:
	var mean: Array = _stats.mean
	var var_arr: Array = _stats.variance()
	var out: Array = []
	for i in x.size():
		var m: float = mean[i] if i < mean.size() else 0.0
		var v: float = var_arr[i] if i < var_arr.size() else 0.0
		var z := (float(x[i]) - m) / sqrt(v + epsilon)
		out.append(clampf(z, -clip_obs, clip_obs))
	return out

func _ensure_stats() -> void:
	if _stats == null:
		_stats = RunningStats.new()

func _find_inner() -> Node:
	var found: Node = null
	for child in get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			if found != null:
				_warn_inner("RunningNormSensor: more than one inner sensor child; expected exactly one.")
				return null
			found = child
	if found == null:
		_warn_inner("RunningNormSensor: no inner sensor child (need a child with get_observation()/obs_size()).")
	return found

func _warn_inner(msg: String) -> void:
	if not _warned_no_inner:
		push_error(msg)
		_warned_no_inner = true
```

> Note: the test calls `wrap2._ready()` explicitly because nodes created with `.new()` in a headless `SceneTree` test aren't added to the tree, so `_ready()` doesn't fire automatically. `_ensure_stats()` guards the case where `_ready()` never ran (e.g. `wrap` in the test was used before any add-to-tree), so `get_observation()` is always safe.

- [ ] **Step 4: Run test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_running_norm_sensor.gd`
Expected: PASS — all assertions, `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/running_norm_sensor.gd test/unit/test_running_norm_sensor.gd
git commit -m "feat: RunningNormSensor online VecNormalize-parity sensor wrapper (#18)"
```

---

## Task 5: `collect_sensors` leaf semantics

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd` (function `_gather_sensor_obs`, ~line 138)
- Modify: `test/unit/test_collect_sensors.gd` (update the nested-sensor assertion)
- Test: `test/unit/test_collect_sensors_leaf.gd` (new)

- [ ] **Step 1: Update the existing test's nested-sensor case to the NEW contract**

In `test/unit/test_collect_sensors.gd`, replace the post-order block (the one with `parent_sensor`) so it expects the leaf semantic. Replace these lines:

```gdscript
	# Pre-order is load-bearing: a sensor that is ITSELF the parent of another sensor.
	# Depth-first PRE-order yields parent-before-child [10, 11]; a post-order traversal
	# would yield [11, 10], so this assertion distinguishes the two strategies.
	var po_root := Node.new()
	var parent_sensor := _make_sensor([10.0])
	parent_sensor.add_child(_make_sensor([11.0]))     # sensor nested under a sensor
	po_root.add_child(parent_sensor)
	po_root.add_child(_make_sensor([12.0]))
	h.assert_eq(NcnnControllerCore.collect_sensors(po_root), [10.0, 11.0, 12.0], "pre-order: parent sensor before its child sensor")
	po_root.free()
```

with:

```gdscript
	# Leaf semantics: an obs-producing node OWNS its subtree. A sensor nested under another
	# sensor is NOT separately collected (this is what lets wrappers hold an inner sensor child
	# without double-counting). The parent emits [10]; its child [11] is skipped; sibling [12] is
	# collected. (Pre-existing pre-order behavior was changed deliberately — see spec #17/#18.)
	var leaf_root := Node.new()
	var parent_sensor := _make_sensor([10.0])
	parent_sensor.add_child(_make_sensor([11.0]))     # inner sensor, owned by the parent
	leaf_root.add_child(parent_sensor)
	leaf_root.add_child(_make_sensor([12.0]))
	h.assert_eq(NcnnControllerCore.collect_sensors(leaf_root), [10.0, 12.0], "obs-producing node is a leaf: inner child not double-counted")
	leaf_root.free()
```

- [ ] **Step 2: Run the updated test to verify it now FAILS against current code**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_collect_sensors.gd`
Expected: FAIL — the new assertion expects `[10.0, 12.0]` but current code returns `[10.0, 11.0, 12.0]`.

- [ ] **Step 3: Implement the leaf semantic**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, change `_gather_sensor_obs`. Current:

```gdscript
static func _gather_sensor_obs(node: Node, out: Array) -> void:
	for child in node.get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			out.append_array(child.get_observation())
		_gather_sensor_obs(child, out)
```

Replace with:

```gdscript
static func _gather_sensor_obs(node: Node, out: Array) -> void:
	for child in node.get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			out.append_array(child.get_observation())
			continue  # leaf: an obs-producing node owns its subtree (wrappers hold an inner sensor)
		_gather_sensor_obs(child, out)
```

Also update the doc comment just above `collect_sensors` (~line 129-132): change the "Recursively gather flat sensors" note to state that an obs-producing node is a leaf and owns its subtree (so wrapper sensors can hold an inner sensor child).

- [ ] **Step 4: Run both tests to verify they pass**

Run:
```bash
"$GODOT" --headless --path . --script res://test/unit/test_collect_sensors.gd
"$GODOT" --headless --path . --script res://test/unit/test_controller_collect_sensors.gd
```
Expected: both PASS, `0 failed`. (The second is run to confirm controller-level discovery still works under the new semantic.)

- [ ] **Step 5: Write the focused leaf-semantics regression test**

Create `test/unit/test_collect_sensors_leaf.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")
const ObsHistoryBuffer = preload("res://addons/godot_native_rl/sensors/obs_history_buffer.gd")

class MockSensor extends Node:
	var _obs: Array = []
	func setup(o: Array) -> void:
		_obs = o
	func get_observation() -> Array:
		return _obs
	func obs_size() -> int:
		return _obs.size()

func _initialize() -> void:
	var h := Harness.new()

	# A real wrapper holding an inner sensor: collect_sensors returns ONLY the wrapper's stacked
	# output, never the inner sensor's raw obs separately.
	var root := Node.new()
	var wrap = ObsHistoryBuffer.new()
	wrap.history_length = 2
	var inner := MockSensor.new()
	inner.setup([1.0, 2.0])
	wrap.add_child(inner)
	root.add_child(wrap)
	# history_length 2 * inner size 2 = 4; one push -> [0,0,1,2].
	h.assert_eq(NcnnControllerCore.collect_sensors(root), [0.0, 0.0, 1.0, 2.0], "wrapper collected as leaf, inner not double-counted")
	root.free()

	h.finish(self)
```

- [ ] **Step 6: Run the new regression test**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_collect_sensors_leaf.gd`
Expected: PASS — `Results: 1 passed, 0 failed`.

- [ ] **Step 7: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd test/unit/test_collect_sensors.gd test/unit/test_collect_sensors_leaf.gd
git commit -m "feat: collect_sensors treats obs-producing nodes as leaves (wrapper support, #17/#18)"
```

---

## Task 6: Reset propagation to sensors

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd` (add `collect_sensors_nodes` static helper)
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` (`reset()`, ~line 179)
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` (`reset()`, ~line 179)
- Test: `test/unit/test_sensor_reset_propagation.gd` (new)

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_sensor_reset_propagation.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

class ResettableSensor extends Node:
	var reset_calls: int = 0
	func get_observation() -> Array:
		return [0.0]
	func obs_size() -> int:
		return 1
	func reset() -> void:
		reset_calls += 1

class PlainSensor extends Node:
	func get_observation() -> Array:
		return [0.0]
	func obs_size() -> int:
		return 1

func _initialize() -> void:
	var h := Harness.new()

	# collect_sensors_nodes returns the leaf sensor NODES (not their obs).
	var root := Node.new()
	var a := ResettableSensor.new()
	var b := PlainSensor.new()
	root.add_child(a)
	root.add_child(b)
	var nodes: Array = NcnnControllerCore.collect_sensors_nodes(root)
	h.assert_eq(nodes.size(), 2, "two sensor nodes discovered")
	h.assert_true(nodes.has(a) and nodes.has(b), "both nodes present")

	# Calling reset on those with a reset() method increments only the resettable one.
	for n in nodes:
		if n.has_method("reset"):
			n.reset()
	h.assert_eq(a.reset_calls, 1, "resettable sensor reset once")
	root.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_sensor_reset_propagation.gd`
Expected: FAIL — `collect_sensors_nodes` does not exist.

- [ ] **Step 3: Add `collect_sensors_nodes` to the core**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, just below `_gather_sensor_obs`, add:

```gdscript
# Discover the obs-producing leaf sensor NODES under `root` (same leaf rule as collect_sensors).
# Used to propagate lifecycle calls (e.g. reset()) to stateful sensor wrappers.
static func collect_sensors_nodes(root: Node) -> Array:
	var out: Array = []
	_gather_sensor_nodes(root, out)
	return out

static func _gather_sensor_nodes(node: Node, out: Array) -> void:
	for child in node.get_children():
		if child.has_method("get_observation") and child.has_method("obs_size"):
			out.append(child)
			continue  # leaf: owns its subtree
		_gather_sensor_nodes(child, out)
```

- [ ] **Step 4: Propagate reset from both controllers**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`, change `reset()`:

```gdscript
func reset() -> void:
	_core.reset()
	for sensor in NcnnControllerCore.collect_sensors_nodes(self):
		if sensor.has_method("reset"):
			sensor.reset()
```

Apply the identical change to `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`'s `reset()`.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
"$GODOT" --headless --path . --script res://test/unit/test_sensor_reset_propagation.gd
"$GODOT" --headless --path . --script res://test/unit/test_controller.gd
"$GODOT" --headless --path . --script res://test/unit/test_controller_3d.gd
```
Expected: all PASS, `0 failed`. (Controller tests confirm the modified `reset()` didn't break existing behavior.)

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd test/unit/test_sensor_reset_propagation.gd
git commit -m "feat: propagate episode reset() to stateful sensor wrappers (#17/#18)"
```

---

## Task 7: Full suite + docs

**Files:**
- Modify: `README.md` (sensors section)
- Modify: `CLAUDE.md` (Done list)
- Modify: `docs/BACKLOG.md` (items 46, 47)

- [ ] **Step 1: Run the full test suite**

Run: `GODOT=/opt/homebrew/bin/godot-mono ./test/run_tests.sh`
Expected: ends with `All tests passed.` and exit code 0. If anything fails, fix it before continuing (do not edit tests to pass — fix the implementation). Do NOT gate on grepping `failed`/`ERROR` (both appear in passing runs); gate on the final `All tests passed.` line / exit 0.

- [ ] **Step 2: Update README sensors section**

In `README.md`, in the Sensors section, add entries for the two wrappers. Add this prose (adapt heading level to the surrounding section):

```markdown
#### Sensor wrappers (dimension-agnostic)

Two sensors *wrap* another sensor rather than reading geometry, so — unlike the geometry sensors —
they have no `2D`/`3D` split. Place the inner sensor as their **single child**:

- **`ObsHistoryBuffer`** (`history_length`) — frame-stacking. Emits the last N observations of its
  inner sensor concatenated (oldest-first, newest-last; zero-filled until the window fills).
  `obs_size()` = `N × inner.obs_size()`. Gives a feed-forward policy short-term memory.
- **`RunningNormSensor`** (`update_stats`, `epsilon`, `clip_obs`, `stats_path`) — normalizes its
  inner sensor online with a running mean/variance (Welford), matching SB3 `VecNormalize`
  (`(x-mean)/sqrt(var+eps)`, clipped to ±`clip_obs`). Train with `update_stats = true`; call
  `save_stats(path)` at the end of training; at deploy set `stats_path` to that file and
  `update_stats = false`. No Python `VecNormalize` needed at deploy.

They compose: `ObsHistoryBuffer → RunningNormSensor → RaycastSensor3D` stacks normalized frames.
```

- [ ] **Step 3: Update CLAUDE.md Done list**

In `CLAUDE.md`, in the "Done:" enumeration under Roadmap & backlog, add:

```
17 (ObsHistoryBuffer — frame-stacking sensor wrapper),
18 (RunningNormSensor — online VecNormalize-parity normalization, game-side stats + freeze/persist),
```

- [ ] **Step 4: Update docs/BACKLOG.md**

In `docs/BACKLOG.md`, find items 46 and 47 (the open-item map shows 46→#17, 47→#18) and flip their checkboxes / status to done, referencing this work.

- [ ] **Step 5: Re-run the full suite (docs changes don't affect it, but confirm clean tree state)**

Run: `GODOT=/opt/homebrew/bin/godot-mono ./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: document ObsHistoryBuffer + RunningNormSensor; close items 46/47 (#17, #18)"
```

---

## Task 8: PR

- [ ] **Step 1: Fetch + rebase onto latest origin/main** (main moves fast in this repo)

```bash
git fetch origin
git rebase origin/main
```
Resolve conflicts if any (CLAUDE.md/BACKLOG.md are common conflict points), then re-run `./test/run_tests.sh`.

- [ ] **Step 2: Push the feature branch and open the PR**

```bash
git push -u origin HEAD
gh pr create --title "feat: frame-stacking + running-norm sensor wrappers (#17, #18)" --body "$(cat <<'EOF'
## Summary
- `ObsHistoryBuffer` (#17): frame-stacking wrapper — stacks the last N inner-sensor observations.
- `RunningNormSensor` (#18): online VecNormalize-parity normalization, game-side, with freeze + sidecar persistence.
- `collect_sensors` now treats an obs-producing node as a discovery leaf (wrappers own their inner sensor child — no double-count).
- Episode `reset()` is propagated to stateful sensor wrappers from the controller.

Deferred alternative (explicit per-step advance hook) tracked in #70.

## Test plan
- [x] New unit tests: FrameRing, RunningStats, ObsHistoryBuffer, RunningNormSensor, collect_sensors leaf, reset propagation.
- [x] Updated `test_collect_sensors.gd` for the new leaf contract.
- [x] `./test/run_tests.sh` → `All tests passed.`

Closes #17
Closes #18
EOF
)"
```

---

## Self-review notes (addressed)

- **Spec coverage:** Component 1 → Task 5; Component 2 (inner resolution) → `_find_inner` in Tasks 2 & 4; Component 3 → Tasks 1–2; Component 4 → Tasks 3–4; Component 5 (reset propagation) → Task 6; Component 6 (dimension-agnostic note) → Task 7 docs; Testing table → tests across Tasks 1–6.
- **Type/name consistency:** `FrameRing.new(frame_size, length)`, `.push/.flat/.clear`; `RunningStats.count/mean/M2/update/variance/to_dict/from_dict`; wrapper exports `history_length` / `update_stats,epsilon,clip_obs,stats_path`; core helpers `collect_sensors` (obs) and `collect_sensors_nodes` (nodes) — all consistent across tasks.
- **Known deferral:** explicit advance hook + demo-recording-mode frame-stacking → #70 (out of scope here).
