# Addon Structure + `NcnnAIController` Base Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the reusable GDScript library under `addons/godot_native_rl/` with a `plugin.cfg`, and split `NcnnAIController2D` into a shared `RefCounted` core + thin `Node2D`/`Node3D` wrappers — backward-compatible, full suite staying green.

**Architecture:** A new `NcnnControllerCore` (RefCounted) owns the node-agnostic episode/reward state machine; `NcnnAIController2D extends Node2D` and a new `NcnnAIController3D extends Node3D` delegate to it and expose the historical state via forwarding properties. `sync.gd`, `reward/`, `sensors/`, and the controllers move into the addon; the compiled GDExtension stays at the repo root.

**Tech Stack:** Godot 4.6 GDScript (TAB indentation), headless `extends SceneTree` harness (`test/harness.gd`), `preload` refs (no bare `class_name` in headless `--script` entrypoints — but `extends ClassName` in preloaded scripts is fine, as the existing `stub_agent.gd extends NcnnAIController2D` proves).

**Spec:** `docs/superpowers/specs/2026-05-30-addon-structure-and-controller-refactor-design.md`

**Conventions for every task:**
- GDScript uses **TAB** indentation (code blocks below already use tabs — preserve exactly).
- The `godot` binary is on PATH (`/opt/homebrew/bin/godot`, v4.6.2). Run a single unit test with
  `godot --headless --path . --script "res://test/unit/test_NAME.gd"`.
- Full suite: `./test/run_tests.sh` — must end with `All tests passed.` (includes trained-chase
  inference + golden regression). New `test/unit/test_*.gd` are auto-discovered; helper scripts
  not named `test_*` are not.
- Verify branch is `feat/addon-structure` (`git branch --show-current`) before each commit. Never commit on main.
- Path rewrites use `perl -pi -e` (portable on macOS). The literal old prefix (e.g. `res://reward/`)
  never matches the new addon path (`res://addons/godot_native_rl/reward/`), so a global rewrite is safe.

---

## File structure (end state)

```
addons/godot_native_rl/
  plugin.cfg                       # NEW
  plugin.gd                        # NEW (minimal @tool EditorPlugin)
  sync.gd                          # moved from res://sync.gd
  controllers/
    ncnn_controller_core.gd        # NEW (RefCounted core)
    ncnn_ai_controller_2d.gd       # moved + refactored
    ncnn_ai_controller_3d.gd       # NEW (thin Node3D wrapper)
  reward/    …moved (internal preloads repathed)
  sensors/   …moved (internal preloads repathed)
test/unit/
  test_controller_core.gd          # NEW
  test_controller_3d.gd            # NEW
  stub_agent_3d.gd                 # NEW (helper, not test_*)
```
Unchanged at root: `ncnn_runner.gdextension`, `bin/`, `src/`, `SConstruct`, `NcnnAgent.gd`, `node_2d.gd`, `main.tscn`.

---

## Task 1: Addon skeleton (`plugin.cfg` + `plugin.gd`)

**Files:**
- Create: `addons/godot_native_rl/plugin.cfg`
- Create: `addons/godot_native_rl/plugin.gd`

- [ ] **Step 1: Create `addons/godot_native_rl/plugin.gd`**

```gdscript
@tool
extends EditorPlugin

# Marker EditorPlugin so this is a recognized, toggleable addon for the Asset Library.
# The GDExtension (NcnnRunner) and all class_names auto-register independently of this
# plugin being enabled, so enabling it is optional for using the library.
func _enter_tree() -> void:
	pass

func _exit_tree() -> void:
	pass
```

- [ ] **Step 2: Create `addons/godot_native_rl/plugin.cfg`**

```ini
[plugin]

name="Godot Native RL"
description="GDExtension RL framework: native ncnn inference + godot_rl_agents-compatible training bridge, declarative reward authoring, and sensors."
author="minigraphx"
version="0.1.0"
script="plugin.gd"
```

- [ ] **Step 3: Run the full suite (nothing references these yet → still green)**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 4: Commit**

```bash
git add addons/godot_native_rl/plugin.cfg addons/godot_native_rl/plugin.gd
git commit -m "feat: addon skeleton (plugin.cfg + minimal EditorPlugin)"
```

---

## Task 2: `NcnnControllerCore` (shared RefCounted core, TDD)

**Files:**
- Create: `addons/godot_native_rl/controllers/ncnn_controller_core.gd`
- Test: `test/unit/test_controller_core.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_controller_core.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

class StubSource:
	var amount: float
	var last_ctx = null
	func _init(a: float) -> void:
		amount = a
	func evaluate(ctx) -> float:
		last_ctx = ctx
		return amount

class StubAdapter:
	var amount: float
	func _init(a: float) -> void:
		amount = a
	func drain() -> float:
		return amount

func _initialize() -> void:
	var h := Harness.new()
	var c = NcnnControllerCore.new()

	# step() / reset_after threshold (godot_rl: done once n_steps > reset_after)
	c.step(3)
	h.assert_eq(c.n_steps, 1, "step increments n_steps")
	h.assert_true(not c.done, "not done before threshold")
	c.step(3)
	c.step(3)
	h.assert_true(not c.done, "not done at n_steps == reset_after")
	c.step(3)
	h.assert_eq(c.n_steps, 4, "n_steps past threshold")
	h.assert_true(c.done, "done once n_steps > reset_after")
	h.assert_true(c.needs_reset, "needs_reset set past threshold")

	# reset()
	c.reset()
	h.assert_eq(c.n_steps, 0, "reset zeroes n_steps")
	h.assert_true(not c.needs_reset, "reset clears needs_reset")

	# reset_if_done()
	c.done = true
	c.n_steps = 5
	c.reset_if_done()
	h.assert_eq(c.n_steps, 0, "reset_if_done resets when done")

	# done helpers
	c.done = true
	h.assert_true(c.get_done(), "get_done reflects done")
	c.set_done_false()
	h.assert_true(not c.get_done(), "set_done_false clears done")

	# heuristic
	c.set_heuristic("noop")
	h.assert_eq(c.heuristic, "noop", "set_heuristic stores value")

	# zero_reward
	c.reward = 5.0
	c.zero_reward()
	h.assert_eq(c.reward, 0.0, "zero_reward clears reward")

	# accumulate(): reward_source + adapters, ctx passed through
	c.reward = 0.0
	var src := StubSource.new(1.5)
	c.reward_source = src
	var ctx := RefCounted.new()
	c.accumulate([StubAdapter.new(0.25), StubAdapter.new(0.1)], ctx)
	h.assert_true(absf(c.reward - 1.85) < 1e-6, "accumulate sums reward_source + adapters")
	h.assert_eq(src.last_ctx, ctx, "accumulate passes ctx to reward_source.evaluate")

	# accumulate() with null reward_source: adapters only
	c.reward = 0.0
	c.reward_source = null
	c.accumulate([StubAdapter.new(0.5)], ctx)
	h.assert_true(absf(c.reward - 0.5) < 1e-6, "accumulate works with null reward_source")

	# obs_space_from_obs() static
	var space := NcnnControllerCore.obs_space_from_obs({"obs": [0.0, 0.0, 0.0]})
	h.assert_eq(space, {"obs": {"size": [3], "space": "box"}}, "obs_space_from_obs shape")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_controller_core.gd"`
Expected: FAIL — preload of `res://addons/godot_native_rl/controllers/ncnn_controller_core.gd` fails (file does not exist).

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/controllers/ncnn_controller_core.gd`:

```gdscript
class_name NcnnControllerCore
extends RefCounted

# Node-agnostic episode + reward state machine shared by NcnnAIController2D/3D.
# Holds no Node references; reset_after is passed into step() by the wrapper so the
# wrapper stays the single source of truth for that exported value.

var done: bool = false
var reward: float = 0.0
var n_steps: int = 0
var needs_reset: bool = false
var heuristic: String = "human"
var reward_source = null

func step(reset_after: int) -> void:
	n_steps += 1
	if n_steps > reset_after:
		# Signal episode termination (godot_rl convention): the trainer reads `done`,
		# which gives proper episode boundaries and reward statistics.
		needs_reset = true
		done = true

func reset() -> void:
	n_steps = 0
	needs_reset = false

func reset_if_done() -> void:
	if done:
		reset()

func zero_reward() -> void:
	reward = 0.0

func set_done_false() -> void:
	done = false

func get_done() -> bool:
	return done

func set_heuristic(h: String) -> void:
	heuristic = h

func accumulate(adapters: Array, ctx) -> void:
	if reward_source != null:
		reward += reward_source.evaluate(ctx)
	for adapter in adapters:
		reward += adapter.drain()

static func obs_space_from_obs(obs: Dictionary) -> Dictionary:
	return {"obs": {"size": [obs["obs"].size()], "space": "box"}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_controller_core.gd"`
Expected: PASS — `Results: 14 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd test/unit/test_controller_core.gd
git commit -m "feat: NcnnControllerCore shared episode/reward state machine (TDD)"
```

---

## Task 3: Move `reward/` into the addon

**Files:**
- Move: `reward/**` → `addons/godot_native_rl/reward/**`
- Modify (repath `res://reward/` → `res://addons/godot_native_rl/reward/`): the moved reward
  files' internal refs, `ncnn_ai_controller_2d.gd`, `examples/chase_the_target/chase_agent.gd`,
  and 8 reward tests.

- [ ] **Step 1: Move the directory (preserving history)**

```bash
git mv reward addons/godot_native_rl/reward
```

- [ ] **Step 2: Rewrite every `res://reward/` reference to the addon path**

```bash
grep -rl 'res://reward/' --include='*.gd' . | grep -v '/godot-cpp/' | grep -v '/thirdparty/' \
  | xargs perl -pi -e 's|res://reward/|res://addons/godot_native_rl/reward/|g'
```

This covers: `addons/godot_native_rl/reward/reward_builder.gd` (5 preloads),
`addons/godot_native_rl/reward/terms/{step_penalty,event_bonus,alive_bonus,progress_shaping}_term.gd`
(`extends "res://reward/terms/reward_term.gd"`), `ncnn_ai_controller_2d.gd` (reward_adapter preload),
`examples/chase_the_target/chase_agent.gd` (reward_builder preload), and
`test/unit/{test_reward_progress_term,test_reward_builder,test_reward_simple_terms,test_reward_evaluator,test_reward_event_bonus_term,test_reward_adapter,test_controller_reward_accumulation,test_chase_reward_parity}.gd`.

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -rn '"res://reward/' --include='*.gd' . | grep -v '/godot-cpp/' | grep -v '/thirdparty/'`
Expected: **no output** (every reference now points at the addon path).

- [ ] **Step 4: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.` (reward unit tests + chase reward parity + trained-chase all green from the new location).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move reward/ into addons/godot_native_rl/ (repath refs)"
```

---

## Task 4: Move `sensors/` into the addon

**Files:**
- Move: `sensors/**` → `addons/godot_native_rl/sensors/**`
- Modify (repath `res://sensors/`): the moved sensors' internal `RaycastMath` preloads + 3 raycast tests.

- [ ] **Step 1: Move the directory**

```bash
git mv sensors addons/godot_native_rl/sensors
```

- [ ] **Step 2: Rewrite every `res://sensors/` reference**

```bash
grep -rl 'res://sensors/' --include='*.gd' . | grep -v '/godot-cpp/' | grep -v '/thirdparty/' \
  | xargs perl -pi -e 's|res://sensors/|res://addons/godot_native_rl/sensors/|g'
```

Covers: `addons/godot_native_rl/sensors/raycast_sensor_2d.gd`, `…/raycast_sensor_3d.gd` (RaycastMath
preload), and `test/unit/{test_raycast_math,test_raycast_sensor_2d,test_raycast_sensor_3d}.gd`.

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -rn '"res://sensors/' --include='*.gd' . | grep -v '/godot-cpp/' | grep -v '/thirdparty/'`
Expected: **no output**.

- [ ] **Step 4: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move sensors/ into addons/godot_native_rl/ (repath refs)"
```

---

## Task 5: Move `sync.gd` into the addon

**Files:**
- Move: `sync.gd` → `addons/godot_native_rl/sync.gd`
- Modify (repath `res://sync.gd`): 5 `.tscn` scenes + 2 sync tests.

- [ ] **Step 1: Move the file**

```bash
git mv sync.gd addons/godot_native_rl/sync.gd
```

- [ ] **Step 2: Rewrite every `res://sync.gd` reference (in `.gd` and `.tscn`)**

```bash
grep -rl 'res://sync.gd' --include='*.gd' --include='*.tscn' . | grep -v '/godot-cpp/' | grep -v '/thirdparty/' \
  | xargs perl -pi -e 's|res://sync\.gd|res://addons/godot_native_rl/sync.gd|g'
```

Covers `.tscn`: `examples/chase_the_target/chase_the_target.tscn`, `…_train.tscn`,
`test/integration/{protocol_test_scene,inference_smoke_scene,trained_chase_scene}.tscn`; and `.gd`:
`test/unit/{test_sync_inference,test_sync_messages}.gd`. (These `ext_resource` lines use `path=` only,
no `uid=`, and `sync.gd` has no `.uid` companion — updating the path string is sufficient.)

- [ ] **Step 3: Verify no stale references remain**

Run: `grep -rn 'res://sync\.gd' --include='*.gd' --include='*.tscn' . | grep -v '/godot-cpp/' | grep -v '/thirdparty/'`
Expected: **no output**.

- [ ] **Step 4: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.` (protocol test, inference smoke, trained-chase, sync unit tests all load sync from the new path).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move sync.gd into addons/godot_native_rl/ (repath scene + test refs)"
```

---

## Task 6: Move the controller into the addon (no refactor yet)

**Files:**
- Move: `ncnn_ai_controller_2d.gd` → `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`

- [ ] **Step 1: Move the file**

```bash
git mv ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd
```

The controller is referenced only via `class_name NcnnAIController2D` (by `chase_agent.gd`,
`test/unit/stub_agent.gd`, `test/integration/protocol_stub_agent.gd`), so moving the file changes
no references. Its own `reward_adapter` preload is already the absolute addon path (rewritten in Task 3).

- [ ] **Step 2: Verify the controller has no stale relative refs**

Run: `grep -n 'res://reward\|res://sensors\|res://sync' addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`
Expected: only `res://addons/godot_native_rl/reward/reward_adapter.gd` (the repathed preload).

- [ ] **Step 3: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.` (controller resolves via class_name from its new location).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: move NcnnAIController2D into addons/godot_native_rl/controllers/"
```

---

## Task 7: Refactor `NcnnAIController2D` onto the shared core

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` (full rewrite)

This is a pure refactor: the public API (class_name, enum, exports, methods, state member names)
is unchanged — state now lives in `_core` behind forwarding properties, and the bookkeeping methods
delegate. The existing `test_controller*.gd` + `chase_*` tests passing again is the backward-compat proof.

- [ ] **Step 1: Rewrite the controller to delegate to `NcnnControllerCore`**

Replace the entire contents of `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` with:

```gdscript
class_name NcnnAIController2D
extends Node2D

const RewardAdapterScript = preload("res://addons/godot_native_rl/reward/reward_adapter.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING, NCNN_INFERENCE }
@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC  # read/written by NcnnSync
@export var reset_after := 1000
@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"

var _core := NcnnControllerCore.new()
var _ncnn_runner = null
var _reward_adapters: Array = []

# --- Forwarding properties: preserve the historical public state API (subclasses + NcnnSync) ---
var done: bool:
	get:
		return _core.done
	set(value):
		_core.done = value
var reward: float:
	get:
		return _core.reward
	set(value):
		_core.reward = value
var n_steps: int:
	get:
		return _core.n_steps
	set(value):
		_core.n_steps = value
var needs_reset: bool:
	get:
		return _core.needs_reset
	set(value):
		_core.needs_reset = value
var heuristic: String:
	get:
		return _core.heuristic
	set(value):
		_core.heuristic = value
var reward_source:
	get:
		return _core.reward_source
	set(value):
		_core.reward_source = value

func _ready() -> void:
	add_to_group("AGENT")
	collect_reward_adapters()
	if control_mode == ControlModes.NCNN_INFERENCE:
		_setup_ncnn_runner()

func _setup_ncnn_runner() -> void:
	if model_param_path.is_empty() or model_bin_path.is_empty():
		push_error("NcnnAIController2D: NCNN_INFERENCE mode requires model_param_path and model_bin_path.")
		return
	_ncnn_runner = NcnnRunner.new()
	_ncnn_runner.input_blob_name = input_blob_name
	_ncnn_runner.output_blob_name = output_blob_name
	add_child(_ncnn_runner)
	var absolute_param := ProjectSettings.globalize_path(model_param_path)
	var absolute_bin := ProjectSettings.globalize_path(model_bin_path)
	if not _ncnn_runner.load_model(absolute_param, absolute_bin):
		push_error("NcnnAIController2D: failed to load ncnn model.")
		_ncnn_runner.queue_free()
		_ncnn_runner = null

func set_ncnn_runner_for_test(runner) -> void:
	_ncnn_runner = runner

func infer_and_act() -> void:
	if _ncnn_runner == null or not _ncnn_runner.is_model_loaded():
		return
	var obs_dict := get_obs()
	assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
	var obs_flat := PackedFloat32Array(obs_dict["obs"])
	var action_index: int = _ncnn_runner.run_discrete_action(obs_flat)
	if action_index < 0:
		push_error("NcnnAIController2D: run_discrete_action returned error sentinel; skipping action.")
		return
	# Single discrete action branch: use the first (and only) action key.
	var action_key: String = get_action_space().keys()[0]
	set_action({action_key: action_index})

# --- Abstract: implemented by the concrete agent ---
func get_obs() -> Dictionary:
	assert(false, "get_obs must be implemented by the agent extending NcnnAIController2D")
	return {"obs": []}

func get_reward() -> float:
	assert(false, "get_reward must be implemented by the agent extending NcnnAIController2D")
	return 0.0

func get_action_space() -> Dictionary:
	assert(false, "get_action_space must be implemented by the agent extending NcnnAIController2D")
	return {}

func set_action(_action) -> void:
	assert(false, "set_action must be implemented by the agent extending NcnnAIController2D")

# --- Concrete contract methods used by NcnnSync (delegate to the shared core) ---
func get_obs_space() -> Dictionary:
	return NcnnControllerCore.obs_space_from_obs(get_obs())

func reset() -> void:
	_core.reset()

func reset_if_done() -> void:
	_core.reset_if_done()

func set_heuristic(h: String) -> void:
	_core.set_heuristic(h)

func get_done() -> bool:
	return _core.get_done()

func set_done_false() -> void:
	_core.set_done_false()

func zero_reward() -> void:
	_core.zero_reward()

func collect_reward_adapters() -> void:
	_reward_adapters.clear()
	for child in get_children():
		if child is RewardAdapterScript:
			_reward_adapters.append(child)

# Sum the declarative reward for this step into the accumulator that NcnnSync drains.
# Call from the concrete agent's _physics_process AFTER world state is updated.
func accumulate_reward() -> void:
	_core.accumulate(_reward_adapters, self)

func _physics_process(_delta) -> void:
	_core.step(reset_after)
```

- [ ] **Step 2: Run the controller-focused tests**

Run each:
```
godot --headless --path . --script "res://test/unit/test_controller.gd"
godot --headless --path . --script "res://test/unit/test_controller_inference.gd"
godot --headless --path . --script "res://test/unit/test_controller_reward_accumulation.gd"
```
Expected: each ends `Results: N passed, 0 failed` (no failures). These exercise the forwarding
properties, `get_obs_space`, `infer_and_act`, and `accumulate_reward` against the new core.

- [ ] **Step 3: Run the full suite (backward-compat proof, incl. trained-chase + golden)**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

If a controller test fails, STOP and use the systematic-debugging skill — do not weaken the test.

- [ ] **Step 4: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd
git commit -m "refactor: NcnnAIController2D delegates to NcnnControllerCore (forwarding properties)"
```

---

## Task 8: `NcnnAIController3D` thin wrapper (TDD)

**Files:**
- Create: `test/unit/stub_agent_3d.gd` (helper, not `test_*`)
- Create: `test/unit/test_controller_3d.gd`
- Create: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`

- [ ] **Step 1: Write the helper stub agent**

Create `test/unit/stub_agent_3d.gd`:

```gdscript
extends NcnnAIController3D

func get_obs() -> Dictionary:
	return {"obs": [0.0, 1.0, 2.0]}

func get_reward() -> float:
	return reward

func get_action_space() -> Dictionary:
	return {"move": {"size": 3, "action_type": "discrete"}}

func set_action(_action) -> void:
	pass
```

- [ ] **Step 2: Write the failing test**

Create `test/unit/test_controller_3d.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const StubAgent3D = preload("res://test/unit/stub_agent_3d.gd")

class StubSource:
	var amount: float
	func _init(a: float) -> void:
		amount = a
	func evaluate(_ctx) -> float:
		return amount

func _initialize() -> void:
	var h := Harness.new()
	var a = StubAgent3D.new()

	# obs_space derived from the stub's get_obs via the shared core helper
	h.assert_eq(a.get_obs_space(), {"obs": {"size": [3], "space": "box"}}, "3D get_obs_space shape")

	# forwarding properties read/write through to the core
	a.done = true
	h.assert_true(a.get_done(), "3D done forwards to core (get_done)")
	a.set_done_false()
	h.assert_true(not a.done, "3D set_done_false clears forwarded done")

	a.reward = 2.0
	h.assert_true(absf(a.reward - 2.0) < 1e-6, "3D reward forwards")
	a.zero_reward()
	h.assert_eq(a.reward, 0.0, "3D zero_reward via core")

	a.needs_reset = true
	a.reset()
	h.assert_true(not a.needs_reset, "3D reset clears needs_reset via core")

	# accumulate_reward delegates to core (no adapters collected without _ready)
	a.reward = 0.0
	a.reward_source = StubSource.new(0.75)
	a.accumulate_reward()
	h.assert_true(absf(a.reward - 0.75) < 1e-6, "3D accumulate_reward sums reward_source via core")

	a.free()
	h.finish(self)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_controller_3d.gd"`
Expected: FAIL — `stub_agent_3d.gd` can't parse because `NcnnAIController3D` is unknown (class does not exist yet).

- [ ] **Step 4: Write the 3D wrapper**

Create `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` (mirrors the 2D wrapper; only
the node type and the diagnostic message strings differ):

```gdscript
class_name NcnnAIController3D
extends Node3D

const RewardAdapterScript = preload("res://addons/godot_native_rl/reward/reward_adapter.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING, NCNN_INFERENCE }
@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC  # read/written by NcnnSync
@export var reset_after := 1000
@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"

var _core := NcnnControllerCore.new()
var _ncnn_runner = null
var _reward_adapters: Array = []

# --- Forwarding properties: preserve the historical public state API (subclasses + NcnnSync) ---
var done: bool:
	get:
		return _core.done
	set(value):
		_core.done = value
var reward: float:
	get:
		return _core.reward
	set(value):
		_core.reward = value
var n_steps: int:
	get:
		return _core.n_steps
	set(value):
		_core.n_steps = value
var needs_reset: bool:
	get:
		return _core.needs_reset
	set(value):
		_core.needs_reset = value
var heuristic: String:
	get:
		return _core.heuristic
	set(value):
		_core.heuristic = value
var reward_source:
	get:
		return _core.reward_source
	set(value):
		_core.reward_source = value

func _ready() -> void:
	add_to_group("AGENT")
	collect_reward_adapters()
	if control_mode == ControlModes.NCNN_INFERENCE:
		_setup_ncnn_runner()

func _setup_ncnn_runner() -> void:
	if model_param_path.is_empty() or model_bin_path.is_empty():
		push_error("NcnnAIController3D: NCNN_INFERENCE mode requires model_param_path and model_bin_path.")
		return
	_ncnn_runner = NcnnRunner.new()
	_ncnn_runner.input_blob_name = input_blob_name
	_ncnn_runner.output_blob_name = output_blob_name
	add_child(_ncnn_runner)
	var absolute_param := ProjectSettings.globalize_path(model_param_path)
	var absolute_bin := ProjectSettings.globalize_path(model_bin_path)
	if not _ncnn_runner.load_model(absolute_param, absolute_bin):
		push_error("NcnnAIController3D: failed to load ncnn model.")
		_ncnn_runner.queue_free()
		_ncnn_runner = null

func set_ncnn_runner_for_test(runner) -> void:
	_ncnn_runner = runner

func infer_and_act() -> void:
	if _ncnn_runner == null or not _ncnn_runner.is_model_loaded():
		return
	var obs_dict := get_obs()
	assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
	var obs_flat := PackedFloat32Array(obs_dict["obs"])
	var action_index: int = _ncnn_runner.run_discrete_action(obs_flat)
	if action_index < 0:
		push_error("NcnnAIController3D: run_discrete_action returned error sentinel; skipping action.")
		return
	# Single discrete action branch: use the first (and only) action key.
	var action_key: String = get_action_space().keys()[0]
	set_action({action_key: action_index})

# --- Abstract: implemented by the concrete agent ---
func get_obs() -> Dictionary:
	assert(false, "get_obs must be implemented by the agent extending NcnnAIController3D")
	return {"obs": []}

func get_reward() -> float:
	assert(false, "get_reward must be implemented by the agent extending NcnnAIController3D")
	return 0.0

func get_action_space() -> Dictionary:
	assert(false, "get_action_space must be implemented by the agent extending NcnnAIController3D")
	return {}

func set_action(_action) -> void:
	assert(false, "set_action must be implemented by the agent extending NcnnAIController3D")

# --- Concrete contract methods used by NcnnSync (delegate to the shared core) ---
func get_obs_space() -> Dictionary:
	return NcnnControllerCore.obs_space_from_obs(get_obs())

func reset() -> void:
	_core.reset()

func reset_if_done() -> void:
	_core.reset_if_done()

func set_heuristic(h: String) -> void:
	_core.set_heuristic(h)

func get_done() -> bool:
	return _core.get_done()

func set_done_false() -> void:
	_core.set_done_false()

func zero_reward() -> void:
	_core.zero_reward()

func collect_reward_adapters() -> void:
	_reward_adapters.clear()
	for child in get_children():
		if child is RewardAdapterScript:
			_reward_adapters.append(child)

func accumulate_reward() -> void:
	_core.accumulate(_reward_adapters, self)

func _physics_process(_delta) -> void:
	_core.step(reset_after)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_controller_3d.gd"`
Expected: PASS — `Results: 7 passed, 0 failed`.

- [ ] **Step 6: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 7: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd test/unit/test_controller_3d.gd test/unit/stub_agent_3d.gd
git commit -m "feat: NcnnAIController3D thin Node3D wrapper over the shared core (TDD)"
```

---

## Task 9: README migration note + new backlog item

**Files:**
- Modify: `README.md`
- Modify: `docs/BACKLOG.md`

- [ ] **Step 1: Add a migration note to the README**

In `README.md`, just under the top intro/decision-guide blockquote (before `## What This Repository
Provides`), insert:

```markdown
> **Library moved (2026-05-30):** the reusable scripts now live under
> `addons/godot_native_rl/` (controllers, `sync.gd`, `reward/`, `sensors/`). `class_name`-based
> usage is unchanged — `extends NcnnAIController2D` / `NcnnAIController3D`, `NcnnSync`,
> `RewardBuilder`, `RaycastSensor2D/3D`, etc. all still resolve. If you `preload` old paths like
> `res://sync.gd` or `res://reward/…`, update them to `res://addons/godot_native_rl/…`. The
> compiled GDExtension (`ncnn_runner.gdextension`, `bin/`) still lives at the project root.
```

- [ ] **Step 2: Add the "Asset Library release" backlog item**

In `docs/BACKLOG.md`, append to the "Deploy-side inference gaps" section's numbered list (after item 24):

```markdown
25. ⬜ **Asset Library release (extension packaging)** — move `ncnn_runner.gdextension` + a `bin/`
    of prebuilt per-platform binaries into `addons/godot_native_rl/`, repoint the manifest's library
    paths + the `SConstruct` output target, build macOS/Windows/Linux (+ web/mobile) binaries, fill
    `plugin.cfg` metadata, and submit. *(surfaced by item 5; the addon layout is already in place)*
```

- [ ] **Step 3: Run the full suite (docs-only change → still green)**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 4: Commit**

```bash
git add README.md docs/BACKLOG.md
git commit -m "docs: addon migration note + Asset Library release backlog item"
```

---

## Self-review notes (author)

- **Spec coverage:** addon layout + plugin.cfg/plugin.gd (Task 1), shared core (Task 2),
  reward/sensors/sync/controller moves with repathing (Tasks 3–6), 2D refactor onto core via
  forwarding properties (Task 7), thin 3D wrapper + tests (Task 8), README migration note + Asset
  Library follow-up item (Task 9). Backward-compat is proven by the unchanged `test_controller*`,
  `chase_*`, trained-chase, and golden tests passing after each move/refactor.
- **Refinement vs spec:** the core does **not** store `reset_after`; the wrapper passes it into
  `step(reset_after)` each `_physics_process`. This keeps the exported value single-sourced on the
  wrapper and preserves the original "re-read each step" behavior — strictly better than caching it
  in `_ready`. All other core members match the spec.
- **Type/name consistency:** `NcnnControllerCore` API (`step`, `reset`, `reset_if_done`,
  `zero_reward`, `set_done_false`, `get_done`, `set_heuristic`, `accumulate`,
  `obs_space_from_obs`, state vars) is identical across Task 2, the wrappers (Tasks 7–8), and the
  tests. `RewardAdapterScript` / `NcnnControllerCore` preload consts match the moved addon paths.
- **Placeholders:** none — every code/command step is concrete.
```
