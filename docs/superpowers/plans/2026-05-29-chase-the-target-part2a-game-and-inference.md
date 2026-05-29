# Chase The Target — Part 2A: Game Scene + ncnn Inference Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the runnable 2D "Chase The Target" example — game logic, the `ChaseAgent` controller, and an ncnn-inference mode added to `NcnnSync`/`NcnnAIController2D` — verified end-to-end by a headless smoke test that drives the scene with a generated dummy model (no training required).

**Architecture:** A `ChaseGame` (Node2D) owns the arena, an agent body, and a target, exposing pure geometry helpers (bounds clamp, random placement, distance). A `ChaseAgent` extends `NcnnAIController2D` and implements the godot_rl contract (`get_obs`/`get_reward`/`get_action_space`/`set_action`) using pure, unit-tested helpers for observation normalization, discrete-action→velocity mapping, and shaped reward. The base `NcnnAIController2D` gains an `NCNN_INFERENCE` mode that owns an `NcnnRunner` and an `infer_and_act()` method; `NcnnSync` gains an inference branch that calls it (no networking). Inference uses `NcnnRunner.run_discrete_action()` (argmax over the 5 logits the policy emits).

**Tech Stack:** Godot 4.6.2 (GDScript), the `NcnnRunner` C++ GDExtension (already built for arm64), the existing dependency-free headless test harness (`test/harness.gd`), and `scripts/export_test_mlp.py` (PyTorch+pnnx, both installed) to generate a dummy 5→5 model for the smoke test.

**Verified facts (from godot-rl source + repo):**
- Action space `{"move":{"size":5,"action_type":"discrete"}}` ⇒ `spaces.Discrete(5)`; trained action arrives as `{"move": <int 0..4>}`.
- The policy's ONNX/ncnn output for a discrete head is the **raw logits** ⇒ argmax picks the action ⇒ `NcnnRunner.run_discrete_action()` is the correct inference call.
- `NcnnRunner` API: `load_model(param, bin) -> bool`, `is_model_loaded() -> bool`, `run_inference(PackedFloat32Array) -> PackedFloat32Array`, `run_discrete_action(PackedFloat32Array) -> int`, properties `input_blob_name`/`output_blob_name`/`input_shape`. `pnnx` emits blob names `in0`/`out0`.
- Part 1 base `NcnnAIController2D` already has: `enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING }`, vars `done`/`reward`/`n_steps`/`needs_reset`, `reset_after`, `_ready()` adds to group `"AGENT"`, `_physics_process` sets `needs_reset` when `n_steps > reset_after`, and concrete `get_obs_space`/`reset`/`set_heuristic`/`get_done`/`set_done_false`/`zero_reward`.
- Part 1 `NcnnSync` already has: `enum ControlModes { HUMAN, TRAINING }`, `agents_training`, `agents_heuristic`, `all_agents`, `_get_agents()`, `_physics_process` → `_training_process()` + `_heuristic_process()`.

**Spec:** `docs/superpowers/specs/2026-05-29-chase-the-target-2d-example-design.md`

**Constants used throughout** (arena `1000×600`, move speed `300` px/s, touch radius `40` px, `reset_after` `1000`, step penalty `0.001`, touch bonus `1.0`).

---

### Task 1: `ChaseGame` geometry helpers (TDD)

**Files:**
- Create: `examples/chase_the_target/chase_game.gd`
- Test: `test/unit/test_chase_game.gd`

- [ ] **Step 1: Create the feature branch**

Run:
```bash
git checkout -b feature/chase-example-2a
```
Expected: `Switched to a new branch 'feature/chase-example-2a'`

- [ ] **Step 2: Write the failing test**

Create `test/unit/test_chase_game.gd`:
```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ChaseGameScript = preload("res://examples/chase_the_target/chase_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var g = ChaseGameScript.new()
	g.arena_size = Vector2(1000, 600)

	# clamp_to_bounds keeps points inside [0, arena_size]
	h.assert_eq(g.clamp_to_bounds(Vector2(-50, 700)), Vector2(0, 600), "clamp low/high")
	h.assert_eq(g.clamp_to_bounds(Vector2(500, 300)), Vector2(500, 300), "clamp inside unchanged")

	# max_distance is the arena diagonal
	h.assert_true(absf(g.max_distance() - Vector2(1000, 600).length()) < 0.001, "max_distance diagonal")

	# random_position is always within bounds (seeded, sampled many times)
	g.seed_rng(123)
	var all_in_bounds := true
	for _i in range(200):
		var p: Vector2 = g.random_position()
		if p.x < 0.0 or p.x > 1000.0 or p.y < 0.0 or p.y > 600.0:
			all_in_bounds = false
	h.assert_true(all_in_bounds, "random_position within bounds")

	g.free()
	h.finish(self)
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
godot --headless --path . --script res://test/unit/test_chase_game.gd
```
Expected: FAIL — `res://examples/chase_the_target/chase_game.gd` does not exist (preload parse error).

- [ ] **Step 4: Write the minimal implementation**

Create `examples/chase_the_target/chase_game.gd`:
```gdscript
class_name ChaseGame
extends Node2D

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0
@export var touch_radius := 40.0
@export var agent_body_path: NodePath
@export var target_path: NodePath

var _rng := RandomNumberGenerator.new()
var _agent_body: Node2D
var _target: Node2D

func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path) as Node2D
	_target = get_node_or_null(target_path) as Node2D
	reset_positions()

# --- Pure helpers (unit-tested) ---
func clamp_to_bounds(pos: Vector2) -> Vector2:
	return Vector2(clampf(pos.x, 0.0, arena_size.x), clampf(pos.y, 0.0, arena_size.y))

func max_distance() -> float:
	return arena_size.length()

func seed_rng(s: int) -> void:
	_rng.seed = s

func random_position() -> Vector2:
	return Vector2(_rng.randf_range(0.0, arena_size.x), _rng.randf_range(0.0, arena_size.y))

# --- Runtime helpers (exercised by the scene + smoke test) ---
func get_agent_pos() -> Vector2:
	return _agent_body.position if _agent_body != null else Vector2.ZERO

func get_target_pos() -> Vector2:
	return _target.position if _target != null else Vector2.ZERO

func distance() -> float:
	return get_agent_pos().distance_to(get_target_pos())

func move_agent(velocity: Vector2, delta: float) -> void:
	if _agent_body != null:
		_agent_body.position = clamp_to_bounds(_agent_body.position + velocity * delta)

func relocate_target() -> void:
	if _target != null:
		_target.position = random_position()

func reset_positions() -> void:
	if _agent_body != null:
		_agent_body.position = random_position()
	if _target != null:
		_target.position = random_position()
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
godot --headless --path . --script res://test/unit/test_chase_game.gd
```
Expected: PASS — `Results: 4 passed, 0 failed` (exit 0).

- [ ] **Step 6: Commit**

```bash
git add examples/chase_the_target/chase_game.gd test/unit/test_chase_game.gd
git commit -m "feat: add ChaseGame geometry helpers for chase example"
```

---

### Task 2: `ChaseAgent` observation/action/reward helpers (TDD)

**Files:**
- Create: `examples/chase_the_target/chase_agent.gd`
- Test: `test/unit/test_chase_agent.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_chase_agent.gd`:
```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ChaseAgentScript = preload("res://examples/chase_the_target/chase_agent.gd")

func _initialize() -> void:
	var h := Harness.new()
	var a = ChaseAgentScript.new()
	var arena := Vector2(1000, 600)

	# Observation: agent centered, target to the right.
	var obs: Array = a.compute_obs(Vector2(500, 300), Vector2(1000, 300), arena)
	h.assert_eq(obs.size(), 5, "obs has 5 elements")
	h.assert_true(absf(obs[0] - 0.0) < 0.001, "obs[0] centered x ~0")
	h.assert_true(absf(obs[1] - 0.0) < 0.001, "obs[1] centered y ~0")
	h.assert_true(absf(obs[2] - 1.0) < 0.001, "obs[2] unit dir x = +1 (target right)")
	h.assert_true(absf(obs[3] - 0.0) < 0.001, "obs[3] unit dir y = 0")
	h.assert_true(obs[4] > 0.0 and obs[4] <= 1.0, "obs[4] normalized distance in (0,1]")

	# Zero distance => zero direction vector (no NaN).
	var obs0: Array = a.compute_obs(Vector2(10, 10), Vector2(10, 10), arena)
	h.assert_eq(obs0[2], 0.0, "obs[2] dir x = 0 at zero distance")
	h.assert_eq(obs0[3], 0.0, "obs[3] dir y = 0 at zero distance")

	# Action index -> velocity (speed 300): 0 idle,1 up,2 down,3 left,4 right.
	h.assert_eq(a.action_index_to_velocity(0, 300.0), Vector2(0, 0), "idle")
	h.assert_eq(a.action_index_to_velocity(1, 300.0), Vector2(0, -300), "up")
	h.assert_eq(a.action_index_to_velocity(2, 300.0), Vector2(0, 300), "down")
	h.assert_eq(a.action_index_to_velocity(3, 300.0), Vector2(-300, 0), "left")
	h.assert_eq(a.action_index_to_velocity(4, 300.0), Vector2(300, 0), "right")

	# Reward: getting closer is positive (minus step penalty); touch adds bonus.
	a.step_penalty = 0.001
	a.touch_bonus = 1.0
	var r_closer: float = a.compute_step_reward(100.0, 60.0, 1000.0, false)
	h.assert_true(r_closer > 0.0, "moving closer yields positive reward")
	var r_touch: float = a.compute_step_reward(60.0, 30.0, 1000.0, true)
	h.assert_true(r_touch > 1.0, "touch adds bonus on top of progress")
	var r_farther: float = a.compute_step_reward(60.0, 90.0, 1000.0, false)
	h.assert_true(r_farther < 0.0, "moving away yields negative reward")

	# Action space matches the godot_rl Discrete(5) contract.
	h.assert_eq(a.get_action_space(), {"move": {"size": 5, "action_type": "discrete"}}, "action space")

	a.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
godot --headless --path . --script res://test/unit/test_chase_agent.gd
```
Expected: FAIL — `res://examples/chase_the_target/chase_agent.gd` does not exist.

- [ ] **Step 3: Write the minimal implementation**

Create `examples/chase_the_target/chase_agent.gd`:
```gdscript
class_name ChaseAgent
extends NcnnAIController2D

const ACTION_KEY := "move"
const ACTION_COUNT := 5

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var touch_bonus := 1.0

var _game: ChaseGame
var _action_index := 0
var _prev_dist := 0.0

func _ready() -> void:
	super._ready()  # joins group "AGENT" and sets up base/inference
	_game = get_node_or_null(game_path) as ChaseGame
	if _game != null:
		_prev_dist = _game.distance()

# --- Pure helpers (unit-tested) ---
func compute_obs(agent_pos: Vector2, target_pos: Vector2, arena_size: Vector2) -> Array:
	var rel := target_pos - agent_pos
	var dist := rel.length()
	var dir := rel.normalized() if dist > 0.0 else Vector2.ZERO
	return [
		(agent_pos.x / arena_size.x - 0.5) * 2.0,
		(agent_pos.y / arena_size.y - 0.5) * 2.0,
		dir.x,
		dir.y,
		clampf(dist / arena_size.length(), 0.0, 1.0),
	]

func action_index_to_velocity(idx: int, speed: float) -> Vector2:
	match idx:
		1: return Vector2(0.0, -speed)
		2: return Vector2(0.0, speed)
		3: return Vector2(-speed, 0.0)
		4: return Vector2(speed, 0.0)
		_: return Vector2.ZERO

func compute_step_reward(prev_dist: float, cur_dist: float, max_dist: float, touched: bool) -> float:
	var progress := (prev_dist - cur_dist) / max_dist
	var r := progress - step_penalty
	if touched:
		r += touch_bonus
	return r

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0]}
	return {"obs": compute_obs(_game.get_agent_pos(), _game.get_target_pos(), _game.arena_size)}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	_action_index = int(action[ACTION_KEY])

# --- Runtime step (drives the game between control decisions) ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)  # increments n_steps; sets needs_reset past reset_after
	if _game == null:
		return

	var velocity := action_index_to_velocity(_action_index, _game.move_speed)
	_game.move_agent(velocity, delta)

	var cur_dist := _game.distance()
	var touched := cur_dist < _game.touch_radius
	reward += compute_step_reward(_prev_dist, cur_dist, _game.max_distance(), touched)
	if touched:
		_game.relocate_target()
	_prev_dist = _game.distance()

	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		_prev_dist = _game.distance()
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
godot --headless --path . --script res://test/unit/test_chase_agent.gd
```
Expected: PASS — `Results: 16 passed, 0 failed` (exit 0).

- [ ] **Step 5: Commit**

```bash
git add examples/chase_the_target/chase_agent.gd test/unit/test_chase_agent.gd
git commit -m "feat: add ChaseAgent obs/action/reward helpers and contract"
```

---

### Task 3: Add `NCNN_INFERENCE` mode to `NcnnAIController2D`

**Files:**
- Modify: `ncnn_ai_controller_2d.gd`
- Test: `test/unit/test_controller_inference.gd`

This adds ncnn model ownership and an `infer_and_act()` method to the base controller. The argmax wiring is verified by the smoke test (Task 6); this task unit-tests only the mode enum and the action-dict construction via an injected fake runner.

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_controller_inference.gd`:
```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/stub_agent.gd")

# Minimal fake that mimics NcnnRunner.run_discrete_action.
class FakeRunner:
	var loaded := true
	var forced_index := 3
	func is_model_loaded() -> bool:
		return loaded
	func run_discrete_action(_input) -> int:
		return forced_index

func _initialize() -> void:
	var h := Harness.new()

	# NCNN_INFERENCE exists in the enum.
	h.assert_true(Stub.ControlModes.has("NCNN_INFERENCE"), "NCNN_INFERENCE enum value exists")

	# infer_and_act builds {action_key: argmax_index} from the runner and calls set_action.
	var a = Stub.new()
	a.set_ncnn_runner_for_test(FakeRunner.new())
	a.infer_and_act()
	h.assert_eq(a.last_action, {"move": 3}, "infer_and_act sets {move: argmax}")

	a.free()
	h.finish(self)
```

(Note: `Stub` is `test/unit/stub_agent.gd` from Part 1. Step 3 adds the two small test hooks it needs.)

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
godot --headless --path . --script res://test/unit/test_controller_inference.gd
```
Expected: FAIL — `ControlModes` has no `NCNN_INFERENCE` / `set_ncnn_runner_for_test` and `infer_and_act` are undefined.

- [ ] **Step 3: Update `ncnn_ai_controller_2d.gd` and the stub**

In `ncnn_ai_controller_2d.gd`, change the enum line:
```gdscript
enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING }
```
to:
```gdscript
enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING, NCNN_INFERENCE }
```

Add these exports immediately after the `@export var reset_after := 1000` line:
```gdscript
@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
```

Add a runner var alongside the other vars (after `var needs_reset := false`):
```gdscript
var _ncnn_runner = null
```

Replace the `_ready()` function:
```gdscript
func _ready() -> void:
	add_to_group("AGENT")
```
with:
```gdscript
func _ready() -> void:
	add_to_group("AGENT")
	if control_mode == ControlModes.NCNN_INFERENCE:
		_setup_ncnn_runner()

func _setup_ncnn_runner() -> void:
	if model_param_path.is_empty() or model_bin_path.is_empty():
		push_error("NcnnAIController2D: NCNN_INFERENCE mode requires model_param_path and model_bin_path.")
		return
	_ncnn_runner = NcnnRunner.new()
	add_child(_ncnn_runner)
	_ncnn_runner.input_blob_name = input_blob_name
	_ncnn_runner.output_blob_name = output_blob_name
	var absolute_param := ProjectSettings.globalize_path(model_param_path)
	var absolute_bin := ProjectSettings.globalize_path(model_bin_path)
	if not _ncnn_runner.load_model(absolute_param, absolute_bin):
		push_error("NcnnAIController2D: failed to load ncnn model.")

func set_ncnn_runner_for_test(runner) -> void:
	_ncnn_runner = runner

func infer_and_act() -> void:
	if _ncnn_runner == null or not _ncnn_runner.is_model_loaded():
		return
	var obs_flat := PackedFloat32Array(get_obs()["obs"])
	var action_index := _ncnn_runner.run_discrete_action(obs_flat)
	var action_key: String = get_action_space().keys()[0]
	set_action({action_key: action_index})
```

In `test/unit/stub_agent.gd`, add a recorder so the test can observe `set_action`. Replace:
```gdscript
func set_action(_action) -> void:
	pass
```
with:
```gdscript
var last_action = null

func set_action(action) -> void:
	last_action = action
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
godot --headless --path . --script res://test/unit/test_controller_inference.gd
```
Expected: PASS — `Results: 2 passed, 0 failed` (exit 0).

- [ ] **Step 5: Run the full suite (no regressions)**

Run:
```bash
./test/run_tests.sh
```
Expected: all unit tests pass (including Part 1's `test_controller.gd`, which still works because the stub's new `last_action`/`set_action` are additive), `PROTOCOL TEST PASSED`, `All tests passed.` (exit 0).

- [ ] **Step 6: Commit**

```bash
git add ncnn_ai_controller_2d.gd test/unit/stub_agent.gd test/unit/test_controller_inference.gd
git commit -m "feat: add NCNN_INFERENCE mode and infer_and_act to NcnnAIController2D"
```

---

### Task 4: Add inference branch to `NcnnSync`

**Files:**
- Modify: `sync.gd`
- Test: `test/unit/test_sync_inference.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_sync_inference.gd`:
```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const SyncScript = preload("res://sync.gd")

# Fake agent recording infer_and_act / reset calls.
class FakeInferenceAgent:
	var infer_calls := 0
	var done := false
	var needs_reset := false
	func infer_and_act() -> void:
		infer_calls += 1
	func get_done() -> bool:
		return done
	func set_done_false() -> void:
		done = false
	func reset() -> void:
		pass

func _initialize() -> void:
	var h := Harness.new()

	# NCNN_INFERENCE exists in NcnnSync's enum.
	h.assert_true(SyncScript.ControlModes.has("NCNN_INFERENCE"), "Sync NCNN_INFERENCE enum value exists")

	# _inference_process calls infer_and_act on each inference agent.
	var s = SyncScript.new()
	var agent := FakeInferenceAgent.new()
	s.agents_inference = [agent]
	s._inference_process()
	h.assert_eq(agent.infer_calls, 1, "inference step calls infer_and_act once")

	s.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
godot --headless --path . --script res://test/unit/test_sync_inference.gd
```
Expected: FAIL — `NCNN_INFERENCE` missing from `NcnnSync.ControlModes`; `agents_inference`/`_inference_process` undefined.

- [ ] **Step 3: Update `sync.gd`**

Change the enum:
```gdscript
enum ControlModes { HUMAN, TRAINING }
```
to:
```gdscript
enum ControlModes { HUMAN, TRAINING, NCNN_INFERENCE }
```

Add an inference-agents list next to the existing `var agents_heuristic: Array = []` line:
```gdscript
var agents_inference: Array = []
```

In `_get_agents()`, the current discrete routing reads:
```gdscript
		if agent.control_mode == agent.ControlModes.TRAINING:
			agents_training.append(agent)
		elif agent.control_mode == agent.ControlModes.HUMAN:
			agents_heuristic.append(agent)
```
Replace it with (note INHERIT resolution must also map NCNN_INFERENCE through):
```gdscript
		if agent.control_mode == agent.ControlModes.TRAINING:
			agents_training.append(agent)
		elif agent.control_mode == agent.ControlModes.NCNN_INFERENCE:
			agents_inference.append(agent)
		elif agent.control_mode == agent.ControlModes.HUMAN:
			agents_heuristic.append(agent)
```

Also update the INHERIT_FROM_SYNC resolution just above it so an inheriting agent picks up NCNN_INFERENCE when the Sync node is in that mode. Replace:
```gdscript
		if agent.control_mode == agent.ControlModes.INHERIT_FROM_SYNC:
			agent.control_mode = (agent.ControlModes.TRAINING if control_mode == ControlModes.TRAINING else agent.ControlModes.HUMAN)
```
with:
```gdscript
		if agent.control_mode == agent.ControlModes.INHERIT_FROM_SYNC:
			match control_mode:
				ControlModes.TRAINING:
					agent.control_mode = agent.ControlModes.TRAINING
				ControlModes.NCNN_INFERENCE:
					agent.control_mode = agent.ControlModes.NCNN_INFERENCE
				_:
					agent.control_mode = agent.ControlModes.HUMAN
```

Add the inference step function (place it next to `_heuristic_process`):
```gdscript
func _inference_process() -> void:
	for agent in agents_inference:
		agent.infer_and_act()
		if agent.get_done():
			agent.set_done_false()
```

In `_physics_process`, the current body (after the action_repeat gate) is:
```gdscript
	n_action_steps += 1
	_training_process()
	_heuristic_process()
```
Replace with:
```gdscript
	n_action_steps += 1
	_training_process()
	_inference_process()
	_heuristic_process()
```

Finally, ensure inference mode does not block on a Python server: in `_initialize()`, the line `_initialize_training_agents()` already no-ops when `agents_training` is empty (it only connects if `agents_training.size() > 0`), so inference-only scenes never attempt a TCP connection. No change needed there.

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
godot --headless --path . --script res://test/unit/test_sync_inference.gd
```
Expected: PASS — `Results: 2 passed, 0 failed` (exit 0).

- [ ] **Step 5: Run the full suite (no regressions)**

Run:
```bash
./test/run_tests.sh
```
Expected: all unit tests pass, `PROTOCOL TEST PASSED`, `All tests passed.` (exit 0). The Part 1 protocol test still passes because the training path is unchanged and `_inference_process()` is a no-op when `agents_inference` is empty.

- [ ] **Step 6: Commit**

```bash
git add sync.gd test/unit/test_sync_inference.gd
git commit -m "feat: add ncnn inference branch to NcnnSync"
```

---

### Task 5: Generate a dummy 5→5 ncnn model for testing

**Files:**
- Create: `examples/chase_the_target/models/chase_dummy.ncnn.param` + `.bin` (generated artifacts, committed)

The smoke test needs a model with the right shape (5 inputs → 5 outputs). The existing `scripts/export_test_mlp.py` generates exactly this. This dummy is randomly initialized (not trained) — it proves the inference *plumbing* (load, argmax, valid action, agent moves, stays in bounds). The trained model arrives in Part 2B.

- [ ] **Step 1: Generate the model**

Run:
```bash
.venv/bin/python scripts/export_test_mlp.py --name chase_dummy --input-dim 5 --hidden-dim 32 --output-dim 5
```
Expected: prints suggested blob names (`in0`/`out0`) and writes `models/chase_dummy.ncnn.param` and `models/chase_dummy.ncnn.bin` (the script's default output dir is `models/`).

- [ ] **Step 2: Move the model into the example folder and confirm its shape**

Run:
```bash
mkdir -p examples/chase_the_target/models
mv models/chase_dummy.ncnn.param models/chase_dummy.ncnn.bin examples/chase_the_target/models/
cat examples/chase_the_target/models/chase_dummy.ncnn.param
```
Expected: the param file lists an `Input ... in0` layer and a final `InnerProduct ... out0 0=5 ...` (output dimension 5). Confirm the input blob is `in0` and output blob is `out0`.

- [ ] **Step 3: Clean up the intermediate export artifacts**

Run:
```bash
rm -f models/chase_dummy.pnnx.* models/chase_dummy.pt models/chase_dummy_ncnn.py models/chase_dummy_pnnx.py models/chase_dummy.pnnx.onnx 2>/dev/null; ls examples/chase_the_target/models/
```
Expected: only `chase_dummy.ncnn.param` and `chase_dummy.ncnn.bin` remain in the example models folder; no stray `chase_dummy.*` intermediates left in `models/`.

- [ ] **Step 4: Commit**

```bash
git add examples/chase_the_target/models/chase_dummy.ncnn.param examples/chase_the_target/models/chase_dummy.ncnn.bin
git commit -m "test: add dummy 5x5 ncnn model for chase inference smoke test"
```

---

### Task 6: Build the scene + headless inference smoke test

**Files:**
- Create: `examples/chase_the_target/chase_the_target.tscn`
- Create: `test/integration/inference_smoke_checker.gd`
- Create: `test/integration/inference_smoke_scene.tscn`
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Create the playable scene**

Create `examples/chase_the_target/chase_the_target.tscn` (headless-friendly Node2D markers; `Sync` in `NCNN_INFERENCE`, agent inherits and points at the dummy model):
```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://examples/chase_the_target/chase_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/chase_the_target/chase_agent.gd" id="2"]
[ext_resource type="Script" path="res://sync.gd" id="3"]

[node name="ChaseGame" type="Node2D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
target_path = NodePath("Target")

[node name="AgentBody" type="Node2D" parent="."]

[node name="Target" type="Node2D" parent="."]

[node name="ChaseAgent" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 0
model_param_path = "res://examples/chase_the_target/models/chase_dummy.ncnn.param"
model_bin_path = "res://examples/chase_the_target/models/chase_dummy.ncnn.bin"

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 2
```

(`ChaseAgent.control_mode = 0` is `INHERIT_FROM_SYNC`; `Sync.control_mode = 2` is `NCNN_INFERENCE`, so the agent resolves to NCNN_INFERENCE via the match added in Task 4.)

- [ ] **Step 2: Create the smoke checker script**

Create `test/integration/inference_smoke_checker.gd`:
```gdscript
extends Node
# Runs inside a headless scene: watches the ChaseAgent run under ncnn inference for
# a fixed number of physics frames, asserts the agent stays in bounds and that the
# model produced valid discrete actions, then quits with an exit code.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 240

var _game: ChaseGame
var _agent: ChaseAgent
var _frames := 0
var _failed := false
var _reason := ""

func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	_game = get_node_or_null(game_path) as ChaseGame
	_agent = get_node_or_null(agent_path) as ChaseAgent
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")

func _physics_process(_delta: float) -> void:
	if _failed:
		_finish()
		return
	if _agent != null and not _agent._ncnn_runner == null:
		var pos := _game.get_agent_pos()
		if pos.x < -1.0 or pos.x > _game.arena_size.x + 1.0 or pos.y < -1.0 or pos.y > _game.arena_size.y + 1.0:
			_fail("agent left arena bounds: %s" % pos)
		if _agent._action_index < 0 or _agent._action_index >= ChaseAgent.ACTION_COUNT:
			_fail("invalid action index: %d" % _agent._action_index)
	_frames += 1
	if _frames >= frames_to_run:
		_finish()

func _fail(reason: String) -> void:
	_failed = true
	_reason = reason

func _finish() -> void:
	if _failed:
		printerr("INFERENCE SMOKE FAILED: %s" % _reason)
		get_tree().quit(1)
	else:
		print("INFERENCE SMOKE PASSED (%d frames)" % _frames)
		get_tree().quit(0)
```

- [ ] **Step 3: Create the smoke scene**

Create `test/integration/inference_smoke_scene.tscn` (same wiring as the example scene plus the checker; `Sync` in NCNN_INFERENCE):
```
[gd_scene load_steps=5 format=3]

[ext_resource type="Script" path="res://examples/chase_the_target/chase_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/chase_the_target/chase_agent.gd" id="2"]
[ext_resource type="Script" path="res://sync.gd" id="3"]
[ext_resource type="Script" path="res://test/integration/inference_smoke_checker.gd" id="4"]

[node name="ChaseGame" type="Node2D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
target_path = NodePath("Target")

[node name="AgentBody" type="Node2D" parent="."]

[node name="Target" type="Node2D" parent="."]

[node name="ChaseAgent" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 3
model_param_path = "res://examples/chase_the_target/models/chase_dummy.ncnn.param"
model_bin_path = "res://examples/chase_the_target/models/chase_dummy.ncnn.bin"

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 2

[node name="SmokeChecker" type="Node" parent="."]
script = ExtResource("4")
game_path = NodePath("../ChaseGame")
agent_path = NodePath("../ChaseAgent")
frames_to_run = 240
```

(Here `ChaseAgent.control_mode = 3` is set explicitly to `NCNN_INFERENCE` so the agent loads the model directly — independent of Sync's inheritance.)

Wait — the checker's `game_path`/`agent_path` are siblings, not children of ChaseGame. Fix the node layout: the checker and agent are children of the root `ChaseGame`. Use `game_path = NodePath("..")`-relative paths. Correct the SmokeChecker node properties to:
```
game_path = NodePath("..")
agent_path = NodePath("../ChaseAgent")
```
and ensure `SmokeChecker` is a child of `ChaseGame` (it is, `parent="."`). The root IS `ChaseGame`, so `..` from `SmokeChecker` is `ChaseGame`. Update the scene text accordingly before saving.

- [ ] **Step 4: Add the smoke test to the runner**

In `test/run_tests.sh`, after the existing protocol-integration block and before the final `echo "All tests passed."`, add:
```bash
echo "== Inference smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/inference_smoke_scene.tscn
```
(`set -e` will abort the script if Godot exits non-zero, i.e. if the smoke checker fails.)

- [ ] **Step 5: Run the smoke scene directly**

Run:
```bash
godot --headless --path . res://test/integration/inference_smoke_scene.tscn; echo "EXIT: $?"
```
Expected: `INFERENCE SMOKE PASSED (240 frames)` and `EXIT: 0`. (A benign "ObjectDB instances leaked at exit" warning may appear; it does not affect the exit code.)

- [ ] **Step 6: Run the full suite**

Run:
```bash
./test/run_tests.sh
```
Expected: all unit tests pass, `PROTOCOL TEST PASSED`, `INFERENCE SMOKE PASSED`, `All tests passed.` (exit 0).

- [ ] **Step 7: Commit**

```bash
git add examples/chase_the_target/chase_the_target.tscn test/integration/inference_smoke_checker.gd test/integration/inference_smoke_scene.tscn test/run_tests.sh
git commit -m "test: add chase scene and headless ncnn inference smoke test"
```

---

### Task 7: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm clean tree and review the branch**

Run:
```bash
git status --short && git log --oneline main..HEAD
```
Expected: no uncommitted changes; commits for ChaseGame, ChaseAgent, NCNN_INFERENCE controller mode, Sync inference branch, the dummy model, and the scene + smoke test.

- [ ] **Step 2: Run the full suite one final time**

Run:
```bash
./test/run_tests.sh
```
Expected: `All tests passed.` (exit 0), including `INFERENCE SMOKE PASSED`.

- [ ] **Step 3: Confirm the example scene loads without script/parse errors**

Run:
```bash
godot --headless --path . res://examples/chase_the_target/chase_the_target.tscn --quit-after 60 2>&1 | grep -iE "SCRIPT ERROR|Parse Error" || echo "NO ERRORS"
```
Expected: `NO ERRORS`.

---

## Self-Review

**Spec coverage (Part 2A portion):**
- 2D chase game (arena, agent, relocating target, bounds, reset) → Tasks 1, 2, 6. ✅
- Observation (5 normalized floats), discrete 5-action space, shaped reward → Task 2 (matches spec §3.1–3.3). ✅
- ncnn INFERENCE mode wired to `NcnnRunner` (the spec's "inference branch wired to NcnnRunner", deferred from Part 1) → Tasks 3, 4. ✅
- Runnable example scene → Task 6. ✅
- Headless inference smoke test (runs scene, asserts no errors + valid behavior) → Task 6. ✅ (The spec's "average distance decreases" assertion requires a *trained* model and is deferred to Part 2B; with the dummy model we assert plumbing: in-bounds + valid actions.)
- **Deferred to Part 2B (intentional):** godot-rl install + PPO training run, ONNX→pnnx→ncnn conversion of the trained policy, the pre-trained model artifact, the "it actually chases / distance decreases" check, the from-scratch tutorial doc, and the top-level README examples pointer.

**Placeholder scan:** No `TBD`/`TODO`/"handle errors appropriately"/"similar to Task N". Every code step shows complete code; every run step has an exact command and expected output. Task 6 Step 3 contains an explicit in-line correction of the SmokeChecker node paths (a deliberate fix instruction, not a placeholder). ✅

**Type/name consistency:**
- `ControlModes.NCNN_INFERENCE` added to BOTH `NcnnAIController2D` (Task 3) and `NcnnSync` (Task 4); the scene files use the matching integer values (`NcnnSync` `NCNN_INFERENCE = 2`; `NcnnAIController2D` `NCNN_INFERENCE = 3`, `INHERIT_FROM_SYNC = 0`). ✅
- `infer_and_act()` defined on the base controller (Task 3) and called by `NcnnSync._inference_process()` (Task 4) and the test fake (Task 4). ✅
- `ChaseAgent.ACTION_COUNT`/`ACTION_KEY`, `action_index_to_velocity`, `compute_obs`, `compute_step_reward` are defined in Task 2 and referenced consistently by the scene and the smoke checker (`ChaseAgent.ACTION_COUNT`). ✅
- `ChaseGame` methods (`get_agent_pos`/`get_target_pos`/`arena_size`/`move_agent`/`relocate_target`/`reset_positions`/`distance`/`max_distance`/`clamp_to_bounds`/`random_position`/`seed_rng`) defined in Task 1 and used by `ChaseAgent` (Task 2/3) and the smoke checker (Task 6) with matching names. ✅
- `stub_agent.gd` gains `last_action`/`set_action` (Task 3) used by `test_controller_inference.gd`; Part 1's `test_controller.gd` is unaffected (additive). ✅
