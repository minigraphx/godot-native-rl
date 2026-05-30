# 3D Raycast Rover Example — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tank-steered 3D rover example that uses `RaycastSensor3D` + `NcnnAIController3D` to reach a goal while avoiding a fixed obstacle field — full scaffold + headless tests, with the real training run/ncnn model/golden regression as a separate final step.

**Architecture:** `RoverGame` (Node3D) holds world state + pure helpers (bounds, obstacle blocking, bearing, free-position). `RoverAgent` extends the 3D controller, composing `RaycastSensor3D` output with egocentric goal features, driving tank motion via discrete actions, and authoring reward with `RewardBuilder` + `RewardAdapter`. Mirrors `examples/chase_the_target/`.

**Tech Stack:** Godot 4.6 GDScript (TAB indentation), headless `extends SceneTree` harness (`test/harness.gd`), `preload` refs, godot_rl + SB3 PPO training pipeline.

**Spec:** `docs/superpowers/specs/2026-05-31-rover-3d-example-design.md`

**Conventions for every task:**
- GDScript uses **TAB** indentation (code blocks below use tabs — preserve exactly).
- `godot` is on PATH (v4.6.2). Single unit test: `godot --headless --path . --script "res://test/unit/test_NAME.gd"`.
- Full suite: `./test/run_tests.sh` → ends `All tests passed.` New `test/unit/test_*.gd` auto-discovered.
- Verify branch is `feat/rover-3d-example` before each commit. Never commit on main.
- Reward API (confirmed from `chase_agent.gd`): `RewardBuilder.new().add_progress_shaping(dist_callable, max_dist_callable, reset_events_array).add_event_bonus(event_name, amount).add_step_penalty(amount).build()`; `RewardAdapter.new()` added as a child, then `adapter.on_signal_event(emitter, signal_name, event_name)`.
- Controller subclasses use **path-based** `extends` (cache-independent headless resolution — see CLAUDE.md / item 5).

---

## File structure

- `examples/rover_3d/rover_game.gd` — `RoverGame extends Node3D`: world state + pure helpers + runtime motion/blocking/reset.
- `examples/rover_3d/rover_agent.gd` — `RoverAgent` extends the 3D controller: obs composition, tank actions, reward wiring.
- `examples/rover_3d/rover_3d.tscn` — world/play scene (game + AgentBody + RaycastSensor3D + Goal + Obstacles + RoverAgent).
- `examples/rover_3d/rover_3d_train.tscn` — world + `NcnnSync` (training).
- `test/integration/rover_smoke_checker.gd` — drives the scene headless, asserts invariants, quits.
- `test/integration/rover_smoke_scene.tscn` — world + smoke checker.
- `test/unit/test_rover_game.gd`, `test/unit/test_rover_game_runtime.gd`, `test/unit/test_rover_agent.gd` — unit tests.
- `scripts/train_rover.py`, `scripts/train_rover.sh` — training (clone of chase).
- `test/run_tests.sh` — add the rover smoke scene. `README.md` — example pointer.

---

## Task 1: `RoverGame` pure helpers (TDD)

**Files:**
- Create: `examples/rover_3d/rover_game.gd`
- Test: `test/unit/test_rover_game.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_rover_game.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RoverGameScript = preload("res://examples/rover_3d/rover_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var g = RoverGameScript.new()
	g.arena_size = Vector2(40.0, 40.0)

	# clamp_to_bounds (X/Z clamped, Y preserved)
	h.assert_eq(g.clamp_to_bounds(Vector3(-5.0, 1.0, 50.0)), Vector3(0.0, 1.0, 40.0), "clamp low/high")
	h.assert_eq(g.clamp_to_bounds(Vector3(20.0, 0.0, 20.0)), Vector3(20.0, 0.0, 20.0), "clamp inside unchanged")

	# is_blocked vs an obstacle AABB centered (12,_,12) half (2,_,2)
	var obs := [{"center": Vector3(12.0, 0.0, 12.0), "half_extent": Vector3(2.0, 1.0, 2.0)}]
	h.assert_true(g.is_blocked(Vector3(12.5, 0.0, 11.0), obs), "inside obstacle -> blocked")
	h.assert_true(not g.is_blocked(Vector3(20.0, 0.0, 20.0), obs), "free cell -> not blocked")
	h.assert_true(not g.is_blocked(Vector3(15.0, 0.0, 12.0), obs), "just outside half-extent -> not blocked")

	# max_distance is the XZ diagonal
	h.assert_true(absf(g.max_distance() - Vector2(40.0, 40.0).length()) < 0.001, "max_distance diagonal")

	# bearing_to: ahead(-Z)->0, +X->-PI/2, -X->+PI/2, behind(+Z)->+/-PI
	h.assert_true(absf(g.bearing_to(Vector3.ZERO, 0.0, Vector3(0.0, 0.0, -5.0))) < 1e-5, "goal ahead -> bearing 0")
	h.assert_true(absf(g.bearing_to(Vector3.ZERO, 0.0, Vector3(5.0, 0.0, 0.0)) - (-PI / 2.0)) < 1e-5, "goal +X -> -PI/2")
	h.assert_true(absf(g.bearing_to(Vector3.ZERO, 0.0, Vector3(-5.0, 0.0, 0.0)) - (PI / 2.0)) < 1e-5, "goal -X -> +PI/2")
	h.assert_true(absf(absf(g.bearing_to(Vector3.ZERO, 0.0, Vector3(0.0, 0.0, 5.0))) - PI) < 1e-5, "goal behind -> +/-PI")
	# bearing is heading-relative: facing +X-ish cancels a +X goal
	h.assert_true(absf(g.bearing_to(Vector3.ZERO, -PI / 2.0, Vector3(5.0, 0.0, 0.0))) < 1e-5, "goal +X while facing +X -> 0")

	# seeded RNG determinism + random_free_position avoids obstacles & stays in bounds
	g.seed_rng(123)
	var rng := RandomNumberGenerator.new()
	rng.seed = 123
	var all_ok := true
	for _i in range(200):
		var p: Vector3 = g.random_free_position(rng, obs)
		if p.x < 0.0 or p.x > 40.0 or p.z < 0.0 or p.z > 40.0 or g.is_blocked(p, obs):
			all_ok = false
	h.assert_true(all_ok, "random_free_position in-bounds and not blocked")

	g.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_rover_game.gd"`
Expected: FAIL — `res://examples/rover_3d/rover_game.gd` does not exist (preload parse error).

- [ ] **Step 3: Write minimal implementation**

Create `examples/rover_3d/rover_game.gd`:

```gdscript
class_name RoverGame
extends Node3D

@export var arena_size := Vector2(40.0, 40.0)  ## XZ extent (meters)
@export var move_speed := 6.0
@export var turn_speed := 2.5  ## radians/sec
@export var goal_radius := 2.0
@export var agent_body_path: NodePath
@export var goal_path: NodePath
@export var obstacles_path: NodePath

signal goal_reached
signal bumped

var obstacles: Array = []  # [{center: Vector3, half_extent: Vector3}]
var reaches := 0
var _rng := RandomNumberGenerator.new()
var _agent_body: Node3D
var _goal: Node3D

func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path) as Node3D
	_goal = get_node_or_null(goal_path) as Node3D
	obstacles = read_obstacles(get_node_or_null(obstacles_path))
	reset_positions()

# --- Pure helpers (unit-tested) ---
func clamp_to_bounds(pos: Vector3) -> Vector3:
	return Vector3(clampf(pos.x, 0.0, arena_size.x), pos.y, clampf(pos.z, 0.0, arena_size.y))

func is_blocked(pos: Vector3, obs: Array) -> bool:
	for o in obs:
		var c: Vector3 = o["center"]
		var hh: Vector3 = o["half_extent"]
		if absf(pos.x - c.x) <= hh.x and absf(pos.z - c.z) <= hh.z:
			return true
	return false

func max_distance() -> float:
	return Vector2(arena_size.x, arena_size.y).length()

# Signed angle (radians) from the rover's heading to the goal direction, in the XZ plane.
# Heading convention matches move_agent's forward = (-sin yaw, 0, -cos yaw).
func bearing_to(agent_pos: Vector3, agent_yaw: float, goal_pos: Vector3) -> float:
	var dx := goal_pos.x - agent_pos.x
	var dz := goal_pos.z - agent_pos.z
	if Vector2(dx, dz).length() < 1e-6:
		return 0.0
	var goal_angle := atan2(-dx, -dz)
	return wrapf(goal_angle - agent_yaw, -PI, PI)

func seed_rng(s: int) -> void:
	_rng.seed = s

func random_free_position(rng: RandomNumberGenerator, obs: Array) -> Vector3:
	var candidate := Vector3.ZERO
	for _i in range(64):
		candidate = Vector3(rng.randf_range(0.0, arena_size.x), 0.0, rng.randf_range(0.0, arena_size.y))
		if not is_blocked(candidate, obs):
			return candidate
	return candidate

# Read obstacle AABBs from StaticBody3D children (each with a "Col" CollisionShape3D / BoxShape3D).
func read_obstacles(parent: Node) -> Array:
	var result: Array = []
	if parent == null:
		return result
	for child in parent.get_children():
		var half := Vector3(1.0, 1.0, 1.0)
		var col = child.get_node_or_null("Col")
		if col != null and col.shape is BoxShape3D:
			half = (col.shape as BoxShape3D).size * 0.5
		result.append({"center": child.position, "half_extent": half})
	return result

# --- Runtime helpers (exercised by the scene + smoke test) ---
func get_agent_pos() -> Vector3:
	return _agent_body.position if _agent_body != null else Vector3.ZERO

func get_agent_yaw() -> float:
	return _agent_body.rotation.y if _agent_body != null else 0.0

func get_goal_pos() -> Vector3:
	return _goal.position if _goal != null else Vector3.ZERO

func distance() -> float:
	return get_agent_pos().distance_to(get_goal_pos())

func move_agent(forward: float, yaw_delta: float, delta: float) -> void:
	if _agent_body == null:
		return
	_agent_body.rotation.y += yaw_delta * delta
	var yaw := _agent_body.rotation.y
	var fwd := Vector3(-sin(yaw), 0.0, -cos(yaw))
	var next_pos := _agent_body.position + fwd * forward * delta
	if is_blocked(next_pos, obstacles):
		bumped.emit()
	else:
		_agent_body.position = clamp_to_bounds(next_pos)

func relocate_goal() -> void:
	reaches += 1
	if _goal != null:
		_goal.position = random_free_position(_rng, obstacles)
	goal_reached.emit()

func reset_positions() -> void:
	if _agent_body != null:
		_agent_body.position = random_free_position(_rng, obstacles)
		_agent_body.rotation.y = _rng.randf_range(-PI, PI)
	if _goal != null:
		_goal.position = random_free_position(_rng, obstacles)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_rover_game.gd"`
Expected: PASS — `Results: 12 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add examples/rover_3d/rover_game.gd test/unit/test_rover_game.gd
git commit -m "feat: RoverGame world + pure helpers (bounds, blocking, bearing, free-position)"
```

---

## Task 2: `RoverGame` runtime motion + signals (TDD)

**Files:**
- Test: `test/unit/test_rover_game_runtime.gd`
- (No new production code — exercises `move_agent`, `relocate_goal`, `reset_positions` from Task 1.)

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_rover_game_runtime.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RoverGameScript = preload("res://examples/rover_3d/rover_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var g = RoverGameScript.new()
	g.arena_size = Vector2(40.0, 40.0)

	# Inject a body + goal (Node3D, not added to tree — only local position/rotation used).
	var body := Node3D.new()
	var goal := Node3D.new()
	g._agent_body = body
	g._goal = goal
	g.obstacles = [{"center": Vector3(20.0, 0.0, 10.0), "half_extent": Vector3(2.0, 1.0, 2.0)}]

	# Unobstructed forward move from (20,0,20) facing -Z advances toward -Z and clamps in bounds.
	body.position = Vector3(20.0, 0.0, 20.0)
	body.rotation.y = 0.0
	g.move_agent(g.move_speed, 0.0, 0.1)
	h.assert_true(body.position.z < 20.0, "forward move advances toward -Z")
	h.assert_true(body.position.x > -0.001 and body.position.x < 40.001, "stays in X bounds")

	# A move that would enter the obstacle is blocked (position held) and emits `bumped`.
	var bumps := [0]
	g.bumped.connect(func() -> void: bumps[0] += 1)
	body.position = Vector3(20.0, 0.0, 13.0)  # just below the obstacle (z=10, half 2 => blocks z in [8,12])
	body.rotation.y = 0.0  # facing -Z (decreasing z) -> moves toward the obstacle
	var held := body.position
	g.move_agent(g.move_speed, 0.0, 0.5)  # large step pushes into z<=12
	h.assert_eq(body.position, held, "blocked move holds position")
	h.assert_eq(bumps[0], 1, "blocked move emits bumped once")

	# relocate_goal increments reaches and emits goal_reached
	var reached := [0]
	g.goal_reached.connect(func() -> void: reached[0] += 1)
	g.seed_rng(7)
	g.relocate_goal()
	g.relocate_goal()
	h.assert_eq(g.reaches, 2, "relocate_goal increments reaches")
	h.assert_eq(reached[0], 2, "relocate_goal emits goal_reached each call")

	# reset_positions keeps body + goal in free, in-bounds cells
	g.reset_positions()
	h.assert_true(not g.is_blocked(body.position, g.obstacles), "reset body not blocked")
	h.assert_true(not g.is_blocked(goal.position, g.obstacles), "reset goal not blocked")

	body.free()
	goal.free()
	g.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_rover_game_runtime.gd"`
Expected: PASS already? No — the production code from Task 1 implements these. So this test should PASS immediately. If it FAILS, fix the discrepancy in `rover_game.gd` (this test is a regression guard for the runtime helpers). Run it and confirm it passes.

Run: `godot --headless --path . --script "res://test/unit/test_rover_game_runtime.gd"`
Expected: PASS — `Results: 7 passed, 0 failed`.

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_rover_game_runtime.gd
git commit -m "test: RoverGame runtime motion (blocking + bumped), relocate_goal, reset"
```

---

## Task 3: `RoverAgent` pure helpers (TDD)

**Files:**
- Create: `examples/rover_3d/rover_agent.gd`
- Test: `test/unit/test_rover_agent.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_rover_agent.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RoverAgentScript = preload("res://examples/rover_3d/rover_agent.gd")

func _initialize() -> void:
	var h := Harness.new()
	var a = RoverAgentScript.new()

	# action_index_to_motion: 0 idle, 1 forward, 2 turn-left (-yaw), 3 turn-right (+yaw)
	h.assert_eq(a.action_index_to_motion(0, 6.0, 2.5), {"forward": 0.0, "yaw": 0.0}, "idle")
	h.assert_eq(a.action_index_to_motion(1, 6.0, 2.5), {"forward": 6.0, "yaw": 0.0}, "forward")
	h.assert_eq(a.action_index_to_motion(2, 6.0, 2.5), {"forward": 0.0, "yaw": -2.5}, "turn left")
	h.assert_eq(a.action_index_to_motion(3, 6.0, 2.5), {"forward": 0.0, "yaw": 2.5}, "turn right")

	# compute_goal_obs(bearing, dist, max_dist) -> [sin, cos, clamped distance]
	var goal_obs: Array = a.compute_goal_obs(0.0, 10.0, 40.0)
	h.assert_eq(goal_obs.size(), 3, "goal obs has 3 elements")
	h.assert_true(absf(goal_obs[0] - 0.0) < 1e-6, "sin(0)=0")
	h.assert_true(absf(goal_obs[1] - 1.0) < 1e-6, "cos(0)=1")
	h.assert_true(absf(goal_obs[2] - 0.25) < 1e-6, "distance normalized 10/40")
	var far: Array = a.compute_goal_obs(PI / 2.0, 100.0, 40.0)
	h.assert_true(absf(far[0] - 1.0) < 1e-6, "sin(PI/2)=1")
	h.assert_true(absf(far[2] - 1.0) < 1e-6, "distance clamps to 1.0")

	# compose_obs concatenates rays + goal in order
	var composed: Array = a.compose_obs([0.1, 0.2, 0.3, 0.4, 0.5], [0.0, 1.0, 0.25])
	h.assert_eq(composed.size(), 8, "composed obs length = rays(5) + goal(3)")
	h.assert_true(absf(composed[0] - 0.1) < 1e-6, "rays come first")
	h.assert_true(absf(composed[5] - 0.0) < 1e-6, "goal obs appended after rays")

	# action space
	h.assert_eq(a.get_action_space(), {"move": {"size": 4, "action_type": "discrete"}}, "action space")

	a.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_rover_agent.gd"`
Expected: FAIL — `res://examples/rover_3d/rover_agent.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `examples/rover_3d/rover_agent.gd`:

```gdscript
class_name RoverAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"

const ACTION_KEY := "move"
const ACTION_COUNT := 4
const GOAL_OBS_SIZE := 3
const DEFAULT_RAY_COUNT := 5
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from the controller — do not redeclare.

@export var game_path: NodePath
@export var sensor_path: NodePath
@export var goal_bonus := 1.0
@export var step_penalty := 0.005
@export var collision_penalty := 0.25

var _game
var _sensor
var _action_index := 0

# --- Pure helpers (unit-tested) ---
func action_index_to_motion(idx: int, move_speed: float, turn_speed: float) -> Dictionary:
	match idx:
		1: return {"forward": move_speed, "yaw": 0.0}
		2: return {"forward": 0.0, "yaw": -turn_speed}
		3: return {"forward": 0.0, "yaw": turn_speed}
		_: return {"forward": 0.0, "yaw": 0.0}

func compute_goal_obs(bearing: float, dist: float, max_dist: float) -> Array:
	var norm := clampf(dist / max_dist, 0.0, 1.0) if max_dist > 0.0 else 0.0
	return [sin(bearing), cos(bearing), norm]

func compose_obs(ray_obs: Array, goal_obs: Array) -> Array:
	return ray_obs + goal_obs

func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

# --- Wiring (Task 4 fills _ready / get_obs / set_action / _physics_process) ---
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_rover_agent.gd"`
Expected: PASS — `Results: 11 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add examples/rover_3d/rover_agent.gd test/unit/test_rover_agent.gd
git commit -m "feat: RoverAgent pure helpers (action->motion, goal obs, compose, action space)"
```

---

## Task 4: `RoverAgent` runtime wiring (obs / action / reward / step)

**Files:**
- Modify: `examples/rover_3d/rover_agent.gd`
- Test: `test/unit/test_rover_agent.gd` (add a null-game fallback + set_action case)

- [ ] **Step 1: Add the failing test cases**

Append to `test/unit/test_rover_agent.gd` **before** `a.free()`:

```gdscript
	# get_obs with no game/sensor -> a correctly-sized zero vector (no crash)
	var obs_dict: Dictionary = a.get_obs()
	h.assert_true("obs" in obs_dict, "get_obs returns an obs key")
	h.assert_eq(obs_dict["obs"].size(), DEFAULT_RAY_COUNT + 3, "fallback obs size = default rays + goal")
	var any_nonzero := false
	for v in obs_dict["obs"]:
		if absf(v) > 1e-9:
			any_nonzero = true
	h.assert_true(not any_nonzero, "fallback obs is all zeros")

	# set_action stores a valid discrete index
	a.set_action({"move": 3})
	h.assert_eq(a._action_index, 3, "set_action stores the index")
```

Note `DEFAULT_RAY_COUNT` is a const on `RoverAgent`; reference it via the instance script (`a` is a `RoverAgentScript` instance, so the const is in scope as `DEFAULT_RAY_COUNT` only inside the script — in the test use the literal `5 + 3`). Replace the assert line with the literal to avoid scope issues:

```gdscript
	h.assert_eq(obs_dict["obs"].size(), 8, "fallback obs size = default rays(5) + goal(3)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script "res://test/unit/test_rover_agent.gd"`
Expected: FAIL — `get_obs`/`set_action` not implemented yet (assertion failure or nonexistent function).

- [ ] **Step 3: Add the runtime methods**

Append to `examples/rover_3d/rover_agent.gd` (after the pure helpers):

```gdscript
func _expected_obs_size() -> int:
	var ray_count := _sensor.obs_size() if _sensor != null else DEFAULT_RAY_COUNT
	return ray_count + GOAL_OBS_SIZE

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(_expected_obs_size())
	z.fill(0.0)
	return z

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	_sensor = get_node_or_null(sensor_path)
	if _game == null:
		push_warning("RoverAgent: game_path not set or invalid — producing zero observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["goal_reached"]) \
		.add_event_bonus("goal_reached", goal_bonus) \
		.add_event_bonus("bumped", -collision_penalty) \
		.add_step_penalty(step_penalty) \
		.build()
	var goal_adapter := RewardAdapterScript.new()
	add_child(goal_adapter)
	goal_adapter.on_signal_event(_game, "goal_reached", "goal_reached")
	var bump_adapter := RewardAdapterScript.new()
	add_child(bump_adapter)
	bump_adapter.on_signal_event(_game, "bumped", "bumped")

func get_obs() -> Dictionary:
	if _game == null or _sensor == null:
		return {"obs": _zero_obs()}
	var ray_obs: Array = _sensor.get_observation()
	var bearing: float = _game.bearing_to(_game.get_agent_pos(), _game.get_agent_yaw(), _game.get_goal_pos())
	var goal_obs := compute_goal_obs(bearing, _game.distance(), _game.max_distance())
	return {"obs": compose_obs(ray_obs, goal_obs)}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "RoverAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	var motion := action_index_to_motion(_action_index, _game.move_speed, _game.turn_speed)
	_game.move_agent(motion["forward"], motion["yaw"], delta)
	# Accumulate reward against the CURRENT goal BEFORE relocating (matches the chase pattern).
	accumulate_reward()
	if _game.distance() < _game.goal_radius:
		_game.relocate_goal()
	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script "res://test/unit/test_rover_agent.gd"`
Expected: PASS — `Results: 13 passed, 0 failed`.

- [ ] **Step 5: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 6: Commit**

```bash
git add examples/rover_3d/rover_agent.gd test/unit/test_rover_agent.gd
git commit -m "feat: RoverAgent runtime wiring (obs/sensor compose, reward builder+adapters, step)"
```

---

## Task 5: World/play scene `rover_3d.tscn`

**Files:**
- Create: `examples/rover_3d/rover_3d.tscn`

- [ ] **Step 1: Author the scene**

Create `examples/rover_3d/rover_3d.tscn`:

```
[gd_scene load_steps=5 format=3]

[ext_resource type="Script" path="res://examples/rover_3d/rover_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/rover_3d/rover_agent.gd" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sensors/raycast_sensor_3d.gd" id="3"]

[sub_resource type="BoxShape3D" id="Box"]
size = Vector3(4, 2, 4)

[node name="RoverGame" type="Node3D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
goal_path = NodePath("Goal")
obstacles_path = NodePath("Obstacles")

[node name="AgentBody" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 6, 0, 6)

[node name="RaycastSensor3D" type="Node3D" parent="AgentBody"]
script = ExtResource("3")
n_rays_width = 5
n_rays_height = 1
ray_length = 20.0
horizontal_fov = 120.0
vertical_fov = 0.0
collision_mask = 1

[node name="Goal" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 34, 0, 34)

[node name="Obstacles" type="Node3D" parent="."]

[node name="Obstacle1" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 12)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle1"]
shape = SubResource("Box")

[node name="Obstacle2" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 28, 0, 12)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle2"]
shape = SubResource("Box")

[node name="Obstacle3" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 28)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle3"]
shape = SubResource("Box")

[node name="Obstacle4" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 28, 0, 28)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle4"]
shape = SubResource("Box")

[node name="RoverAgent" type="Node3D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
sensor_path = NodePath("../AgentBody/RaycastSensor3D")
```

- [ ] **Step 2: Verify the scene loads headless without errors**

Run: `godot --headless --path . --quit-after 3 res://examples/rover_3d/rover_3d.tscn 2>&1`
Expected: no `SCRIPT ERROR` / `Parse Error` / `Could not find` lines (the scene loads; the rover sits idle with no Sync). A warning-free load is success. (If `--quit-after` is unavailable, the implementer may run it backgrounded briefly and inspect output — the gate is "no errors on load".)

- [ ] **Step 3: Commit**

```bash
git add examples/rover_3d/rover_3d.tscn
git commit -m "feat: rover_3d world/play scene (game + agent + RaycastSensor3D + obstacles)"
```

---

## Task 6: Training scene `rover_3d_train.tscn`

**Files:**
- Create: `examples/rover_3d/rover_3d_train.tscn`

- [ ] **Step 1: Author the training scene**

Create `examples/rover_3d/rover_3d_train.tscn` (same world as Task 5 plus a `Sync` node; agent `control_mode = 2` = TRAINING, Sync `control_mode = 1`, mirroring `chase_the_target_train.tscn`):

```
[gd_scene load_steps=6 format=3]

[ext_resource type="Script" path="res://examples/rover_3d/rover_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/rover_3d/rover_agent.gd" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sensors/raycast_sensor_3d.gd" id="3"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="4"]

[sub_resource type="BoxShape3D" id="Box"]
size = Vector3(4, 2, 4)

[node name="RoverGame" type="Node3D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
goal_path = NodePath("Goal")
obstacles_path = NodePath("Obstacles")

[node name="AgentBody" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 6, 0, 6)

[node name="RaycastSensor3D" type="Node3D" parent="AgentBody"]
script = ExtResource("3")
n_rays_width = 5
n_rays_height = 1
ray_length = 20.0
horizontal_fov = 120.0
vertical_fov = 0.0
collision_mask = 1

[node name="Goal" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 34, 0, 34)

[node name="Obstacles" type="Node3D" parent="."]

[node name="Obstacle1" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 12)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle1"]
shape = SubResource("Box")

[node name="Obstacle2" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 28, 0, 12)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle2"]
shape = SubResource("Box")

[node name="Obstacle3" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 28)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle3"]
shape = SubResource("Box")

[node name="Obstacle4" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 28, 0, 28)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle4"]
shape = SubResource("Box")

[node name="RoverAgent" type="Node3D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
sensor_path = NodePath("../AgentBody/RaycastSensor3D")
control_mode = 2

[node name="Sync" type="Node" parent="."]
script = ExtResource("4")
control_mode = 1
```

- [ ] **Step 2: Verify it loads headless without errors**

Run: `godot --headless --path . --quit-after 3 res://examples/rover_3d/rover_3d_train.tscn 2>&1`
Expected: no `SCRIPT ERROR`/`Parse Error`; the `NcnnSync: couldn't connect to Python server; using human controls` warning is expected (no trainer running) and is success.

- [ ] **Step 3: Commit**

```bash
git add examples/rover_3d/rover_3d_train.tscn
git commit -m "feat: rover_3d training scene (world + NcnnSync)"
```

---

## Task 7: Headless smoke test scene + checker + wire into run_tests.sh

**Files:**
- Create: `test/integration/rover_smoke_checker.gd`
- Create: `test/integration/rover_smoke_scene.tscn`
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Write the smoke checker**

Create `test/integration/rover_smoke_checker.gd`:

```gdscript
extends Node
# Headless smoke test: drives the rover scene for a fixed number of physics frames with
# random actions, exercising the real RaycastSensor3D physics queries + observation pipeline
# + movement/blocking, asserting invariants, then quitting with an exit code.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 180
@export var expected_obs_size := 8
@export var action_count := 4

var _game
var _agent
var _frames := 0
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_rng.seed = 12345
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	# Drive a random discrete action so the rover moves and the ray fan sweeps the obstacles.
	_agent.set_action({"move": _rng.randi_range(0, action_count - 1)})
	var obs_dict = _agent.get_obs()
	if not ("obs" in obs_dict) or obs_dict["obs"].size() != expected_obs_size:
		_fail("bad obs shape: %s" % obs_dict)
		return
	for v in obs_dict["obs"]:
		if not is_finite(v):
			_fail("non-finite observation value")
			return
	var p = _game.get_agent_pos()
	if p.x < 0.0 or p.x > _game.arena_size.x or p.z < 0.0 or p.z > _game.arena_size.y:
		_fail("rover left arena bounds: %s" % p)
		return
	_frames += 1
	if _frames >= frames_to_run:
		print("ROVER SMOKE PASSED (%d frames)" % _frames)
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("ROVER SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
```

- [ ] **Step 2: Author the smoke scene**

Create `test/integration/rover_smoke_scene.tscn` (same world as Task 5 plus a `SmokeChecker`; no Sync; agent stays `control_mode = 0` INHERIT so no ncnn runner is created):

```
[gd_scene load_steps=6 format=3]

[ext_resource type="Script" path="res://examples/rover_3d/rover_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/rover_3d/rover_agent.gd" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sensors/raycast_sensor_3d.gd" id="3"]
[ext_resource type="Script" path="res://test/integration/rover_smoke_checker.gd" id="4"]

[sub_resource type="BoxShape3D" id="Box"]
size = Vector3(4, 2, 4)

[node name="RoverGame" type="Node3D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
goal_path = NodePath("Goal")
obstacles_path = NodePath("Obstacles")

[node name="AgentBody" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 6, 0, 6)

[node name="RaycastSensor3D" type="Node3D" parent="AgentBody"]
script = ExtResource("3")
n_rays_width = 5
n_rays_height = 1
ray_length = 20.0
horizontal_fov = 120.0
vertical_fov = 0.0
collision_mask = 1

[node name="Goal" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 34, 0, 34)

[node name="Obstacles" type="Node3D" parent="."]

[node name="Obstacle1" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 12)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle1"]
shape = SubResource("Box")

[node name="Obstacle2" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 28, 0, 12)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle2"]
shape = SubResource("Box")

[node name="Obstacle3" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 28)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle3"]
shape = SubResource("Box")

[node name="Obstacle4" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 28, 0, 28)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle4"]
shape = SubResource("Box")

[node name="RoverAgent" type="Node3D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
sensor_path = NodePath("../AgentBody/RaycastSensor3D")

[node name="SmokeChecker" type="Node" parent="."]
script = ExtResource("4")
game_path = NodePath("..")
agent_path = NodePath("../RoverAgent")
frames_to_run = 180
expected_obs_size = 8
action_count = 4
```

- [ ] **Step 3: Run the smoke scene directly**

Run: `godot --headless --path . res://test/integration/rover_smoke_scene.tscn`
Expected: prints `ROVER SMOKE PASSED (180 frames)` and exits 0.

If it fails, STOP and report the exact `ROVER SMOKE FAILED:` reason (do not weaken the checker).

- [ ] **Step 4: Wire it into `test/run_tests.sh`**

In `test/run_tests.sh`, after the existing `Trained chase check` block (the line running `res://test/integration/trained_chase_scene.tscn`), add:

```bash
echo "== Rover 3D smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/rover_smoke_scene.tscn
```

- [ ] **Step 5: Run the full suite**

Run: `./test/run_tests.sh`
Expected: `All tests passed.` (now including `ROVER SMOKE PASSED`).

- [ ] **Step 6: Commit**

```bash
git add test/integration/rover_smoke_checker.gd test/integration/rover_smoke_scene.tscn test/run_tests.sh
git commit -m "test: rover_3d headless smoke scene (real raycasts + obs pipeline) wired into run_tests.sh"
```

---

## Task 8: Training scripts + README pointer

**Files:**
- Create: `scripts/train_rover.py`
- Create: `scripts/train_rover.sh`
- Modify: `README.md`

- [ ] **Step 1: Create `scripts/train_rover.py`** (clone of `scripts/train_chase.py` with rover paths)

```python
#!/usr/bin/env python3
"""Train the 3D Rover agent with Stable-Baselines3 PPO over the godot-rl bridge.

Run this FIRST (it opens the server on port 11008 and waits), THEN launch the Godot
training scene which connects as the client. See scripts/train_rover.sh for orchestration.
"""
import argparse
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model_path", type=str, default="models/rover_policy.zip")
    parser.add_argument("--onnx_export_path", type=str, default="models/rover_policy.onnx")
    args = parser.parse_args()

    # env_path=None => in-editor training: opens the server and waits for a Godot client.
    env = StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    # Note: do NOT pass seed= to PPO — StableBaselinesGodotEnv.seed() raises
    # NotImplementedError. The env seed is set via the env constructor above.
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        tensorboard_log="logs/sb3",
    )
    model.learn(args.timesteps)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    onnx_path = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
    export_model_as_onnx(model, str(onnx_path))
    print("Exported ONNX to:", onnx_path)

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/train_rover.sh`** (mirror `scripts/train_chase.sh`)

First read `scripts/train_chase.sh` to copy its exact orchestration, then create `scripts/train_rover.sh` identical except: the Godot scene is `res://examples/rover_3d/rover_3d_train.tscn`, the python entry is `scripts/train_rover.py`, and any `TIMESTEPS` default mirrors chase's pattern. Make it executable:

```bash
chmod +x scripts/train_rover.sh
```

- [ ] **Step 3: Add a README pointer**

In `README.md`, in the `## Examples` section (after the chase entry), add:

```markdown
### 3D Raycast Rover

A tank-steered 3D rover (`examples/rover_3d/`) that uses a `RaycastSensor3D` to avoid a fixed
obstacle field and reach a goal it senses egocentrically. Demonstrates `NcnnAIController3D` +
`RaycastSensor3D` + declarative `RewardBuilder`/`RewardAdapter` reward. Discrete tank actions
(`idle / forward / turn-left / turn-right`); observation = 5 ray closeness values + `[sin, cos]`
of the goal bearing + normalized distance. Train with `scripts/train_rover.sh`; the headless
smoke test (`test/integration/rover_smoke_scene.tscn`) exercises the full obs + physics-raycast
pipeline. *(The pre-trained ncnn model + golden regression land in a follow-up training step.)*
```

- [ ] **Step 4: Run the full suite (no behavior change → still green)**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 5: Commit**

```bash
git add scripts/train_rover.py scripts/train_rover.sh README.md
git commit -m "feat: rover training scripts (train_rover.py/.sh) + README example pointer"
```

---

## Deferred final step (separate, after this scaffold merges)

Run the real training → convert → ship → regress loop (matching chase's bar), as its own task:
1. `TIMESTEPS=400000 ./scripts/train_rover.sh` (launch trainer + headless `rover_3d_train.tscn`).
2. `.venv-train/bin/python scripts/export_to_ncnn.py models/rover_policy.onnx` → `models/rover_policy.ncnn.{param,bin}` (copy into `examples/rover_3d/models/`).
3. Add a `trained_rover_scene.tscn` (agent `control_mode = 3` NCNN_INFERENCE pointing at the model) + a golden-inference regression, wired into `run_tests.sh`, mirroring the trained-chase check.
4. Optional: `docs/examples/rover_3d_tutorial.md` paralleling the chase tutorial.

---

## Self-review notes (author)

- **Spec coverage:** RoverGame pure helpers (T1) + runtime/signals (T2); RoverAgent pure helpers (T3) + wiring incl. RewardBuilder + two RewardAdapters for `goal_reached`/`bumped` (T4); play scene (T5); training scene (T6); smoke checker+scene wired into run_tests.sh, which also serves as the real-physics raycast verification (T7); training scripts + README (T8); trained model + golden regression explicitly deferred.
- **Deviation from spec:** the standalone `rover_3d.tscn` is built now (T5) as both the play/world scene; the smoke scene (T7) re-authors the world (matching chase's per-scene authoring rather than scene-instancing — lower authoring risk). Egocentric `bearing_to` lives on `RoverGame` (tested) and `compute_goal_obs` takes the precomputed bearing (kept trivially pure) — a small, DRY refinement of the spec's "compute_goal_obs(agent_pos, …)" signature.
- **Type/name consistency:** `RoverGame` API (`clamp_to_bounds`, `is_blocked(pos, obs)`, `max_distance`, `bearing_to`, `random_free_position`, `read_obstacles`, `move_agent(forward, yaw_delta, delta)`, `relocate_goal`, `reset_positions`, signals `goal_reached`/`bumped`) and `RoverAgent` API (`action_index_to_motion`, `compute_goal_obs(bearing, dist, max_dist)`, `compose_obs`, `get_action_space`, `get_obs`, `set_action`, `_action_index`) are consistent across tasks and scenes. Obs size = 5 rays + 3 goal = 8 everywhere. Action count = 4 everywhere.
- **Placeholders:** none — every code/scene/command step is concrete.
```
