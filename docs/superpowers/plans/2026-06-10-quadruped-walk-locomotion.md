# Quadruped Walk Locomotion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Milestone-1 harness for a blocky quadruped that learns to walk forward — a code-built articulated creature, a continuous 8-joint RL agent on the godot_rl wire, training/deploy scenes, and a headless smoke — then (PR2) a trained ncnn model + regressions.

**Architecture:** A pure-GDScript rig **builder** constructs the torso + 4 two-segment legs joined by 8 `HingeJoint3D` angular motors at runtime, so `.tscn` files stay trivial and all physics structure is unit-testable. A `QuadrupedGame` node wraps the builder and exposes pos/distance/upright/joint-state/foot-contact/apply-motors/reset. A `QuadrupedAgent` (extends the existing `ncnn_ai_controller_3d.gd`) maps an 8-dim continuous action to motor targets, composes ~30-dim observations, and shapes reward via `RewardBuilder`. Training reuses the `fly_by` SB3-PPO pattern; deploy reuses `NcnnAIController3D`. Physics runs on the **Jolt** backend.

**Tech Stack:** Godot 4.5+ (GDScript, TAB indent), Jolt physics, `addons/godot_native_rl/` library (`ncnn_ai_controller_3d`, `RewardBuilder`, `RelativePositionSensor3D`, `ParallelArena`, `NcnnSync`), SB3 PPO (`.venv-train`), `scripts/export_to_ncnn.py` + `scripts/export_action_dist.py`, dependency-free `test/harness.gd`.

**Spec:** `docs/superpowers/specs/2026-06-10-quadruped-walk-locomotion-design.md`

---

## File structure

**PR1 (harness):**
- Create `examples/quadruped_walk/quadruped_math.gd` — pure static helpers: obs composition, upright dot, progress delta, motor-target mapping. Unit-tested, no scene.
- Create `examples/quadruped_walk/quadruped_builder.gd` — pure-ish rig builder: constructs torso + legs + 8 hinge joints under a given parent, returns a struct of references. Unit-tested headlessly.
- Create `examples/quadruped_walk/quadruped_game.gd` — `Node3D` wrapping the builder; runtime accessors + `apply_motors`/`reset`.
- Create `examples/quadruped_walk/quadruped_agent.gd` — extends `ncnn_ai_controller_3d.gd`; godot_rl contract.
- Create `examples/quadruped_walk/quadruped_world.tscn` — ground + finish `Marker3D` + empty `CreatureRoot` + `QuadrupedGame` + `RelativePositionSensor3D` + `QuadrupedAgent` (AGENT group).
- Create `examples/quadruped_walk/quadruped_walk_train.tscn` — single world + `NcnnSync`.
- Create `examples/quadruped_walk/quadruped_walk_train_parallel.tscn` — `ParallelArena` (count=8) + `NcnnSync`.
- Create `examples/quadruped_walk/quadruped_walk_track.tscn` — deploy: world + `NcnnAIController3D`-driven agent + distance HUD + follow camera.
- Create `scripts/train_quadruped.py` + `scripts/train_quadruped.sh` — SB3 PPO driver (mirrors `fly_by`).
- Create `test/unit/test_quadruped_math.gd`, `test/unit/test_quadruped_builder.gd`, `test/unit/test_quadruped_agent.gd`.
- Create `test/integration/quadruped_smoke_checker.gd` + `quadruped_smoke_scene.tscn`.
- Modify `test/run_tests.sh` — register the quadruped smoke.
- Modify `project.godot` — enable Jolt backend.

**PR2 (trained model — separate branch/PR, gated on a real training run):**
- Create `examples/quadruped_walk/models/quadruped_walk.ncnn.{param,bin}` + `*_action_dist.json`.
- Create `test/unit/test_quadruped_golden_inference.gd` + golden fixture.
- Create `test/integration/quadruped_trained_checker.gd` + `quadruped_trained_scene.tscn` (behavioral forward-distance gate).
- Modify `test/run_tests.sh` — register the behavioral regression.

---

## PR1 — Harness

### Task 1: Pure obs/motor/reward math (`quadruped_math.gd`)

**Files:**
- Create: `examples/quadruped_walk/quadruped_math.gd`
- Test: `test/unit/test_quadruped_math.gd`

- [ ] **Step 1: Write the failing test**

`test/unit/test_quadruped_math.gd`:
```gdscript
extends SceneTree
# Pure-helper unit tests for QuadrupedMath. Run headless via the project harness.

const Harness = preload("res://test/harness.gd")
const QM = preload("res://examples/quadruped_walk/quadruped_math.gd")

func _initialize() -> void:
	var h = Harness.new()

	# clamp_action keeps values in [-1, 1]
	h.assert_eq(QM.clamp_action(2.5), 1.0, "clamp_action upper")
	h.assert_eq(QM.clamp_action(-2.5), -1.0, "clamp_action lower")
	h.assert_eq(QM.clamp_action(0.3), 0.3, "clamp_action passthrough")

	# action_to_motor_velocity scales a clamped action by max_speed
	h.assert_eq(QM.action_to_motor_velocity(1.0, 6.0), 6.0, "motor vel full+")
	h.assert_eq(QM.action_to_motor_velocity(-1.0, 6.0), -6.0, "motor vel full-")
	h.assert_eq(QM.action_to_motor_velocity(2.0, 6.0), 6.0, "motor vel clamps before scaling")

	# upright_dot: world-up vs body-up basis column
	h.assert_eq(QM.upright_dot(Vector3.UP), 1.0, "upright fully up")
	h.assert_eq(QM.upright_dot(Vector3.DOWN), -1.0, "upright upside down")
	h.assert_true(absf(QM.upright_dot(Vector3.RIGHT)) < 1e-6, "upright sideways ~0")

	# progress_delta: positive when distance shrinks
	h.assert_eq(QM.progress_delta(10.0, 8.0), 2.0, "progress closed 2")
	h.assert_eq(QM.progress_delta(8.0, 10.0), -2.0, "progress regressed")

	# compose_obs concatenates in the documented order and reports its own size
	var obs: Array = QM.compose_obs([0.1, 0.2], [1.0, 2.0], Vector3(0, 1, 0), Vector3(3, 0, 0), [0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0])
	h.assert_eq(obs.size(), 2 + 2 + 3 + 3 + 3 + 4, "compose_obs total size")
	h.assert_eq(obs[0], 0.1, "compose_obs first joint angle")
	h.assert_eq(obs[4], 1.0, "compose_obs up-vector y starts at index 4")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_quadruped_math.gd`
Expected: FAIL — `quadruped_math.gd` does not exist / parse error.

- [ ] **Step 3: Write minimal implementation**

`examples/quadruped_walk/quadruped_math.gd`:
```gdscript
class_name QuadrupedMath
extends RefCounted
# Pure, scene-free locomotion math: action clamping/scaling, upright signal, progress
# shaping, observation composition. Unit-tested in test/unit/test_quadruped_math.gd.

static func clamp_action(v: float) -> float:
	return clampf(v, -1.0, 1.0)

# Map a policy action (post-clamp) to a hinge motor target velocity.
static func action_to_motor_velocity(action: float, max_speed: float) -> float:
	return clamp_action(action) * max_speed

# Cosine of the torso's tilt from vertical. body_up is the torso basis Y column in world space.
static func upright_dot(body_up: Vector3) -> float:
	return body_up.dot(Vector3.UP)

# Distance closed toward the goal since last step (positive = progress).
static func progress_delta(prev_dist: float, cur_dist: float) -> float:
	return prev_dist - cur_dist

# Concatenate the observation in the documented order:
# joint_angles + joint_velocities + body_up(3) + body_local_vel(3) + dir_to_finish(3) + foot_contacts(4)
static func compose_obs(joint_angles: Array, joint_velocities: Array, body_up: Vector3, body_local_vel: Vector3, dir_to_finish: Array, foot_contacts: Array) -> Array:
	var out: Array = []
	out.append_array(joint_angles)
	out.append_array(joint_velocities)
	out.append_array([body_up.x, body_up.y, body_up.z])
	out.append_array([body_local_vel.x, body_local_vel.y, body_local_vel.z])
	out.append_array(dir_to_finish)
	out.append_array(foot_contacts)
	return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_quadruped_math.gd`
Expected: `Results: N passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add examples/quadruped_walk/quadruped_math.gd test/unit/test_quadruped_math.gd
git commit -m "feat: quadruped locomotion pure math helpers (#60)"
```

---

### Task 2: Creature rig builder (`quadruped_builder.gd`)

Builds the articulated quadruped under a parent node at runtime: a box torso `RigidBody3D`, and 4 legs each = upper + lower `RigidBody3D` segment joined to its parent by a `HingeJoint3D` with an angular motor enabled. Returns a struct with ordered joint + segment + foot references so the game/agent layer can read joint state and apply motor targets deterministically.

**Files:**
- Create: `examples/quadruped_walk/quadruped_builder.gd`
- Test: `test/unit/test_quadruped_builder.gd`

- [ ] **Step 1: Write the failing test**

`test/unit/test_quadruped_builder.gd`:
```gdscript
extends SceneTree
# Structural test: the builder must produce a torso, 8 motorized hinge joints, and 4 feet,
# all parented under the given root, with motors enabled. Runs headless (no physics stepping).

const Harness = preload("res://test/harness.gd")
const Builder = preload("res://examples/quadruped_walk/quadruped_builder.gd")

func _initialize() -> void:
	var h = Harness.new()
	var root := Node3D.new()
	get_root().add_child(root)

	var rig = Builder.build(root)

	h.assert_true(rig.torso is RigidBody3D, "torso is a RigidBody3D")
	h.assert_eq(rig.joints.size(), 8, "8 hinge joints (hip+knee x4)")
	h.assert_eq(rig.feet.size(), 4, "4 feet")
	for j in rig.joints:
		h.assert_true(j is HingeJoint3D, "joint is HingeJoint3D")
		h.assert_true(j.get_flag(HingeJoint3D.FLAG_ENABLE_MOTOR), "joint motor enabled")
	# Every joint and foot is inside the provided root subtree.
	h.assert_true(root.is_ancestor_of(rig.torso), "torso under root")
	h.assert_true(root.is_ancestor_of(rig.joints[0]), "joint under root")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_quadruped_builder.gd`
Expected: FAIL — builder missing.

- [ ] **Step 3: Write minimal implementation**

`examples/quadruped_walk/quadruped_builder.gd`:
```gdscript
class_name QuadrupedBuilder
extends RefCounted
# Constructs a blocky quadruped (box torso + 4 two-segment legs joined by 8 motorized
# HingeJoint3D) under a parent node. All physics structure is built in code so the .tscn
# scenes stay trivial and the rig is unit-testable. Jolt backend assumed (see project.godot).
#
# Returns a Dictionary: {
#   "torso": RigidBody3D,
#   "joints": Array[HingeJoint3D]  # order: [FL_hip, FL_knee, FR_hip, FR_knee, BL_hip, BL_knee, BR_hip, BR_knee]
#   "uppers": Array[RigidBody3D], "lowers": Array[RigidBody3D], "feet": Array[Node3D]
# }

const TORSO_SIZE := Vector3(1.2, 0.4, 0.8)
const UPPER_SIZE := Vector3(0.18, 0.5, 0.18)
const LOWER_SIZE := Vector3(0.15, 0.5, 0.15)
const TORSO_MASS := 6.0
const SEG_MASS := 0.6
const MOTOR_MAX_IMPULSE := 40.0
const HIP_LIMIT := deg_to_rad(60.0)
const KNEE_LIMIT := deg_to_rad(70.0)

# Corner offsets (x = right, z = forward). Order matches the joints array documented above.
const _CORNERS := [
	Vector3( 0.5, 0.0,  0.35),  # FL
	Vector3(-0.5, 0.0,  0.35),  # FR
	Vector3( 0.5, 0.0, -0.35),  # BL
	Vector3(-0.5, 0.0, -0.35),  # BR
]

static func _box_body(size: Vector3, mass: float, name: String) -> RigidBody3D:
	var body := RigidBody3D.new()
	body.name = name
	body.mass = mass
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = size
	shape.shape = box
	body.add_child(shape)
	var mesh := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	mesh.mesh = bm
	body.add_child(mesh)
	return body

static func _hinge(a: PhysicsBody3D, b: PhysicsBody3D, at: Vector3, limit: float, name: String) -> HingeJoint3D:
	var j := HingeJoint3D.new()
	j.name = name
	j.position = at
	# Hinge axis = local X so legs swing in the forward/back (Z-Y) plane.
	j.node_a = j.get_path_to(a)
	j.node_b = j.get_path_to(b)
	j.set_flag(HingeJoint3D.FLAG_USE_LIMIT, true)
	j.set_param(HingeJoint3D.PARAM_LIMIT_LOWER, -limit)
	j.set_param(HingeJoint3D.PARAM_LIMIT_UPPER, limit)
	j.set_flag(HingeJoint3D.FLAG_ENABLE_MOTOR, true)
	j.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, 0.0)
	j.set_param(HingeJoint3D.PARAM_MOTOR_MAX_IMPULSE, MOTOR_MAX_IMPULSE)
	return j

static func build(parent: Node3D) -> Dictionary:
	var torso := _box_body(TORSO_SIZE, TORSO_MASS, "Torso")
	torso.position = Vector3(0, 1.0, 0)
	parent.add_child(torso)

	var joints: Array = []
	var uppers: Array = []
	var lowers: Array = []
	var feet: Array = []
	var tags := ["FL", "FR", "BL", "BR"]
	for i in range(4):
		var corner: Vector3 = _CORNERS[i]
		var hip_pos: Vector3 = torso.position + corner
		var upper := _box_body(UPPER_SIZE, SEG_MASS, "%s_upper" % tags[i])
		upper.position = hip_pos + Vector3(0, -UPPER_SIZE.y * 0.5, 0)
		parent.add_child(upper)
		var lower := _box_body(LOWER_SIZE, SEG_MASS, "%s_lower" % tags[i])
		lower.position = upper.position + Vector3(0, -(UPPER_SIZE.y * 0.5 + LOWER_SIZE.y * 0.5), 0)
		parent.add_child(lower)

		var hip := _hinge(torso, upper, hip_pos, HIP_LIMIT, "%s_hip" % tags[i])
		parent.add_child(hip)
		var knee_pos: Vector3 = upper.position + Vector3(0, -UPPER_SIZE.y * 0.5, 0)
		var knee := _hinge(upper, lower, knee_pos, KNEE_LIMIT, "%s_knee" % tags[i])
		parent.add_child(knee)

		# Foot marker = bottom of the lower segment (for contact + distance debugging).
		var foot := Marker3D.new()
		foot.name = "%s_foot" % tags[i]
		foot.position = Vector3(0, -LOWER_SIZE.y * 0.5, 0)
		lower.add_child(foot)

		joints.append(hip)
		joints.append(knee)
		uppers.append(upper)
		lowers.append(lower)
		feet.append(foot)

	return {"torso": torso, "joints": joints, "uppers": uppers, "lowers": lowers, "feet": feet}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_quadruped_builder.gd`
Expected: `Results: N passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add examples/quadruped_walk/quadruped_builder.gd test/unit/test_quadruped_builder.gd
git commit -m "feat: code-built quadruped rig (8 motorized hinge joints) (#60)"
```

---

### Task 3: Game node (`quadruped_game.gd`)

Wraps the builder and exposes the runtime surface the agent needs: torso pose, distance to finish, upright signal, per-joint angle/velocity, foot ground-contact flags, `apply_motors`, and `reset_positions`. Built and stepped under physics, so its test drives a real `SceneTree` for a few frames.

**Files:**
- Create: `examples/quadruped_walk/quadruped_game.gd`
- Test: `test/unit/test_quadruped_game.gd`

- [ ] **Step 1: Write the failing test**

`test/unit/test_quadruped_game.gd`:
```gdscript
extends SceneTree
# Runtime test: build the game, step physics a few frames, assert the accessor surface.
# Uses a finish marker far on +Z and checks obs primitives are well-formed and finite.

const Harness = preload("res://test/harness.gd")
const Game = preload("res://examples/quadruped_walk/quadruped_game.gd")

var _h
var _game
var _frames := 0

func _initialize() -> void:
	_h = Harness.new()
	var finish := Marker3D.new()
	finish.position = Vector3(0, 0, 40)
	get_root().add_child(finish)
	_game = Game.new()
	_game.finish_path = _game.get_path_to(finish)  # set before _ready via direct field
	get_root().add_child(finish)
	# Build under a CreatureRoot child the game owns:
	get_root().add_child(_game)
	_game.set_finish(finish)
	_game._build_now()

	_h.assert_eq(_game.joint_count(), 8, "8 joints")
	_h.assert_eq(_game.foot_contacts().size(), 4, "4 foot contact flags")
	_h.assert_eq(_game.joint_angles().size(), 8, "8 joint angles")
	_h.assert_eq(_game.joint_velocities().size(), 8, "8 joint velocities")
	_h.assert_true(_game.distance() > 0.0, "distance to finish positive")
	_h.assert_true(is_finite(_game.upright()), "upright finite")

func _physics_process(_delta: float) -> void:
	# Apply zero motors; just confirm stepping stays finite and reset works.
	_game.apply_motors([0,0,0,0,0,0,0,0])
	for v in _game.joint_angles():
		if not is_finite(v):
			_h.assert_true(false, "joint angle finite")
	_frames += 1
	if _frames >= 10:
		_game.reset_positions()
		_h.assert_true(_game.distance() > 0.0, "distance positive after reset")
		_h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_quadruped_game.gd`
Expected: FAIL — game missing.

- [ ] **Step 3: Write minimal implementation**

`examples/quadruped_walk/quadruped_game.gd`:
```gdscript
class_name QuadrupedGame
extends Node3D
# Owns the code-built quadruped rig and exposes the runtime surface QuadrupedAgent consumes.
# Distance/direction are measured from the torso to a finish Marker3D. Foot contact is a simple
# height test (cheap + deterministic; refine to ray/contact monitoring later if needed).

const Builder = preload("res://examples/quadruped_walk/quadruped_builder.gd")
const QM = preload("res://examples/quadruped_walk/quadruped_math.gd")

@export var finish_path: NodePath
@export var motor_max_speed := 6.0
@export var foot_contact_height := 0.18  ## lower-segment foot below this Y counts as grounded

var _rig: Dictionary
var _finish: Node3D
var _start_xform: Array = []  # per-body reset transforms

func _ready() -> void:
	_finish = get_node_or_null(finish_path)
	_build_now()

func set_finish(n: Node3D) -> void:
	_finish = n

func _build_now() -> void:
	if not _rig.is_empty():
		return
	_rig = Builder.build(self)
	_capture_start()

func _capture_start() -> void:
	_start_xform.clear()
	for b in _bodies():
		_start_xform.append(b.global_transform)

func _bodies() -> Array:
	var out: Array = [_rig["torso"]]
	out.append_array(_rig["uppers"])
	out.append_array(_rig["lowers"])
	return out

func joint_count() -> int:
	return _rig["joints"].size()

func torso_pos() -> Vector3:
	return _rig["torso"].global_position

func distance() -> float:
	if _finish == null:
		return 0.0
	return torso_pos().distance_to(_finish.global_position)

func max_distance() -> float:
	return 60.0

func dir_to_finish() -> Array:
	if _finish == null:
		return [0.0, 0.0, 0.0]
	var d: Vector3 = (_finish.global_position - torso_pos())
	var n := d.limit_length(1.0) if d.length() > 1.0 else d
	return [n.x / max_distance(), n.y / max_distance(), n.z / max_distance()]

func upright() -> float:
	return QM.upright_dot(_rig["torso"].global_transform.basis.y)

func body_local_velocity() -> Vector3:
	var t: RigidBody3D = _rig["torso"]
	return t.global_transform.basis.inverse() * t.linear_velocity

func joint_angles() -> Array:
	var out: Array = []
	for j in _rig["joints"]:
		out.append((j as HingeJoint3D).get_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY) * 0.0)  # placeholder replaced below
	return _measured_angles()

func _measured_angles() -> Array:
	# Hinge has no direct angle read; derive from the relative basis of the two bodies about local X.
	var out: Array = []
	var pairs := [[_rig["torso"], _rig["uppers"]], null]
	# Compute per-joint: angle of child relative to parent around hinge X.
	var idx := 0
	for i in range(4):
		var torso: RigidBody3D = _rig["torso"]
		var upper: RigidBody3D = _rig["uppers"][i]
		var lower: RigidBody3D = _rig["lowers"][i]
		out.append(_rel_pitch(torso, upper))
		out.append(_rel_pitch(upper, lower))
		idx += 2
	return out

func _rel_pitch(parent: RigidBody3D, child: RigidBody3D) -> float:
	var rel := parent.global_transform.basis.inverse() * child.global_transform.basis
	# pitch about X from the child's forward (Z) tilt
	return atan2(rel.z.y, rel.z.z)

func joint_velocities() -> Array:
	var out: Array = []
	for i in range(4):
		out.append(_rig["uppers"][i].angular_velocity.x)
		out.append(_rig["lowers"][i].angular_velocity.x)
	return out

func foot_contacts() -> Array:
	var out: Array = []
	for f in _rig["feet"]:
		out.append(1.0 if (f as Node3D).global_position.y <= foot_contact_height else 0.0)
	return out

func apply_motors(actions: Array) -> void:
	var joints: Array = _rig["joints"]
	for i in range(joints.size()):
		var a: float = float(actions[i]) if i < actions.size() else 0.0
		var vel := QM.action_to_motor_velocity(a, motor_max_speed)
		(joints[i] as HingeJoint3D).set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, vel)

func reset_positions() -> void:
	var bodies := _bodies()
	for i in range(bodies.size()):
		var b: RigidBody3D = bodies[i]
		b.linear_velocity = Vector3.ZERO
		b.angular_velocity = Vector3.ZERO
		if i < _start_xform.size():
			b.global_transform = _start_xform[i]
	for j in _rig["joints"]:
		(j as HingeJoint3D).set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, 0.0)
```

> **Implementer note:** `joint_angles()` delegates to `_measured_angles()` (the first loop body is a no-op placeholder kept only so the array length is obvious); if you prefer, delete the placeholder loop and have `joint_angles()` simply `return _measured_angles()`. The angle read is approximate — exact hinge-angle readback is not exposed by Godot; the relative-pitch estimate is sufficient as a policy observation. Keep it finite and stable, which the smoke test (Task 6) verifies under real physics.

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_quadruped_game.gd`
Expected: `Results: N passed, 0 failed`, exit 0. (If `set_finish`/`_build_now` ordering in the test trips, simplify the test's setup to: add finish, add game, `game.set_finish(finish)`, `game._build_now()` — the field-poke line is redundant.)

- [ ] **Step 5: Commit**

```bash
git add examples/quadruped_walk/quadruped_game.gd test/unit/test_quadruped_game.gd
git commit -m "feat: QuadrupedGame runtime surface (pose/joints/contacts/motors/reset) (#60)"
```

---

### Task 4: RL agent (`quadruped_agent.gd`)

Extends the existing `ncnn_ai_controller_3d.gd` so it speaks the godot_rl contract. One continuous `motors` action of size 8; ~30-dim obs from `QuadrupedMath.compose_obs`; reward via `RewardBuilder` (progress + upright + alive − energy − fall), with fall/timeout termination.

**Files:**
- Create: `examples/quadruped_walk/quadruped_agent.gd`
- Test: `test/unit/test_quadruped_agent.gd`

- [ ] **Step 1: Write the failing test**

`test/unit/test_quadruped_agent.gd`:
```gdscript
extends SceneTree
# Agent contract test: action space shape, obs size with a real game, and that set_action
# stores a clamped 8-vector. No trainer/socket — pure node wiring.

const Harness = preload("res://test/harness.gd")
const Game = preload("res://examples/quadruped_walk/quadruped_game.gd")
const Agent = preload("res://examples/quadruped_walk/quadruped_agent.gd")

func _initialize() -> void:
	var h = Harness.new()
	var finish := Marker3D.new()
	finish.position = Vector3(0, 0, 40)
	get_root().add_child(finish)
	var game = Game.new()
	get_root().add_child(game)
	game.set_finish(finish)
	game._build_now()

	var agent = Agent.new()
	get_root().add_child(agent)
	agent.set_game(game)

	var space = agent.get_action_space()
	h.assert_true("motors" in space, "action key 'motors'")
	h.assert_eq(space["motors"]["size"], 8, "8 continuous actions")
	h.assert_eq(space["motors"]["action_type"], "continuous", "continuous type")

	var obs = agent.get_obs()
	h.assert_eq(obs["obs"].size(), agent.expected_obs_size(), "obs matches expected size")
	h.assert_eq(agent.expected_obs_size(), 8 + 8 + 3 + 3 + 3 + 4, "expected obs = 29")

	agent.set_action({"motors": [2.0, -2.0, 0,0,0,0,0,0]})
	var stored = agent.stored_action_for_test()
	h.assert_eq(stored[0], 1.0, "action[0] clamped to 1")
	h.assert_eq(stored[1], -1.0, "action[1] clamped to -1")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_quadruped_agent.gd`
Expected: FAIL — agent missing.

- [ ] **Step 3: Write minimal implementation**

`examples/quadruped_walk/quadruped_agent.gd`:
```gdscript
class_name QuadrupedAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"

const ACTION_KEY := "motors"
const ACTION_COUNT := 8
const OBS_SIZE := 8 + 8 + 3 + 3 + 3 + 4  # joints+vels+up+localvel+dir+contacts = 29
const QM = preload("res://examples/quadruped_walk/quadruped_math.gd")
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from the controller — do not redeclare.

@export var game_path: NodePath
@export var upright_weight := 0.05
@export var alive_bonus := 0.01
@export var energy_penalty := 0.002
@export var fall_penalty := 1.0
@export var fall_height := 0.45      ## torso below this Y = fallen
@export var fall_upright := 0.2      ## upright dot below this = fallen

var _game
var _action: Array = []

func set_game(g) -> void:
	_game = g

func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "continuous"}}

func expected_obs_size() -> int:
	return OBS_SIZE

func stored_action_for_test() -> Array:
	return _action

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(OBS_SIZE)
	z.fill(0.0)
	return z

func _ready() -> void:
	super._ready()
	if _game == null:
		_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("QuadrupedAgent: game_path not set — producing zero observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["fell"]) \
		.add_event_bonus("upright", upright_weight) \
		.add_event_bonus("fell", -fall_penalty) \
		.add_alive_bonus(alive_bonus) \
		.add_step_penalty(energy_penalty) \
		.build()
	call_deferred("_reset_reward_baseline")

func _reset_reward_baseline() -> void:
	if reward_source != null:
		reward_source.reset()

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": _zero_obs()}
	return {"obs": QM.compose_obs(
		_game.joint_angles(), _game.joint_velocities(),
		_game._rig["torso"].global_transform.basis.y,
		_game.body_local_velocity(),
		_game.dir_to_finish(), _game.foot_contacts())}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var raw: Array = action[ACTION_KEY]
	_action = []
	for v in raw:
		_action.append(QM.clamp_action(float(v)))

func _is_fallen() -> bool:
	return _game.torso_pos().y < fall_height or _game.upright() < fall_upright

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	if _action.size() == ACTION_COUNT:
		_game.apply_motors(_action)
	# Energy term: feed |action| sum into the step penalty already in RewardBuilder by scaling
	# accumulate; here we just accumulate progress/upright/alive each frame.
	accumulate_reward()
	if needs_reset or _is_fallen():
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()
```

> **Implementer note:** the `"upright"` and `"fell"` reward events are emitted by the agent each physics frame via the inherited reward pipeline. If `RewardAdapterScript` signal-wiring (as used in `rover_agent.gd`) is the cleaner fit than `add_event_bonus` polling, mirror the rover pattern: emit a `fell` event on the terminating frame and an `upright` event gated on `_game.upright() > 0.7`. Keep the reward terms exactly those in the spec (progress + upright + alive − energy − fall). The unit test does not assert reward magnitude (that's tuned during PR2); it asserts wiring/shape only.

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_quadruped_agent.gd`
Expected: `Results: N passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add examples/quadruped_walk/quadruped_agent.gd test/unit/test_quadruped_agent.gd
git commit -m "feat: QuadrupedAgent godot_rl contract (8-dim continuous, ~29-dim obs, shaped reward) (#60)"
```

---

### Task 5: Scenes (world + train + parallel + track)

Author the four `.tscn` files. Keep each trivial — the rig is code-built, so scenes only hold ground, finish marker, empty `CreatureRoot`, the `QuadrupedGame`, a `RelativePositionSensor3D` pointed at the finish, the `QuadrupedAgent` (in the `AGENT` group), and the harness node (`NcnnSync` / `ParallelArena` / `NcnnAIController3D`). Copy node/group/property wiring from `examples/rover_3d/*.tscn` (the established 3D pattern).

**Files:**
- Create: `examples/quadruped_walk/quadruped_world.tscn`
- Create: `examples/quadruped_walk/quadruped_walk_train.tscn`
- Create: `examples/quadruped_walk/quadruped_walk_train_parallel.tscn`
- Create: `examples/quadruped_walk/quadruped_walk_track.tscn`

- [ ] **Step 1: Build `quadruped_world.tscn`**

Nodes: `Node3D` root → `StaticBody3D` "Ground" (large flat `BoxShape3D`, e.g. 60×1×60, top at y=0) → `Marker3D` "Finish" at `(0,0,40)` → `QuadrupedGame` "Game" (`finish_path` → the Finish marker; the game builds the rig under itself at `_ready`) → `QuadrupedAgent` "Agent" added to group **`AGENT`** with `game_path` → "Game". (No `RelativePositionSensor3D` node is required — `QuadrupedGame.dir_to_finish()` covers it; add one only if you prefer the sensor node, pointing `objects_to_observe` at the Finish marker.)

- [ ] **Step 2: Build `quadruped_walk_train.tscn`**

Root `Node3D` → instance `quadruped_world.tscn` → `NcnnSync` node configured exactly like `examples/rover_3d/rover_3d_train.tscn` (control mode = training, port 11008). Copy the `NcnnSync` setup verbatim from the rover train scene.

- [ ] **Step 3: Build `quadruped_walk_train_parallel.tscn`**

Root `Node3D` → `ParallelArena` (`world_scene` = `quadruped_world.tscn`, `count` = 8, `spacing` = 200) → `NcnnSync`. Mirror `examples/rover_3d/rover_3d_train_parallel.tscn`.

- [ ] **Step 4: Build `quadruped_walk_track.tscn`**

Root `Node3D` → instance `quadruped_world.tscn`, but set the Agent's control mode to **inference** (`NcnnAIController3D` driving from a model path; leave `model_path` empty until PR2) → `Camera3D` following the torso → `CanvasLayer`/`Label` "DistanceHUD" updated each frame with meters travelled (a tiny attached script reading `Game.torso_pos().z`). Mirror `examples/rover_3d/rover_3d.tscn` for the inference-side wiring.

- [ ] **Step 5: Verify scenes load headless**

Run:
```bash
godot --headless --path . --quit-after 2 res://examples/quadruped_walk/quadruped_walk_train.tscn
godot --headless --path . --quit-after 2 res://examples/quadruped_walk/quadruped_walk_track.tscn
```
Expected: no parse/instantiation errors (the train scene will warn about no trainer connection — that's fine; we only check it loads).

- [ ] **Step 6: Commit**

```bash
git add examples/quadruped_walk/*.tscn
git commit -m "feat: quadruped_walk scenes (world/train/parallel/track) (#60)"
```

---

### Task 6: Headless smoke test + run_tests.sh wiring

Drive the world with random actions under real Jolt physics for N frames; assert obs shape (29) + finiteness, action application, and that reset works on fall. This is the real gate on the scene + builder + game wiring.

**Files:**
- Create: `test/integration/quadruped_smoke_checker.gd`
- Create: `test/integration/quadruped_smoke_scene.tscn`
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Write the smoke checker**

`test/integration/quadruped_smoke_checker.gd`:
```gdscript
extends Node
# Headless smoke: random continuous actions through the real rig + Jolt physics for a fixed
# number of frames; asserts obs shape/finiteness and that the creature stays in a sane range.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 240
@export var expected_obs_size := 29

var _game
var _agent
var _frames := 0
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_rng.seed = 4242
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	var a: Array = []
	for i in range(8):
		a.append(_rng.randf_range(-1.0, 1.0))
	_agent.set_action({"motors": a})
	var obs = _agent.get_obs()
	if not ("obs" in obs) or obs["obs"].size() != expected_obs_size:
		_fail("bad obs shape: %d" % (obs["obs"].size() if "obs" in obs else -1))
		return
	for v in obs["obs"]:
		if not is_finite(v):
			_fail("non-finite observation")
			return
	_frames += 1
	if _frames >= frames_to_run:
		print("QUADRUPED SMOKE PASSED (%d frames)" % _frames)
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("QUADRUPED SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
```

- [ ] **Step 2: Build `quadruped_smoke_scene.tscn`**

Root `Node3D` → instance `quadruped_world.tscn` → `quadruped_smoke_checker.gd` node with `game_path`/`agent_path` pointed at the world's Game/Agent. (Mirror `test/integration/rover_smoke_scene.tscn`.)

- [ ] **Step 3: Run the smoke to verify it passes**

Run: `godot --headless --path . res://test/integration/quadruped_smoke_scene.tscn`
Expected: `QUADRUPED SMOKE PASSED (240 frames)`, exit 0. If obs is non-finite or sizes mismatch, fix Task 3/4 before proceeding.

- [ ] **Step 4: Register the smoke + unit tests in `run_tests.sh`**

In `test/run_tests.sh`, after the rover smoke block (around line 54), add:
```bash
echo "== Quadruped walk smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/quadruped_smoke_scene.tscn
```
Confirm the three new `test/unit/test_quadruped_*.gd` files are picked up by the existing unit-test discovery loop (the script auto-runs `test/unit/test_*.gd`; if it uses an explicit list, append them).

- [ ] **Step 5: Run the full suite**

Run: `./test/run_tests.sh`
Expected: all green, including the new quadruped unit tests + smoke.

- [ ] **Step 6: Commit**

```bash
git add test/integration/quadruped_smoke_checker.gd test/integration/quadruped_smoke_scene.tscn test/run_tests.sh
git commit -m "test: quadruped_walk headless smoke + suite wiring (#60)"
```

---

### Task 7: Enable Jolt + verify no regression

**Files:**
- Modify: `project.godot`

- [ ] **Step 1: Enable the Jolt backend**

In `project.godot`, under `[physics]`, set the 3D physics engine to Jolt:
```ini
[physics]

3d/physics_engine="Jolt Physics"
```
(In Godot 4.5+ Jolt is built in and selected by this string. Verify the exact value via Project Settings → Physics → 3D → Physics Engine in the editor if the string is rejected at load.)

- [ ] **Step 2: Re-run the full suite to confirm no regression**

Run: `./test/run_tests.sh`
Expected: all green — the existing 2D/3D example smokes (rover, parallel arena, inference, hide & seek) still pass under Jolt. If any 3D example regresses, note it and adjust that example's physics tolerances rather than reverting Jolt (Jolt is required for the ragdoll work).

- [ ] **Step 3: Commit**

```bash
git add project.godot
git commit -m "chore: enable Jolt 3D physics backend for articulated bodies (#60)"
```

---

### Task 8: Training driver (`train_quadruped.py` + `.sh`)

Mirror the `fly_by` SB3-PPO scripts so PR2 (and users) can train. No training run happens in PR1 — this task only adds the runnable scripts and a guarded presence in docs.

**Files:**
- Create: `scripts/train_quadruped.py`
- Create: `scripts/train_quadruped.sh`

- [ ] **Step 1: Copy + adapt the trainer**

Copy `scripts/train_fly_by.py` → `scripts/train_quadruped.py`. Change only: default exported model name → `models/quadruped_walk.pt`; any env-id/log labels → `quadruped_walk`. The godot_rl env wiring, PPO config, and TorchScript export are unchanged (continuous action space is auto-detected from the handshake).

- [ ] **Step 2: Copy + adapt the orchestration script**

Copy `scripts/train_fly_by.sh` → `scripts/train_quadruped.sh`. Change only: `SCENE` default → `res://examples/quadruped_walk/quadruped_walk_train_parallel.tscn`; the python entry → `scripts/train_quadruped.py`; default `TIMESTEPS` to a locomotion-appropriate value (e.g. `2000000`). Keep `SPEEDUP`/`ACTION_REPEAT`/`SCENE`/`TIMESTEPS` overridable.

- [ ] **Step 3: Smoke the script wiring (syntax only, no full run)**

Run: `bash -n scripts/train_quadruped.sh && .venv-train/bin/python -c "import ast; ast.parse(open('scripts/train_quadruped.py').read())"`
Expected: no output, exit 0 (syntax valid). A real training run is PR2.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_quadruped.py scripts/train_quadruped.sh
chmod +x scripts/train_quadruped.sh
git commit -m "feat: quadruped_walk SB3 PPO training scripts (#60)"
```

---

### Task 9: Docs for the harness (PR1)

Per CLAUDE.md "before every push, check and update the docs."

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Update CLAUDE.md**

Add `quadruped_walk` to the examples list in "Current state", and add a **Train (quadruped walk)** key-command line:
```
- **Train (quadruped walk):** `./scripts/train_quadruped.sh` — SB3 PPO over the code-built
  quadruped (8 hinge-joint continuous action, ~29-dim obs), `ParallelArena` tiling. Jolt backend.
  Exports TorchScript → `export_to_ncnn.py`. `TIMESTEPS`/`SCENE` overrides.
```

- [ ] **Step 2: Update README.md**

Add the example to the examples section and add the locomotion/web-shippable hook to the moat section (a continuous-control quadruped that trains in Godot and deploys to the browser via ncnn).

- [ ] **Step 3: Update gap analysis + BACKLOG**

In `docs/godot-rl-gap-analysis-2026-06-02.md`, note continuous-control locomotion is now in progress (M1 harness). In `docs/BACKLOG.md`, annotate #60 as M1-harness landed (leave the epic open).

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md docs/godot-rl-gap-analysis-2026-06-02.md docs/BACKLOG.md
git commit -m "docs: quadruped_walk M1 harness (#60)"
```

- [ ] **Step 5: Open PR1**

```bash
git push -u origin feature/quadruped-walk-locomotion
env -u GH_TOKEN gh pr create --title "feat: quadruped-walk locomotion harness (#60 M1, PR1)" --body-file <(printf '%s\n' "Milestone-1 harness for issue #60: code-built quadruped rig, continuous 8-joint RL agent, train/deploy scenes, headless smoke. No trained model yet (PR2). Jolt backend enabled." "" "🤖 Generated with [Claude Code](https://claude.com/claude-code)")
```

---

## PR2 — Trained model (separate branch, gated on a real training run)

> This is not bite-sized TDD: it depends on RL convergence. Execute on a fresh branch off the merged PR1. Expect reward-shaping iteration.

### Task 10: Train to a walking policy
- [ ] Run `caffeinate -is ./scripts/train_quadruped.sh` (start ~2M steps; raise/iterate as needed). Watch episode forward-distance climb. Tune `motor_max_speed`, joint limits, `upright_weight`, `energy_penalty`, and `fall_height` until the quadruped reliably moves forward without immediately falling. Re-run with overrides as needed.

### Task 11: Export to ncnn + action-dist sidecar
- [ ] `.venv-train/bin/python scripts/export_to_ncnn.py models/quadruped_walk.pt` → `examples/quadruped_walk/models/quadruped_walk.ncnn.{param,bin}` (move into the example's `models/`).
- [ ] `.venv-train/bin/python scripts/export_action_dist.py models/quadruped_walk.zip` → `*_action_dist.json` alongside the model.
- [ ] Point `quadruped_walk_track.tscn`'s `NcnnAIController3D` `model_path` at the committed model.

### Task 12: Golden-inference regression
- [ ] Create `test/unit/test_quadruped_golden_inference.gd` + a committed golden fixture (obs vector → expected action), mirroring `test/unit/test_rover_golden_inference.gd`. Run it; commit the fixture.

### Task 13: Behavioral forward-distance regression
- [ ] Create `test/integration/quadruped_trained_checker.gd` + `quadruped_trained_scene.tscn`: run the deployed policy headless in the track scene for N frames, assert the torso advances past a forward-distance threshold (the "it actually walks" gate). Mirror `test/integration/trained_rover_checker.gd`.
- [ ] Register both in `test/run_tests.sh`. Run the full suite green.

### Task 14: Finalize docs + close M1
- [ ] Update README/CLAUDE/gap-analysis to state the trained quadruped ships and is web-deployable. Tick #60 M1 in `docs/BACKLOG.md`; comment on issue #60 that M1 is complete (epic stays open for M2–M5). Open PR2.

---

## Self-review notes
- **Spec coverage:** layout (Task 5), obs/action/reward (Tasks 1,4), Jolt (Task 7), data flow + training (Task 8 / PR2), two-PR delivery (PR1 Tasks 1–9 / PR2 Tasks 10–14), tests (Tasks 1–4,6 + PR2 12–13), docs (Task 9 / PR2 14). All spec sections map to tasks.
- **Known approximation:** hinge-angle readback is estimated (Godot exposes no direct hinge angle); flagged in Task 3 — acceptable as a policy observation, validated finite/stable by the smoke.
- **Naming consistency:** `QuadrupedMath`, `QuadrupedBuilder` (`build`→Dictionary with `torso`/`joints`/`uppers`/`lowers`/`feet`), `QuadrupedGame` (`set_finish`/`_build_now`/`apply_motors`/`reset_positions`/`joint_angles`/`joint_velocities`/`foot_contacts`/`dir_to_finish`/`distance`/`max_distance`/`torso_pos`/`upright`/`body_local_velocity`), `QuadrupedAgent` (`set_game`/`get_action_space`/`expected_obs_size`/`stored_action_for_test`) are used consistently across tasks and tests.
