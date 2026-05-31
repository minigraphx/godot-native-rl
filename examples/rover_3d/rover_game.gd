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
		result.append({"center": child.global_position, "half_extent": half})
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
