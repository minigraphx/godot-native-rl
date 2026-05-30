class_name ChaseGame
extends Node2D

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0  ## read by the agent controller to scale velocity
@export var touch_radius := 40.0  ## read by the reward logic to detect a catch
@export var agent_body_path: NodePath
@export var target_path: NodePath

signal target_caught  ## emitted when the target is caught and relocated

var _rng := RandomNumberGenerator.new()
var _agent_body: Node2D
var _target: Node2D
var catches := 0

func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path) as Node2D
	_target = get_node_or_null(target_path) as Node2D
	reset_positions()

# --- Pure helpers (unit-tested) ---
func clamp_to_bounds(pos: Vector2) -> Vector2:
	return Vector2(clampf(pos.x, 0.0, arena_size.x), clampf(pos.y, 0.0, arena_size.y))

func max_distance() -> float:
	return arena_size.length()

## s must be a non-negative integer (RandomNumberGenerator.seed is uint64; negatives wrap).
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
	catches += 1
	if _target != null:
		_target.position = random_position()
	target_caught.emit()

func reset_positions() -> void:
	if _agent_body != null:
		_agent_body.position = random_position()
	if _target != null:
		_target.position = random_position()
