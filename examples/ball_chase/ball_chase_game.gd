class_name BallChaseGame
extends Node2D
# Minimal continuous-control env: a 2D agent applies continuous thrust toward a target.
# Structurally "chase_the_target, but continuous" — same arena/target/reaches, continuous action.

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0  ## agent scales the [-1,1] thrust by this
@export var touch_radius := 40.0  ## reach detection radius
@export var agent_body_path: NodePath
@export var target_path: NodePath
@export var rng_seed := -1  ## >= 0 seeds the RNG at _ready for reproducible runs; -1 leaves it random

signal target_caught  ## emitted when the target is reached and relocated

var _rng := RandomNumberGenerator.new()
var _agent_body: Node2D
var _target: Node2D
var reaches := 0
# Headless/test fallbacks: used by get_*_pos when no child Node2D is attached (the
# `_agent_body == null` / `_target == null` guards select these).
var _agent_pos_override := Vector2.ZERO
var _target_pos_override := Vector2.ZERO

func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path) as Node2D
	_target = get_node_or_null(target_path) as Node2D
	if rng_seed >= 0:
		seed_rng(rng_seed)
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

# --- Runtime + test accessors ---
func set_agent_pos_for_test(p: Vector2) -> void:
	_agent_pos_override = p

func set_target_pos_for_test(p: Vector2) -> void:
	_target_pos_override = p

func get_agent_pos() -> Vector2:
	if _agent_body != null:
		return _agent_body.position
	return _agent_pos_override

func get_target_pos() -> Vector2:
	if _target != null:
		return _target.position
	return _target_pos_override

func distance() -> float:
	return get_agent_pos().distance_to(get_target_pos())

func move_agent(velocity: Vector2, delta: float) -> void:
	var new_pos := clamp_to_bounds(get_agent_pos() + velocity * delta)
	if _agent_body != null:
		_agent_body.position = new_pos
	else:
		_agent_pos_override = new_pos

func relocate_target() -> void:
	reaches += 1
	if _target != null:
		_target.position = random_position()
	else:
		_target_pos_override = random_position()
	target_caught.emit()

func reset_positions() -> void:
	if _agent_body != null:
		_agent_body.position = random_position()
	if _target != null:
		_target.position = random_position()

# Lightweight visualizer for the standalone deploy scene. Gameplay remains Node2D-only, so these
# draw calls do not affect observations, physics, training scenes, or headless execution.
func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.08, 0.12, 0.17), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.28, 0.42, 0.55), false, 2.0)
	draw_circle(get_target_pos(), touch_radius, Color(0.95, 0.55, 0.20))
	draw_circle(get_agent_pos(), 16.0, Color(0.25, 0.85, 0.95))
