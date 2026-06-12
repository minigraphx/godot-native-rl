class_name BallBalanceGame
extends Node3D
# Unity 3DBall parity (#47): a tilting platform balances a ball. The agent tilts around X/Z
# (2 continuous rates); the episode ends when the ball falls off. Platform is an
# AnimatableBody3D — kinematic bodies impart contact motion to the ball correctly (a moved
# StaticBody3D does not; a RigidBody3D would fight direct rotation control). Jolt backend.

@export var platform_path: NodePath
@export var ball_path: NodePath
@export var tilt_speed := 1.5        ## rad/s at full action deflection
@export var max_tilt := 0.35         ## rad (~20°), Unity-like clamp
@export var platform_half_extent := 2.5
@export var fall_margin := 1.0       ## beyond half-extent (xz) or below (y) counts as fallen
@export var spawn_height := 1.2
@export var spawn_jitter := 0.8      ## random initial xz offset so episodes differ

var _platform: Node3D
var _ball: RigidBody3D
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_platform = get_node_or_null(platform_path)
	_ball = get_node_or_null(ball_path)
	reset_episode()

# --- Pure helpers (unit-tested) ---
static func clamp_tilt(rot: float, limit: float) -> float:
	return clampf(rot, -limit, limit)

static func fallen(rel_pos: Vector3, half_extent: float, margin: float) -> bool:
	if rel_pos.y < -margin:
		return true
	return absf(rel_pos.x) > half_extent + margin or absf(rel_pos.z) > half_extent + margin

static func assemble_obs(tilt: Vector2, rel_pos: Vector3, vel: Vector3) -> Array:
	return [tilt.x, tilt.y, rel_pos.x, rel_pos.y, rel_pos.z, vel.x, vel.y, vel.z]

# --- Runtime surface ---
func seed_rng(s: int) -> void:
	_rng.seed = s

func apply_tilt(rates: Vector2, delta: float) -> void:
	if _platform == null:
		return
	_platform.rotation.x = clamp_tilt(_platform.rotation.x + rates.x * tilt_speed * delta, max_tilt)
	_platform.rotation.z = clamp_tilt(_platform.rotation.z + rates.y * tilt_speed * delta, max_tilt)

func platform_tilt() -> Vector2:
	if _platform == null:
		return Vector2.ZERO
	return Vector2(_platform.rotation.x, _platform.rotation.z)

func relative_ball_pos() -> Vector3:
	if _ball == null or _platform == null:
		return Vector3.ZERO
	return _ball.global_position - _platform.global_position

func ball_velocity() -> Vector3:
	return _ball.linear_velocity if _ball != null else Vector3.ZERO

func is_fallen() -> bool:
	return fallen(relative_ball_pos(), platform_half_extent, fall_margin)

func get_obs_array() -> Array:
	return assemble_obs(platform_tilt(), relative_ball_pos(), ball_velocity())

func reset_episode() -> void:
	if _platform != null:
		_platform.rotation = Vector3.ZERO
	if _ball != null:
		_ball.linear_velocity = Vector3.ZERO
		_ball.angular_velocity = Vector3.ZERO
		var off := Vector3(_rng.randf_range(-spawn_jitter, spawn_jitter), spawn_height,
			_rng.randf_range(-spawn_jitter, spawn_jitter))
		_ball.global_position = (_platform.global_position if _platform != null else Vector3.ZERO) + off
