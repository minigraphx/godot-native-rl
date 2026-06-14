class_name CrowdChaseAgent
extends Node2D
# A lightweight chaser unit for the batched-crowd example. Self-contained world state — its own
# arena-local position (_arena_pos) + target — with NO per-agent NcnnRunner (the parent
# CrowdController owns the one shared net and drives inference for the whole crowd). Implements the
# duck-typed agent contract (get_obs/get_action_space/set_action) the controller discovers.
#
# Node2D.position is the unit's TILE OFFSET in the grid (set by ChaseCrowdGame._layout) and is never
# touched by the physics step; the chaser moves within its own local arena via _arena_pos, drawn
# relative to the tile origin. Keeping the two separate is what lets the per-unit arenas tile side
# by side instead of collapsing onto each other.

const ChaseObs = preload("res://examples/chase_the_target/chase_obs.gd")

# Emitted once per batched decision by the parent CrowdController (which owns the shared net), so the
# PolicyDebugOverlay can auto-discover and render this unit. Same payload shape as the single-agent
# controllers. Inert when nothing listens. (#232)
signal inference_step(debug: Dictionary)

const ACTION_KEY := "move"
const ACTION_COUNT := 5

@export var arena_size := Vector2(280.0, 200.0)  # per-unit local arena (tiled by the game)
@export var move_speed := 120.0
@export var touch_radius := 16.0

var _rng := RandomNumberGenerator.new()
var _arena_pos := Vector2.ZERO   # chaser position WITHIN this unit's local arena (not the tile offset)
var _target_pos := Vector2.ZERO
var _action_index := 0
var catches := 0

func _ready() -> void:
	_rng.randomize()
	_reset_positions()

func _reset_positions() -> void:
	_arena_pos = _random_local()
	_target_pos = _random_local()

func _random_local() -> Vector2:
	return Vector2(_rng.randf_range(0.0, arena_size.x), _rng.randf_range(0.0, arena_size.y))

func get_unit_pos() -> Vector2:
	return _arena_pos

func get_target_pos() -> Vector2:
	return _target_pos

# --- duck-typed agent contract (read/written by CrowdController) ---
func get_obs() -> Dictionary:
	return {"obs": ChaseObs.compute_obs(_arena_pos, _target_pos, arena_size)}

func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func set_action(action) -> void:
	_action_index = int(action[ACTION_KEY])

# Advance this unit's world by one step using the last decided action. Called from the game's
# _physics_process (the controller decides; the unit moves). Relocates the target on a catch.
func apply_step(delta: float) -> void:
	var velocity := ChaseObs.action_index_to_velocity(_action_index, move_speed)
	_arena_pos = Vector2(
		clampf(_arena_pos.x + velocity.x * delta, 0.0, arena_size.x),
		clampf(_arena_pos.y + velocity.y * delta, 0.0, arena_size.y))
	if _arena_pos.distance_to(_target_pos) < touch_radius:
		catches += 1
		_target_pos = _random_local()
	queue_redraw()  # per-CanvasItem: the unit must request its own redraw to animate (the game's
	# _process redraw only covers the game node's own _draw, not each unit's)

func _draw() -> void:
	# Drawn in local space; Node2D.position (the tile offset) places this whole arena in the grid.
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.10, 0.11, 0.15), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.25, 0.27, 0.36), false, 1.0)
	draw_circle(_target_pos, touch_radius, Color(0.92, 0.33, 0.33))
	draw_circle(_arena_pos, 8.0, Color(0.30, 0.80, 1.0))
