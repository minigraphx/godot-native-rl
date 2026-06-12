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

<<<<<<< HEAD
# Episode replay hooks (#39): minimal state for exact playback (kinematic + seeded game).
# rng_state is REQUIRED for exactness: relocate_target() consumes the RNG, so a replay without
# the recorded RNG state diverges after the first catch. Stored as a String — the uint64 RNG
# state does not survive JSON's float64 numbers.
func get_replay_state() -> Dictionary:
	return {"agent_x": get_agent_pos().x, "agent_y": get_agent_pos().y,
		"target_x": get_target_pos().x, "target_y": get_target_pos().y,
		"catches": catches, "rng_state": str(_rng.state)}

func apply_replay_state(state: Dictionary) -> void:
	if _agent_body != null and state.has("agent_x"):
		_agent_body.position = Vector2(float(state["agent_x"]), float(state["agent_y"]))
	if _target != null and state.has("target_x"):
		_target.position = Vector2(float(state["target_x"]), float(state["target_y"]))
	catches = int(state.get("catches", 0))
	if state.has("rng_state"):
		_rng.state = int(String(state["rng_state"]))
=======
# Curriculum hook (#28): stage params applied at episode boundaries by CurriculumController.
# Flat floats only (params arrive from JSON / the wire — no Vector2).
func apply_curriculum(params: Dictionary) -> void:
	if params.has("touch_radius"):
		touch_radius = float(params["touch_radius"])
	if params.has("arena_size_x"):
		arena_size.x = float(params["arena_size_x"])
	if params.has("arena_size_y"):
		arena_size.y = float(params["arena_size_y"])
>>>>>>> origin/main

# --- Lightweight visualizer ---
# The agent/target are bare Node2Ds (no sprites), so the scene renders nothing on its own. This
# draws the arena, the target, and the agent so the deploy scene is watchable (e.g. the web export
# proof). Headless runs never call _draw()/_process redraws, so this is free for the test suite.
func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.10, 0.11, 0.15), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.30, 0.32, 0.42), false, 2.0)
	draw_circle(get_target_pos(), touch_radius, Color(0.92, 0.33, 0.33))
	draw_circle(get_agent_pos(), 15.0, Color(0.30, 0.80, 1.0))
