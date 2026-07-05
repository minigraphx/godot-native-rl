class_name SeekGame
extends Node2D

# Seek-the-goal / avoid-the-hazard (#38): the RelativePositionSensor2D worked example. The agent
# must reach a goal while a hazard patrols the arena; BOTH are observed exclusively through one
# RelativePositionSensor2D on the agent (separate-direction mode: unit dir + distance per target,
# 6 obs floats total) — no hand-coded obs at all. Kinematic and fully seeded (seed_rng + deterministic patrol), so it is exact under the ES
# trainer's common-random-numbers evaluation; the shipped net is trained IN-ENGINE by ESTrainer
# with the sep-CMA-ES optimizer — no Python anywhere in this example's loop.

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0  ## read by the agent controller to scale velocity
@export var hazard_speed := 220.0  ## patrol speed toward the current waypoint
@export var goal_radius := 40.0  ## reach distance for the goal
@export var hazard_radius := 55.0  ## touch distance that counts as a hit
@export var agent_body_path: NodePath
@export var goal_path: NodePath
@export var hazard_path: NodePath

signal goal_reached  ## emitted when the goal is reached and relocated
signal hazard_hit  ## emitted when the hazard touches the agent (hazard re-routes)

var _rng := RandomNumberGenerator.new()
var _agent_body: Node2D
var _goal: Node2D
var _hazard: Node2D
var _hazard_waypoint := Vector2.ZERO
var goals_reached := 0
var hazard_hits := 0

func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path) as Node2D
	_goal = get_node_or_null(goal_path) as Node2D
	_hazard = get_node_or_null(hazard_path) as Node2D
	reset_positions()

# --- Pure helpers (unit-tested) ---
func clamp_to_bounds(pos: Vector2) -> Vector2:
	return Vector2(clampf(pos.x, 0.0, arena_size.x), clampf(pos.y, 0.0, arena_size.y))

func max_distance() -> float:
	return arena_size.length()

## Constant-speed kinematic step toward a destination, never overshooting. Static so the patrol
## rule is unit-testable without a scene.
static func step_toward(pos: Vector2, dest: Vector2, speed: float, delta: float) -> Vector2:
	return pos.move_toward(dest, speed * delta)

## s must be a non-negative integer (RandomNumberGenerator.seed is uint64; negatives wrap).
func seed_rng(s: int) -> void:
	_rng.seed = s

func random_position() -> Vector2:
	return Vector2(_rng.randf_range(0.0, arena_size.x), _rng.randf_range(0.0, arena_size.y))

# --- Runtime helpers (exercised by the scene + tests) ---
func get_agent_pos() -> Vector2:
	return _agent_body.position if _agent_body != null else Vector2.ZERO

func get_goal_pos() -> Vector2:
	return _goal.position if _goal != null else Vector2.ZERO

func get_hazard_pos() -> Vector2:
	return _hazard.position if _hazard != null else Vector2.ZERO

func goal_distance() -> float:
	return get_agent_pos().distance_to(get_goal_pos())

func hazard_distance() -> float:
	return get_agent_pos().distance_to(get_hazard_pos())

func move_agent(velocity: Vector2, delta: float) -> void:
	if _agent_body != null:
		_agent_body.position = clamp_to_bounds(_agent_body.position + velocity * delta)

## Advance the hazard patrol one tick: constant speed toward the current waypoint, re-rolled
## (seeded) on arrival. Called by the agent's _physics_process so the whole sim shares one clock.
func step_hazard(delta: float) -> void:
	if _hazard == null:
		return
	_hazard.position = step_toward(_hazard.position, _hazard_waypoint, hazard_speed, delta)
	if _hazard.position.distance_to(_hazard_waypoint) < 1.0:
		_hazard_waypoint = random_position()

func relocate_goal() -> void:
	goals_reached += 1
	if _goal != null:
		_goal.position = random_position()
	goal_reached.emit()

## A hazard touch is a DISCRETE scored event: count it, then send the hazard elsewhere (new spot
## + waypoint) so overlap can't re-trigger every tick and the penalty stays comparable across
## candidates.
func trigger_hazard_hit() -> void:
	hazard_hits += 1
	if _hazard != null:
		_hazard.position = random_position()
		_hazard_waypoint = random_position()
	hazard_hit.emit()

func reset_positions() -> void:
	if _agent_body != null:
		_agent_body.position = random_position()
	if _goal != null:
		_goal.position = random_position()
	if _hazard != null:
		_hazard.position = random_position()
	_hazard_waypoint = random_position()

# Episode replay hooks (#39/#335): minimal state for exact playback (kinematic + seeded game).
# rng_state is REQUIRED for exactness: goal/hazard relocations and waypoint re-rolls consume the
# RNG, so a replay without the recorded state diverges after the first event. Stored as String —
# the uint64 RNG state does not survive JSON's float64 numbers.
func get_replay_state() -> Dictionary:
	return {"agent_x": get_agent_pos().x, "agent_y": get_agent_pos().y,
		"goal_x": get_goal_pos().x, "goal_y": get_goal_pos().y,
		"hazard_x": get_hazard_pos().x, "hazard_y": get_hazard_pos().y,
		"waypoint_x": _hazard_waypoint.x, "waypoint_y": _hazard_waypoint.y,
		"goals_reached": goals_reached, "hazard_hits": hazard_hits, "rng_state": str(_rng.state)}


func apply_replay_state(state: Dictionary) -> void:
	if _agent_body != null and state.has("agent_x"):
		_agent_body.position = Vector2(float(state["agent_x"]), float(state["agent_y"]))
	if _goal != null and state.has("goal_x"):
		_goal.position = Vector2(float(state["goal_x"]), float(state["goal_y"]))
	if _hazard != null and state.has("hazard_x"):
		_hazard.position = Vector2(float(state["hazard_x"]), float(state["hazard_y"]))
	if state.has("waypoint_x"):
		_hazard_waypoint = Vector2(float(state["waypoint_x"]), float(state["waypoint_y"]))
	goals_reached = int(state.get("goals_reached", 0))
	hazard_hits = int(state.get("hazard_hits", 0))
	if state.has("rng_state"):
		_rng.state = int(String(state["rng_state"]))


# Curriculum hook (#28/#335): stage params applied at episode boundaries by CurriculumController.
# Flat floats only (params arrive from JSON / the wire — no Vector2).
func apply_curriculum(params: Dictionary) -> void:
	if params.has("hazard_speed"):
		hazard_speed = float(params["hazard_speed"])
	if params.has("goal_radius"):
		goal_radius = float(params["goal_radius"])
	if params.has("hazard_radius"):
		hazard_radius = float(params["hazard_radius"])
	if params.has("arena_size_x"):
		arena_size.x = float(params["arena_size_x"])
	if params.has("arena_size_y"):
		arena_size.y = float(params["arena_size_y"])


# --- Lightweight visualizer (free headless: _draw never runs without a renderer) ---
func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.10, 0.11, 0.15), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.30, 0.32, 0.42), false, 2.0)
	draw_circle(get_goal_pos(), goal_radius, Color(0.35, 0.85, 0.45))
	draw_circle(get_hazard_pos(), hazard_radius, Color(0.95, 0.45, 0.20))
	draw_circle(get_agent_pos(), 15.0, Color(0.30, 0.80, 1.0))
