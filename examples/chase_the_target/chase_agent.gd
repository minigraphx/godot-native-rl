class_name ChaseAgent
extends NcnnAIController2D

const ACTION_KEY := "move"
const ACTION_COUNT := 5

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var touch_bonus := 1.0

var _game  # ChaseGame (typed at runtime via duck-typing to avoid class_name scope issues)
var _action_index := 0
var _prev_dist := 0.0

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("ChaseAgent: game_path is not set or invalid — agent will produce null observations.")
		return
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
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "ChaseAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

# --- Runtime step (drives the game between control decisions) ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return

	var velocity := action_index_to_velocity(_action_index, _game.move_speed)
	_game.move_agent(velocity, delta)

	var cur_dist: float = _game.distance()
	var touched: bool = cur_dist < _game.touch_radius
	reward += compute_step_reward(_prev_dist, cur_dist, _game.max_distance(), touched)
	if touched:
		_game.relocate_target()
		_prev_dist = _game.distance()  # baseline is distance to the NEW target after relocate
	else:
		_prev_dist = cur_dist

	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		_prev_dist = _game.distance()
