class_name SorterAgent
# Path-based extends (not bare class_name) so the class resolves headless — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"
# Sorter agent (#46 M2): THIS node is the moving body (the game's agent_body_path points here),
# so the child EntitySensor2D rides at the agent's position and its egocentric entity block is
# correct by construction. Obs = collect_sensors() = the sensor's [N*F entities][N flags] block —
# the attention encoder's input contract, nothing else. Discrete chase-style movement.

const ACTION_KEY := "move"
const ACTION_COUNT := 5  # stay + 4 directions
const ChaseObs = preload("res://examples/chase_the_target/chase_obs.gd")

@export var game_path: NodePath
@export var step_penalty := 0.002
@export var correct_reward := 1.0
@export var wrong_penalty := 0.1

var _game
var _action_index := 0


func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("SorterAgent: game_path is not set — agent will produce zero observations.")
		return
	_game.correct_visit.connect(func() -> void: reward += correct_reward)
	_game.wrong_visit.connect(func() -> void: reward -= wrong_penalty)
	_game.all_sorted.connect(func() -> void:
		done = true
		needs_reset = true)


func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}


func get_obs() -> Dictionary:
	return {"obs": collect_sensors()}


func get_reward() -> float:
	return reward


func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "SorterAgent: action index %d out of range" % idx)
	_action_index = idx


func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	reward -= step_penalty
	var velocity: Vector2 = ChaseObs.action_index_to_velocity(_action_index, _game.move_speed)
	_game.move_agent(velocity, delta)
	if needs_reset:
		needs_reset = false
		_game.reset_episode()
		reset()
		zero_reward()
