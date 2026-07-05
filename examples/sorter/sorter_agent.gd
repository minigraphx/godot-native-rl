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
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
const ControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

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
	# Signal-driven rewards through the addon reward system, like the sibling agents (#317).
	reward_source = RewardBuilderScript.new() \
		.add_event_bonus("correct_visit", correct_reward) \
		.add_event_bonus("wrong_visit", -wrong_penalty) \
		.add_step_penalty(step_penalty) \
		.build()
	var adapter := RewardAdapterScript.new()
	add_child(adapter)
	adapter.on_signal_event(_game, "correct_visit", "correct_visit")
	adapter.on_signal_event(_game, "wrong_visit", "wrong_visit")
	_game.all_sorted.connect(func() -> void:
		done = true
		needs_reset = true)
	# Point the sensor at THIS world's tile group (#313): groups are tree-global, so the shared
	# literal group leaked every ParallelArena2D world's tiles into every sensor.
	for sensor in ControllerCore.collect_sensors_nodes(self):
		if "group_name" in sensor:
			sensor.group_name = _game.instance_group()


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
	var velocity: Vector2 = ChaseObs.action_index_to_velocity(_action_index, _game.move_speed)
	_game.move_agent(velocity, delta)
	accumulate_reward()  # applies the step penalty + any visit events this move just fired
	if needs_reset:
		needs_reset = false
		_game.reset_episode()
		reset()
		# Do NOT zero_reward(): the bridge reads reward+done together THEN zeroes (hide&seek
		# contract) — zeroing here wiped the +1.0 completion bonus of the final tile before the
		# trainer ever saw it, silently deleting the sparse success signal (#314).
		if reward_source != null:
			reward_source.reset()
