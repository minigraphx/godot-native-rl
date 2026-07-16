class_name GoToGoalAgent
# Path-based extends (not bare `extends NcnnAIController2D`) so the class resolves in
# headless/CLI runs that have no editor-generated global_script_class_cache.cfg — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

# The #386 goal-conditioning worked example's controller: ALL observations come from the child
# sensors via collect_sensors() — a 3-target RelativePositionSensor2D (unit dir + distance per
# target) plus a GoalSensor (num_goals one-hot of the signaled target). get_obs() has no hand-coded
# features: the same weights pursue whichever target the GoalSensor channel signals.

const ACTION_KEY := "move"
const ACTION_COUNT := 5
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# The ONE discrete-action convention the 2D examples share (#335): 0=stay, 1=up, 2=down, 3=left,
# 4=right via ChaseObs.action_index_to_velocity — a hand-rolled mapping here silently swapped axes
# for any tooling (scripted experts, demo recorders) written against the shared convention.
const ChaseObs = preload("res://examples/chase_the_target/chase_obs.gd")

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var goal_bonus := 2.0
@export var wrong_penalty := 1.0

var _game  # GoToGoalGame (duck-typed to avoid class_name scope issues headless)
var _action_index := 0

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("GoToGoalAgent: game_path is not set or invalid — agent will produce null observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.signaled_distance, _game.max_distance, ["goal_reached"]) \
		.add_event_bonus("goal_reached", goal_bonus) \
		.add_event_bonus("wrong_target_touched", -wrong_penalty) \
		.add_step_penalty(step_penalty) \
		.build()
	var adapter := RewardAdapterScript.new()
	add_child(adapter)
	adapter.on_signal_event(_game, "goal_reached", "goal_reached")
	adapter.on_signal_event(_game, "wrong_target_touched", "wrong_target_touched")

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	# The whole observation IS the sensor sweep: the RelativePositionSensor2D (unit dir + distance
	# per target) plus the GoalSensor one-hot naming which target is currently signaled.
	return {"obs": collect_sensors()}

func get_reward() -> float:
	return reward

func get_debug_status() -> Dictionary:
	if _game == null:
		return {}
	return {"signaled_dist": _game.signaled_distance(), "goal": _game.get_current_goal(),
		"goals": _game.goals_reached, "wrong": _game.wrong_touches}

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "GoToGoalAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

# --- Runtime step (drives the game between control decisions) ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return

	_game.move_agent(ChaseObs.action_index_to_velocity(_action_index, _game.move_speed), delta)

	# Score against the CURRENT layout BEFORE resolve_touches() relocates (seek baseline-rebase
	# gotcha): the progress baseline rebases on goal_reached, the queued bonuses/penalties land next
	# step.
	accumulate_reward()
	_game.resolve_touches()

	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()  # rebase baseline to post-reset distance; clear pending events
