class_name SeekAgent
# Path-based extends (not bare `extends NcnnAIController2D`) so the class resolves in
# headless/CLI runs that have no editor-generated global_script_class_cache.cfg — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

# The #38 worked example's controller: ALL observations come from the child
# RelativePositionSensor2D (goal slot + hazard slot -> 4 floats) via the controllers'
# collect_sensors() auto-discovery — get_obs() has no hand-coded features. Compare ChaseAgent,
# which computes its obs by hand: this is the drop-in-sensor version of the same shape of task.

const ACTION_KEY := "move"
const ACTION_COUNT := 5
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")

## Discrete action set: idle, left, right, up, down.
const DIRECTIONS: Array[Vector2] = [
	Vector2.ZERO, Vector2.LEFT, Vector2.RIGHT, Vector2.UP, Vector2.DOWN,
]

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var goal_bonus := 2.0
@export var hazard_penalty := 1.0

var _game  # SeekGame (duck-typed to avoid class_name scope issues headless)
var _action_index := 0

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("SeekAgent: game_path is not set or invalid — agent will produce null observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.goal_distance, _game.max_distance, ["goal_reached"]) \
		.add_event_bonus("goal_reached", goal_bonus) \
		.add_event_bonus("hazard_hit", -hazard_penalty) \
		.add_step_penalty(step_penalty) \
		.build()
	var adapter := RewardAdapterScript.new()
	add_child(adapter)
	adapter.on_signal_event(_game, "goal_reached", "goal_reached")
	adapter.on_signal_event(_game, "hazard_hit", "hazard_hit")

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	# The whole observation IS the sensor sweep (use_separate_direction mode): per target a UNIT
	# direction + a normalized distance -> [goal dir_x, dir_y, dist, hazard dir_x, dir_y, dist].
	# Unit directions matter: with plain normalized offsets the direction signal shrinks to ~0
	# near the target — the first training run barely learned to seek until this switch.
	return {"obs": collect_sensors()}

func get_reward() -> float:
	return reward

func get_debug_status() -> Dictionary:
	if _game == null:
		return {}
	return {"goal_dist": _game.goal_distance(), "hazard_dist": _game.hazard_distance(),
		"goals": _game.goals_reached, "hits": _game.hazard_hits}

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "SeekAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

# --- Runtime step (drives the game between control decisions) ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return

	_game.move_agent(DIRECTIONS[_action_index] * _game.move_speed, delta)
	_game.step_hazard(delta)

	# Score events against the CURRENT layout BEFORE any relocation (chase pattern): the
	# progress baseline rebases on goal_reached, the queued bonuses/penalties land next step.
	accumulate_reward()

	if _game.goal_distance() < _game.goal_radius:
		_game.relocate_goal()
	if _game.hazard_distance() < _game.hazard_radius:
		_game.trigger_hazard_hit()

	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()  # rebase baseline to post-reset distance; clear pending events
