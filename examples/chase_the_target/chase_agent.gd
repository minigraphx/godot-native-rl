class_name ChaseAgent
extends NcnnAIController2D

const ACTION_KEY := "move"
const ACTION_COUNT := 5
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from NcnnAIController2D — do not redeclare.

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var touch_bonus := 1.0

var _game  # ChaseGame (typed at runtime via duck-typing to avoid class_name scope issues)
var _action_index := 0

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("ChaseAgent: game_path is not set or invalid — agent will produce null observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["target_caught"]) \
		.add_event_bonus("target_caught", touch_bonus) \
		.add_step_penalty(step_penalty) \
		.build()
	var adapter := RewardAdapterScript.new()
	add_child(adapter)
	adapter.on_signal_event(_game, "target_caught", "target_caught")

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

	# Accumulate reward against the CURRENT target BEFORE relocating. The catch is
	# signalled by relocate_target() -> RewardAdapter -> Reward: it rebases the progress
	# baseline to the new target immediately and queues the catch bonus for next step.
	accumulate_reward()

	if _game.distance() < _game.touch_radius:
		_game.relocate_target()

	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()   # rebase baseline to post-reset distance; clear pending bonus
