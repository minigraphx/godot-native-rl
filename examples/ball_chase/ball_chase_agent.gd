class_name BallChaseAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

const ACTION_KEY := "move"
const ACTION_SIZE := 2
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from NcnnAIController2D — do not redeclare.

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var touch_bonus := 1.0

var _game  # BallChaseGame (duck-typed at runtime)
var _thrust := Vector2.ZERO

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("BallChaseAgent: game_path not set or invalid — null observations.")
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

func clamp_thrust(a: Array) -> Vector2:
	# SAC outputs tanh-squashed actions in [-1,1]; clamp defensively (training samples may graze).
	# Assert size so a train/deploy shape mismatch fails loud here, not as an opaque a[1] crash.
	assert(a.size() >= ACTION_SIZE, "BallChaseAgent: action 'move' has %d elements, expected %d" % [a.size(), ACTION_SIZE])
	return Vector2(clampf(a[0], -1.0, 1.0), clampf(a[1], -1.0, 1.0))

func get_thrust_for_test() -> Vector2:
	return _thrust

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	# NOTE: no "squash" key — SAC's exported actor already applies tanh; squashing again
	# would distort the action. Deploy decode passes these values through raw.
	return {ACTION_KEY: {"size": ACTION_SIZE, "action_type": "continuous"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0]}
	return {"obs": compute_obs(_game.get_agent_pos(), _game.get_target_pos(), _game.arena_size)}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	_thrust = clamp_thrust(action[ACTION_KEY])

# --- Runtime step ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	_game.move_agent(_thrust * _game.move_speed, delta)
	accumulate_reward()
	if _game.distance() < _game.touch_radius:
		_game.relocate_target()
	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()
