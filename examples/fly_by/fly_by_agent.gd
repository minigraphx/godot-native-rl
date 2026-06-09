class_name FlyByAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"

const PITCH_KEY := "pitch"
const TURN_KEY := "turn"
const OBS_SIZE := 8
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from the controller — do not redeclare.

@export var game_path: NodePath
@export var goal_bonus := 2.0
@export var step_penalty := 0.002  ## per physics frame; ~ -2.0 over a full reset_after=1000 episode
# Must exceed the max per-episode step penalty (~2.0) so diving out of bounds to escape the step
# penalty early is never worth it — otherwise the plane learns to leave the arena. Episode ends on
# exit (done=true), so this fires at most once per episode.
@export var exit_penalty := 5.0

var _game  # FlyByGame (duck-typed at runtime)
var _pitch := 0.0
var _turn := 0.0

# --- Pure helpers (unit-tested via accessors) ---
func clamp_input(v: float) -> float:
	return clampf(v, -1.0, 1.0)

func get_pitch_for_test() -> float:
	return _pitch

func get_turn_for_test() -> float:
	return _turn

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	# Two continuous size-1 keys. No "squash": PPO's mean is unbounded and the #64 DiagGaussian
	# sample can exceed [-1,1], so we clamp game-side in set_action (NOT tanh-squash at decode).
	return {
		PITCH_KEY: {"size": 1, "action_type": "continuous"},
		TURN_KEY: {"size": 1, "action_type": "continuous"},
	}

func get_obs() -> Dictionary:
	if _game == null:
		var z: Array = []
		z.resize(OBS_SIZE)
		z.fill(0.0)
		return {"obs": z}
	return {"obs": _game.get_obs_array()}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	_pitch = clamp_input(float(action[PITCH_KEY][0]))
	_turn = clamp_input(float(action[TURN_KEY][0]))

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("FlyByAgent: game_path not set or invalid — null observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["goal_reached"]) \
		.add_event_bonus("goal_reached", goal_bonus) \
		.add_event_bonus("exited", -exit_penalty) \
		.add_step_penalty(step_penalty) \
		.build()
	var goal_adapter := RewardAdapterScript.new()
	add_child(goal_adapter)
	goal_adapter.on_signal_event(_game, "goal_reached", "goal_reached")
	var exit_adapter := RewardAdapterScript.new()
	add_child(exit_adapter)
	exit_adapter.on_signal_event(_game, "exited_arena", "exited")
	# Children _ready runs before the parent FlyByGame._ready that positions the plane, so rebase
	# the progress-shaping baseline once the world is initialized (mirrors RoverAgent).
	call_deferred("_reset_reward_baseline")

func _reset_reward_baseline() -> void:
	if reward_source != null:
		reward_source.reset()

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	_game.move_plane(_pitch, _turn, delta)
	# Accumulate reward against the CURRENT goal BEFORE advancing it (matches chase/rover).
	accumulate_reward()
	_game.try_reach_goal()
	# End the episode when the plane leaves the arena (the exited_arena signal already penalized).
	if _game.out_of_bounds(_game.get_plane_xform().origin, _game.arena_half):
		done = true
	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()
