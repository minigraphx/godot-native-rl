class_name HideSeekAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

const ACTION_KEY := "move"
const ACTION_COUNT := 5
const HideSeekMath = preload("res://examples/hide_and_seek/hide_seek_math.gd")

@export var game_path: NodePath
@export var is_seeker := false
@export var catch_bonus := 5.0
@export var ray_count := 8
@export var ray_length := 400.0

var _game  # HideSeekGame (duck-typed to avoid class_name scope issues headless)
var _action_index := 0
var _selfplay: Node = null  # optional SelfPlayManager (group SELF_PLAY), null in plain scenes (#29)

# --- Pure helpers (unit-tested) ---
func action_to_velocity(idx: int, speed: float) -> Vector2:
	match idx:
		1: return Vector2(0.0, -speed)
		2: return Vector2(0.0, speed)
		3: return Vector2(-speed, 0.0)
		4: return Vector2(speed, 0.0)
		_: return Vector2.ZERO

func _obs_size() -> int:
	return 2 + ray_count + 4 + 1

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(_obs_size())
	z.fill(0.0)
	return z

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("HideSeekAgent: game_path not set or invalid — producing zero observations.")
	# Distinct-policy identity ("seeker"/"hider") is baked into hide_seek_world.tscn as `policy_group`
	# and honored only when the training scene's Sync sets multi_policy=true (#73). No cmdline gate.
	# Self-play (#29): optional SelfPlayManager in the scene; only the TRAINING-side (learner)
	# agent reports match outcomes — the frozen ghost must not double-report.
	_selfplay = get_tree().get_first_node_in_group("SELF_PLAY")

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": _zero_obs()}
	var self_pos: Vector2 = _game.seeker_pos() if is_seeker else _game.hider_pos()
	var opp_pos: Vector2 = _game.hider_pos() if is_seeker else _game.seeker_pos()
	var own: Array = HideSeekMath.own_pos_obs(self_pos, _game.arena_size)
	var dirs: Array = HideSeekMath.ray_directions_surround(ray_count)
	var wall: Array = HideSeekMath.wall_ray_closeness(self_pos, dirs, ray_length, _game.walls)
	var opp: Array = HideSeekMath.encode_opponent(self_pos, opp_pos, _game.walls, _game.opp_max_dist)
	return {"obs": HideSeekMath.assemble_obs(own, wall, opp, HideSeekMath.role_flag(is_seeker))}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "HideSeekAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

# --- Runtime step ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)  # core.step (reset_after acts only as a safety net)
	if _game == null:
		return
	var vel := action_to_velocity(_action_index, _game.move_speed)
	if is_seeker:
		_game.set_seeker_velocity(vel)
	else:
		_game.set_hider_velocity(vel)
	# Inline role-signed reward read from the game's shared, single-source cached state.
	# Note: like the other examples, reward accumulates across the whole action_repeat window. If a
	# terminal (catch) lands mid-window the game resets next frame and a few new-episode step rewards
	# join this bucket — bounded (<=1 terminal/window via min_separation; small vs catch_bonus) and
	# harmless to PPO, matching how chase/rover treat the action_repeat window.
	reward += HideSeekMath.step_reward(is_seeker, _game.has_los(), _game.was_caught(), catch_bonus)
	# Both agents read the same terminal flag in the same frame -> they end together. The game
	# resets positions itself (next frame); agents only reset their own controller state. Do NOT
	# zero_reward() here — the bridge reads reward + done together, then zeroes reward.
	var terminal: bool = _game.is_terminal()
	if terminal or needs_reset:
		if terminal:
			done = true
		# Self-play (#29): the learner reports the match outcome. Caught = seeker win;
		# timeout = hider win (no draws). Ghost (NCNN_INFERENCE) agents never report.
		if _selfplay != null and control_mode != ControlModes.NCNN_INFERENCE:
			var seeker_won: bool = _game.was_caught()
			_selfplay.report_match(seeker_won if is_seeker else not seeker_won)
		needs_reset = false
		reset()
		if is_seeker:
			_game.request_reset()
