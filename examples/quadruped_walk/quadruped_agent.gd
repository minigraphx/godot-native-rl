class_name QuadrupedAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"

const ACTION_KEY := "motors"
const ACTION_COUNT := 8  # quadruped fallback (used before the rig is built); 2 motors/leg
const OBS_SIZE := 8 + 8 + 3 + 3 + 3 + 4  # quadruped fallback = 29; general = 5*legs + 9 (see _obs_dim)
const QM = preload("res://examples/quadruped_walk/quadruped_math.gd")
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from the controller — do not redeclare.

@export var game_path: NodePath
# Locomotion reward (tuned 2026-06-11, v3 robustness pass). The arc:
#   v1 (upright 0.05 + alive 0.01, no velocity): balanced in place (~3m).
#   v2 (forward 0.2, upright 0.02, alive 0.005): walked but lunged + fell fast (ep_len ~64) and
#       drifted sideways/backward in some physics realizations (unreliable).
#   v3 (this): keep forward velocity as the driver but penalize LATERAL drift (track straight),
#       and restore enough upright/alive + a stronger fall penalty that the creature stays up
#       AND moves — for a robust, consistently-forward gait.
@export var forward_weight := 0.15   ## reward +Z (toward finish) velocity — main driver
@export var lateral_weight := 0.06   ## penalize |X| velocity — go straight, don't drift sideways
@export var upright_weight := 0.04
@export var alive_bonus := 0.02
@export var energy_penalty := 0.0005
@export var fall_penalty := 2.0
@export var fall_height := 0.45      ## torso below this Y = fallen
@export var fall_upright := 0.2      ## upright dot below this = fallen

var _game
var _action: Array = []

func set_game(g) -> void:
	_game = g

# Motor count = one per joint (2/leg). Falls back to the quadruped const before the rig exists.
func _n_motors() -> int:
	if _game != null and _game.joint_count() > 0:
		return _game.joint_count()
	return ACTION_COUNT

# Obs dim is leg-count-agnostic: joints(2L) + vels(2L) + up(3) + localvel(3) + dir(3) + contacts(L)
# = 5L + 9. Quadruped (L=4) -> 29; hexapod (L=6) -> 39.
func _obs_dim() -> int:
	if _game != null and _game.joint_count() > 0:
		var legs: int = _game.joint_count() / 2
		return 5 * legs + 9
	return OBS_SIZE

func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": _n_motors(), "action_type": "continuous"}}

func expected_obs_size() -> int:
	return _obs_dim()

func stored_action_for_test() -> Array:
	return _action

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(_obs_dim())
	z.fill(0.0)
	return z

func _ready() -> void:
	super._ready()
	if _game == null:
		_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("QuadrupedAgent: game_path not set — producing zero observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance) \
		.add_alive_bonus(alive_bonus) \
		.build()
	call_deferred("_reset_reward_baseline")

func _reset_reward_baseline() -> void:
	if reward_source != null:
		reward_source.reset()

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": _zero_obs()}
	return {"obs": QM.compose_obs(
		_game.joint_angles(), _game.joint_velocities(),
		_game.torso_up(), _game.body_local_velocity(),
		_game.dir_to_finish(), _game.foot_contacts())}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var raw: Array = action[ACTION_KEY]
	_action = []
	for v in raw:
		_action.append(QM.clamp_action(float(v)))

func _sum_abs(a: Array) -> float:
	var s := 0.0
	for v in a:
		s += absf(v)
	return s

func _is_fallen() -> bool:
	return _game.torso_pos().y < fall_height or _game.upright() < fall_upright

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	if _action.size() == _n_motors():
		_game.apply_motors(_action)
	# Progress + alive come from reward_source; forward velocity (main driver) + upright + energy
	# applied directly. Forward velocity is the dense locomotion signal: reward moving toward +Z.
	accumulate_reward()
	reward += forward_weight * _game.forward_velocity()
	reward -= lateral_weight * absf(_game.lateral_velocity())
	reward += upright_weight * _game.upright()
	reward -= energy_penalty * _sum_abs(_action)
	# A fall is a terminal state: signal `done` (and `needs_reset`) so the trainer gets a real
	# episode boundary — mirrors the core's reset_after timeout (which sets done+needs_reset on
	# step()). WITHOUT this, the constant early-training falls reset n_steps every few frames, so
	# the 1000-step timeout never fires and `done` never reaches the trainer → no episode ever
	# completes → no learning signal (SB3 logs no ep_rew_mean).
	if _is_fallen():
		reward -= fall_penalty
		done = true
		needs_reset = true
	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()
