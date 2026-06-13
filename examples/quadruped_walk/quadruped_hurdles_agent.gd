class_name QuadrupedHurdlesAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://examples/quadruped_walk/quadruped_agent.gd"

# #60 M2: the M1 quadruped + forward hurdle perception and a clear-the-hurdle bonus.
# Obs = M1's 29 + 6 closeness rays (RaycastSensor3D, collision_mask = hurdle layer 2) = 35.
# The sensor is a world-scene node (the rig is code-built, so it can't be a torso child in the
# scene): each frame we snap its position to the torso and keep its fixed, level orientation
# (scene sets rotation.y = PI so the ray fan's local -Z looks down +Z, the running direction).

const RAY_OBS_SIZE := 6

@export var hurdle_track_path: NodePath
@export var ray_sensor_path: NodePath
@export var curriculum_path: NodePath  ## this world's CurriculumController (per-world under ParallelArena)
@export var clear_bonus := 1.0
@export var sensor_height := 0.5  ## ray origin Y above the ground (level, torso-independent)

var _track
var _sensor
var _warned_no_sensor := false
var _curriculum: Node = null
var _episode_reward := 0.0
var _episode_clears := 0

func _ready() -> void:
	super._ready()
	_track = get_node_or_null(hurdle_track_path)
	_sensor = get_node_or_null(ray_sensor_path)
	# Per-world controller via path (each tiled world is self-contained); group fallback for
	# single-world scenes that keep the controller at the top level.
	_curriculum = get_node_or_null(curriculum_path)
	if _curriculum == null and is_inside_tree():
		_curriculum = get_tree().get_first_node_in_group("CURRICULUM")

func get_info() -> Dictionary:
	if _curriculum == null:
		return {}
	return {"curriculum_stage": _curriculum.stage_index()}

func expected_obs_size() -> int:
	return OBS_SIZE + RAY_OBS_SIZE

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(expected_obs_size())
	z.fill(0.0)
	return z

# Fixed-size ray slice: zero-filled without a sensor (one warning), padded/truncated otherwise —
# the wire contract must not drift with scene wiring.
func _ray_obs() -> Array:
	var out: Array = []
	if _sensor == null:
		if not _warned_no_sensor:
			push_warning("QuadrupedHurdlesAgent: ray_sensor_path not set — zero-filled ray observations.")
			_warned_no_sensor = true
	else:
		out = _sensor.get_observation()
	while out.size() < RAY_OBS_SIZE:
		out.append(0.0)
	if out.size() > RAY_OBS_SIZE:
		out.resize(RAY_OBS_SIZE)
	return out

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": _zero_obs()}
	var base: Array = super.get_obs()["obs"]
	base.append_array(_ray_obs())
	return {"obs": base}

func _snap_sensor() -> void:
	if _sensor == null or _game == null:
		return
	var p: Vector3 = _game.torso_pos()
	if _sensor.is_inside_tree():
		_sensor.global_position = Vector3(p.x, sensor_height, p.z)
	else:
		_sensor.position = Vector3(p.x, sensor_height, p.z)

# Full override of QuadrupedAgent's loop (GDScript can't skip one super level): same v3
# locomotion reward + hurdle-clear bonus, and the corrected terminal ordering — the fall
# penalty must NOT be zeroed before the sync reads it (#207; reward+done are read together).
func _physics_process(_delta: float) -> void:
	_core.step(reset_after)  # the controller layer's episode bookkeeping
	if _game == null:
		return
	if _action.size() == ACTION_COUNT:
		_game.apply_motors(_action)
	_snap_sensor()
	var reward_before := reward
	accumulate_reward()
	reward += forward_weight * _game.forward_velocity()
	reward -= lateral_weight * absf(_game.lateral_velocity())
	reward += upright_weight * _game.upright()
	reward -= energy_penalty * _sum_abs(_action)
	if _track != null:
		var cleared: int = _track.count_newly_passed(_game.torso_pos().z)
		_episode_clears += cleared
		reward += clear_bonus * cleared
	if _is_fallen():
		reward -= fall_penalty
		done = true
		needs_reset = true
	_episode_reward += reward - reward_before
	if needs_reset:
		needs_reset = false
		if _curriculum != null:
			_curriculum.record_episode(_episode_reward, _episode_clears > 0)
		_episode_reward = 0.0
		_episode_clears = 0
		_game.reset_positions()
		if _track != null:
			_track.reset_progress()
		reset()
		# NO zero_reward() here: this step's reward (incl. the fall penalty and any clear
		# bonus) is read by the sync together with done. Zeroing would wipe it (#207).
		if reward_source != null:
			reward_source.reset()
