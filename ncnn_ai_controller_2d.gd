class_name NcnnAIController2D
extends Node2D

const RewardAdapterScript = preload("res://reward/reward_adapter.gd")

enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING, NCNN_INFERENCE }
@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC  # read by NcnnSync
@export var reset_after := 1000
@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"

var heuristic := "human"
var done := false
var reward := 0.0
var n_steps := 0
var needs_reset := false
var _ncnn_runner = null
var reward_source = null         # optional Reward (from RewardBuilder.build()); null = legacy behavior
var _reward_adapters: Array = []

func _ready() -> void:
	add_to_group("AGENT")
	_collect_reward_adapters()
	if control_mode == ControlModes.NCNN_INFERENCE:
		_setup_ncnn_runner()

func _setup_ncnn_runner() -> void:
	if model_param_path.is_empty() or model_bin_path.is_empty():
		push_error("NcnnAIController2D: NCNN_INFERENCE mode requires model_param_path and model_bin_path.")
		return
	_ncnn_runner = NcnnRunner.new()
	_ncnn_runner.input_blob_name = input_blob_name
	_ncnn_runner.output_blob_name = output_blob_name
	add_child(_ncnn_runner)
	var absolute_param := ProjectSettings.globalize_path(model_param_path)
	var absolute_bin := ProjectSettings.globalize_path(model_bin_path)
	if not _ncnn_runner.load_model(absolute_param, absolute_bin):
		push_error("NcnnAIController2D: failed to load ncnn model.")
		_ncnn_runner.queue_free()
		_ncnn_runner = null

func set_ncnn_runner_for_test(runner) -> void:
	_ncnn_runner = runner

func infer_and_act() -> void:
	if _ncnn_runner == null or not _ncnn_runner.is_model_loaded():
		return
	var obs_dict := get_obs()
	assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
	var obs_flat := PackedFloat32Array(obs_dict["obs"])
	var action_index: int = _ncnn_runner.run_discrete_action(obs_flat)
	if action_index < 0:
		push_error("NcnnAIController2D: run_discrete_action returned error sentinel; skipping action.")
		return
	# Single discrete action branch: use the first (and only) action key.
	var action_key: String = get_action_space().keys()[0]
	set_action({action_key: action_index})

# --- Abstract: implemented by the concrete agent ---
func get_obs() -> Dictionary:
	assert(false, "get_obs must be implemented by the agent extending NcnnAIController2D")
	return {"obs": []}

func get_reward() -> float:
	assert(false, "get_reward must be implemented by the agent extending NcnnAIController2D")
	return 0.0

func get_action_space() -> Dictionary:
	assert(false, "get_action_space must be implemented by the agent extending NcnnAIController2D")
	return {}

func set_action(_action) -> void:
	assert(false, "set_action must be implemented by the agent extending NcnnAIController2D")

# --- Concrete contract methods used by NcnnSync ---
func get_obs_space() -> Dictionary:
	var obs := get_obs()
	return {"obs": {"size": [obs["obs"].size()], "space": "box"}}

func reset() -> void:
	n_steps = 0
	needs_reset = false

func reset_if_done() -> void:
	if done:
		reset()

func set_heuristic(h: String) -> void:
	heuristic = h

func get_done() -> bool:
	return done

func set_done_false() -> void:
	done = false

func zero_reward() -> void:
	reward = 0.0

func _collect_reward_adapters() -> void:
	_reward_adapters.clear()
	for child in get_children():
		if child is RewardAdapterScript:
			_reward_adapters.append(child)

# Sum the declarative reward for this step into the accumulator that NcnnSync drains.
# Call this from the concrete agent's _physics_process AFTER world state is updated.
func accumulate_reward() -> void:
	if reward_source != null:
		reward += reward_source.evaluate(self)
	for adapter in _reward_adapters:
		reward += adapter.drain()

func _physics_process(_delta) -> void:
	n_steps += 1
	if n_steps > reset_after:
		needs_reset = true
		# Signal episode termination (godot_rl convention): the trainer reads this as
		# `done`, which gives proper episode boundaries and reward statistics.
		done = true
