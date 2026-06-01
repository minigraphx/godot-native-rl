class_name NcnnAIController2D
extends Node2D

const RewardAdapterScript = preload("res://addons/godot_native_rl/reward/reward_adapter.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING, NCNN_INFERENCE }
@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC  # read/written by NcnnSync
@export var reset_after := 1000
@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"

var _core := NcnnControllerCore.new()
var _ncnn_runner = null
var _reward_adapters: Array = []

# --- Forwarding properties: preserve the historical public state API (subclasses + NcnnSync) ---
var done: bool:
	get:
		return _core.done
	set(value):
		_core.done = value
var reward: float:
	get:
		return _core.reward
	set(value):
		_core.reward = value
var n_steps: int:
	get:
		return _core.n_steps
	set(value):
		_core.n_steps = value
var needs_reset: bool:
	get:
		return _core.needs_reset
	set(value):
		_core.needs_reset = value
var heuristic: String:
	get:
		return _core.heuristic
	set(value):
		_core.heuristic = value
var reward_source:
	get:
		return _core.reward_source
	set(value):
		_core.reward_source = value

func _ready() -> void:
	add_to_group("AGENT")
	collect_reward_adapters()
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

# Optional per-agent info (godot_rl reads response.get("info", ...)); default empty.
# Agents may override to return e.g. {"is_success": true}.
func get_info() -> Dictionary:
	return {}

# --- Concrete contract methods used by NcnnSync (delegate to the shared core) ---
func get_obs_space() -> Dictionary:
	return NcnnControllerCore.obs_space_from_obs(get_obs())

func reset() -> void:
	_core.reset()

func reset_if_done() -> void:
	_core.reset_if_done()

func set_heuristic(h: String) -> void:
	_core.set_heuristic(h)

func get_done() -> bool:
	return _core.get_done()

func set_done_false() -> void:
	_core.set_done_false()

func zero_reward() -> void:
	_core.zero_reward()

func collect_reward_adapters() -> void:
	_reward_adapters.clear()
	for child in get_children():
		if child is RewardAdapterScript:
			_reward_adapters.append(child)

# Sum the declarative reward for this step into the accumulator that NcnnSync drains.
# Call from the concrete agent's _physics_process AFTER world state is updated.
func accumulate_reward() -> void:
	_core.accumulate(_reward_adapters, self)

func _physics_process(_delta) -> void:
	_core.step(reset_after)
