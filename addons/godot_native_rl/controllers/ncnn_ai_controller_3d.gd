class_name NcnnAIController3D
extends Node3D

const RewardAdapterScript = preload("res://addons/godot_native_rl/reward/reward_adapter.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")
const ObsNormalize = preload("res://addons/godot_native_rl/controllers/obs_normalize.gd")

enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING, NCNN_INFERENCE }
@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC  # read/written by NcnnSync
@export var reset_after := 1000
@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
@export_file("*.json") var obs_norm_stats_path: String = ""
@export var policy_name: String = "shared_policy"  # multi-policy routing (PettingZoo/RLlib)

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
		_load_obs_norm_stats()

func _setup_ncnn_runner() -> void:
	if model_param_path.is_empty() or model_bin_path.is_empty():
		push_error("NcnnAIController3D: NCNN_INFERENCE mode requires model_param_path and model_bin_path.")
		return
	_ncnn_runner = NcnnRunner.new()
	_ncnn_runner.input_blob_name = input_blob_name
	_ncnn_runner.output_blob_name = output_blob_name
	add_child(_ncnn_runner)
	var absolute_param := ProjectSettings.globalize_path(model_param_path)
	var absolute_bin := ProjectSettings.globalize_path(model_bin_path)
	if not _ncnn_runner.load_model(absolute_param, absolute_bin):
		push_error("NcnnAIController3D: failed to load ncnn model.")
		_ncnn_runner.queue_free()
		_ncnn_runner = null

func set_ncnn_runner_for_test(runner) -> void:
	_ncnn_runner = runner

func _load_obs_norm_stats() -> void:
	if obs_norm_stats_path.is_empty():
		return
	var f := FileAccess.open(obs_norm_stats_path, FileAccess.READ)
	if f == null:
		push_error("NcnnAIController3D: cannot open obs_norm_stats_path '%s'." % obs_norm_stats_path)
		return
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary) or not ObsNormalize.validate(parsed):
		push_error("NcnnAIController3D: invalid obs-norm stats JSON at '%s'." % obs_norm_stats_path)
		return
	_core.obs_norm_stats = ObsNormalize.to_typed(parsed)

func set_obs_norm_stats_for_test(stats: Dictionary) -> void:
	_core.obs_norm_stats = stats

func infer_and_act() -> void:
	_core.choose_and_apply_action(self, _ncnn_runner)

# --- Abstract: implemented by the concrete agent ---
func get_obs() -> Dictionary:
	assert(false, "get_obs must be implemented by the agent extending NcnnAIController3D")
	return {"obs": []}

func get_reward() -> float:
	assert(false, "get_reward must be implemented by the agent extending NcnnAIController3D")
	return 0.0

func get_action_space() -> Dictionary:
	assert(false, "get_action_space must be implemented by the agent extending NcnnAIController3D")
	return {}

func set_action(_action) -> void:
	assert(false, "set_action must be implemented by the agent extending NcnnAIController3D")

# Override in an image agent to return the live frame for native inference, e.g.
# `return _camera.get_image()`. Non-null routes infer_and_act through run_inference_image.
func get_inference_image() -> Image:
	return null

# Optional per-agent info (godot_rl reads response.get("info", ...)); default empty.
# Agents may override to return e.g. {"is_success": true}.
func get_info() -> Dictionary:
	return {}

# --- Concrete contract methods used by NcnnSync (delegate to the shared core) ---
func get_obs_space() -> Dictionary:
	return NcnnControllerCore.obs_space_from_obs(get_obs())

# Convenience: concatenate all child flat-sensor observations (recursive, tree order).
# Agents can write `return {"obs": collect_sensors()}` in get_obs().
func collect_sensors() -> Array:
	return NcnnControllerCore.collect_sensors(self)

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

func accumulate_reward() -> void:
	_core.accumulate(_reward_adapters, self)

func _physics_process(_delta) -> void:
	_core.step(reset_after)
