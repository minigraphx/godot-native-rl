class_name NcnnAgentHelper
extends Node

signal training_action_received(action: Variant)

enum AgentMode {
	INFERENCE,
	TRAINING,
}

enum ActionMode {
	CONTINUOUS,
	DISCRETE_ARGMAX,
}

@export_file("*.param") var model_param_path: String = "res://models/test_mlp.ncnn.param"
@export_file("*.bin") var model_bin_path: String = "res://models/test_mlp.ncnn.bin"
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
@export var input_shape: PackedInt32Array = PackedInt32Array()
@export_enum("Inference", "Training") var agent_mode: int = AgentMode.INFERENCE
@export_enum("Continuous", "Discrete Argmax") var action_mode: int = ActionMode.CONTINUOUS
@export var training_group_name: String = "ncnn_training_agents"

var _native_runner: NcnnRunner
var _last_training_observation: PackedFloat32Array = PackedFloat32Array()
var _latest_training_action: Variant = null

func _ready() -> void:
	_native_runner = NcnnRunner.new()
	add_child(_native_runner)

	_native_runner.input_blob_name = input_blob_name
	_native_runner.output_blob_name = output_blob_name
	_native_runner.input_shape = input_shape

	var absolute_param = ProjectSettings.globalize_path(model_param_path)
	var absolute_bin = ProjectSettings.globalize_path(model_bin_path)

	var ok = _native_runner.load_model(absolute_param, absolute_bin)
	if not ok:
		push_error("Failed to load ncnn model.")
		return

	if agent_mode == AgentMode.TRAINING:
		add_to_group(training_group_name)
	else:
		remove_from_group(training_group_name)

	print("Model loaded")
	var sample_obs := PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
	if agent_mode == AgentMode.INFERENCE:
		if action_mode == ActionMode.DISCRETE_ARGMAX:
			print("Sample discrete action: ", _native_runner.run_discrete_action(sample_obs))
		else:
			print("Sample inference output: ", _native_runner.run_inference(sample_obs))

func set_mode(mode: int) -> void:
	agent_mode = mode
	if is_inside_tree():
		if agent_mode == AgentMode.TRAINING:
			add_to_group(training_group_name)
		else:
			remove_from_group(training_group_name)

func get_action(observations: Array[float]) -> Variant:
	if _native_runner == null or not _native_runner.is_model_loaded():
		push_error("NcnnAgentHelper.get_action: model not loaded.")
		return null

	var packed_obs := PackedFloat32Array(observations)
	if agent_mode == AgentMode.TRAINING:
		_last_training_observation = packed_obs
		return _latest_training_action

	if action_mode == ActionMode.DISCRETE_ARGMAX:
		return _native_runner.run_discrete_action(packed_obs)

	return _native_runner.run_inference(packed_obs)


func get_action_from_image(image: Image, normalize_to_zero_one: bool = true) -> PackedFloat32Array:
	if _native_runner == null or not _native_runner.is_model_loaded():
		push_error("NcnnAgentHelper.get_action_from_image: native runner is not ready.")
		return PackedFloat32Array()

	if agent_mode == AgentMode.TRAINING:
		push_warning("NcnnAgentHelper.get_action_from_image: training mode is active; using native image inference anyway.")

	return _native_runner.run_inference_image(image, normalize_to_zero_one)

func set_training_observation(observation: Array[float]) -> void:
	_last_training_observation = PackedFloat32Array(observation)

func collect_observation() -> Array:
	return Array(_last_training_observation)

func apply_training_action(action: Variant) -> void:
	_latest_training_action = action
	emit_signal("training_action_received", action)

func get_latest_training_action() -> Variant:
	return _latest_training_action

func clear_latest_training_action() -> void:
	_latest_training_action = null

func _process(_delta: float) -> void:
	pass
