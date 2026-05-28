class_name NcnnAgentHelper
extends Node

enum ActionMode {
	CONTINUOUS,
	DISCRETE_ARGMAX,
}

@export_file("*.param") var model_param_path: String = "res://models/test_mlp.ncnn.param"
@export_file("*.bin") var model_bin_path: String = "res://models/test_mlp.ncnn.bin"
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
@export_enum("Continuous", "Discrete Argmax") var action_mode: int = ActionMode.CONTINUOUS

var _native_runner: NcnnRunner

func _ready() -> void:
	_native_runner = NcnnRunner.new()
	add_child(_native_runner)

	_native_runner.input_blob_name = input_blob_name
	_native_runner.output_blob_name = output_blob_name

	var absolute_param = ProjectSettings.globalize_path(model_param_path)
	var absolute_bin = ProjectSettings.globalize_path(model_bin_path)

	var ok = _native_runner.load_model(absolute_param, absolute_bin)
	if not ok:
		push_error("Failed to load ncnn model.")
		return

	print("Model loaded")
	var sample_obs := PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
	if action_mode == ActionMode.DISCRETE_ARGMAX:
		print("Sample discrete action: ", _native_runner.run_discrete_action(sample_obs))
	else:
		print("Sample inference output: ", _native_runner.run_inference(sample_obs))

func get_action(observations: Array[float]) -> Variant:
	if _native_runner == null or not _native_runner.is_model_loaded():
		push_error("NcnnAgentHelper.get_action: native runner is not ready.")
		return null

	var packed_obs := PackedFloat32Array(observations)
	if action_mode == ActionMode.DISCRETE_ARGMAX:
		return _native_runner.run_discrete_action(packed_obs)

	return _native_runner.run_inference(packed_obs)

func _process(_delta: float) -> void:
	pass
