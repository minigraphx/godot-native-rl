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
@export var input_shape: PackedInt32Array = PackedInt32Array()
@export_enum("Continuous", "Discrete Argmax") var action_mode: int = ActionMode.CONTINUOUS

var _native_runner: NcnnRunner

func _ready() -> void:
	_native_runner = NcnnRunner.new()
	add_child(_native_runner)
	_native_runner.input_blob_name = input_blob_name
	_native_runner.output_blob_name = output_blob_name
	_native_runner.input_shape = input_shape

	var absolute_param := ProjectSettings.globalize_path(model_param_path)
	var absolute_bin := ProjectSettings.globalize_path(model_bin_path)
	if not _native_runner.load_model(absolute_param, absolute_bin):
		push_error("NcnnAgentHelper: failed to load ncnn model.")

func get_action(observations: Array[float]) -> Variant:
	if _native_runner == null or not _native_runner.is_model_loaded():
		push_error("NcnnAgentHelper.get_action: model not loaded.")
		return null

	var packed_obs := PackedFloat32Array(observations)
	if action_mode == ActionMode.DISCRETE_ARGMAX:
		return _native_runner.run_discrete_action(packed_obs)
	return _native_runner.run_inference(packed_obs)

func get_action_from_image(image: Image, normalize_to_zero_one: bool = true) -> PackedFloat32Array:
	if _native_runner == null or not _native_runner.is_model_loaded():
		push_error("NcnnAgentHelper.get_action_from_image: native runner is not ready.")
		return PackedFloat32Array()
	return _native_runner.run_inference_image(image, normalize_to_zero_one)
