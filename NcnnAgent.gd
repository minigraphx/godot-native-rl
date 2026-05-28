# Called when the node enters the scene tree for the first time.
class_name NcnnAgentHelper
extends Node

@export_file("*.param") var model_param_path: String = "res://models/test_mlp.ncnn.param"
@export_file("*.bin") var model_bin_path: String = "res://models/test_mlp.ncnn.bin"
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"

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
	print("Model loaded")
	
	var out = _native_runner.run_inference(PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
	print("Inference output: ", out)


func get_action(observations: Array[float]) -> PackedFloat32Array:
	return _native_runner.run_inference(PackedFloat32Array(observations))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
