extends SceneTree
# Golden inference regression for the shipped rover ncnn model. Loads it via NcnnRunner and
# asserts the argmax action for FIXED 8-dim observations (5 ray closeness + [sin, cos] goal
# bearing + normalized distance) matches a recorded baseline. Guards against silent regressions
# in the runner / conversion / model file. If the model is retrained, recompute these argmax
# values (set the blob names to in0/out0, run the four obs) and update GOLDEN intentionally.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/rover_3d/models/rover_policy.ncnn.param"
const BIN := "res://examples/rover_3d/models/rover_policy.ncnn.bin"

# Recorded baseline for the shipped rover_policy (225k-step checkpoint). [obs, expected argmax].
const GOLDEN: Array = [
	[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5], 1],
	[[0.2, 0.5, 0.9, 0.5, 0.2, 1.0, 0.0, 0.5], 2],
	[[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0, 0.2], 1],
	[[0.8, 0.8, 0.8, 0.8, 0.8, -1.0, 0.0, 0.9], 3],
]

func _initialize() -> void:
	var h := Harness.new()
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "rover model loads")
	if ok:
		var n := 0
		for pair in GOLDEN:
			var got := runner.run_discrete_action(PackedFloat32Array(pair[0]))
			h.assert_eq(got, pair[1], "golden case %d argmax" % n)
			n += 1
	runner.free()
	h.finish(self)
