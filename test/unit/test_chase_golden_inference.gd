extends SceneTree
# Golden inference regression test for the Chase The Target trained ncnn model.
# Loads the shipped model via NcnnRunner and asserts that run_discrete_action()
# returns the expected argmax for 5 fixed observations (computed at conversion time
# and verified against onnxruntime). If this test fails after a model swap, re-run
# verify_ncnn_parity.py and recompute golden values with scripts/export_to_ncnn.py.
#
# Golden pairs (seed=42, obs in [-1,1]^5 -> expected discrete action):
#   obs=[ 0.5479,-0.1222, 0.7172, 0.3947,-0.8116] -> 2
#   obs=[ 0.9512, 0.5223, 0.5721,-0.7438,-0.0992] -> 1
#   obs=[-0.2584, 0.8535, 0.2877, 0.6455,-0.1132] -> 2
#   obs=[-0.5455, 0.1092,-0.8724, 0.6553, 0.2633] -> 3
#   obs=[ 0.5162,-0.2909, 0.9414, 0.7862, 0.5568] -> 2

const MODEL_PARAM := "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
const MODEL_BIN   := "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
const Harness = preload("res://test/harness.gd")

# PackedFloat32Array(...) is not a constant expression in GDScript 4 — use a plain Array
# of Arrays; each inner pair is [obs_values, expected_action].
const GOLDEN: Array = [
	[[ 0.5479, -0.1222,  0.7172,  0.3947, -0.8116], 2],
	[[ 0.9512,  0.5223,  0.5721, -0.7438, -0.0992], 1],
	[[-0.2584,  0.8535,  0.2877,  0.6455, -0.1132], 2],
	[[-0.5455,  0.1092, -0.8724,  0.6553,  0.2633], 3],
	[[ 0.5162, -0.2909,  0.9414,  0.7862,  0.5568], 2],
]

func _initialize() -> void:
	var h := Harness.new()

	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var param := ProjectSettings.globalize_path(MODEL_PARAM)
	var bin  := ProjectSettings.globalize_path(MODEL_BIN)
	var ok := runner.load_model(param, bin)
	h.assert_true(ok, "chase_the_target model loads")

	if ok:
		for pair in GOLDEN:
			var obs := PackedFloat32Array(pair[0])
			var expected: int = pair[1]
			var got := runner.run_discrete_action(obs)
			h.assert_eq(got, expected, "golden argmax for obs %s" % str(pair[0]))

	runner.free()
	h.finish(self)
