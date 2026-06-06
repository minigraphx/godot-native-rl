extends SceneTree
# Golden inference regression for the SampleFactory-trained Chase The Target ncnn model
# (scripts/train_sf.sh -> export_sf_to_torchscript.py -> export_to_ncnn.py ->
# models/chase_sf_policy.ncnn.*). Mirrors test_chase_cleanrl_golden_inference.gd (CleanRL) and
# test_chase_golden_inference.gd (SB3): loads the shipped model via NcnnRunner and asserts
# run_discrete_action() returns the expected argmax for 5 fixed observations. Values were captured
# from the real ncnn deploy path and the TorchScript<->ncnn parity was verified (50/50 argmax match,
# atol=1e-2) by scripts/export_to_ncnn.py at conversion time. If this fails after a retrain/model
# swap, recapture the goldens from the new model and update them here.
#
# The 5 observations are the same fixed set as the SB3 and CleanRL chase golden tests, so all three
# backends' models are probed on identical inputs (the expected actions may differ — different
# trained weights). This pins SF deploy-side behavior against accidental conversion/runtime
# regressions without re-training (the committed fixture; no training run needed).

const MODEL_PARAM := "res://models/chase_sf_policy.ncnn.param"
const MODEL_BIN   := "res://models/chase_sf_policy.ncnn.bin"
const Harness = preload("res://test/harness.gd")

# Each inner pair is [obs_values, expected_action].
const GOLDEN: Array = [
	[[ 0.5479, -0.1222,  0.7172,  0.3947, -0.8116], 4],
	[[ 0.9512,  0.5223,  0.5721, -0.7438, -0.0992], 1],
	[[-0.2584,  0.8535,  0.2877,  0.6455, -0.1132], 2],
	[[-0.5455,  0.1092, -0.8724,  0.6553,  0.2633], 3],
	[[ 0.5162, -0.2909,  0.9414,  0.7862,  0.5568], 4],
]

func _initialize() -> void:
	var h := Harness.new()

	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var param := ProjectSettings.globalize_path(MODEL_PARAM)
	var bin  := ProjectSettings.globalize_path(MODEL_BIN)
	var ok := runner.load_model(param, bin)
	h.assert_true(ok, "chase_sf_policy model loads")

	if ok:
		for pair in GOLDEN:
			var obs := PackedFloat32Array(pair[0])
			var expected: int = pair[1]
			var got := runner.run_discrete_action(obs)
			h.assert_eq(got, expected, "golden argmax for obs %s" % str(pair[0]))

	runner.free()
	h.finish(self)
