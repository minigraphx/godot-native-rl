extends SceneTree
# Golden inference regression for the two multi-policy hide & seek ncnn models
# (scripts/train_hide_seek_multipolicy.py -> export_to_ncnn.py --via torchscript). Loads each model
# via NcnnRunner and asserts run_discrete_action() returns the captured argmax for 5 fixed
# observations. ncnn<->torch.jit parity (50/50 argmax) was verified at conversion time by
# export_to_ncnn.py. If this fails after a retrain/model swap, recapture the goldens from the new
# models and update them here.
#
# obs is 15 floats; index 14 is the role flag (seeker=1.0, hider=0.0). Each policy was trained only
# on its own role's observations, so it is probed with its role flag set accordingly.
#
# The obs are chosen so BOTH policies decide with a comfortable top-2 logit margin (>= ~0.86 here):
# a near-tie argmax can flip between CPU architectures (mac arm64 capture vs Linux x86_64 CI) under
# ncnn's float math, which made an earlier near-tie golden (margin 0.02) fail only on CI (#241).

const Harness = preload("res://test/harness.gd")

const OBS: Array = [
	[0.04,-0.62,0.32,0.88,-0.88,-0.29,0.62,0.84,0.41,0.38,-0.47,0.76,0.01,0.82, 1.0],
	[-0.1,-0.27,0.43,-0.47,0.74,0.7,0.67,0.75,-0.06,0.24,-0.63,-0.39,-0.94,0.45, 1.0],
	[0.76,-0.94,0.67,-0.11,0.4,-0.21,-0.33,0.5,0.84,0.27,0.61,0.8,0.52,0.49, 1.0],
	[0.57,-0.34,-0.88,0.27,-0.58,0.52,0.99,0.36,-0.98,-0.01,0.16,0.16,-0.65,0.34, 1.0],
	[-0.5,-0.4,-0.23,0.34,-0.67,0.44,-0.46,0.11,0.49,0.45,-0.15,0.27,-0.4,0.65, 1.0],
]
const EXPECTED_SEEKER: Array = [0, 0, 4, 0, 0]  # captured from the real ncnn deploy path (role 1.0)
const EXPECTED_HIDER: Array  = [4, 4, 3, 4, 4]  # captured from the real ncnn deploy path (role 0.0)

func _check(h, tag: String, base: String, expected: Array, role_flag: float) -> void:
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(base + ".param"),
		ProjectSettings.globalize_path(base + ".bin"))
	h.assert_true(ok, "%s model loads" % tag)
	if ok:
		for i in range(OBS.size()):
			var o: Array = OBS[i].duplicate()
			o[14] = role_flag
			var got := runner.run_discrete_action(PackedFloat32Array(o))
			h.assert_eq(got, int(expected[i]), "%s golden argmax #%d" % [tag, i])
	runner.free()

func _initialize() -> void:
	var h := Harness.new()
	_check(h, "seeker", "res://examples/hide_and_seek/models/hide_seek_seeker.ncnn", EXPECTED_SEEKER, 1.0)
	_check(h, "hider", "res://examples/hide_and_seek/models/hide_seek_hider.ncnn", EXPECTED_HIDER, 0.0)
	h.finish(self)
