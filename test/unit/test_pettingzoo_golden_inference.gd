extends SceneTree
# Golden inference regression for the two PettingZoo-path multi-policy hide & seek ncnn models
# (scripts/train_pettingzoo.sh -> export_to_ncnn.py --via torchscript ->
# models/pettingzoo_{seeker,hider}.ncnn.*). Mirrors
# test_hide_seek_multipolicy_golden_inference.gd (the custom-PPO multi-policy example): loads each
# model via NcnnRunner and asserts run_discrete_action() returns the captured argmax for 5 fixed
# observations. ncnn<->torch.jit parity (50/50 argmax, atol=1e-2) was verified at conversion time
# by export_to_ncnn.py. If this fails after a retrain/model swap, recapture the goldens from the
# new models and update them here.
#
# obs is 15 floats; index 14 is the role flag (seeker=1.0, hider=0.0). Each policy was trained only
# on its own role's observations, so it is probed with its role flag set accordingly.

const Harness = preload("res://test/harness.gd")

const OBS: Array = [
	[0.5,-0.1,0.7,0.4,-0.8,0.1,0.2,0.3,0.0,0.1,0.5,0.6,0.2,0.9, 1.0],
	[0.9,0.5,0.5,-0.7,-0.1,0.3,0.1,0.2,0.4,0.0,0.7,0.1,0.3,0.2, 1.0],
	[-0.2,0.8,0.2,0.6,-0.1,0.5,0.6,0.1,0.2,0.3,0.4,0.5,0.6,0.1, 1.0],
	[-0.5,0.1,-0.8,0.6,0.2,0.1,0.0,0.2,0.3,0.4,0.1,0.2,0.3,0.4, 1.0],
	[0.5,-0.2,0.9,0.7,0.5,0.2,0.1,0.3,0.4,0.5,0.6,0.1,0.2,0.3, 1.0],
]
# Captured from the real ncnn deploy path in Task 5 of the implementation plan (run the capture
# script against the trained fixtures, paste the printed arrays here).
const EXPECTED_SEEKER: Array = [0, 4, 0, 0, 0]  # captured from the real ncnn deploy path (role 1.0)
const EXPECTED_HIDER: Array  = [1, 0, 0, 1, 1]  # captured from the real ncnn deploy path (role 0.0)

func _check(h, tag: String, base: String, expected: Array, role_flag: float) -> void:
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(base + ".param"),
		ProjectSettings.globalize_path(base + ".bin"))
	h.assert_true(ok, "%s model loads" % tag)
	h.assert_eq(expected.size(), OBS.size(), "%s goldens captured (one per obs)" % tag)
	if ok and expected.size() == OBS.size():
		for i in range(OBS.size()):
			var o: Array = OBS[i].duplicate()
			o[14] = role_flag
			var got := runner.run_discrete_action(PackedFloat32Array(o))
			h.assert_eq(got, int(expected[i]), "%s golden argmax #%d" % [tag, i])
	runner.free()

func _initialize() -> void:
	var h := Harness.new()
	_check(h, "seeker", "res://models/pettingzoo_seeker.ncnn", EXPECTED_SEEKER, 1.0)
	_check(h, "hider", "res://models/pettingzoo_hider.ncnn", EXPECTED_HIDER, 0.0)
	h.finish(self)
