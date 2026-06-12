extends SceneTree
# Golden inference regression for the shipped quadruped_walk ncnn model (continuous control:
# 8 hinge-motor action means from a 29-dim obs). Loads it via NcnnRunner and asserts the raw
# output vector for FIXED observations matches a recorded baseline within tolerance — guards
# against silent regressions in the runner / TorchScript->ncnn conversion / model file. If the
# policy is retrained, flip RECORD=true, rerun, and paste the printed GOLDEN intentionally.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/quadruped_walk/models/quadruped_walk.ncnn.param"
const BIN := "res://examples/quadruped_walk/models/quadruped_walk.ncnn.bin"

const RECORD := false  # true -> print outputs for the OBS cases instead of asserting

# Fixed 29-dim observation cases (joints+vels+up+localvel+dir+contacts). Values are arbitrary but
# fixed; only that the conversion+runner reproduce the same outputs matters.
const OBS: Array = [
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.6,  1.0, 1.0, 1.0, 1.0],
	[0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2,  0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3,  0.05, 0.98, 0.0,  0.4, 0.0, 0.1,  0.0, 0.0, 0.55,  1.0, 0.0, 1.0, 0.0],
	[-0.3, 0.4, -0.2, 0.5, -0.3, 0.4, -0.2, 0.5,  -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6,  -0.1, 0.95, 0.05,  0.6, 0.1, 0.2,  0.0, 0.0, 0.5,  0.0, 1.0, 0.0, 1.0],
]

# Recorded baseline (8 action means per OBS case) from the shipped 6M-step v3 policy.
const GOLDEN: Array = [
	[-2.02734375, -1.509765625, -1.9306640625, 0.06768798828125, -2.888671875, -3.328125, -3.8671875, -3.958984375],
	[-1.90625, -1.9716796875, -1.892578125, 0.325439453125, -2.861328125, -2.046875, -3.455078125, -3.728515625],
	[-1.595703125, -1.4111328125, 0.61376953125, 1.7607421875, 0.26416015625, -2.607421875, -3.22265625, -2.3828125],
]

func _initialize() -> void:
	var h := Harness.new()
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "quadruped model loads")
	if ok:
		for i in range(OBS.size()):
			var out := runner.run_inference(PackedFloat32Array(OBS[i]))
			if RECORD:
				print("GOLDEN_%d = %s" % [i, JSON.stringify(Array(out))])
				continue
			var golden: Array = GOLDEN[i]
			h.assert_eq(out.size(), golden.size(), "case %d output size (8 motor means)" % i)
			var close := true
			for j in range(min(out.size(), golden.size())):
				# rtol 1e-2 with an atol floor of 1e-2: the baseline is recorded on macOS arm64;
				# CI runs Linux x86 ncnn whose SIMD paths drift more than a 1e-3 floor allows on
				# small-magnitude outputs. 1e-2 matches the continuous-mean tolerance the
				# algorithm-agnostic golden test uses (CI-proven cross-platform); real regressions
				# (wrong weights/conversion) differ by orders of magnitude more.
				if absf(out[j] - float(golden[j])) > 1e-2 * absf(float(golden[j])) + 1e-2:
					close = false
			h.assert_true(close, "case %d outputs within tolerance of golden" % i)
	runner.free()
	h.finish(self)
