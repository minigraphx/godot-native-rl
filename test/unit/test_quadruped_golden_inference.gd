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

# Recorded baseline (8 action means per OBS case) from the finish-reaching 8M-step policy (#252).
const GOLDEN: Array = [
	[-0.7724609375, -0.60888671875, 0.13427734375, 0.599609375, -1.875, -3.77734375, -3.435546875, 1.1064453125],
	[-1.169921875, -0.85302734375, 0.810546875, 1.6064453125, -1.845703125, -2.533203125, -2.1953125, 0.5048828125],
	[0.34033203125, -0.83154296875, 0.91845703125, 0.5078125, -0.302978515625, -3.5234375, -2.919921875, -1.3642578125],
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
