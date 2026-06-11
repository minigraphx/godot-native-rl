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

# Recorded baseline (8 action means per OBS case) from the shipped 5M-step policy.
const GOLDEN: Array = [
	[-1.419921875, 1.44140625, -0.50341796875, -0.97119140625, -0.427978515625, 0.481201171875, -1.5, -0.51611328125],
	[-0.8310546875, 0.654296875, 0.284912109375, 0.11181640625, -0.367919921875, -0.062103271484375, -0.80126953125, -0.74658203125],
	[-1.1708984375, 1.5068359375, -3.375, -1.9462890625, 0.3671875, 0.638671875, -0.90966796875, 0.0],
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
				if absf(out[j] - float(golden[j])) > 1e-2 * absf(float(golden[j])) + 1e-3:
					close = false
			h.assert_true(close, "case %d outputs within tolerance of golden" % i)
	runner.free()
	h.finish(self)
