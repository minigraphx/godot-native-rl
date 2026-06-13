extends SceneTree
# Golden inference regression for the shipped quadruped_hurdles ncnn model (#60 M2 — continuous
# control: 8 hinge-motor action means from a 35-dim obs = M1's 29 + 6 hurdle closeness rays).
# Same contract as the M1 golden: fixed obs -> recorded outputs within tolerance. Retrained
# policy? Flip RECORD=true, rerun, paste the printed GOLDEN intentionally.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/quadruped_walk/models/quadruped_hurdles.ncnn.param"
const BIN := "res://examples/quadruped_walk/models/quadruped_hurdles.ncnn.bin"

const RECORD := false  # true -> print outputs for the OBS cases instead of asserting

# Fixed 35-dim observation cases (joints+vels+up+localvel+dir+contacts+rays). Arbitrary but
# fixed; the last 6 slots are the hurdle rays (0 = none in range .. ~1 = hurdle close).
const OBS: Array = [
	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.6,  1.0, 1.0, 1.0, 1.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	[0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2,  0.3, -0.3, 0.3, -0.3, 0.3, -0.3, 0.3, -0.3,  0.05, 0.98, 0.0,  0.4, 0.0, 0.1,  0.0, 0.0, 0.55,  1.0, 0.0, 1.0, 0.0,  0.5, 0.5, 0.5, 0.4, 0.4, 0.4],
	[-0.3, 0.4, -0.2, 0.5, -0.3, 0.4, -0.2, 0.5,  -0.6, 0.6, -0.6, 0.6, -0.6, 0.6, -0.6, 0.6,  -0.1, 0.95, 0.05,  0.6, 0.1, 0.2,  0.0, 0.0, 0.5,  0.0, 1.0, 0.0, 1.0,  0.9, 0.9, 0.9, 0.8, 0.8, 0.8],
]

# Recorded baseline (8 action means per OBS case) from the shipped 6M-step race-curriculum policy.
const GOLDEN: Array = [
	[-2.67578125, -1.427734375, 0.89892578125, 0.00360870361328125, -3.30859375, -3.75, -3.41796875, -2.62890625],
	[-2.236328125, -0.7626953125, 1.201171875, 0.2276611328125, -3.033203125, -1.52734375, -2.94140625, 0.395751953125],
	[1.744140625, -1.396484375, -1.53515625, 0.09033203125, 0.07232666015625, -2.390625, -1.33984375, -2.833984375],
]

func _initialize() -> void:
	var h := Harness.new()
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "quadruped hurdles model loads")
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
				# rtol 1e-2 + atol floor 1e-2: baseline recorded on macOS arm64, CI is Linux x86
				# (same tolerance rationale as the M1 golden — CI-proven cross-platform).
				if absf(out[j] - float(golden[j])) > 1e-2 * absf(float(golden[j])) + 1e-2:
					close = false
			h.assert_true(close, "case %d outputs within tolerance of golden" % i)
	runner.free()
	h.finish(self)
