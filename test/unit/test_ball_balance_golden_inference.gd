extends SceneTree
# Golden inference regression for the shipped 3DBall ncnn model (#47): continuous 2-out (tilt
# means) from an 8-dim obs, asserted within cross-platform tolerance (rtol 1e-2 + atol 1e-2 —
# baseline recorded on macOS arm64, CI runs Linux x86 ncnn; quadruped lesson). If retrained,
# flip RECORD=true, rerun, paste the printed GOLDEN values.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/3dball/models/ball_balance.ncnn.param"
const BIN := "res://examples/3dball/models/ball_balance.ncnn.bin"

const RECORD := false

const OBS: Array = [
	[0.0, 0.0,  0.0, 0.7, 0.0,  0.0, 0.0, 0.0],
	[0.2, -0.1,  1.5, 0.6, -1.0,  0.8, -0.2, 0.5],
	[-0.3, 0.3,  -2.0, 0.5, 2.0,  -1.5, 0.0, -1.0],
]

const GOLDEN: Array = [
	[0.09075927734375, -0.70654296875],
	[-0.10333251953125, 1.794921875],
	[0.1796875, -1.9755859375],
]

func _initialize() -> void:
	var h := Harness.new()
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "3dball model loads")
	if ok:
		for i in range(OBS.size()):
			var out := runner.run_inference(PackedFloat32Array(OBS[i]))
			if RECORD:
				print("GOLDEN_%d = %s" % [i, JSON.stringify(Array(out))])
				continue
			var golden: Array = GOLDEN[i]
			h.assert_eq(out.size(), golden.size(), "case %d output size (2 tilt means)" % i)
			var close := true
			for j in range(min(out.size(), golden.size())):
				if absf(out[j] - float(golden[j])) > 1e-2 * absf(float(golden[j])) + 1e-2:
					close = false
			h.assert_true(close, "case %d within tolerance of golden" % i)
	runner.free()
	h.finish(self)
