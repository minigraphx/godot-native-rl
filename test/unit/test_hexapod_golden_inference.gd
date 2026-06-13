extends SceneTree
# Golden inference regression for the shipped hexapod walk ncnn model (#60 M3 — continuous control:
# 12 hinge-motor action means from a 39-dim obs). Fixed obs -> recorded baseline within tolerance;
# guards the runner / TorchScript->ncnn conversion / model file. Retrained? Flip RECORD=true, rerun,
# paste the printed GOLDEN intentionally.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/quadruped_walk/models/hexapod_walk.ncnn.param"
const BIN := "res://examples/quadruped_walk/models/hexapod_walk.ncnn.bin"

const RECORD := false

func _obs(pattern: Array, last: float) -> Array:
	var o: Array = []
	while o.size() < 38:
		o.append_array(pattern)
	o.resize(38)
	o.append(last)
	return o

func _initialize() -> void:
	var h := Harness.new()
	var OBS := [_obs([0.0], 0.0), _obs([0.1, -0.1], 1.0), _obs([-0.2, 0.15], 0.5)]
	var GOLDEN := [
		[1.953125, 1.463867, 0.628418, 1.383789, 1.317383, -0.235596, 1.332031, -1.68457, -1.366211, 0.347168, -0.208008, 1.981445],
		[3.154297, 2.011719, 0.741211, 1.958008, 0.187866, 1.151367, 0.631836, -0.459229, -0.992188, -0.200928, -2.265625, -0.181274],
		[5.300781, 1.773438, -0.340332, 1.637695, 2.222656, -0.755859, 1.082031, -1.967773, -1.243164, 1.786133, 0.036163, 1.533203],
	]
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "hexapod model loads")
	if ok:
		for i in range(OBS.size()):
			var out := runner.run_inference(PackedFloat32Array(OBS[i]))
			if RECORD:
				print("GOLDEN_%d = %s" % [i, JSON.stringify(Array(out))])
				continue
			var golden: Array = GOLDEN[i]
			h.assert_eq(out.size(), golden.size(), "case %d output size (12 motor means)" % i)
			var close := true
			for j in range(min(out.size(), golden.size())):
				# rtol/atol 5e-2: strong policy -> larger logits -> more cross-platform drift (macOS
				# arm64 baseline vs Linux x86 CI) than the quadruped; same continuous-mean approach.
				if absf(out[j] - float(golden[j])) > 5e-2 * absf(float(golden[j])) + 5e-2:
					close = false
			h.assert_true(close, "case %d outputs within tolerance of golden" % i)
	runner.free()
	h.finish(self)
