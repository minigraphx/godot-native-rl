extends SceneTree
# Golden inference regression for the shipped MA-POCA actor (#30 M2): the deployed shared actor maps
# a 16-dim coop_collect obs to 5 discrete move logits. Fixed obs -> recorded logits within tolerance,
# guarding the runner / TorchScript->ncnn conversion / model file against silent regressions. The
# actor is what deploys (the centralized critic is training-only and discarded). Retrained? Flip
# RECORD=true, rerun, paste the printed GOLDEN intentionally.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/coop_collect/models/coop_mapoca.ncnn.param"
const BIN := "res://examples/coop_collect/models/coop_mapoca.ncnn.bin"

const RECORD := false

const OBS: Array = [
	[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
	[0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5],
	[0.0, 0.0, 0.3, -0.2, 0.4, 0.1, 1.0, -0.3, 0.2, 0.0, 0.1, 0.5, 1.0, -0.4, -0.1, 0.0],
]

# Recorded baseline (5 move logits per obs) from the shipped 2M-step MA-POCA actor (#242 retrain).
const GOLDEN: Array = [
	[-2.359375, -4.679688, 7.125000, -3.273438, -2.925781],
	[0.037720, -0.836426, 7.617188, 1.229492, -7.429688],
	[-1.341797, -3.503906, 8.125000, -1.528320, -5.382812],
]

func _initialize() -> void:
	var h := Harness.new()
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "coop MA-POCA actor loads")
	if ok:
		for i in range(OBS.size()):
			var out := runner.run_inference(PackedFloat32Array(OBS[i]))
			if RECORD:
				print("GOLDEN_%d = %s" % [i, JSON.stringify(Array(out))])
				continue
			var golden: Array = GOLDEN[i]
			h.assert_eq(out.size(), golden.size(), "case %d output size (5 move logits)" % i)
			var close := true
			for j in range(min(out.size(), golden.size())):
				# rtol/atol 1e-2: baseline on macOS arm64, CI on Linux x86 (small MLP -> tiny drift,
				# unlike the visual CNN; same tolerance the other golden tests use).
				if absf(out[j] - float(golden[j])) > 1e-2 * absf(float(golden[j])) + 1e-2:
					close = false
			h.assert_true(close, "case %d outputs within tolerance of golden" % i)
	runner.free()
	h.finish(self)
