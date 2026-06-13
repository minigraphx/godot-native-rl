extends SceneTree
# Golden inference regression for the shipped MA-POCA posthumous-credit actor (#30 M3): the deployed
# shared actor maps a 17-dim early-finish obs (M2's 16 + a per-agent active flag) to 5 move logits.
# Fixed obs -> recorded logits within tolerance, guarding the runner / conversion / model file.
# Retrained? Flip RECORD=true, rerun, paste the printed GOLDEN intentionally.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://examples/coop_collect/models/coop_mapoca_bank.ncnn.param"
const BIN := "res://examples/coop_collect/models/coop_mapoca_bank.ncnn.bin"

const RECORD := false

const OBS: Array = [
	[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0],
	[0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 1.0],
	[0.0, 0.0, 0.3, -0.2, 0.4, 0.1, 1.0, -0.3, 0.2, 0.0, 0.1, 0.5, 1.0, -0.4, -0.1, 0.0, 0.0],
]

# Recorded baseline (5 move logits per obs) from the shipped 1.5M-step early-finish actor.
const GOLDEN: Array = [
	[-5.710938, -6.972656, 1.84082, -6.546875, 3.607422],
	[-4.867188, -5.96875, 1.053711, -5.585938, 3.660156],
	[-5.464844, -6.679688, -0.126343, -6.296875, 5.574219],
]

func _initialize() -> void:
	var h := Harness.new()
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "coop MA-POCA bank actor loads")
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
				# rtol/atol 2e-2: strong policy -> larger logits -> a touch more cross-platform drift
				# (macOS arm64 baseline vs Linux x86 CI) than the M2 net; argmax is exact regardless.
				if absf(out[j] - float(golden[j])) > 2e-2 * absf(float(golden[j])) + 2e-2:
					close = false
			h.assert_true(close, "case %d outputs within tolerance of golden" % i)
	runner.free()
	h.finish(self)
