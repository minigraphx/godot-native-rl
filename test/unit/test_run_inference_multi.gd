extends SceneTree
# Exercises NcnnRunner.run_inference_multi against the synthetic LSTM fixture: one zero-state
# step must reproduce golden step 0 (zero-init), and all 3 output blobs must come back.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://models/synthetic_lstm.ncnn.param"
const BIN := "res://models/synthetic_lstm.ncnn.bin"
const SIDECAR := "res://models/synthetic_lstm.recurrent.json"
const GOLDEN := "res://models/synthetic_lstm_golden.json"

func _load_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	return JSON.parse_string(f.get_as_text())

func _initialize() -> void:
	var h := Harness.new()
	var sc := _load_json(SIDECAR)
	var golden := _load_json(GOLDEN)
	h.assert_true(not sc.is_empty() and not golden.is_empty(), "sidecar + golden load")

	var runner := NcnnRunner.new()
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "synthetic LSTM loads")
	if ok:
		var obs_shape := PackedInt32Array(sc["obs_shape"])
		var hidden := int(golden["hidden"])
		var step0: Dictionary = golden["steps"][0]
		var obs := PackedFloat32Array(step0["obs"])
		var zero := PackedFloat32Array()
		zero.resize(hidden)  # zero-filled
		var pairs: Array = sc["state_pairs"]
		var inputs: Array = [{"name": sc["obs_input"], "data": obs, "shape": obs_shape}]
		var out_names := PackedStringArray([sc["action_output"]])
		for pair in pairs:
			inputs.append({"name": pair["in"], "data": zero, "shape": PackedInt32Array(pair["shape"])})
			out_names.append(pair["out"])

		var result: Dictionary = runner.run_inference_multi(inputs, out_names)
		h.assert_eq(result.size(), out_names.size(), "all output blobs returned")
		var logits: PackedFloat32Array = result[sc["action_output"]]
		var ref: Array = step0["logits"]
		h.assert_eq(logits.size(), ref.size(), "action logit count matches golden")
		var within := logits.size() == ref.size()
		for i in range(mini(logits.size(), ref.size())):
			if absf(logits[i] - float(ref[i])) > 1e-2:
				within = false
		h.assert_true(within, "zero-state logits within atol 1e-2 of golden step 0")

		# Error path: missing required key -> empty dict.
		var bad: Array = [{"name": sc["obs_input"], "data": obs}]  # no shape
		h.assert_true(runner.run_inference_multi(bad, out_names).is_empty(), "missing shape -> empty result")
	h.finish(self)
