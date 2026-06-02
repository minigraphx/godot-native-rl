extends SceneTree
# Golden regression for native continuous-action inference: loads the committed synthetic MLP,
# asserts NcnnRunner.run_inference() matches the onnxruntime golden output within atol=1e-2
# (numerical closeness, not argmax), then checks ActionDecode continuous decoding (raw + tanh).
# Regenerate with: .venv-train/bin/python scripts/make_synthetic_continuous.py

const Harness = preload("res://test/harness.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")
const GOLDEN := "res://models/synthetic_continuous_golden.json"
const PARAM := "res://models/synthetic_continuous.ncnn.param"
const BIN := "res://models/synthetic_continuous.ncnn.bin"

func _initialize() -> void:
	var h := Harness.new()

	var f := FileAccess.open(GOLDEN, FileAccess.READ)
	h.assert_true(f != null, "golden json opens")
	if f == null:
		h.finish(self)
		return
	var data: Dictionary = JSON.parse_string(f.get_as_text())

	var obs := PackedFloat32Array()
	for v in data["obs"]:
		obs.append(float(v))

	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "synthetic continuous model loads")
	if ok:
		var output: PackedFloat32Array = runner.run_inference(obs)
		var golden: Array = data["output"]
		h.assert_eq(output.size(), golden.size(), "output count matches golden")
		var within := output.size() == golden.size()
		for i in range(mini(output.size(), golden.size())):
			if absf(output[i] - float(golden[i])) > 1e-2:
				within = false
		h.assert_true(within, "output within atol 1e-2 of onnxruntime golden (numerical closeness)")

		# Continuous decode, no squash -> raw values (within tolerance of golden output).
		var space := {"steer": {"size": golden.size(), "action_type": "continuous"}}
		var raw := ActionDecode.decode_actions(output, space)
		var raw_ok: bool = raw.has("steer") and raw["steer"].size() == golden.size()
		for i in range(golden.size()):
			if not raw_ok or absf(raw["steer"][i] - float(golden[i])) > 1e-2:
				raw_ok = false
		h.assert_true(raw_ok, "continuous no-squash decode matches golden output")

		# Continuous decode, squash -> tanh(values) (within tolerance of golden squashed).
		var space_sq := {"steer": {"size": golden.size(), "action_type": "continuous", "squash": true}}
		var sq := ActionDecode.decode_actions(output, space_sq)
		var sq_golden: Array = data["squashed"]
		var sq_ok: bool = sq.has("steer") and sq["steer"].size() == sq_golden.size()
		for i in range(sq_golden.size()):
			if not sq_ok or absf(sq["steer"][i] - float(sq_golden[i])) > 1e-2:
				sq_ok = false
		h.assert_true(sq_ok, "continuous squash decode matches tanh(golden)")

	h.finish(self)
