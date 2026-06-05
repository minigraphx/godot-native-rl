extends SceneTree
# Golden regression guarding the algorithm-agnostic deploy contract (issue #45): non-PPO networks
# deploy through the SAME pure forward pass + ActionDecode path as PPO. Two committed synthetic
# fixtures exported via the real ncnn pipeline:
#   * synthetic_dqn  — discrete Q-net with UNBOUNDED Q-values -> argmax preserved end-to-end.
#   * synthetic_sac  — continuous actor (raw means) -> tanh(mean) via squash, end-to-end.
# Regenerate with:
#   .venv-train/bin/python scripts/make_synthetic_dqn.py
#   .venv-train/bin/python scripts/make_synthetic_sac.py

const Harness = preload("res://test/harness.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

const DQN_GOLDEN := "res://models/synthetic_dqn_golden.json"
const DQN_PARAM := "res://models/synthetic_dqn.ncnn.param"
const DQN_BIN := "res://models/synthetic_dqn.ncnn.bin"

const SAC_GOLDEN := "res://models/synthetic_sac_golden.json"
const SAC_PARAM := "res://models/synthetic_sac.ncnn.param"
const SAC_BIN := "res://models/synthetic_sac.ncnn.bin"

func _load_golden(path: String, h) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	h.assert_true(f != null, "golden json opens: %s" % path)
	if f == null:
		return {}
	return JSON.parse_string(f.get_as_text())

func _obs_of(data: Dictionary) -> PackedFloat32Array:
	var obs := PackedFloat32Array()
	for v in data["obs"]:
		obs.append(float(v))
	return obs

func _run(param: String, bin: String, obs: PackedFloat32Array, h, label: String) -> PackedFloat32Array:
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(param), ProjectSettings.globalize_path(bin))
	h.assert_true(ok, "%s model loads" % label)
	if not ok:
		return PackedFloat32Array()
	return runner.run_inference(obs)

func _initialize() -> void:
	var h := Harness.new()

	# --- DQN: unbounded Q-values -> argmax preserved through real fp32 ncnn. ---
	# (Distinct variable names per block — headless GDScript treats locals as function-scoped.)
	var dqn: Dictionary = _load_golden(DQN_GOLDEN, h)
	if not dqn.is_empty():
		var dqn_obs := _obs_of(dqn)
		var dqn_out := _run(DQN_PARAM, DQN_BIN, dqn_obs, h, "synthetic DQN")
		var dqn_golden: Array = dqn["output"]
		h.assert_eq(dqn_out.size(), dqn_golden.size(), "DQN output count matches golden")

		# Argmax is the behaviorally-meaningful invariant — assert it EXACTLY.
		var dqn_space := {"move": {"size": dqn_golden.size(), "action_type": "discrete"}}
		var dqn_decoded := ActionDecode.decode_actions(dqn_out, dqn_space)
		h.assert_eq(dqn_decoded.get("move", -1), int(dqn["argmax"]),
			"DQN unbounded Q-values -> argmax preserved end-to-end (same path as PPO/DQN)")

		# Raw-value parity held to a RELATIVE tolerance (proportional to each Q-value's magnitude),
		# more precise than a flat atol for large unbounded outputs.
		var rtol := 1e-2
		var atol_floor := 1e-3
		var rel_ok := dqn_out.size() == dqn_golden.size()
		for i in range(mini(dqn_out.size(), dqn_golden.size())):
			var g: float = float(dqn_golden[i])
			if absf(dqn_out[i] - g) > rtol * absf(g) + atol_floor:
				rel_ok = false
		h.assert_true(rel_ok, "DQN raw Q-values within relative tolerance (rtol=1e-2) of golden")

	# --- SAC: continuous actor raw means -> tanh(mean) via squash, through real ncnn. ---
	var sac: Dictionary = _load_golden(SAC_GOLDEN, h)
	if not sac.is_empty():
		var sac_obs := _obs_of(sac)
		var sac_out := _run(SAC_PARAM, SAC_BIN, sac_obs, h, "synthetic SAC")
		var sac_golden: Array = sac["output"]
		h.assert_eq(sac_out.size(), sac_golden.size(), "SAC output count matches golden")

		# Means are tanh-bounded and small -> standard atol=1e-2 closeness.
		var raw_ok := sac_out.size() == sac_golden.size()
		for i in range(mini(sac_out.size(), sac_golden.size())):
			if absf(sac_out[i] - float(sac_golden[i])) > 1e-2:
				raw_ok = false
		h.assert_true(raw_ok, "SAC raw means within atol 1e-2 of golden")

		# Squash decode -> tanh(mean), the SAC deterministic deploy.
		var sac_space := {"steer": {"size": sac_golden.size(), "action_type": "continuous", "squash": true}}
		var sac_decoded := ActionDecode.decode_actions(sac_out, sac_space)
		var sq_golden: Array = sac["squashed"]
		var sq_ok: bool = sac_decoded.has("steer") and sac_decoded["steer"].size() == sq_golden.size()
		for i in range(sq_golden.size()):
			if not sq_ok or absf(sac_decoded["steer"][i] - float(sq_golden[i])) > 1e-2:
				sq_ok = false
		h.assert_true(sq_ok, "SAC squashed actor -> tanh(mean) decode matches golden end-to-end")

	h.finish(self)
