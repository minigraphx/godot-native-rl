extends SceneTree
# Golden regression for native recurrent inference: drives the synthetic LSTM through the controller
# core with state carried frame-to-frame, asserting each step's argmax matches the torch golden, and
# that reset_recurrent_state() reproduces step 0. Mirrors test_image_inference_golden.gd.
# Regenerate fixture with: .venv-train/bin/python scripts/make_synthetic_lstm.py

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/recurrent_stub_agent.gd")
const PARAM := "res://models/synthetic_lstm.ncnn.param"
const BIN := "res://models/synthetic_lstm.ncnn.bin"
const SIDECAR := "res://models/synthetic_lstm.recurrent.json"
const GOLDEN := "res://models/synthetic_lstm_golden.json"

func _load_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	return {} if f == null else JSON.parse_string(f.get_as_text())

func _initialize() -> void:
	var h := Harness.new()
	var golden := _load_json(GOLDEN)
	h.assert_true(not golden.is_empty(), "golden loads")

	var runner := NcnnRunner.new()
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "synthetic LSTM loads")
	if not ok:
		h.finish(self)
		return

	var agent = Stub.new()
	agent.set_ncnn_runner_for_test(runner)
	agent.set_recurrent_contract_for_test(SIDECAR)

	var steps: Array = golden["steps"]
	var first_action = null
	for i in steps.size():
		var step: Dictionary = steps[i]
		agent.obs_to_return = PackedFloat32Array(step["obs"])
		agent.infer_and_act()
		h.assert_eq(agent.last_action, {"move": int(step["argmax"])}, "step %d argmax matches golden" % i)
		if i == 0:
			first_action = agent.last_action

	# Reset reproduces step 0 (state cleared, same obs -> same action).
	agent.reset_recurrent_state()
	agent.obs_to_return = PackedFloat32Array(steps[0]["obs"])
	agent.infer_and_act()
	h.assert_eq(agent.last_action, first_action, "reset_recurrent_state reproduces step 0")

	# Numerical state-carry coverage: drive run_inference_multi directly, carrying state exactly
	# like the core does, and assert per-step LOGIT parity with the torch golden (atol=1e-2). The
	# argmax checks above can't catch a no-state-carry regression (all golden argmaxes are 1), but
	# the carried-state logits diverge and compound if state is dropped — this catches that.
	var sc := _load_json(SIDECAR)
	var obs_shape := PackedInt32Array(sc["obs_shape"])
	var pairs: Array = sc["state_pairs"]
	# Zero-initialized carried state, keyed by input blob name.
	var carried := {}
	for pair in pairs:
		var n := 1
		for d in pair["shape"]:
			n *= int(d)
		var z := PackedFloat32Array()
		z.resize(n)
		carried[pair["in"]] = z
	var all_within := true
	for i in steps.size():
		var step: Dictionary = steps[i]
		var inputs: Array = [{"name": sc["obs_input"], "data": PackedFloat32Array(step["obs"]), "shape": obs_shape}]
		var out_names := PackedStringArray([sc["action_output"]])
		for pair in pairs:
			inputs.append({"name": pair["in"], "data": carried[pair["in"]], "shape": PackedInt32Array(pair["shape"])})
			out_names.append(pair["out"])
		var result: Dictionary = runner.run_inference_multi(inputs, out_names)
		if result.is_empty():
			all_within = false
			break
		var logits: PackedFloat32Array = result[sc["action_output"]]
		var ref: Array = step["logits"]
		if logits.size() != ref.size():
			all_within = false
			break
		for j in range(logits.size()):
			if absf(logits[j] - float(ref[j])) > 1e-2:
				all_within = false
		for pair in pairs:
			carried[pair["in"]] = result[pair["out"]]
	h.assert_true(all_within, "carried-state logits match golden within atol 1e-2 across all steps")

	agent.free()
	h.finish(self)
