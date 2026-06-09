extends SceneTree
# End-to-end #64 guard on the REAL trained net: with deterministic_inference=false + a fixed
# inference_seed and the shipped action_dist sidecar, two FlyBy agents over identical observations
# sample IDENTICAL continuous actions (reproducible), and those sampled actions differ from the
# deterministic mean (sampling is actually live). Headless: drives the controller via infer_and_act.

const Harness = preload("res://test/harness.gd")
const AgentScript = preload("res://examples/fly_by/fly_by_agent.gd")

const PARAM := "res://examples/fly_by/models/fly_by_policy.ncnn.param"
const BIN := "res://examples/fly_by/models/fly_by_policy.ncnn.bin"
const DIST := "res://examples/fly_by/models/fly_by_action_dist.json"

func _make_agent(deterministic: bool, seed_value: int):
	var a = AgentScript.new()
	a.control_mode = 3  # NCNN_INFERENCE
	a.model_param_path = PARAM
	a.model_bin_path = BIN
	a.action_dist_stats_path = DIST
	a.deterministic_inference = deterministic
	a.inference_seed = seed_value
	# root.add_child() does NOT enter the tree in a --script SceneTree (_ready never fires), so call
	# _ready() directly to run the real NCNN_INFERENCE setup: load the ncnn model + std sidecar + rng.
	a._ready()
	return a

func _initialize() -> void:
	var h := Harness.new()

	var det = _make_agent(true, -1)
	h.assert_true(det._ncnn_runner != null and det._ncnn_runner.is_model_loaded(), "model loads")
	det.infer_and_act()
	var det_pitch: float = det.get_pitch_for_test()

	# Two stochastic agents, same fixed seed -> identical sampled actions.
	var s1 = _make_agent(false, 123)
	var s2 = _make_agent(false, 123)
	s1.infer_and_act()
	s2.infer_and_act()
	h.assert_true(absf(s1.get_pitch_for_test() - s2.get_pitch_for_test()) < 1e-6
		and absf(s1.get_turn_for_test() - s2.get_turn_for_test()) < 1e-6,
		"same seed -> identical sampled action")
	# Sampling actually perturbs away from the deterministic mean (std > 0 in the sidecar).
	h.assert_true(absf(s1.get_pitch_for_test() - det_pitch) > 1e-5,
		"stochastic sample differs from the deterministic mean")

	det.free(); s1.free(); s2.free()
	h.finish(self)
