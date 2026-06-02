extends SceneTree

const Harness = preload("res://test/harness.gd")
const Controller2D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd")

# Minimal size-6 agent (matches models/synthetic_vecnormalize.json obs_size) to exercise the
# real JSON loader -> core -> normalize path end-to-end.
class Agent6 extends Controller2D:
	var last_action = null
	func get_obs() -> Dictionary:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
	func get_reward() -> float:
		return 0.0
	func get_action_space() -> Dictionary:
		return {"move": {"size": 3, "action_type": "discrete"}}
	func set_action(action) -> void:
		last_action = action

class CaptureRunner:
	var last_input := PackedFloat32Array()
	func is_model_loaded() -> bool:
		return true
	func run_inference(input) -> PackedFloat32Array:
		last_input = input
		return PackedFloat32Array([0.1, 0.2, 0.9])  # argmax 2 over size-3

func _initialize() -> void:
	var h := Harness.new()

	# Load the committed stats fixture through the REAL wrapper loader.
	var agent = Agent6.new()
	agent.obs_norm_stats_path = "res://models/synthetic_vecnormalize.json"
	agent._load_obs_norm_stats()

	# Feed an all-zero size-6 obs; the runner should receive the normalized vector
	# (= (0 - mean)/sqrt(var+eps), clipped). Compare against the fixture's own mean/var.
	var cr := CaptureRunner.new()
	agent.set_ncnn_runner_for_test(cr)
	agent.infer_and_act()
	h.assert_eq(cr.last_input.size(), 6, "loader applied size-6 stats to obs before inference")

	# Recompute expected normalization for obs=0 from the fixture file.
	var f := FileAccess.open("res://models/synthetic_vecnormalize.json", FileAccess.READ)
	var stats: Dictionary = JSON.parse_string(f.get_as_text())
	f.close()
	var mean: Array = stats["mean"]
	var var_: Array = stats["var"]
	var eps: float = float(stats["epsilon"])
	var clip: float = float(stats["clip_obs"])
	var max_diff := 0.0
	for i in 6:
		var expected: float = clampf((0.0 - float(mean[i])) / sqrt(float(var_[i]) + eps), -clip, clip)
		max_diff = maxf(max_diff, absf(cr.last_input[i] - expected))
	h.assert_true(max_diff < 1e-5, "loaded stats normalize obs correctly (max diff %f)" % max_diff)

	# Backward-compat: a bad path leaves stats unset (no crash), so obs passes through raw.
	var agent2 = Agent6.new()
	agent2.obs_norm_stats_path = "res://models/does_not_exist.json"
	agent2._load_obs_norm_stats()  # should push_error but not crash
	var cr2 := CaptureRunner.new()
	agent2.set_ncnn_runner_for_test(cr2)
	agent2.infer_and_act()
	h.assert_true(absf(cr2.last_input[0]) < 1e-6, "missing stats file -> raw obs (no crash)")

	agent.free()
	agent2.free()
	h.finish(self)
