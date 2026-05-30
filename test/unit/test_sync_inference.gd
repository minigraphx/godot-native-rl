extends SceneTree

const Harness = preload("res://test/harness.gd")
const SyncScript = preload("res://addons/godot_native_rl/sync.gd")

# Fake agent recording infer_and_act / reset calls.
class FakeInferenceAgent:
	var infer_calls := 0
	var done := false
	var needs_reset := false
	func infer_and_act() -> void:
		infer_calls += 1
	func get_done() -> bool:
		return done
	func set_done_false() -> void:
		done = false
	func reset() -> void:
		pass

func _initialize() -> void:
	var h := Harness.new()

	h.assert_true(SyncScript.ControlModes.has("NCNN_INFERENCE"), "Sync NCNN_INFERENCE enum value exists")

	var s := SyncScript.new()
	var agent := FakeInferenceAgent.new()
	s.agents_inference = [agent]
	s._inference_process()
	h.assert_eq(agent.infer_calls, 1, "inference step calls infer_and_act once")

	# A done inference agent has its flag cleared by the inference step.
	agent.done = true
	s._inference_process()
	h.assert_eq(agent.done, false, "done flag cleared after inference step")

	s.free()
	h.finish(self)
