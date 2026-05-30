extends SceneTree

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/stub_agent.gd")

# Minimal fake that mimics NcnnRunner.run_discrete_action.
class FakeRunner:
	var loaded := true
	var forced_index := 3
	func is_model_loaded() -> bool:
		return loaded
	func run_discrete_action(_input) -> int:
		return forced_index

func _initialize() -> void:
	var h := Harness.new()

	h.assert_true(Stub.ControlModes.has("NCNN_INFERENCE"), "NCNN_INFERENCE enum value exists")

	var a = Stub.new()
	a.set_ncnn_runner_for_test(FakeRunner.new())
	a.infer_and_act()
	h.assert_eq(a.last_action, {"move": 3}, "infer_and_act sets {move: argmax}")

	a.free()

	# infer_and_act on an agent with no runner is a safe no-op.
	var b = Stub.new()
	b.infer_and_act()
	h.assert_eq(b.last_action, null, "infer_and_act with no runner leaves last_action null")
	b.free()

	h.finish(self)
