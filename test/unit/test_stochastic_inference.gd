extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")
const Stub = preload("res://test/unit/stub_agent.gd")

# Fake runner returning fixed logits over the stub's "move" space.
class FakeRunner:
	var loaded := true
	# Length must match stub_agent.gd get_action_space() ({"move": size 5}). Peak is index 1.
	var output := PackedFloat32Array([0.5, 2.0, 0.5, 1.0, 0.5])
	func is_model_loaded() -> bool:
		return loaded
	func run_inference(_input) -> PackedFloat32Array:
		return output

func _initialize() -> void:
	var h := Harness.new()

	# Defaults: deterministic, with a live RNG instance.
	var core := NcnnControllerCore.new()
	h.assert_eq(core.deterministic_inference, true, "core defaults to deterministic")
	h.assert_true(core.rng != null, "core has an RNG instance")

	# setup_rng(seed): fixed seed is reproducible; setup_rng(-1) randomizes.
	core.setup_rng(42)
	var first: float = core.rng.randf()
	core.setup_rng(42)
	var again: float = core.rng.randf()
	h.assert_true(absf(first - again) < 1e-9, "setup_rng(42) is reproducible")
	# setup_rng(-1) takes the randomize path: rng must stay usable and yield a draw in [0,1).
	core.setup_rng(-1)
	var rand_draw: float = core.rng.randf()
	h.assert_true(core.rng != null and rand_draw >= 0.0 and rand_draw < 1.0,
		"setup_rng(-1) randomizes, rng still usable")

	# Controller exports default to deterministic.
	var dflt := Stub.new()
	h.assert_eq(dflt.deterministic_inference, true, "controller export defaults deterministic")
	h.assert_eq(dflt.inference_seed, -1, "controller export inference_seed defaults -1")
	dflt.free()

	# Deterministic wiring -> argmax (peak of the fixed logits is index 1).
	var det := Stub.new()
	det.set_ncnn_runner_for_test(FakeRunner.new())
	det.set_stochastic_for_test(true, 0)  # seed irrelevant on the deterministic (argmax) path
	det.infer_and_act()
	h.assert_eq(det.last_action, {"move": 1}, "deterministic controller -> argmax index 1")
	det.free()

	# Stochastic + same seed on two controllers -> identical sampled action (reproducible),
	# and the sampled index is a valid bucket in [0,5).
	var s1 := Stub.new(); s1.set_ncnn_runner_for_test(FakeRunner.new()); s1.set_stochastic_for_test(false, 77)
	var s2 := Stub.new(); s2.set_ncnn_runner_for_test(FakeRunner.new()); s2.set_stochastic_for_test(false, 77)
	s1.infer_and_act(); s2.infer_and_act()
	h.assert_eq(s1.last_action, s2.last_action, "same seed -> identical sampled action")
	var picked: int = s1.last_action["move"]
	h.assert_true(picked >= 0 and picked < 5, "sampled action in [0,5)")
	s1.free(); s2.free()

	h.finish(self)
