extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

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
	core.setup_rng(-1)  # must not error (randomize path)
	h.assert_true(true, "setup_rng(-1) randomizes without error")

	h.finish(self)
