extends SceneTree
# reload_model() on the controllers (#29): runtime policy swap for self-play ghosts.
# Uses the committed chase fixtures (5-in -> 5-out MLPs) so real ncnn loads are exercised.

const Harness = preload("res://test/harness.gd")
const Controller2D = preload("res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd")

const CHASE_PARAM := "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
const CHASE_BIN := "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
const DUMMY_PARAM := "res://examples/chase_the_target/models/chase_dummy.ncnn.param"
const DUMMY_BIN := "res://examples/chase_the_target/models/chase_dummy.ncnn.bin"

func _initialize() -> void:
	var h = Harness.new()

	var ctrl = Controller2D.new()
	get_root().add_child(ctrl)

	# Initial load via reload_model (the path _setup_ncnn_runner now delegates to).
	h.assert_true(ctrl.reload_model(CHASE_PARAM, CHASE_BIN), "initial load succeeds")
	h.assert_true(ctrl._ncnn_runner != null and ctrl._ncnn_runner.is_model_loaded(), "runner has a net")
	var out1: PackedFloat32Array = ctrl._ncnn_runner.run_inference(PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5]))
	h.assert_eq(out1.size(), 5, "trained net produces 5 logits")

	# Swap to a different committed net.
	h.assert_true(ctrl.reload_model(DUMMY_PARAM, DUMMY_BIN), "swap to dummy succeeds")
	h.assert_eq(ctrl.model_param_path, DUMMY_PARAM, "paths updated on success")
	var out2: PackedFloat32Array = ctrl._ncnn_runner.run_inference(PackedFloat32Array([0.1, 0.2, 0.3, 0.4, 0.5]))
	h.assert_eq(out2.size(), 5, "swapped net still produces 5 logits")

	# Bad path: fails loud, keeps the working net + paths.
	h.assert_true(not ctrl.reload_model("res://nope.param", "res://nope.bin"), "bad path refused")
	h.assert_eq(ctrl.model_param_path, DUMMY_PARAM, "paths unchanged on failure")
	h.assert_true(ctrl._ncnn_runner.is_model_loaded(), "old net survives a failed reload")

	h.finish(self)
