extends SceneTree
# Guards NcnnAIController2D/3D.swap_model: the live policy hot-swap behind the chase web demo's
# model switcher. Verifies a runtime reload between the trained and untrained (dummy) ncnn models
# succeeds in place, that the reloaded trained model still infers correctly (golden argmax), and
# that the method's preconditions fail loud instead of crashing.

const Stub = preload("res://test/unit/stub_agent.gd")
const Harness = preload("res://test/harness.gd")

const TRAINED_PARAM := "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
const TRAINED_BIN := "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
const DUMMY_PARAM := "res://examples/chase_the_target/models/chase_dummy.ncnn.param"
const DUMMY_BIN := "res://examples/chase_the_target/models/chase_dummy.ncnn.bin"

func _initialize() -> void:
	var h := Harness.new()

	# A runner preloaded with the trained model (as the scene would have at startup).
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var loaded := runner.load_model(
		ProjectSettings.globalize_path(TRAINED_PARAM),
		ProjectSettings.globalize_path(TRAINED_BIN))
	h.assert_true(loaded, "trained model loads into runner")

	var a = Stub.new()
	a.control_mode = Stub.ControlModes.NCNN_INFERENCE
	a.input_blob_name = "in0"
	a.output_blob_name = "out0"
	a.set_ncnn_runner_for_test(runner)

	if loaded:
		# Hot-swap to the untrained policy and back — both reload in place and keep a loaded model.
		h.assert_true(a.swap_model(DUMMY_PARAM, DUMMY_BIN), "swap to untrained model")
		h.assert_true(runner.is_model_loaded(), "runner still loaded after swap to untrained")
		h.assert_true(a.swap_model(TRAINED_PARAM, TRAINED_BIN), "swap back to trained model")
		h.assert_eq(a.model_param_path, TRAINED_PARAM, "swap updates model_param_path")

		# The reloaded trained model reproduces a known golden argmax (same fixture as
		# test_chase_golden_inference.gd), proving inference is intact after the swap-back.
		var got := runner.run_discrete_action(
			PackedFloat32Array([0.5479, -0.1222, 0.7172, 0.3947, -0.8116]))
		h.assert_eq(got, 2, "trained model reproduces golden argmax after swap-back")

	# Precondition guards fail loud (return false), they do not crash. The push_error these emit is
	# expected and does not fail the suite (failures are by exit code, not stderr).
	h.assert_true(not a.swap_model("", ""), "empty paths rejected")
	h.assert_true(not a.swap_model("res://no/such.param", "res://no/such.bin"), "missing files rejected")

	a.free()
	runner.free()
	h.finish(self)
