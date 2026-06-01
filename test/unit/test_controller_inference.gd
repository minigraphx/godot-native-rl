extends SceneTree

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/stub_agent.gd")
const ImageStub = preload("res://test/unit/image_stub_agent.gd")

# Minimal fake that mimics NcnnRunner.run_discrete_action (float path).
class FakeRunner:
	var loaded := true
	var forced_index := 3
	func is_model_loaded() -> bool:
		return loaded
	func run_discrete_action(_input) -> int:
		return forced_index

# Fake that mimics NcnnRunner.run_inference_image (image path) -> raw logits.
class FakeImageRunner:
	var loaded := true
	var logits := PackedFloat32Array([0.1, 0.9, 0.2, 0.0])  # argmax == 1
	var last_normalize := false
	func is_model_loaded() -> bool:
		return loaded
	func run_inference_image(_img, normalize) -> PackedFloat32Array:
		last_normalize = normalize
		return logits

func _initialize() -> void:
	var h := Harness.new()

	h.assert_true(Stub.ControlModes.has("NCNN_INFERENCE"), "NCNN_INFERENCE enum value exists")

	# Float path: no inference image -> run_discrete_action argmax.
	var a = Stub.new()
	a.set_ncnn_runner_for_test(FakeRunner.new())
	a.infer_and_act()
	h.assert_eq(a.last_action, {"move": 3}, "float path sets {move: run_discrete_action}")
	a.free()

	# Image path: get_inference_image() non-null -> run_inference_image + argmax.
	var img := Image.create(2, 2, false, Image.FORMAT_RGB8)
	var ia = ImageStub.new()
	ia.image_to_return = img
	var fir := FakeImageRunner.new()
	ia.set_ncnn_runner_for_test(fir)
	ia.infer_and_act()
	h.assert_eq(ia.last_action, {"move": 1}, "image path sets {move: argmax(logits)}")
	h.assert_true(fir.last_normalize, "image path requests /255 normalization")
	ia.free()

	# No runner -> safe no-op.
	var b = Stub.new()
	b.infer_and_act()
	h.assert_eq(b.last_action, null, "no runner leaves last_action null")
	b.free()

	h.finish(self)
