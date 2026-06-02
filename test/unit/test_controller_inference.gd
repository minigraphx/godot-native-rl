extends SceneTree

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/stub_agent.gd")
const ImageStub = preload("res://test/unit/image_stub_agent.gd")
const ContinuousStub = preload("res://test/unit/continuous_stub_agent.gd")

# Fake that mimics NcnnRunner.run_inference (float path) -> raw output vector.
class FakeRunner:
	var loaded := true
	var output := PackedFloat32Array([0.0, 0.0, 0.0, 0.9, 0.0])  # argmax == 3 over size-5
	var last_input := PackedFloat32Array()
	func is_model_loaded() -> bool:
		return loaded
	func run_inference(input) -> PackedFloat32Array:
		last_input = input
		return output

# Fake that mimics NcnnRunner.run_inference_image (image path) -> raw logits.
class FakeImageRunner:
	var loaded := true
	var logits := PackedFloat32Array([0.1, 0.9, 0.2, 0.0])  # argmax == 1 over size-4
	var last_normalize := false
	func is_model_loaded() -> bool:
		return loaded
	func run_inference_image(_img, normalize) -> PackedFloat32Array:
		last_normalize = normalize
		return logits

func _initialize() -> void:
	var h := Harness.new()

	h.assert_true(Stub.ControlModes.has("NCNN_INFERENCE"), "NCNN_INFERENCE enum value exists")

	# Float path: run_inference -> decode (single discrete key, size 5).
	var a = Stub.new()
	a.set_ncnn_runner_for_test(FakeRunner.new())
	a.infer_and_act()
	h.assert_eq(a.last_action, {"move": 3}, "float path sets {move: argmax(run_inference)}")
	a.free()

	# Image path: run_inference_image -> decode (single discrete key, size 4).
	var img := Image.create(2, 2, false, Image.FORMAT_RGB8)
	var ia = ImageStub.new()
	ia.image_to_return = img
	var fir := FakeImageRunner.new()
	ia.set_ncnn_runner_for_test(fir)
	ia.infer_and_act()
	h.assert_eq(ia.last_action, {"move": 1}, "image path sets {move: argmax(logits)}")
	h.assert_true(fir.last_normalize, "image path requests /255 normalization")
	ia.free()

	# Mixed action space (discrete "fire" size 2 + continuous "steer" size 2, squashed).
	var ca = ContinuousStub.new()
	var cr := FakeRunner.new()
	cr.output = PackedFloat32Array([0.1, 0.9, 0.4, -0.4])  # fire -> argmax=1; steer -> tanh([0.4,-0.4])
	ca.set_ncnn_runner_for_test(cr)
	ca.infer_and_act()
	h.assert_eq(ca.last_action["fire"], 1, "mixed: discrete key decoded")
	h.assert_true(absf(ca.last_action["steer"][0] - tanh(0.4)) < 1e-6
		and absf(ca.last_action["steer"][1] - tanh(-0.4)) < 1e-6,
		"mixed: continuous key tanh-squashed")
	ca.free()

	# No runner -> safe no-op.
	var b = Stub.new()
	b.infer_and_act()
	h.assert_eq(b.last_action, null, "no runner leaves last_action null")
	b.free()

	# Obs normalization: identity stats (mean 0, var 1) -> runner sees raw obs unchanged.
	var na = Stub.new()
	var nr := FakeRunner.new()
	na.set_ncnn_runner_for_test(nr)
	na.set_obs_norm_stats_for_test({
		"mean": PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0]),
		"var": PackedFloat32Array([1.0, 1.0, 1.0, 1.0, 1.0]),
		"epsilon": 0.0, "clip_obs": 10.0})
	na.infer_and_act()
	h.assert_true(absf(nr.last_input[2] - 1.0) < 1e-6 and absf(nr.last_input[4] - 0.5) < 1e-6,
		"identity stats feed obs unchanged to runner")
	na.free()

	# Non-identity stats actually transform: obs[2]=1, mean=1, var=4 -> (1-1)/sqrt(4)=0.
	var na2 = Stub.new()
	var nr2 := FakeRunner.new()
	na2.set_ncnn_runner_for_test(nr2)
	na2.set_obs_norm_stats_for_test({
		"mean": PackedFloat32Array([0.0, 0.0, 1.0, 0.0, 0.0]),
		"var": PackedFloat32Array([1.0, 1.0, 4.0, 1.0, 1.0]),
		"epsilon": 0.0, "clip_obs": 10.0})
	na2.infer_and_act()
	h.assert_true(absf(nr2.last_input[2]) < 1e-6, "non-identity stats transform obs[2] -> 0")
	na2.free()

	# Empty stats (default) -> raw obs (backward compatible).
	var nb = Stub.new()
	var nbr := FakeRunner.new()
	nb.set_ncnn_runner_for_test(nbr)
	nb.infer_and_act()
	h.assert_true(absf(nbr.last_input[2] - 1.0) < 1e-6, "empty stats feeds raw obs (backward compatible)")
	nb.free()

	# Size-mismatch stats -> normalize returns empty -> action skipped (no set_action).
	var nc = Stub.new()
	var ncr := FakeRunner.new()
	nc.set_ncnn_runner_for_test(ncr)
	nc.set_obs_norm_stats_for_test({
		"mean": PackedFloat32Array([0.0]), "var": PackedFloat32Array([1.0]),
		"epsilon": 0.0, "clip_obs": 10.0})
	nc.infer_and_act()
	h.assert_eq(nc.last_action, null, "size-mismatch stats skips action")
	nc.free()

	h.finish(self)
