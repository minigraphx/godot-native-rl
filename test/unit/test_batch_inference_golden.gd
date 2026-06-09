extends SceneTree
# Batched inference parity + threading regression for run_inference_batch.
# Reuses the committed chase_the_target.ncnn net (5-dim obs, in0/out0). Asserts:
#   (a) run_inference_batch(inputs)[i] == run_inference(inputs[i]) for every i (same op path),
#   (b) serial (num_threads=1) == threaded (num_threads=8) outputs (determinism / no race),
#   (c) an empty inputs Array yields an empty result,
#   (d) a wrong-sized input (with input_shape pinned) yields an empty slot while others succeed.

const MODEL_PARAM := "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
const MODEL_BIN   := "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
const Harness = preload("res://test/harness.gd")

const OBS: Array = [
	[ 0.5479, -0.1222,  0.7172,  0.3947, -0.8116],
	[ 0.9512,  0.5223,  0.5721, -0.7438, -0.0992],
	[-0.2584,  0.8535,  0.2877,  0.6455, -0.1132],
	[-0.5455,  0.1092, -0.8724,  0.6553,  0.2633],
	[ 0.5162, -0.2909,  0.9414,  0.7862,  0.5568],
]

func _approx_eq(a: PackedFloat32Array, b: PackedFloat32Array) -> bool:
	if a.size() != b.size() or a.size() == 0:
		return false
	for i in a.size():
		if absf(a[i] - b[i]) > 1e-6:
			return false
	return true

func _make_runner() -> NcnnRunner:
	var r := NcnnRunner.new()
	r.input_blob_name = "in0"
	r.output_blob_name = "out0"
	var ok := r.load_model(ProjectSettings.globalize_path(MODEL_PARAM),
		ProjectSettings.globalize_path(MODEL_BIN))
	return r if ok else null

func _initialize() -> void:
	var h := Harness.new()

	var runner := _make_runner()
	h.assert_true(runner != null, "chase model loads")
	if runner == null:
		h.finish(self)
		return

	var inputs: Array = []
	for o in OBS:
		inputs.append(PackedFloat32Array(o))

	# (a) batch == per-agent single inference.
	var batch: Array = runner.run_inference_batch(inputs, -1)
	h.assert_eq(batch.size(), inputs.size(), "batch returns one output per input")
	for i in inputs.size():
		var single: PackedFloat32Array = runner.run_inference(inputs[i])
		h.assert_true(_approx_eq(batch[i], single), "batch[%d] == single inference" % i)

	# (b) serial == threaded.
	var serial: Array = runner.run_inference_batch(inputs, 1)
	var threaded: Array = runner.run_inference_batch(inputs, 8)
	h.assert_eq(serial.size(), threaded.size(), "serial/threaded same length")
	var all_match := true
	for i in serial.size():
		if not _approx_eq(serial[i], threaded[i]):
			all_match = false
	h.assert_true(all_match, "serial outputs == threaded outputs")

	# (c) empty input -> empty result.
	var empty: Array = runner.run_inference_batch([], -1)
	h.assert_eq(empty.size(), 0, "empty inputs -> empty result")

	# (d) malformed slot: pin input_shape so a wrong-sized vector fails at mat-build.
	var pinned := _make_runner()
	pinned.input_shape = PackedInt32Array([5])
	var mixed: Array = [PackedFloat32Array(OBS[0]), PackedFloat32Array([1.0, 2.0, 3.0])]
	var mixed_out: Array = pinned.run_inference_batch(mixed, -1)
	h.assert_eq(mixed_out.size(), 2, "malformed batch keeps slot count")
	h.assert_true((mixed_out[0] as PackedFloat32Array).size() > 0, "valid slot produced output")
	h.assert_eq((mixed_out[1] as PackedFloat32Array).size(), 0, "malformed slot is empty")
	pinned.free()
	runner.free()
	h.finish(self)
