extends SceneTree
# Guards NcnnRunner.run_inference_async (#19): a non-blocking forward pass on a worker thread that
# emits inference_completed(output) on the main thread. Verifies the request is accepted, the
# in-flight flag tracks state, a second concurrent request is rejected, the async output matches
# the synchronous run_inference exactly (worker parity), and that a not-loaded runner refuses.
#
# Async needs the main loop to run so the worker's call_deferred flushes — so unlike the synchronous
# harness tests this drives _process() and only quits once the signal fires (with a frame-budget
# timeout so a regression fails loud instead of hanging CI).

const Harness = preload("res://test/harness.gd")

const PARAM := "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
const BIN := "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
var INPUT := PackedFloat32Array([0.5479, -0.1222, 0.7172, 0.3947, -0.8116])

var _h: RefCounted
var _runner
var _captured: Array = []
var _frames := 0
var _done := false
var _quitting := false  # one-shot guard: _process keeps firing after a queued quit, so without this
                        # the timeout branch re-enters every frame, double-frees the runner, and spams
                        # errors forever instead of exiting (#208).

func _initialize() -> void:
	_h = Harness.new()

	# A not-loaded runner must refuse (and not emit).
	var empty_runner = NcnnRunner.new()
	empty_runner.input_blob_name = "in0"
	empty_runner.output_blob_name = "out0"
	_h.assert_true(not empty_runner.run_inference_async(INPUT), "not-loaded runner rejects async request")
	_h.assert_true(not empty_runner.is_inference_running(), "not-loaded runner stays idle")
	empty_runner.free()

	_runner = NcnnRunner.new()
	_runner.input_blob_name = "in0"
	_runner.output_blob_name = "out0"
	var loaded: bool = _runner.load_model(
		ProjectSettings.globalize_path(PARAM),
		ProjectSettings.globalize_path(BIN))
	_h.assert_true(loaded, "chase model loads into runner")
	if not loaded:
		_runner.free()
		_h.finish(self)
		return

	_runner.inference_completed.connect(func(out): _captured.append(out))

	var started: bool = _runner.run_inference_async(INPUT)
	_h.assert_true(started, "async inference request accepted")
	_h.assert_true(_runner.is_inference_running(), "is_inference_running true while in flight")
	# A second request while one is in flight is rejected (one at a time).
	_h.assert_true(not _runner.run_inference_async(INPUT), "concurrent async request rejected")
	# Reloading the model while a worker holds net_ must be refused (else use-after-free) and
	# leave the loaded model intact.
	_h.assert_true(not _runner.load_model(
		ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN)),
		"load_model refused while async inference in flight")
	_h.assert_true(_runner.is_model_loaded(), "model still loaded after refused reload")
	# _process below waits for the completion signal.

func _process(_delta: float) -> bool:
	if _quitting:  # quit is queued, not immediate — stop doing anything until the tree tears down.
		return true
	_frames += 1
	if not _done and not _captured.is_empty():
		_done = true
		_quitting = true
		var async_out: PackedFloat32Array = _captured[0]
		_h.assert_true(async_out.size() > 0, "async inference produced non-empty output")
		_h.assert_true(not _runner.is_inference_running(), "is_inference_running false after completion")
		# Worker parity: the off-thread result must match the synchronous path bit-for-bit.
		var sync_out: PackedFloat32Array = _runner.run_inference(INPUT)
		_h.assert_eq(async_out, sync_out, "async output matches synchronous run_inference")
		# After the signal the runner accepts a fresh request (flag was cleared).
		_h.assert_true(_runner.run_inference_async(INPUT), "runner accepts a new request after completion")
		_free_runner()
		_h.finish(self)
		return true
	if _frames > 600:  # ~10s at 60 Hz — generous; a working path completes in a frame or two.
		_quitting = true
		# Fail loud and force the exit code directly — `finish()` queues a quit that _process would
		# otherwise outrun, re-firing this branch and double-freeing the runner forever (#208).
		printerr("ASYNC INFERENCE FAILED: timed out (inference_completed never fired)")
		_free_runner()
		quit(1)  # this script IS the SceneTree; call quit() directly (no get_tree()).
		return true
	return false

func _free_runner() -> void:
	if _runner != null:
		_runner.free()
		_runner = null
