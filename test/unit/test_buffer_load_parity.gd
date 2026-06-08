extends SceneTree
# Regression: load_model_from_buffers (bytes via FileAccess) must produce identical inference to
# the path-based load_model, and must work for TWO runners loaded simultaneously (multi-model =
# multiple NcnnRunner instances, the multi-policy pattern). Web cannot fopen inside Godot's .pck,
# so the buffer path is the deploy path on every platform; this pins it against regressions.

const MODEL_PARAM := "res://models/chase_sf_policy.ncnn.param"
const MODEL_BIN   := "res://models/chase_sf_policy.ncnn.bin"
const Harness = preload("res://test/harness.gd")

const OBS: Array = [
	[ 0.5479, -0.1222,  0.7172,  0.3947, -0.8116],
	[ 0.9512,  0.5223,  0.5721, -0.7438, -0.0992],
	[-0.2584,  0.8535,  0.2877,  0.6455, -0.1132],
]

func _make_runner() -> NcnnRunner:
	var r := NcnnRunner.new()
	r.input_blob_name = "in0"
	r.output_blob_name = "out0"
	return r

func _initialize() -> void:
	var h := Harness.new()

	# Path-based reference runner.
	var ref := _make_runner()
	var ok_ref := ref.load_model(
		ProjectSettings.globalize_path(MODEL_PARAM),
		ProjectSettings.globalize_path(MODEL_BIN))
	h.assert_true(ok_ref, "reference path-load succeeds")

	# Two buffer-loaded runners loaded simultaneously (multi-model guarantee).
	var param_bytes := FileAccess.get_file_as_bytes(MODEL_PARAM)
	var bin_bytes := FileAccess.get_file_as_bytes(MODEL_BIN)
	h.assert_true(param_bytes.size() > 0, "param bytes read")
	h.assert_true(bin_bytes.size() > 0, "bin bytes read")

	var a := _make_runner()
	var b := _make_runner()
	var ok_a := a.load_model_from_buffers(param_bytes, bin_bytes)
	var ok_b := b.load_model_from_buffers(param_bytes, bin_bytes)
	h.assert_true(ok_a, "buffer-load runner A succeeds")
	h.assert_true(ok_b, "buffer-load runner B succeeds while A is loaded")

	if ok_ref and ok_a and ok_b:
		for obs_values in OBS:
			var obs := PackedFloat32Array(obs_values)
			var want := ref.run_discrete_action(obs)
			h.assert_eq(a.run_discrete_action(obs), want, "A parity for %s" % str(obs_values))
			h.assert_eq(b.run_discrete_action(obs), want, "B parity for %s" % str(obs_values))

	# Empty buffers must fail closed (the non-empty guard).
	var empty := _make_runner()
	h.assert_true(not empty.load_model_from_buffers(PackedByteArray(), bin_bytes), "empty param buffer rejected")
	h.assert_true(not empty.load_model_from_buffers(param_bytes, PackedByteArray()), "empty bin buffer rejected")
	empty.free()

	ref.free(); a.free(); b.free()
	h.finish(self)
