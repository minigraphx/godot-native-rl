extends SceneTree

# Headless unit tests for the pure θ ⇄ ncnn-buffers MLP codec (native ES trainer, #131).
# The format goldens mirror scripts/export_statedict_to_ncnn.py's tested writer; the forward
# goldens go THROUGH the real NcnnRunner.load_model_from_buffers, so the whole
# "flat weight vector → live ncnn net" path is asserted against hand-computed values.

const Harness = preload("res://test/harness.gd")
const W = preload("res://addons/godot_native_rl/training/ncnn_weights.gd")

func _approx(h, got: PackedFloat32Array, want: Array, msg: String, tol := 1e-5) -> void:
	h.assert_eq(got.size(), want.size(), msg + " (size)")
	if got.size() != want.size():
		return
	for i in range(got.size()):
		h.assert_true(absf(got[i] - float(want[i])) < tol, "%s [%d] got %f want %f" % [msg, i, got[i], want[i]])

func _forward(h, spec: Dictionary, theta: PackedFloat32Array, input: Array, label: String) -> PackedFloat32Array:
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok: bool = runner.load_model_from_buffers(
		W.param_text(spec).to_utf8_buffer(), W.bin_bytes(spec, theta))
	h.assert_true(ok, "%s: buffers load into a live ncnn net" % label)
	if not ok:
		runner.free()
		return PackedFloat32Array()
	var obs := PackedFloat32Array()
	for v in input:
		obs.append(float(v))
	var out := runner.run_inference(obs)
	runner.free()
	return out

func _initialize() -> void:
	var h := Harness.new()

	# --- theta_size ---
	var s1: Dictionary = W.mlp_spec([3, 2])
	h.assert_eq(W.theta_size(s1), 8, "theta_size single linear: 3*2 + 2")
	var s2: Dictionary = W.mlp_spec([5, 32, 5])
	h.assert_eq(W.theta_size(s2), 5 * 32 + 32 + 32 * 5 + 5, "theta_size chase-sized MLP")

	# --- param_text exact golden (format-critical; mirrors the Python writer) ---
	var expected_param := "\n".join([
		"7767517",
		"4 4",
		"Input in0 0 1 in0 0=2",
		"InnerProduct fc0 1 1 in0 fc0 0=2 1=1 2=4",
		"ReLU act1 1 1 fc0 act1 0=0.0",
		"InnerProduct fc2 1 1 act1 out0 0=2 1=1 2=4",
	]) + "\n"
	h.assert_eq(W.param_text(W.mlp_spec([2, 2, 2])), expected_param, "param_text exact for [2,2,2] relu")

	# --- bin layout: fp32 tag before weights, raw bias, per linear layer ---
	var theta1 := PackedFloat32Array([1, 2, 3, 4, 5, 6, 0.5, -0.5])
	var bin1: PackedByteArray = W.bin_bytes(s1, theta1)
	h.assert_eq(bin1.size(), 4 + 6 * 4 + 2 * 4, "bin size: tag + 6 weights + 2 biases")
	h.assert_eq(bin1.decode_u32(0), 0, "fp32 tag is uint32 0")
	h.assert_true(absf(bin1.decode_float(4) - 1.0) < 1e-6, "first weight after tag")
	h.assert_true(absf(bin1.decode_float(4 + 6 * 4) - 0.5) < 1e-6, "bias follows weights untagged")

	# --- encode/decode round-trip (bijective; enables warm-start from a shipped net) ---
	var back: PackedFloat32Array = W.theta_from_bin(s1, bin1)
	_approx(h, back, [1, 2, 3, 4, 5, 6, 0.5, -0.5], "theta_from_bin round-trip")
	var s3: Dictionary = W.mlp_spec([4, 8, 3], "relu", "tanh")
	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	var theta3: PackedFloat32Array = W.init_theta(s3, rng)
	_approx(h, W.theta_from_bin(s3, W.bin_bytes(s3, theta3)),
		Array(theta3), "round-trip on random init [4,8,3]")

	# --- init_theta: right size, deterministic under seed, zero biases ---
	h.assert_eq(theta3.size(), W.theta_size(s3), "init_theta size")
	var rng2 := RandomNumberGenerator.new()
	rng2.seed = 7
	h.assert_eq(W.init_theta(s3, rng2), theta3, "init_theta deterministic under seed")
	# Bias block of the first layer (offset 4*8) is zero.
	h.assert_eq(theta3[4 * 8], 0.0, "init_theta bias starts zero")

	# --- forward golden A: single linear through real ncnn ---
	# W=[[1,2,3],[4,5,6]] b=[0.5,-0.5], x=[1,-1,2] -> [1-2+6+0.5, 4-5+12-0.5] = [5.5, 10.5]
	var out_a := _forward(h, s1, theta1, [1, -1, 2], "single linear")
	_approx(h, out_a, [5.5, 10.5], "single-linear forward golden")

	# --- forward golden B: relu hidden stack ---
	# W1=[[1,-1],[0.5,0.5]] b1=[0,-1]; relu; W2=[[1,1],[-1,1]] b2=[0.1,0.2]; x=[2,1]
	# h_pre=[2-1, 1+0.5-1]=[1,0.5] -> relu unchanged -> out=[1+0.5+0.1, -1+0.5+0.2]=[1.6,-0.3]
	var theta_b := PackedFloat32Array([1, -1, 0.5, 0.5, 0, -1, 1, 1, -1, 1, 0.1, 0.2])
	var out_b := _forward(h, W.mlp_spec([2, 2, 2]), theta_b, [2, 1], "relu stack")
	_approx(h, out_b, [1.6, -0.3], "relu-stack forward golden")

	# --- forward golden C: tanh output activation ---
	# W=[[2]] b=[0], x=[0.5] -> tanh(1.0)
	var out_c := _forward(h, W.mlp_spec([1, 1], "relu", "tanh"), PackedFloat32Array([2, 0]), [0.5], "tanh output")
	_approx(h, out_c, [tanh(1.0)], "tanh-output forward golden", 1e-4)

	# --- cross-language parity: byte-identical to the PYTHON writer's committed fixture ---
	# The fixture (test/unit/fixtures/py_writer_mlp.ncnn.*) was emitted by
	# scripts/export_statedict_to_ncnn.py's pure format functions (dims [3,4,2], relu hidden,
	# tanh output, theta[i] = i*0.03125 - 0.5). If either writer's format drifts, this trips —
	# the two codecs must stay interchangeable or warm-starting from a Python-exported net breaks.
	var fx_spec: Dictionary = W.mlp_spec([3, 4, 2], "relu", "tanh")
	var fx_theta := PackedFloat32Array()
	for i in range(26):
		fx_theta.append(float(i) * 0.03125 - 0.5)
	var fx_param := FileAccess.get_file_as_string("res://test/unit/fixtures/py_writer_mlp.ncnn.param")
	var fx_bin := FileAccess.get_file_as_bytes("res://test/unit/fixtures/py_writer_mlp.ncnn.bin")
	h.assert_true(not fx_param.is_empty() and not fx_bin.is_empty(), "python-writer fixture present")
	h.assert_eq(W.param_text(fx_spec), fx_param, "param_text byte-identical to the Python writer")
	h.assert_eq(W.bin_bytes(fx_spec, fx_theta), fx_bin, "bin_bytes byte-identical to the Python writer")
	_approx(h, W.theta_from_bin(fx_spec, fx_bin), Array(fx_theta), "theta_from_bin decodes a Python-written bin")

	# --- fail-loud guards ---
	h.assert_eq(W.mlp_spec([5]), {}, "mlp_spec rejects fewer than 2 dims")
	h.assert_eq(W.mlp_spec([5, 3], "swish"), {}, "mlp_spec rejects unknown activation")
	h.assert_eq(W.bin_bytes(s1, PackedFloat32Array([1, 2])), PackedByteArray(),
		"bin_bytes rejects wrong theta size")

	h.finish(self)
