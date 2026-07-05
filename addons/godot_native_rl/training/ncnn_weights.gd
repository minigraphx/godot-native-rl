extends RefCounted
# Pure θ ⇄ ncnn-buffers codec for simple MLP policies (native ES trainer, #131).
#
# GDScript port of the tested format writer in scripts/export_statedict_to_ncnn.py: renders the
# ncnn `.param` text and `.bin` bytes for a feed-forward Linear/activation stack directly from a
# flat weight vector θ, so an in-engine optimizer can turn a candidate vector into a live net via
# NcnnRunner.load_model_from_buffers with NO Python and NO file IO. The mapping is bijective
# (theta_from_bin inverts bin_bytes), which also enables warm-starting θ from a shipped net.
#
# θ layout, per linear layer in order: weights ([out, in] row-major — torch order, copied
# verbatim like the Python writer) then bias ([out]). Scope is the same v1 the Python writer
# covers: Input + InnerProduct + ReLU/TanH/Sigmoid. Anything else: fail loud, use the pnnx path.

const NCNN_MAGIC := "7767517"
# Activation name -> [ncnn layer type, params dict]. ReLU carries a float slope param; ncnn
# parses a token as float only if it looks like one, so 0.0 must render with its dot.
const _ACTIVATIONS := {
	"relu": ["ReLU", {0: 0.0}],
	"tanh": ["TanH", {}],
	"sigmoid": ["Sigmoid", {}],
	"": [],  # no activation layer
}


## Build an MLP spec: dims = [in, hidden..., out]. Returns {} (fail loud) on bad input.
static func mlp_spec(dims: Array, hidden_activation: String = "relu", output_activation: String = "") -> Dictionary:
	if dims.size() < 2:
		push_error("NcnnWeights.mlp_spec: need at least [in, out] dims, got %s" % [dims])
		return {}
	for d in dims:
		if int(d) <= 0:
			push_error("NcnnWeights.mlp_spec: non-positive dim in %s" % [dims])
			return {}
	if not _ACTIVATIONS.has(hidden_activation) or not _ACTIVATIONS.has(output_activation):
		push_error("NcnnWeights.mlp_spec: unknown activation '%s'/'%s' (know: relu, tanh, sigmoid, '')"
			% [hidden_activation, output_activation])
		return {}
	var d_int: Array = []
	for d in dims:
		d_int.append(int(d))
	return {"dims": d_int, "hidden_activation": hidden_activation, "output_activation": output_activation}


## Total flat-θ length: Σ per linear (in*out + out).
static func theta_size(spec: Dictionary) -> int:
	var dims: Array = spec.get("dims", [])
	var total := 0
	for i in range(dims.size() - 1):
		total += int(dims[i]) * int(dims[i + 1]) + int(dims[i + 1])
	return total


## Render the ncnn `.param` text: magic, `layer_count blob_count`, one line per layer.
## Layer/blob naming mirrors the Python writer (in0, fc0, act1, ...; final top renamed out0).
static func param_text(spec: Dictionary) -> String:
	var dims: Array = spec.get("dims", [])
	if dims.size() < 2:
		push_error("NcnnWeights.param_text: invalid spec")
		return ""
	var lines: Array = []
	var prev := "in0"
	lines.append("Input in0 0 1 in0 0=%d" % int(dims[0]))
	var idx := 0
	var n_linear := dims.size() - 1
	for i in range(n_linear):
		var fan_in := int(dims[i])
		var fan_out := int(dims[i + 1])
		var act: String = spec["hidden_activation"] if i < n_linear - 1 else spec["output_activation"]
		var is_last_linear := i == n_linear - 1
		var name := "fc%d" % idx
		# The final line's top blob is out0 (the NcnnRunner convention), whichever layer that is.
		var top := "out0" if is_last_linear and act == "" else name
		lines.append("InnerProduct %s 1 1 %s %s 0=%d 1=1 2=%d" % [name, prev, top, fan_out, fan_in * fan_out])
		prev = top
		idx += 1
		if act != "":
			var mapping: Array = _ACTIVATIONS[act]
			name = "act%d" % idx
			top = "out0" if is_last_linear else name
			var parts := "%s %s 1 1 %s %s" % [mapping[0], name, prev, top]
			var params: Dictionary = mapping[1]
			for pid in params:
				parts += " %d=%s" % [pid, _fmt_param(params[pid])]
			lines.append(parts)
			prev = top
			idx += 1
	# Blob count == layer count here: every layer contributes exactly one new top and reuses the
	# previous top as its bottom (same invariant the Python writer's count_blobs computes).
	var header := "%s\n%d %d\n" % [NCNN_MAGIC, lines.size(), lines.size()]
	return header + "\n".join(lines) + "\n"


## Render the `.bin`: per linear layer, a 4-byte fp32 tag (uint32 0) + weight floats, then the
## raw untagged bias floats. Returns an empty array (fail loud) on a θ-size mismatch.
static func bin_bytes(spec: Dictionary, theta: PackedFloat32Array) -> PackedByteArray:
	var want := theta_size(spec)
	if theta.size() != want or want == 0:
		push_error("NcnnWeights.bin_bytes: theta has %d floats, spec needs %d" % [theta.size(), want])
		return PackedByteArray()
	var dims: Array = spec["dims"]
	var out := PackedByteArray()
	var tag := PackedByteArray([0, 0, 0, 0])  # ncnn ModelBin type-0: plain float32 follows
	var offset := 0
	for i in range(dims.size() - 1):
		# Weights then bias are adjacent in θ, so one contiguous slice covers the layer; the
		# fp32 tag prefixes the weights and the bias is raw, so tag + slice is the whole layer.
		# append_array is in-place (`out += x` reallocates and re-copies the accumulator).
		var n_floats := int(dims[i]) * int(dims[i + 1]) + int(dims[i + 1])
		out.append_array(tag)
		out.append_array(theta.slice(offset, offset + n_floats).to_byte_array())
		offset += n_floats
	return out


## Inverse of bin_bytes: recover the flat θ from `.bin` bytes (warm-start from a shipped net).
static func theta_from_bin(spec: Dictionary, bin: PackedByteArray) -> PackedFloat32Array:
	var dims: Array = spec.get("dims", [])
	var expected := 0
	for i in range(dims.size() - 1):
		expected += 4 + (int(dims[i]) * int(dims[i + 1]) + int(dims[i + 1])) * 4
	if dims.size() < 2 or bin.size() != expected:
		push_error("NcnnWeights.theta_from_bin: bin is %d bytes, spec needs %d" % [bin.size(), expected])
		return PackedFloat32Array()
	var theta := PackedFloat32Array()
	var offset := 0
	for i in range(dims.size() - 1):
		offset += 4  # skip the fp32 tag
		var n_floats := int(dims[i]) * int(dims[i + 1]) + int(dims[i + 1])
		theta += bin.slice(offset, offset + n_floats * 4).to_float32_array()
		offset += n_floats * 4
	return theta


# ---- pnnx warm-start adapter (#328): adopt an EXTERNALLY-exported MLP into the θ codec ----
# pnnx-exported nets are architecturally plain MLPs but differ from our canonical form three
# ways: layer names/whitespace (so the exact param-text warm-start check rejects them), an
# occasional weightless Reshape/Flatten after Input, and FP16-TAGGED weight blobs (tag
# 0x01306B47; biases stay raw fp32). The two helpers below parse the STRUCTURE and decode the
# bin regardless, so ESTrainer can warm-start from any shipped PPO net — see warm_start_theta.

const _FP32_TAG := 0
const _FP16_TAG := 0x01306B47  # ncnn ModelBin: float16 data follows, 4-byte aligned

## Parse an MLP-shaped ncnn `.param` into a spec structurally (names/whitespace-independent).
## Accepts Input + optional weightless Reshape/Flatten + InnerProduct/activation stacks; anything
## else fails loud with {} (use the pnnx path for real). Input width derives from the first
## InnerProduct (pnnx omits the Input dim param).
static func spec_from_param_text(param_text: String) -> Dictionary:
	var dims: Array = []
	var acts: Array = []  # activation name after each linear ("" if none)
	var type_to_act := {"ReLU": "relu", "TanH": "tanh", "Sigmoid": "sigmoid"}
	for raw_line in param_text.split("\n"):
		var tokens := raw_line.split(" ", false)
		if tokens.size() < 2 or not tokens[0].is_valid_identifier():
			continue  # magic / counts / blank
		var ltype := String(tokens[0])
		match ltype:
			"Input", "Reshape", "Flatten":
				continue  # weightless plumbing
			"InnerProduct":
				var out_features := 0
				var weight_size := 0
				var has_bias := false
				for t in tokens:
					if t.begins_with("0="):
						out_features = int(t.substr(2))
					elif t.begins_with("1="):
						has_bias = int(t.substr(2)) == 1
					elif t.begins_with("2="):
						weight_size = int(t.substr(2))
				if out_features <= 0 or weight_size <= 0 or weight_size % out_features != 0 or not has_bias:
					push_error("NcnnWeights.spec_from_param_text: unsupported InnerProduct line '%s'." % raw_line.strip_edges())
					return {}
				if dims.is_empty():
					dims.append(weight_size / out_features)
				elif int(dims[-1]) != weight_size / out_features:
					push_error("NcnnWeights.spec_from_param_text: layer fan-in %d does not chain from previous width %d." % [weight_size / out_features, int(dims[-1])])
					return {}
				dims.append(out_features)
				acts.append("")
			_:
				if type_to_act.has(ltype):
					if acts.is_empty():
						push_error("NcnnWeights.spec_from_param_text: activation before any linear layer.")
						return {}
					acts[-1] = type_to_act[ltype]
				else:
					push_error("NcnnWeights.spec_from_param_text: unsupported layer type '%s' — not a plain MLP." % ltype)
					return {}
	if dims.size() < 2:
		push_error("NcnnWeights.spec_from_param_text: no InnerProduct layers found.")
		return {}
	# Hidden activations must be uniform (that's what the canonical writer can express).
	var hidden := ""
	for i in range(acts.size() - 1):
		if i == 0:
			hidden = acts[0]
		elif acts[i] != hidden:
			push_error("NcnnWeights.spec_from_param_text: mixed hidden activations (%s vs %s)." % [hidden, acts[i]])
			return {}
	return mlp_spec(dims, hidden, acts[-1])


## Decode θ from an EXTERNAL model's `.bin` for `spec`: like theta_from_bin, but honours the
## per-blob tag — fp32 (0) or fp16 (0x01306B47, 4-byte aligned) weights; biases always raw fp32.
static func theta_from_model_bin(spec: Dictionary, bin: PackedByteArray) -> PackedFloat32Array:
	var dims: Array = spec.get("dims", [])
	if dims.size() < 2:
		push_error("NcnnWeights.theta_from_model_bin: invalid spec")
		return PackedFloat32Array()
	var theta := PackedFloat32Array()
	var offset := 0
	for i in range(dims.size() - 1):
		var n_weights := int(dims[i]) * int(dims[i + 1])
		var n_bias := int(dims[i + 1])
		if offset + 4 > bin.size():
			push_error("NcnnWeights.theta_from_model_bin: bin truncated at layer %d tag." % i)
			return PackedFloat32Array()
		var tag := bin.decode_u32(offset)
		offset += 4
		match tag:
			_FP32_TAG:
				if offset + n_weights * 4 > bin.size():
					push_error("NcnnWeights.theta_from_model_bin: bin truncated in fp32 weights of layer %d." % i)
					return PackedFloat32Array()
				theta += bin.slice(offset, offset + n_weights * 4).to_float32_array()
				offset += n_weights * 4
			_FP16_TAG:
				var half_bytes := n_weights * 2
				if offset + half_bytes > bin.size():
					push_error("NcnnWeights.theta_from_model_bin: bin truncated in fp16 weights of layer %d." % i)
					return PackedFloat32Array()
				for w in range(n_weights):
					theta.append(bin.decode_half(offset + w * 2))
				offset += half_bytes + (half_bytes % 4)  # ModelBin aligns fp16 blobs to 4 bytes
			_:
				push_error("NcnnWeights.theta_from_model_bin: unsupported weight tag 0x%08X in layer %d (know fp32/fp16)." % [tag, i])
				return PackedFloat32Array()
		if offset + n_bias * 4 > bin.size():
			push_error("NcnnWeights.theta_from_model_bin: bin truncated in bias of layer %d." % i)
			return PackedFloat32Array()
		theta += bin.slice(offset, offset + n_bias * 4).to_float32_array()
		offset += n_bias * 4
	if offset != bin.size():
		push_error("NcnnWeights.theta_from_model_bin: %d trailing bytes after the last layer — architecture mismatch." % (bin.size() - offset))
		return PackedFloat32Array()
	return theta


## Warm-start θ from a net on disk for `spec` (#328): our own canonical pairs decode via the
## strict exact-text path; pnnx-exported MLPs are accepted STRUCTURALLY (same dims/activations)
## and decoded tag-aware. Returns empty (with a pushed error) when the net doesn't match `spec`.
static func warm_start_theta(spec: Dictionary, param_path: String, bin_path: String) -> PackedFloat32Array:
	var have_param := FileAccess.get_file_as_string(param_path)
	var bin := FileAccess.get_file_as_bytes(bin_path)
	if have_param.is_empty() or bin.is_empty():
		push_error("NcnnWeights.warm_start_theta: cannot read '%s' / '%s'." % [param_path, bin_path])
		return PackedFloat32Array()
	if have_param.strip_edges() == param_text(spec).strip_edges():
		return theta_from_bin(spec, bin)  # our own canonical output — strict fp32 decode
	var file_spec := spec_from_param_text(have_param)
	if file_spec.is_empty():
		return PackedFloat32Array()  # parser pushed the specific reason
	if file_spec["dims"] != spec["dims"] or file_spec["hidden_activation"] != spec["hidden_activation"] \
			or file_spec["output_activation"] != spec["output_activation"]:
		push_error("NcnnWeights.warm_start_theta: '%s' is %s/%s/%s but this scene builds %s/%s/%s."
			% [param_path, str(file_spec["dims"]), file_spec["hidden_activation"], file_spec["output_activation"],
			str(spec["dims"]), spec["hidden_activation"], spec["output_activation"]])
		return PackedFloat32Array()
	print("[NcnnWeights] warm-start: adopting externally-exported net '%s' (structural match %s)." % [param_path, str(spec["dims"])])
	return theta_from_model_bin(spec, bin)


## He-scaled gaussian init: weights ~ N(0, sqrt(2/fan_in)), biases 0. Deterministic under the
## caller's seeded RNG (ES runs want reproducibility).
static func init_theta(spec: Dictionary, rng: RandomNumberGenerator) -> PackedFloat32Array:
	var dims: Array = spec.get("dims", [])
	var theta := PackedFloat32Array()
	for i in range(dims.size() - 1):
		var std := sqrt(2.0 / float(dims[i]))
		for _w in range(int(dims[i]) * int(dims[i + 1])):
			theta.append(rng.randfn(0.0, std))
		for _b in range(int(dims[i + 1])):
			theta.append(0.0)
	return theta


static func _fmt_param(value: Variant) -> String:
	if typeof(value) == TYPE_FLOAT:
		var s := str(value)
		if not ("." in s or "e" in s or "inf" in s or "nan" in s):
			s += ".0"
		return s
	return str(value)
