extends SceneTree

# Deploy-contract golden for masked multi-head attention (#46/#258, the ncnn-side half of the
# round-trip spike, provable without torch/pnnx): a hand-authored ncnn graph with ONE
# MultiHeadAttention layer + an additive attn_mask input (fixture from
# scripts/make_synthetic_attention.py) must run through the real NcnnRunner multi-IO path and
# match pure-math goldens — INCLUDING mask invariance: with a slot masked out, its contents
# must not influence the surviving rows' outputs (the property the entity-obs padding relies on).

const Harness = preload("res://test/harness.gd")

const PARAM := "res://models/synthetic_attention.ncnn.param"
const BIN := "res://models/synthetic_attention.ncnn.bin"
const GOLDEN := "res://models/synthetic_attention_golden.json"
const ATOL := 1e-4  # fp32 accumulate vs float64 golden; values are O(1)

func _run_case(runner: NcnnRunner, golden: Dictionary, c: Dictionary, h) -> PackedFloat32Array:
	var inputs: Array = [
		{"name": "x", "data": PackedFloat32Array(c["x"]), "shape": PackedInt32Array(golden["x_shape"])},
		{"name": "mask", "data": PackedFloat32Array(c["mask"]), "shape": PackedInt32Array(golden["mask_shape"])},
	]
	var result: Dictionary = runner.run_inference_multi(inputs, PackedStringArray(["out"]))
	h.assert_true(result.has("out"), "%s: MHA graph produced an output" % c["name"])
	return result.get("out", PackedFloat32Array())

func _initialize() -> void:
	var h := Harness.new()

	var golden_text := FileAccess.get_file_as_string(GOLDEN)
	var golden: Dictionary = JSON.parse_string(golden_text) if golden_text != "" else {}
	h.assert_true(not golden.is_empty(), "golden json parses (loud fail on corrupt fixture)")
	if golden.is_empty():
		h.finish(self)
		return

	var runner := NcnnRunner.new()
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "hand-authored MultiHeadAttention graph loads")
	if not ok:
		runner.free()
		h.finish(self)
		return

	var outputs: Dictionary = {}
	for c in golden["cases"]:
		var out := _run_case(runner, golden, c, h)
		outputs[c["name"]] = out
		var expected: Array = c["expected"]
		h.assert_eq(out.size(), expected.size(), "%s: output size" % c["name"])
		if out.size() != expected.size():
			continue
		var worst := 0.0
		for i in range(out.size()):
			worst = maxf(worst, absf(out[i] - float(expected[i])))
		h.assert_true(worst < ATOL, "%s: matches pure-math golden (worst |err| %f)" % [c["name"], worst])

	# Mask invariance, asserted between two REAL ncnn runs (stronger than golden-vs-golden):
	# the two padded cases differ only in the masked slot's contents; the surviving rows
	# (entities 0..1 -> the first 2*embed_dim floats) must agree bit-for-bit-ish.
	var a: PackedFloat32Array = outputs.get("padded_junk_a", PackedFloat32Array())
	var b: PackedFloat32Array = outputs.get("padded_junk_b", PackedFloat32Array())
	if a.size() == b.size() and a.size() > 0:
		var embed := int(golden["embed_dim"])
		var worst_inv := 0.0
		for i in range(2 * embed):
			worst_inv = maxf(worst_inv, absf(a[i] - b[i]))
		h.assert_true(worst_inv < 1e-6, "masked slot's contents don't leak into surviving rows (worst |err| %f)" % worst_inv)

	runner.free()
	h.finish(self)
