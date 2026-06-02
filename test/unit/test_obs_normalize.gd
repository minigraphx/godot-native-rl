extends SceneTree

const Harness = preload("res://test/harness.gd")
const ObsNormalize = preload("res://addons/godot_native_rl/controllers/obs_normalize.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Basic: (obs - mean)/sqrt(var + eps), in-range so no clipping.
	var out := ObsNormalize.normalize(
		PackedFloat32Array([2.0, 0.0]), PackedFloat32Array([1.0, 0.0]),
		PackedFloat32Array([4.0, 1.0]), 0.0, 10.0)
	h.assert_true(absf(out[0] - 0.5) < 1e-6 and absf(out[1]) < 1e-6, "basic normalize")

	# Clipping at +clip_obs and -clip_obs.
	var hi := ObsNormalize.normalize(
		PackedFloat32Array([100.0]), PackedFloat32Array([0.0]),
		PackedFloat32Array([1.0]), 0.0, 10.0)
	h.assert_true(absf(hi[0] - 10.0) < 1e-6, "clips to +clip_obs")
	var lo := ObsNormalize.normalize(
		PackedFloat32Array([-100.0]), PackedFloat32Array([0.0]),
		PackedFloat32Array([1.0]), 0.0, 10.0)
	h.assert_true(absf(lo[0] + 10.0) < 1e-6, "clips to -clip_obs")

	# Epsilon avoids div-by-zero on zero variance (clip high so it isn't clamped).
	var eps_out := ObsNormalize.normalize(
		PackedFloat32Array([1.0]), PackedFloat32Array([0.0]),
		PackedFloat32Array([0.0]), 1e-8, 1e9)
	h.assert_true(eps_out[0] > 100.0, "epsilon avoids div-by-zero (large finite value)")

	# Size mismatch -> empty.
	var bad := ObsNormalize.normalize(
		PackedFloat32Array([1.0, 2.0]), PackedFloat32Array([0.0]),
		PackedFloat32Array([1.0]), 0.0, 10.0)
	h.assert_eq(bad.size(), 0, "size mismatch returns empty")

	# validate accept/reject.
	h.assert_true(ObsNormalize.validate({"mean": [0.0], "var": [1.0], "epsilon": 1e-8, "clip_obs": 10.0}),
		"validate accepts well-formed")
	h.assert_true(not ObsNormalize.validate({"mean": [0.0], "var": [1.0, 2.0], "epsilon": 1e-8, "clip_obs": 10.0}),
		"validate rejects unequal lengths")
	h.assert_true(not ObsNormalize.validate({"mean": [], "var": [], "epsilon": 1e-8, "clip_obs": 10.0}),
		"validate rejects empty")
	h.assert_true(not ObsNormalize.validate({"var": [1.0], "epsilon": 1e-8, "clip_obs": 10.0}),
		"validate rejects missing key")

	# to_typed coerces JSON arrays into PackedFloat32Array + floats.
	var typed := ObsNormalize.to_typed({"mean": [0.5], "var": [2.0], "epsilon": 1e-8, "clip_obs": 5.0})
	h.assert_true(typed["mean"] is PackedFloat32Array and absf(typed["clip_obs"] - 5.0) < 1e-6,
		"to_typed coerces types")

	# --- Golden parity: GDScript helper must reproduce SB3 vn.normalize_obs (atol 1e-6) ---
	var stats := _load_json("res://models/synthetic_vecnormalize.json")
	var golden := _load_json("res://models/synthetic_vecnormalize_golden.json")
	h.assert_true(ObsNormalize.validate(stats), "committed stats fixture validates")
	var typed_stats := ObsNormalize.to_typed(stats)
	for gcase in golden["cases"]:
		var raw := PackedFloat32Array(gcase["raw"])
		var expected: Array = gcase["normalized"]
		var got := ObsNormalize.normalize(raw, typed_stats["mean"], typed_stats["var"],
			typed_stats["epsilon"], typed_stats["clip_obs"])
		h.assert_eq(got.size(), expected.size(), "golden case length matches")
		var max_diff := 0.0
		for i in got.size():
			max_diff = maxf(max_diff, absf(got[i] - float(expected[i])))
		h.assert_true(max_diff < 1e-6, "golden case parity (max diff %f < 1e-6)" % max_diff)

	h.finish(self)


func _load_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	assert(f != null, "cannot open %s" % path)
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	assert(parsed is Dictionary, "cannot parse %s" % path)
	return parsed
