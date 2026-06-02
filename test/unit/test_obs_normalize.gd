extends SceneTree

const Harness = preload("res://test/harness.gd")
const ObsNormalize = preload("res://addons/godot_native_rl/obs/obs_normalize.gd")

# Pinned parity values (must match the Python parity test in
# test/python/test_export_vecnormalize_stats.py — both anchor to the SB3 formula
# norm = clip((obs - mean) / sqrt(var + eps), -clip, clip)).
const OBS := [0.0, 2.0, 100.0]
const MEAN := [0.0, 1.0, 5.0]
const VAR := [1.0, 4.0, 0.0]
const EPS := 1e-8
const CLIP := 10.0
const EXPECTED := [0.0, 0.5, 10.0]  # el2 = 95/sqrt(1e-8) = 950000 -> clipped to 10

func _approx(h: Harness, out: Array, expected: Array, label: String) -> void:
	var ok := out.size() == expected.size()
	for i in range(mini(out.size(), expected.size())):
		if absf(float(out[i]) - float(expected[i])) > 1e-5:
			ok = false
	h.assert_true(ok, "%s (got %s, want %s)" % [label, str(out), str(expected)])

func _initialize() -> void:
	var h := Harness.new()

	# Pinned formula: hand-computed expected, identical to SB3 normalize_obs.
	_approx(h, ObsNormalize.normalize(OBS, MEAN, VAR, EPS, CLIP), EXPECTED, "pinned formula matches SB3")

	# No clipping needed when values are in range.
	_approx(h, ObsNormalize.normalize([3.0], [1.0], [4.0], EPS, CLIP), [1.0], "centered/scaled, in-range")

	# Negative side clips to -clip.
	_approx(h, ObsNormalize.normalize([-1000.0], [0.0], [1.0], EPS, 10.0), [-10.0], "clips to -clip")

	# epsilon guard: zero variance + epsilon stays finite (no NaN/inf), clips.
	var zv: Array = ObsNormalize.normalize([1.0], [0.0], [0.0], 1e-8, 10.0)
	h.assert_true(is_finite(float(zv[0])) and absf(float(zv[0]) - 10.0) < 1e-5, "zero var + eps finite, clipped")

	# clip <= 0 means no clipping (a default-0 must not zero everything).
	var nc: Array = ObsNormalize.normalize([1000.0], [0.0], [1.0], 1e-8, 0.0)
	h.assert_true(float(nc[0]) > 100.0, "clip<=0 -> no clipping")

	# Length mismatch (mean/var shorter than obs) -> obs returned unchanged (loud, stable shape).
	_approx(h, ObsNormalize.normalize([5.0, 6.0], [0.0], [1.0], EPS, CLIP), [5.0, 6.0], "length mismatch -> unchanged")

	# Immutability: input obs Array is not mutated.
	var src := [2.0, 4.0]
	var _out: Array = ObsNormalize.normalize(src, [0.0, 0.0], [1.0, 1.0], EPS, CLIP)
	h.assert_true(absf(float(src[0]) - 2.0) < 1e-9 and absf(float(src[1]) - 4.0) < 1e-9, "input not mutated")

	h.finish(self)
