extends SceneTree

const Harness = preload("res://test/harness.gd")
const ObsNormalize = preload("res://addons/godot_native_rl/obs/obs_normalize.gd")
const ObsNormalizer = preload("res://addons/godot_native_rl/obs/obs_normalizer.gd")

func _initialize() -> void:
	var h := Harness.new()

	var n = ObsNormalizer.new()

	# Before loading: not loaded, obs_size 0, normalize returns obs unchanged (no crash).
	h.assert_true(not n.is_loaded(), "not loaded before set/load")
	h.assert_eq(n.obs_size(), 0, "obs_size 0 before load")
	var passthrough: Array = n.normalize([1.0, 2.0])
	h.assert_true(passthrough.size() == 2 and absf(float(passthrough[0]) - 1.0) < 1e-9, "normalize before load -> unchanged")

	# After set_stats_for_test: loaded, obs_size matches stats, normalize == pure math.
	var mean := [0.0, 1.0, 5.0]
	var var_arr := [1.0, 4.0, 0.0]
	n.set_stats_for_test(mean, var_arr, 1e-8, 10.0)
	h.assert_true(n.is_loaded(), "loaded after set_stats_for_test")
	h.assert_eq(n.obs_size(), 3, "obs_size == stats length")

	var obs := [0.0, 2.0, 100.0]
	var got: Array = n.normalize(obs)
	var want: Array = ObsNormalize.normalize(obs, mean, var_arr, 1e-8, 10.0)
	var matches := got.size() == want.size()
	for i in range(mini(got.size(), want.size())):
		if absf(float(got[i]) - float(want[i])) > 1e-6:
			matches = false
	h.assert_true(matches, "node.normalize matches pure math (got %s, want %s)" % [str(got), str(want)])

	# Pinned expected (same numbers as test_obs_normalize.gd / the Python parity test).
	h.assert_true(absf(float(got[0])) < 1e-5 and absf(float(got[1]) - 0.5) < 1e-5 and absf(float(got[2]) - 10.0) < 1e-5, "node.normalize matches pinned expected")

	n.free()
	h.finish(self)
