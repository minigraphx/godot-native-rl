extends SceneTree

# Headless unit tests for the pure sep-CMA-ES optimizer (native ES trainer follow-up to #131).
# Mirrors test_es_math.gd: the sphere climb is the load-bearing test; the axis-scaled ellipsoid
# additionally exercises what sep-CMA adds over plain ES — per-coordinate variance adaptation.

const Harness = preload("res://test/harness.gd")
const Cma = preload("res://addons/godot_native_rl/training/cma_math.gd")

func _sphere_fitness(x: PackedFloat32Array, target: Array) -> float:
	# Maximize -||x - target||^2 (optimum 0 at x == target).
	var total := 0.0
	for i in range(x.size()):
		var d := x[i] - float(target[i])
		total -= d * d
	return total

func _ellipsoid_fitness(x: PackedFloat32Array, scales: Array) -> float:
	# Maximize -sum(a_j * x_j^2): badly axis-scaled — per-coordinate variance adaptation territory.
	var total := 0.0
	for i in range(x.size()):
		total -= float(scales[i]) * x[i] * x[i]
	return total

func _run_cma(seed_val: int, sigma0: float, generations: int, start: PackedFloat32Array,
		fitness_fn: Callable) -> Dictionary:
	var cma = Cma.new()
	var ok: bool = cma.setup(start, sigma0, 16)
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_val
	for _gen in range(generations):
		cma.sample(rng)
		var fit: Array = []
		for i in range(cma.population):
			fit.append(fitness_fn.call(cma.candidate_at(i)))
		cma.update(fit)
	return {"ok": ok, "mean": cma.mean_vector(), "sigma": cma.sigma, "c_diag": cma._c_diag}

func _initialize() -> void:
	var h := Harness.new()

	# --- setup: weights + strategy constants ---
	var cma = Cma.new()
	h.assert_true(cma.setup(PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0]), 0.5, 16), "setup accepts a sane config")
	h.assert_eq(cma.mu, 8, "mu = population/2")
	var w_sum := 0.0
	for w in cma.weights:
		w_sum += float(w)
	h.assert_true(absf(w_sum - 1.0) < 1e-6, "recombination weights sum to 1")
	h.assert_true(float(cma.weights[0]) > float(cma.weights[cma.mu - 1]), "weights are descending")
	h.assert_true(cma.mu_eff > 1.0 and cma.mu_eff <= float(cma.mu), "mu_eff in (1, mu]")
	h.assert_true(cma.c_1 + cma.c_mu <= 1.0 + 1e-9, "covariance learning rates bounded (c1+cmu <= 1)")
	h.assert_true(cma.c_sigma > 0.0 and cma.c_sigma < 1.0 and cma.d_sigma >= 1.0, "CSA constants in range")

	# --- setup failure modes fail loud ---
	var bad = Cma.new()
	h.assert_true(not bad.setup(PackedFloat32Array(), 0.5, 16), "empty mean rejected")
	h.assert_true(not bad.setup(PackedFloat32Array([0.0]), 0.0, 16), "sigma0 <= 0 rejected")
	h.assert_true(not bad.setup(PackedFloat32Array([0.0]), 0.5, 3), "population < 4 rejected")

	# --- sampling: shape, lazy candidate access, out-of-range fails loud ---
	var rng := RandomNumberGenerator.new()
	rng.seed = 42
	cma.sample(rng)
	h.assert_eq(cma._z.size(), 16, "sample draws population z vectors")
	h.assert_eq(cma.candidate_at(0).size(), 5, "candidate dimension matches n")
	h.assert_eq(cma.candidate_at(16), PackedFloat32Array(), "candidate_at out of range fails loud")
	# Fresh state: candidates are mean + sigma*z exactly (C starts at identity).
	var c0: PackedFloat32Array = cma.candidate_at(0)
	var z0: PackedFloat32Array = cma._z[0]
	var worst := 0.0
	for j in range(5):
		worst = maxf(worst, absf(c0[j] - 0.5 * z0[j]))
	h.assert_true(worst < 1e-6, "identity-C candidates are mean + sigma*z (worst |err| %f)" % worst)

	# --- update guards ---
	cma.update([1.0, 2.0])  # wrong count: must push an error and leave state untouched
	h.assert_eq(cma.mean_vector(), PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0]),
		"mismatched fitness count leaves the mean untouched")

	# --- the load-bearing test: sphere climb, deterministic ---
	var target := [1.0, -2.0, 0.5, 3.0, -1.0]
	var sphere := func(x: PackedFloat32Array) -> float: return _sphere_fitness(x, target)
	var start := PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0])
	var run_a: Dictionary = _run_cma(7, 0.5, 150, start, sphere)
	var run_b: Dictionary = _run_cma(7, 0.5, 150, start, sphere)
	var initial_fitness := _sphere_fitness(start, target)
	var final_fitness := _sphere_fitness(run_a["mean"], target)
	h.assert_true(final_fitness > initial_fitness + 10.0,
		"sphere fitness climbs (initial %f -> final %f)" % [initial_fitness, final_fitness])
	h.assert_true(final_fitness > -0.01, "sphere converges near target (final %f)" % final_fitness)
	h.assert_eq(run_a["mean"], run_b["mean"], "whole optimization deterministic under seed")
	h.assert_true(float(run_a["sigma"]) < 0.5 and float(run_a["sigma"]) > 0.0,
		"step size contracted at convergence (sigma %f)" % float(run_a["sigma"]))

	# --- sep-CMA's own claim: per-coordinate adaptation on a badly axis-scaled ellipsoid ---
	var scales := [1.0, 100.0, 1.0, 100.0, 1.0]
	var ell := func(x: PackedFloat32Array) -> float: return _ellipsoid_fitness(x, scales)
	var e_start := PackedFloat32Array([2.0, 2.0, 2.0, 2.0, 2.0])
	var e_run: Dictionary = _run_cma(11, 0.5, 200, e_start, ell)
	var e_final := _ellipsoid_fitness(e_run["mean"], scales)
	h.assert_true(e_final > -0.01, "ellipsoid converges (final %f from %f)"
		% [e_final, _ellipsoid_fitness(e_start, scales)])
	# The stiff coordinates (a=100) must have learned SMALLER variances than the soft ones.
	var cd: PackedFloat32Array = e_run["c_diag"]
	h.assert_true(cd[1] < cd[0] and cd[3] < cd[4],
		"diagonal covariance adapted per coordinate (stiff %f/%f < soft %f/%f)" % [cd[1], cd[3], cd[0], cd[4]])

	h.finish(self)
