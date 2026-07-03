extends SceneTree

# Headless unit tests for the pure OpenAI-ES optimizer helpers (native ES trainer, #131).
# The sphere-function test is the load-bearing one: it proves the optimizer actually climbs
# a fitness landscape, deterministically under a fixed seed, with no engine or ncnn involved.

const Harness = preload("res://test/harness.gd")
const ES = preload("res://addons/godot_native_rl/training/es_math.gd")

func _sphere_fitness(x: PackedFloat32Array, target: Array) -> float:
	# Maximize -||x - target||^2 (optimum 0 at x == target).
	var total := 0.0
	for i in range(x.size()):
		var d := x[i] - float(target[i])
		total -= d * d
	return total

func _initialize() -> void:
	var h := Harness.new()

	# --- sample_epsilons: shape + determinism under seed ---
	var rng := RandomNumberGenerator.new()
	rng.seed = 42
	var eps: Array = ES.sample_epsilons(rng, 3, 4)
	h.assert_eq(eps.size(), 3, "sample_epsilons returns half_pop vectors")
	h.assert_eq((eps[0] as PackedFloat32Array).size(), 4, "epsilon dimension matches n")
	var rng_b := RandomNumberGenerator.new()
	rng_b.seed = 42
	var eps_b: Array = ES.sample_epsilons(rng_b, 3, 4)
	h.assert_eq(eps[2], eps_b[2], "sample_epsilons deterministic under seed")

	# --- antithetic_candidates: mirrored pairs around theta ---
	var theta := PackedFloat32Array([1.0, -1.0])
	var one_eps := [PackedFloat32Array([0.5, 2.0])]
	var cands: Array = ES.antithetic_candidates(theta, 0.1, one_eps)
	h.assert_eq(cands.size(), 2, "antithetic pair per epsilon")
	h.assert_eq(cands[0], PackedFloat32Array([1.05, -0.8]), "plus candidate = theta + sigma*eps")
	h.assert_eq(cands[1], PackedFloat32Array([0.95, -1.2]), "minus candidate = theta - sigma*eps")

	# --- centered_ranks: [-0.5, 0.5], order-preserving, mean zero ---
	var shaped: Array = ES.centered_ranks([10.0, -3.0, 5.0, 99.0])
	h.assert_eq(shaped.size(), 4, "centered_ranks size")
	h.assert_true(absf(float(shaped[3]) - 0.5) < 1e-9, "best fitness -> +0.5")
	h.assert_true(absf(float(shaped[1]) + 0.5) < 1e-9, "worst fitness -> -0.5")
	var mean := 0.0
	for v in shaped:
		mean += float(v)
	h.assert_true(absf(mean) < 1e-9, "centered ranks are zero-mean")
	h.assert_true(float(shaped[0]) > float(shaped[2]), "rank order preserved (10 > 5)")
	h.assert_eq(ES.centered_ranks([7.0]), [0.0], "single candidate ranks to 0")

	# --- candidate_at: lazy indexing matches the materialized antithetic ordering ---
	var eps2 := [PackedFloat32Array([0.5, 2.0]), PackedFloat32Array([-1.0, 0.25])]
	var all_cands: Array = ES.antithetic_candidates(theta, 0.1, eps2)
	for idx in range(4):
		h.assert_eq(ES.candidate_at(theta, 0.1, eps2, idx), all_cands[idx],
			"candidate_at(%d) == antithetic_candidates[%d]" % [idx, idx])
	h.assert_eq(ES.candidate_at(theta, 0.1, eps2, 4), PackedFloat32Array(), "candidate_at out of range fails loud")

	# --- episode_seed: candidate-independent, (gen, episode)-distinct, positive ---
	h.assert_eq(ES.episode_seed(7, 3, 1), ES.episode_seed(7, 3, 1), "episode_seed deterministic")
	h.assert_true(ES.episode_seed(7, 3, 1) != ES.episode_seed(7, 4, 1), "episode_seed varies by generation")
	h.assert_true(ES.episode_seed(7, 3, 1) != ES.episode_seed(7, 3, 2), "episode_seed varies by episode")
	h.assert_true(ES.episode_seed(-5, 0, 0) > 0, "episode_seed positive for negative base")

	# --- es_update: hand-computed single-pair step ---
	# theta=[0,0], eps=[[1,0]], shaped=[+0.5 (plus cand), -0.5 (minus cand)], sigma=1, alpha=1
	# grad = (0.5*eps + (-0.5)*(-eps)) = eps -> theta' = theta + 1/(2*1) * [1,0] = [0.5, 0]
	var upd: PackedFloat32Array = ES.es_update(
		PackedFloat32Array([0.0, 0.0]), [PackedFloat32Array([1.0, 0.0])], [0.5, -0.5], 1.0, 1.0)
	h.assert_eq(upd, PackedFloat32Array([0.5, 0.0]), "es_update single mirrored pair")

	# --- the load-bearing test: sphere-function climb, deterministic ---
	var target := [1.0, -2.0, 0.5, 3.0, -1.0]
	var runs: Array = []
	for run in range(2):
		var opt_rng := RandomNumberGenerator.new()
		opt_rng.seed = 7
		var th := PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0])
		for gen in range(400):
			var e: Array = ES.sample_epsilons(opt_rng, 8, th.size())
			var cs: Array = ES.antithetic_candidates(th, 0.1, e)
			var fit: Array = []
			for c in cs:
				fit.append(_sphere_fitness(c, target))
			th = ES.es_update(th, e, ES.centered_ranks(fit), 0.1, 0.05)
		runs.append(th)
	var initial_fitness := _sphere_fitness(PackedFloat32Array([0, 0, 0, 0, 0]), target)
	var final_fitness := _sphere_fitness(runs[0], target)
	h.assert_true(final_fitness > initial_fitness + 10.0,
		"sphere fitness climbs (initial %f -> final %f)" % [initial_fitness, final_fitness])
	h.assert_true(final_fitness > -0.5,
		"sphere converges near target (final %f)" % final_fitness)
	h.assert_eq(runs[0], runs[1], "whole optimization deterministic under seed")

	h.finish(self)
