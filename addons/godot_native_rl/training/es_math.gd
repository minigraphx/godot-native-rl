extends RefCounted
# Pure OpenAI-style evolutionary-strategies optimizer (native ES trainer, #131).
#
# Salimans et al. 2017: perturb a flat weight vector θ with antithetic gaussian noise, score
# every candidate (episodic return = fitness), rank-shape the scores, and step θ along the
# fitness-weighted noise. Needs only forward passes — no gradients — which is exactly what the
# statically linked ncnn runtime can do on every deploy target. Pure and engine-free: the caller
# owns the RNG (seeded → the whole optimization is reproducible) and the fitness evaluation.
#
# Candidate ordering convention used throughout: [θ+σε_0, θ−σε_0, θ+σε_1, θ−σε_1, ...] — the
# fitness array handed back to es_update must match it.


## Fill a packed array with standard-normal draws IN PLACE — the one RNG-stream-consuming
## primitive, shared with CmaMath so both optimizers draw in the same order (a divergence would
## silently break seeded-run comparability). Also the zero-allocation path for per-generation
## buffer reuse (#324).
static func fill_gaussian(rng: RandomNumberGenerator, out: PackedFloat32Array) -> void:
	for j in range(out.size()):
		out[j] = rng.randfn()


## half_pop gaussian noise vectors of dimension n (candidates come in mirrored pairs, so the
## effective population is 2*half_pop).
static func sample_epsilons(rng: RandomNumberGenerator, half_pop: int, n: int) -> Array:
	var epsilons: Array = []
	for _i in range(half_pop):
		var eps := PackedFloat32Array()
		eps.resize(n)
		fill_gaussian(rng, eps)
		epsilons.append(eps)
	return epsilons


## Mirrored candidate pairs around theta: [θ+σε_0, θ−σε_0, θ+σε_1, ...]. Antithetic sampling
## halves the variance of the gradient estimate for free.
static func antithetic_candidates(theta: PackedFloat32Array, sigma: float, epsilons: Array) -> Array:
	var candidates: Array = []
	for eps in epsilons:
		var plus := PackedFloat32Array()
		var minus := PackedFloat32Array()
		plus.resize(theta.size())
		minus.resize(theta.size())
		for j in range(theta.size()):
			var d: float = sigma * eps[j]
			plus[j] = theta[j] + d
			minus[j] = theta[j] - d
		candidates.append(plus)
		candidates.append(minus)
	return candidates


## Single candidate for index `idx` under the antithetic ordering ([θ+σε_0, θ−σε_0, θ+σε_1, ...]),
## computed lazily — callers that consume one candidate at a time (the trainer) need never
## materialize the whole population (O(population × θ) memory saved).
static func candidate_at(theta: PackedFloat32Array, sigma: float, epsilons: Array, idx: int) -> PackedFloat32Array:
	if idx < 0 or idx >= 2 * epsilons.size():
		push_error("EsMath.candidate_at: index %d out of range for %d epsilon pairs." % [idx, epsilons.size()])
		return PackedFloat32Array()
	var eps: PackedFloat32Array = epsilons[idx >> 1]
	var sign_mult := 1.0 if (idx & 1) == 0 else -1.0
	var out := PackedFloat32Array()
	out.resize(theta.size())
	for j in range(theta.size()):
		out[j] = theta[j] + sign_mult * sigma * eps[j]
	return out


## Deterministic per-episode seed for common-random-numbers evaluation: identical for every
## candidate (no candidate term — that is the point), distinct across generations and episode
## indices, always positive (RandomNumberGenerator.seed is uint64; negatives wrap).
static func episode_seed(base_seed: int, generation: int, episode: int) -> int:
	return absi(base_seed * 92821 + (generation + 1) * 1_000_003 + (episode + 1) * 7919) + 1


## Rank-shape fitnesses to evenly spaced values in [-0.5, 0.5] (best -> +0.5). Makes the update
## invariant to fitness scale and robust to outliers/noise — the standard OpenAI-ES transform.
## Ties break by stable candidate order. A single candidate ranks to 0.
static func centered_ranks(fitness: Array) -> Array:
	var n := fitness.size()
	if n <= 1:
		var flat: Array = []
		flat.resize(n)
		flat.fill(0.0)
		return flat
	var order: Array = []
	for i in range(n):
		order.append([float(fitness[i]), i])
	order.sort()  # ascending by fitness, ties by original index
	var shaped: Array = []
	shaped.resize(n)
	for rank in range(n):
		shaped[order[rank][1]] = float(rank) / float(n - 1) - 0.5
	return shaped


## One ES step: θ' = θ + α/(n·σ) Σ_i F_i ε_i, where the mirrored-pair convention folds the minus
## candidates' sign in: contribution per pair j is (F_plus_j − F_minus_j)·ε_j. `shaped_fitness`
## must align with antithetic_candidates' ordering; n is the full candidate count (2·half_pop).
static func es_update(theta: PackedFloat32Array, epsilons: Array, shaped_fitness: Array,
		sigma: float, alpha: float) -> PackedFloat32Array:
	var n := shaped_fitness.size()
	if n != 2 * epsilons.size() or sigma <= 0.0:
		push_error("EsMath.es_update: need 2 fitnesses per epsilon (got %d for %d) and sigma > 0."
			% [n, epsilons.size()])
		return theta
	var scale := alpha / (float(n) * sigma)
	var out := theta.duplicate()
	for j in range(epsilons.size()):
		var weight: float = float(shaped_fitness[2 * j]) - float(shaped_fitness[2 * j + 1])
		var eps: PackedFloat32Array = epsilons[j]
		for k in range(out.size()):
			out[k] += scale * weight * eps[k]
	return out
