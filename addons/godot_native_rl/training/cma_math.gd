extends RefCounted
# Pure sep-CMA-ES optimizer (native ES trainer follow-up to #131).
#
# CMA-ES (Hansen) adapts a full n×n covariance — O(n²) memory, O(n³) eigendecomposition — which
# is the wrong tool for neural-net-sized θ in GDScript. This is the SEPARABLE variant
# (Ros & Hansen 2008): the covariance is restricted to its diagonal, making every operation O(n)
# per candidate while keeping the parts that beat plain OpenAI-ES on many landscapes —
# weighted recombination of the top-μ candidates, cumulative step-size adaptation (CSA), and
# per-coordinate variance adaptation via the rank-one + rank-μ evolution paths. The lost
# correlations are compensated by faster learning rates (the (n+2)/3 factor from the paper).
#
# Pure and engine-free like es_math.gd: the caller owns the RNG (seeded → the whole optimization
# is reproducible) and the fitness evaluation. Fitness is MAXIMIZED. Stateful by nature (mean,
# step size, diagonal covariance, evolution paths), hence a RefCounted instance, not statics.
#
# Usage per generation:
#   sample(rng)                      # draw λ standard-normal z vectors
#   candidate_at(i) for i in 0..λ-1  # x_i = m + σ·√C∘z_i (lazy — one candidate at a time)
#   update(fitness)                  # rank, recombine, adapt σ and C, advance the mean

var n := 0
var population := 0  ## λ — candidates per generation
var mu := 0  ## parents (top-μ recombination)
var mu_eff := 0.0
var weights: Array = []  # recombination weights, descending, sum 1
var c_sigma := 0.0
var d_sigma := 0.0
var c_c := 0.0
var c_1 := 0.0
var c_mu := 0.0
var chi_n := 0.0  # E||N(0,I)||

var sigma := 0.0
var _mean := PackedFloat32Array()
var _c_diag := PackedFloat32Array()  # diagonal of the covariance C
var _p_sigma := PackedFloat32Array()
var _p_c := PackedFloat32Array()
var _z: Array = []  # this generation's λ standard-normal draws
var _generation := 0


## Initialize around mean0 with step size sigma0 and λ = population_size. Returns false (with a
## pushed error) on an unusable configuration.
func setup(mean0: PackedFloat32Array, sigma0: float, population_size: int) -> bool:
	if mean0.is_empty() or sigma0 <= 0.0 or population_size < 4:
		push_error("CmaMath.setup: need a non-empty mean, sigma0 > 0 and population >= 4 (got n=%d, sigma0=%s, population=%d)."
			% [mean0.size(), str(sigma0), population_size])
		return false
	n = mean0.size()
	population = population_size
	mu = population / 2
	# Log-rank recombination weights over the top-μ, normalized to sum 1.
	weights = []
	var w_sum := 0.0
	for i in range(mu):
		var w := log((float(population) + 1.0) / 2.0) - log(float(i + 1))
		weights.append(w)
		w_sum += w
	var w_sq := 0.0
	for i in range(mu):
		weights[i] = float(weights[i]) / w_sum
		w_sq += float(weights[i]) * float(weights[i])
	mu_eff = 1.0 / w_sq
	var nf := float(n)
	c_sigma = (mu_eff + 2.0) / (nf + mu_eff + 5.0)
	d_sigma = 1.0 + 2.0 * maxf(0.0, sqrt((mu_eff - 1.0) / (nf + 1.0)) - 1.0) + c_sigma
	c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf)
	# Default CMA rates × the sep speedup factor (n+2)/3 (Ros & Hansen 2008, eq. for ccov_sep):
	# the diagonal has n free parameters instead of n(n+1)/2, so it can afford to learn faster.
	var sep := (nf + 2.0) / 3.0
	c_1 = minf(1.0, sep * 2.0 / ((nf + 1.3) * (nf + 1.3) + mu_eff))
	c_mu = minf(1.0 - c_1, sep * 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0) * (nf + 2.0) + mu_eff))
	chi_n = sqrt(nf) * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf))
	sigma = sigma0
	_mean = mean0.duplicate()
	_c_diag = PackedFloat32Array()
	_c_diag.resize(n)
	_c_diag.fill(1.0)
	_p_sigma = PackedFloat32Array()
	_p_sigma.resize(n)
	_p_c = PackedFloat32Array()
	_p_c.resize(n)
	_generation = 0
	return true


## Draw this generation's λ standard-normal vectors. Must be called once per generation, before
## candidate_at / update.
func sample(rng: RandomNumberGenerator) -> void:
	_z = []
	for _i in range(population):
		var z := PackedFloat32Array()
		z.resize(n)
		for j in range(n):
			z[j] = rng.randfn()
		_z.append(z)


## Candidate i of the current generation: x_i = m + σ·√C ∘ z_i. Lazy like EsMath.candidate_at —
## the trainer consumes one candidate at a time, so the population never needs materializing.
func candidate_at(idx: int) -> PackedFloat32Array:
	if idx < 0 or idx >= _z.size():
		push_error("CmaMath.candidate_at: index %d out of range for %d sampled candidates." % [idx, _z.size()])
		return PackedFloat32Array()
	var z: PackedFloat32Array = _z[idx]
	var out := PackedFloat32Array()
	out.resize(n)
	for j in range(n):
		out[j] = _mean[j] + sigma * sqrt(_c_diag[j]) * z[j]
	return out


## One sep-CMA-ES step from the λ fitness values (aligned with candidate_at indices; MAXIMIZED).
## Advances the mean, the evolution paths, the step size and the diagonal covariance.
func update(fitness: Array) -> void:
	if fitness.size() != _z.size() or _z.is_empty():
		push_error("CmaMath.update: need one fitness per sampled candidate (got %d for %d)."
			% [fitness.size(), _z.size()])
		return
	# Rank descending (maximize), ties stable by candidate index.
	var order: Array = []
	for i in range(population):
		order.append([-float(fitness[i]), i])
	order.sort()
	# Weighted recombination over the top-μ: z_w (isotropic) and y_w = √C∘z_w-per-candidate.
	var z_w := PackedFloat32Array()
	var y_w := PackedFloat32Array()
	z_w.resize(n)
	y_w.resize(n)
	var y_sel: Array = []  # top-μ y vectors, for the rank-μ covariance term
	for r in range(mu):
		var idx: int = order[r][1]
		var w: float = weights[r]
		var z: PackedFloat32Array = _z[idx]
		var y := PackedFloat32Array()
		y.resize(n)
		for j in range(n):
			y[j] = sqrt(_c_diag[j]) * z[j]
			z_w[j] += w * z[j]
			y_w[j] += w * y[j]
		y_sel.append(y)
	# Mean shift.
	for j in range(n):
		_mean[j] += sigma * y_w[j]
	# Step-size path (C is diagonal, so C^{-1/2}·(y_w/√C) is exactly z_w) + CSA update.
	var cs_norm := sqrt(c_sigma * (2.0 - c_sigma) * mu_eff)
	var ps_sq := 0.0
	for j in range(n):
		_p_sigma[j] = (1.0 - c_sigma) * _p_sigma[j] + cs_norm * z_w[j]
		ps_sq += _p_sigma[j] * _p_sigma[j]
	var ps_norm := sqrt(ps_sq)
	_generation += 1
	# Stall the rank-one update while σ is still adapting fast (standard h_σ gate): a long
	# p_σ means the last steps were correlated — feeding that into C would over-elongate it.
	var expected := sqrt(1.0 - pow(1.0 - c_sigma, 2.0 * float(_generation)))
	var h_sigma := 1.0 if ps_norm / expected < (1.4 + 2.0 / (float(n) + 1.0)) * chi_n else 0.0
	var cc_norm := sqrt(c_c * (2.0 - c_c) * mu_eff)
	for j in range(n):
		_p_c[j] = (1.0 - c_c) * _p_c[j] + h_sigma * cc_norm * y_w[j]
	# Diagonal covariance: decay + rank-one (p_c²) + rank-μ (weighted y²), with the h_σ
	# correction folded into the decay when the rank-one term is gated off.
	var decay := 1.0 - c_1 - c_mu + (1.0 - h_sigma) * c_1 * c_c * (2.0 - c_c)
	for j in range(n):
		var rank_mu := 0.0
		for r in range(mu):
			var yj: float = y_sel[r][j]
			rank_mu += float(weights[r]) * yj * yj
		_c_diag[j] = maxf(1e-20, decay * _c_diag[j] + c_1 * _p_c[j] * _p_c[j] + c_mu * rank_mu)
	sigma = clampf(sigma * exp((c_sigma / d_sigma) * (ps_norm / chi_n - 1.0)), 1e-12, 1e12)
	_z = []


## The current distribution mean — the trainer's θ / the checkpointed net.
func mean_vector() -> PackedFloat32Array:
	return _mean.duplicate()
