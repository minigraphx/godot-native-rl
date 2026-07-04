# Native in-engine evolutionary-strategies trainer (issue #131)

Date: 2026-07-02
Status: accepted (brainstorm → this spec; plan follows in docs/superpowers/plans/)

## What

A **training loop that runs entirely inside the Godot process** — no Python, no socket, no
backprop. An OpenAI-style evolutionary-strategies (ES) optimizer perturbs a flat policy weight
vector θ, evaluates each candidate with the **existing ncnn forward pass** (episodic return from
the **existing reward system** is the fitness), and updates θ from the rank-shaped fitnesses.

This is the "train without leaving the engine" capability: it works on every deploy target the
static-linked ncnn build reaches (web/WASM, console, mobile, edge) because it *is* the deploy
stack. Neither godot_rl (Python server) nor Unity ML-Agents (removed in-editor training) has this.

Two properties fall out for free:

- **The training artifact IS the deploy artifact.** ES's output is an ncnn `.param`/`.bin` pair —
  there is no export step at all.
- **Warm-start / on-device fine-tuning.** θ ⇄ bin is bijective for our MLP layout, so a shipped,
  Python-trained policy can be loaded into θ and adapted on-device (per-player adaptation).

## Why ES (recap from #131)

ncnn is inference-only. ES needs only (a) a forward pass — `NcnnRunner` has it; (b) a fitness
score — the Signal→Reward system + episodic return has it; (c) weight perturbation + selection —
the only new piece. Gradient-based training would mean re-implementing autograd in C++; ES
sidesteps it entirely.

## Key discovery that de-risks the whole issue

`#131` was labelled `needs-C++` for the weight-IO mechanism. **The C++ is already shipped**:
`NcnnRunner.load_model_from_buffers(param: PackedByteArray, bin: PackedByteArray)` exists, is
bound, and is battle-tested by the controllers' hot-swap path (#197). And the repo already knows
the exact ncnn `.param`/`.bin` byte layout for MLPs — `scripts/export_statedict_to_ncnn.py`
hand-writes it (fully unit-tested pure format writer). So the entire trainer is **pure GDScript**:

- port the format writer to GDScript (`NcnnWeights`),
- assemble a candidate's bin from θ (`PackedFloat32Array.to_byte_array()` — raw LE fp32, same as
  the Python `struct.pack("<f")`),
- load it with `load_model_from_buffers`, run episodes, done.

A chase-sized net (5→32→4 ≈ 300 params, ~1.3 KB bin) reloads per candidate in microseconds.

## Architecture (pure helpers + thin node, house style)

### 1. `addons/godot_native_rl/training/ncnn_weights.gd` — pure θ ⇄ ncnn-buffers codec

MLP spec: `{dims: [in, h1, ..., out], hidden_activation: "relu"|"tanh"|"sigmoid"|"",
output_activation: same}`. Static funcs:

- `mlp_spec(dims, hidden_activation="relu", output_activation="") -> Dictionary`
- `theta_size(spec) -> int` — Σ per linear (in·out + out)
- `param_text(spec) -> String` — ncnn magic `7767517`, `layer_count blob_count`, then
  `Input in0 0 1 in0 0=<dim>` + per layer `InnerProduct fcN 1 1 <prev> fcN 0=<out> 1=1 2=<in*out>`
  + activation lines (`ReLU actN … 0=0.0` / `TanH` / `Sigmoid`); final top renamed `out0`
  (NcnnRunner blob-name convention).
- `bin_bytes(spec, theta) -> PackedByteArray` — per linear: 4-byte fp32 tag (uint32 0) + weight
  floats ([out, in] row-major) then untagged bias floats. θ layout per layer: weights then bias.
- `theta_from_bin(spec, bin) -> PackedFloat32Array` — inverse (warm-start from a shipped net).
- `init_theta(spec, rng) -> PackedFloat32Array` — He-scaled gaussian weights, zero bias.

Fail loud on malformed specs. Format-critical + fully unit-tested, mirroring the Python writer's
test posture; a **round-trip golden** loads writer output through the real
`load_model_from_buffers` and asserts the forward pass against hand-computed values.

### 2. `addons/godot_native_rl/training/es_math.gd` — pure OpenAI-ES optimizer

- `sample_epsilons(rng, half_pop, n) -> Array[PackedFloat32Array]` — seeded gaussians
  (`rng.randfn`).
- `antithetic_candidates(theta, sigma, epsilons) -> Array` — mirrored pairs
  `[θ+σε_0, θ−σε_0, θ+σε_1, …]` (2·half_pop candidates).
- `centered_ranks(fitness) -> Array` — rank transform to [−0.5, 0.5] (noise-robust; ties by
  stable order).
- `es_update(theta, epsilons, shaped_fitness, sigma, alpha) -> PackedFloat32Array` —
  θ' = θ + α/(n·σ) Σᵢ Fᵢ εᵢ with the mirrored sign convention.

Unit tests: deterministic under seed; **fitness improves on a synthetic sphere function**
(maximize −‖x − target‖²) over a few hundred generations; centered-rank properties.

### 3. `addons/godot_native_rl/training/es_trainer.gd` — thin `ESTrainer` node

A **drop-in native replacement for the `NcnnSync` node**: same agent contract
(`get_obs()["obs"]`, `set_action(dict)`, `get_reward()`/`zero_reward()`,
`get_done()`/`set_done_false()`, `needs_reset`, `get_action_space()`), same `action_repeat`
cadence, driven from `_physics_process`. Episodes are the fixed horizon the controllers already
implement (`reset_after` via `NcnnControllerCore.step`).

Per generation:
1. `sample_epsilons` → antithetic candidates.
2. Evaluate in **waves of `n_agents`**: candidate k's bin loads into agent k's dedicated
   `NcnnRunner` via `load_model_from_buffers`; run one episode per candidate; fitness = summed
   `get_reward()` drain until `get_done()`. Multi-agent scenes (e.g. `ParallelArena` tiles)
   evaluate `n_agents` candidates in parallel per wave.
3. Rank-shape → `es_update` → optionally write a checkpoint (`.param`/`.bin` — deploy-ready).

Actions decode through the existing `ActionDecode.decode_actions(output, action_space)` — the
same path every inference controller uses, so anything deployable is trainable.

Exports: `dims_hidden` (e.g. `[32]`), `half_population`, `sigma`, `alpha`, `generations`,
`action_repeat`, `seed`, `out_dir`, `warm_start_param_path`/`bin_path`. Signals:
`generation_finished(gen, mean_fitness, best_fitness)`, `training_finished(best_fitness)`.

### 4. Example + tests

- `examples/chase_the_target/chase_es_train.tscn` — chase world + `ESTrainer` (no Sync, no
  Python). Chase is the right vehicle: dense shaped reward, 5-dim obs, 5 discrete actions, tiny
  MLP — squarely in ES's sweet spot.
- CI: pure unit tests prove the optimizer improves on synthetic fitness (fast, deterministic);
  an integration smoke runs ~2 tiny generations headless and asserts the wiring (fitness values
  finite, θ updated, checkpoint written). A long "ES learns chase" behavioral run is a
  local/manual showcase first (CI improvement assertions on few generations of a noisy env would
  flake), committed later like the other trained regressions.

## Milestones

- **M1** (this PR): `ncnn_weights.gd` + round-trip golden through real ncnn; `es_math.gd` + sphere
  tests; `ESTrainer` + chase ES scene + headless wiring smoke; docs.
- **M2**: a real trained-in-engine chase net committed + behavioral regression; watch-it-learn
  demo scene in the launcher (live fitness HUD).
- **M3**: web-export showcase ("this page is training a net in your browser"); warm-start
  fine-tuning example (load shipped chase net → adapt).
- **M4** (stretch): CMA-ES for tiny nets; multi-world wave evaluation via `ParallelArena` tuning;
  fitness averaging over k seeds.

## Non-goals (unchanged from #131)

In-engine backprop; swapping ncnn for a training runtime; Hebbian/plasticity (#132 is separate).
ES is sample-inefficient — the docs frame it as self-contained/edge/small-net training and
per-player adaptation, not a PPO/SAC replacement.

## Risks

- **Fitness noise** (random spawns): fixed per-generation seeds and/or rank shaping; averaging
  over k episodes is an M4 knob.
- **Scale ceiling**: ES degrades past ~10⁴ params — stated up front; chase-sized nets are the
  target.
- **Wall-clock**: physics-bound; reuse the existing `speedup` machinery and `ParallelArena`
  tiling.

## Post-review addendum (2026-07-03)

The first two full runs plus an adversarial self-review reshaped the trainer; the shipped design
differs from the sketch above in these load-bearing ways:

- **The trainer owns the episode horizon** (`episode_decisions`, decision-step units; agents'
  `reset_after` is overridden to effectively-infinite). The controller's horizon self-reset
  zeroes the agent's reward accumulator before the trainer's boundary read — the final window's
  reward (catches!) silently vanished from fitness, biased exactly against good candidates.
- **Fitness comparability is first-class**: common-random-numbers seeding (`seed_games`
  duck-types each agent's `game_path` game's `seed_rng()` per (generation, episode);
  `episode_starting` signal for custom setup), k-episode averaging (`episodes_per_candidate`),
  a candidate-independent neutral action over every reset window, and phantom-done restarts.
  Without CRN, spawn luck drowned the ranking entirely (flat −0.89 mean, run 1).
- **`<stem>_best` is blessed before the ES update** — the measured mean belongs to the
  population around the *current* θ; the post-update θ is unevaluated.
- **`load_model_from_buffers` owns its bytes**: ncnn's memory reader zero-copies
  (`reference()`), which was a production use-after-free; the runner now reads from a private
  owned copy that outlives the net (a DataReader subclass would have been cleaner but doesn't
  link on iOS — ncnn builds without RTTI there; see docs/dev/gotchas.md).
- The Python and GDScript format writers are pinned together by a committed cross-language
  fixture (byte-equality + `theta_from_bin` decode).

## Warm-start fine-tuning experiment (2026-07-04, PR #298)

The M3 "warm-start fine-tuning example" was built and measured honestly. Setup: the shipped
`chase_es` net (trained vs a static target) dropped into `chase_drift_game` — a target that
FLEES (away-vector + seeded jitter, wall reflection; 260 px/s vs the agent's 300). Deterministic
eval (seeded, first layout re-rolled post-seed — a checker fix this experiment forced): the
shipped net falls from ~19 catches/1800 frames (static) to 6/6/6/5 across seeds.

Two identical 300-generation ES runs on the flee env (pop 32 × 3 CRN episodes, σ 0.25, α 0.08):

| | gen-1 mean fitness | blessed best mean |
|---|---|---|
| **Warm-start** (`warm_start_*_path` from the shipped net) | **3.91** | **4.48** |
| **Cold-start** (identical config, random init) | −1.33 | 0.67 |

Replicated at a milder setting (200 px/s: warm 5.86 → 6.73 vs cold −1.35 → 0.94). Two findings:

1. **Warm-start = time-to-competence.** At generation 1 the warm population already outperforms
   300 generations of from-scratch training by ~6×. This is the on-device adaptation value:
   after an environment change, the shipped brain is instantly competent and refining, where
   from-scratch is unusable for the first ~20 minutes.
2. **Same-architecture adaptation headroom on THIS task is ≈ nil** — the warm curve plateaus on
   arrival (deterministic eval: adapted 6/6/7/7 vs un-adapted 6/6/6/5; retention, not gain).
   Why: the 5-dim obs carry no target velocity, so informed pursuit is already the
   representable optimum; interception/cornering strategies need velocity features. Stated in
   the docs so users know when fine-tuning helps (policy near a NEW optimum reachable with the
   SAME inputs) and when it can't (the shift demands features the obs lack — that's a
   retrain-with-new-obs, not a fine-tune).

The committed regression (`trained_es_drift_scene.tscn`) guards the full pipeline — shipped net
→ warm-start decode → in-engine ES → blessed checkpoint → standard inference deploy — not an
improvement claim.

## sep-CMA-ES optimizer (2026-07-04, the M4 stretch — measured)

`training/cma_math.gd` adds the M4 "CMA-ES" stretch as **sep-CMA-ES** (Ros & Hansen 2008):
diagonal covariance only, O(θ) per generation (full CMA's O(θ²) memory + O(θ³) eigendecomposition
is wrong for net-sized θ in GDScript), with the paper's (n+2)/3 learning-rate speedup. Pure and
caller-seeded like es_math.gd; `ESTrainer.optimizer = "cma_es"` (cmdline `optimizer=`) selects it,
`sigma` becomes the initial step size (CSA self-adapts), `alpha` is unused. CRN, two-phase episode
starts, blessing and checkpoints are shared — the optimizer swap changed nothing else.

**Paired 200-generation chase benchmark** (parallel scene, identical seeds/budget; the OpenAI-ES
baseline ran the shipped net's hand-tuned σ=0.4/α=0.1):

| | OpenAI-ES | sep-CMA-ES |
|---|---|---|
| first gen with mean > 0 | 91 | 15 |
| first gen above the baseline's whole-run best (2.314) | 192 | 22 |
| best generation mean | 2.314 | **13.389** |

CMA matched the historical **400-gen** OpenAI-ES result (13.5) in ~140 generations — a ~9× faster
climb to the baseline's endpoint and ~3× less budget to the shipped-net level, from self-adapting
step size + per-coordinate variances. One paired run (indicative, not a statistics claim), but far
beyond run-to-run noise. The Evolution Lab demo now runs `cma_es` (faster visible learning in the
browser); `chase_es_train_parallel.tscn` keeps `openai_es` so the committed net stays reproducible.
