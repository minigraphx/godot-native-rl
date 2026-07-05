# Training In-Engine (Evolutionary Strategies — no Python)

`ESTrainer` is a training loop that runs **entirely inside the Godot process**: no Python, no
socket, no backprop. It evolves a small MLP policy with evolutionary strategies (OpenAI-style ES or sep-CMA-ES) —
each candidate weight vector becomes a live ncnn net in memory, one episode's return (your
existing reward system) is its fitness — and every checkpoint it writes **is already a deploy
artifact** (`.ncnn.param`/`.ncnn.bin`, loadable by every controller, no export step).

Because it needs nothing but the statically-linked ncnn forward pass, it runs on **every deploy
target**, including the web — the [live demo](https://minigraphx.github.io/godot-native-rl/)'s
*Evolution Lab* is this trainer running in a browser tab.

## When to use it (and when not)

**Use it for:** small nets (a few hundred to a few thousand weights) on dense-reward tasks;
self-contained/edge/on-device learning; shipping games that keep adapting; live "watch it learn"
exhibits; fine-tuning a shipped net after an environment change (see Warm-start below).

**Don't use it instead of PPO/SAC** for big problems: ES is sample-inefficient by design. The
Python backends remain the path for large nets, image observations, and maximum sample
efficiency.

## Quick start

Drop an `ESTrainer` node into a training scene **in place of `NcnnSync`** (same agent contract,
same `action_repeat` cadence; place it after the agents in the tree, like Sync). Or run the
shipped chase examples:

```bash
# single world
godot --headless --path . res://examples/chase_the_target/chase_es_train.tscn
# 8 tiled worlds via ParallelArena2D — candidates evaluate in parallel waves
godot --headless --path . res://examples/chase_the_target/chase_es_train_parallel.tscn
```

The parallel run learns chase from scratch in ~25 minutes (mean fitness −0.9 → 13.5 over 400
generations); the resulting net ships in the repo with a CI behavioral regression. Checkpoints
land in `out_dir` as `<stem>_best` (blessed on every improved generation mean, **measured before
the update step** — deploy this one), `<stem>_final`, and optional `<stem>_genN` snapshots
(`checkpoint_every`).

The `examples/seek_target` demo (#38) is the fully in-engine showcase: its shipped net was
trained by `seek_es_train_parallel.tscn` with `optimizer = "cma_es"` — env, training, and deploy
all native, no Python installed at any point.

## The knobs that matter

| Export | What it does |
|---|---|
| `hidden_dims`, `hidden_activation` | The policy MLP between your obs and action space |
| `optimizer` | `openai_es` (default) or `cma_es` — see the optimizer section below. Cmdline `optimizer=` overrides |
| `half_population` | Candidates per generation = 2× this (antithetic pairs; for `cma_es` simply λ = 2× this) |
| `sigma`, `alpha` | Perturbation size / learning rate. He-init weights need σ large enough to change behavior (the chase examples use 0.4 from scratch, 0.25 for fine-tuning). Under `cma_es`, `sigma` is only the initial step size (self-adapts) and `alpha` is unused |
| `episode_decisions` | Episode horizon in decision steps — **the trainer owns it** (agents' `reset_after` is overridden and restored, so a controller self-reset can never silently delete the final window's reward from fitness) |
| `episodes_per_candidate` | Fitness = mean over k seeded episodes (noise ↓, wall-clock ↑) |
| `seed_games` | Common random numbers (below). Leave on |
| `speed_up` | Same mechanism as Sync (`speedup=` cmdline overrides it) |
| `exit_on_finish` | CLI runs exit 0 on finish / 1 on abort — never hangs |

## Choosing the optimizer: OpenAI-ES vs sep-CMA-ES

`optimizer = "openai_es"` (default) is the antithetic gradient estimator described above.
`optimizer = "cma_es"` switches to **sep-CMA-ES** (Ros & Hansen 2008) — the diagonal-covariance
variant of CMA-ES, O(θ) per generation so it scales to net-sized search spaces — which adds what
plain ES lacks: **self-adapting step size** (`sigma` becomes only the *initial* value; CSA grows
it while progress is easy and shrinks it near an optimum) and **per-coordinate variances** (stiff
weights get small perturbations, loose ones large). `alpha` is unused; everything else — CRN
seeding, episode phases, blessing, checkpoints — is identical. Cmdline `optimizer=` (and
`generations=`) override the scene, so the same training scene can run either:

```bash
godot --headless --path . res://examples/chase_the_target/chase_es_train_parallel.tscn optimizer=cma_es
```

**Measured head-to-head** (one paired run of the parallel chase scene: 200 generations each,
identical seeds/CRN/budget, only `optimizer=` differing — and note the OpenAI-ES `sigma`/`alpha`
were the hand-tuned values that produced the shipped net, so this is not a straw-man baseline):

| Mean fitness | OpenAI-ES | sep-CMA-ES |
|---|---|---|
| first generation above 0 | 91 | **15** |
| first generation above 2.314 (the baseline's whole-run best) | 192 | **22** |
| first generation above 10 | never | **33** |
| best generation mean in 200 generations | 2.314 | **13.389** |

sep-CMA-ES reached the baseline's 200-generation endpoint ~9× sooner and matched the historical
**400**-generation OpenAI-ES result (13.5) within ~140 generations. The mechanism is visible in
the logs: CSA grows the step size while progress is easy (fast early climb) and shrinks it as the
population closes in, which fixed-σ ES cannot do. One paired run, so treat the exact ratios as
indicative — but the gap is far beyond run-to-run noise on this task. Default remains
`openai_es` for continuity with the shipped nets; prefer `cma_es` for new from-scratch runs.

## Fitness comparability: why the defaults look the way they do

ES lives or dies on the *ranking* between candidates, so the trainer builds the variance
reduction in:

- **Common random numbers** (`seed_games`): every candidate's episode *k* in generation *g* is
  seeded identically, via the duck-typed `seed_rng()` on the node at each agent's `game_path`.
  Without it, spawn luck drowns the ranking — the first-ever ES run flatlined until CRN landed.
  If your game node has no `seed_rng()`, you'll get one loud warning; connect the
  `episode_starting(slot, candidate, generation, episode)` signal and seed however your env
  needs.
- **Two-phase episode starts**: each episode begins with one decision window under a neutral
  action so leftover game events resolve against the *old* RNG stream — every candidate then
  sees identical fresh draws.
- **Phantom-done restarts**: a terminal that fires before the candidate's first decision
  (hostile spawn) restarts the episode instead of recording a meaningless 0.

## Warm-start: on-device fine-tuning of a shipped net

The θ⇄ncnn codec is bijective, so a shipped, previously-trained net can seed the search:

```
warm_start_param_path = "res://examples/chase_the_target/models/chase_es.ncnn.param"
warm_start_bin_path   = "res://examples/chase_the_target/models/chase_es.ncnn.bin"
```

The `.param` is validated against the scene's architecture (fail loud on mismatch). The shipped
`chase_es_finetune_parallel.tscn` demonstrates it against a fleeing target the original net
never saw.

**This includes your pnnx-exported PPO nets** (#328): warm-start accepts nets this codec never
wrote — a structural adapter parses foreign MLP params (pnnx layer names, stray Reshape,
fp16-tagged weight blobs) and adopts any net whose architecture matches the scene
(`hidden_dims`/`hidden_activation` must mirror it, e.g. `[64, 64]` + `"tanh"` for SB3-default
PPO actors). CI proves it live: the shipped chase PPO net is adopted bit-for-bit and scores
generation-1 fitness ~5 (an untrained init starts near −1), then CMA-ES keeps improving it.
Scope: plain MLP actors — standalone OR fused (`9=`) activations both parse (#340) — which
covers chase, rover, fly_by, quadruped, hexapod and 3dball; CNN policies (visual_chase) are
refused loud by design.

**What warm-start buys — measured honestly:** *time-to-competence*. In the shipped experiment
the warm-started population outperformed 300 generations of identical from-scratch training **at
generation 1** (replicated at two difficulty settings). **What it can't buy:** capability the
observations don't support — same-architecture fine-tuning showed ~no headroom on that task
because the obs carry no target velocity, so informed pursuit was already the representable
optimum. Rule of thumb: fine-tune when the environment shift moves the optimum *within* what
your inputs can express; add observations (and retrain properly) when it doesn't.

## Watching it learn

Open the launcher's **Evolution Lab** (`evolution_lab.tscn`): 8 worlds train on screen while a
champion world hot-swaps onto every blessed checkpoint (`checkpoint_saved` signal →
`reload_model`), with a live learning-curve HUD. Keys 1/2/3 set speed. The same scene runs in
the published web build. The lab trains with `cma_es` (the benchmark above is why — visibly
faster learning while you watch).

## Signals & integration

- `generation_finished(generation, mean_fitness, best_fitness)` — emitted **before** any
  blessing, so a consumer pairing it with `checkpoint_saved` always sees the matching stats.
- `checkpoint_saved(stem, param_path, bin_path)` — react to blessed nets (hot-swap a champion,
  upload, notify) without re-deriving the blessing rule.
- `training_finished(best_mean_fitness)`.

Related reading: the design + measured experiments live in
[`docs/superpowers/specs/2026-07-02-native-es-trainer-design.md`](../superpowers/specs/2026-07-02-native-es-trainer-design.md);
the deploy-side "checkpoint = artifact" flow in [deploying.md](deploying.md).
