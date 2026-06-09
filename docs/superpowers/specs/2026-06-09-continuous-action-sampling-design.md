# Continuous DiagGaussian Action Sampling (`log_std` sidecar) + FlyBy example — Design

**Date:** 2026-06-09
**GitHub issue:** #64 (`area:deploy`, `priority:3`, milestone v0.3 — deploy moat + training depth)
**Builds on:** #16 (stochastic action sampling — discrete softmax sampling; `deterministic_inference` /
`inference_seed` + the `deterministic`/`rng` plumbing already reaching the continuous branch) and #24
(VecNormalize obs-stats sidecar — the loader/validator/wiring pattern this mirrors)
**Upstream reference:** `edbeeching/godot_rl_agents` →
`godot_rl/wrappers/onnx/stable_baselines_export.py` (export emits **mean** for continuous, never a
sampled action and never the std) and `edbeeching/godot_rl_agents_examples` → `FlyBy` (the env ported
in PR 2)
**Status:** Approved (brainstorm → spec)

## Problem & motivation

Our deploy path samples **discrete** actions stochastically (#16) but a continuous (Box) policy always
emits its **mean** — the std is dropped at export, so there is no exploration / variation at deploy.
godot_rl has the same limitation by design: its ONNX/ncnn export contains only the mean head.

For **PPO continuous (Box)**, sb3's std is a **state-independent** learned parameter
(`policy.log_std`) — a fixed per-dim vector that is *never* part of the network output. We can export it
as a **sidecar JSON** (mirroring `obs_norm_stats` / VecNormalize), then sample a DiagGaussian game-side:
`action = mean + std·N(0,1)`, then the existing optional `tanh` squash. This **exceeds godot_rl**
(game-side continuous sampling with nothing extra in the ncnn output) and is squarely on the deploy moat.

**Out of scope (unchanged from #16):** SAC continuous uses a *state-dependent* std that the export
hardcodes to `tanh(mean)`; sampling it would change the SAC export path — separate spec.

This work ships in **two stacked PRs**: PR 1 is the deploy-side capability (synthetic-tested, no training
run); PR 2 ports the **FlyBy** env as a real PPO-continuous example that **ships a runnable ncnn net**
demonstrating the feature out of the box (see the "examples ship runnable nets" principle).

## Sidecar representation (decision)

The issue sketched `{"<key>": {"log_std": [...]}}`, but **SB3's `policy.log_std` is a single flat vector
over the whole continuous action dim and carries no godot_rl action-key names.** Forcing the user to
attach key names at export invites train/deploy mismatch. So the sidecar is **flat**:

```json
{ "std": [s0, s1, ...], "action_dim": N }
```

- `std = exp(policy.log_std)` (length `N` = total continuous action dim).
- `decode_actions` applies `std` **positionally across continuous segments** using a continuous-only
  counter, so single- and multi-continuous-key spaces both work (e.g. FlyBy's `pitch` + `turn`, each
  size 1, map to `std[0]`, `std[1]`).
- Mixed discrete+continuous in one PPO policy is not an SB3 construct, so positional-over-continuous is
  sufficient; the loader validates `std.size()` against the action space at load and fails loud on
  mismatch.

---

# PR 1 — the capability (no training run)

## Components

### 1. `scripts/export_action_dist.py` (new — mirrors `scripts/export_vecnormalize.py`)
- CLI: `.venv-train/bin/python scripts/export_action_dist.py <ppo_checkpoint.zip> [--out action_dist.json]`.
- Loads the SB3 model, reads `model.policy.log_std` (a `torch.nn.Parameter`, shape `[action_dim]`),
  computes `std = exp(log_std)`, writes `{"std": [...], "action_dim": N}`.
- **Fail-fast (ValueError)** when the policy has no `log_std` (e.g. a discrete/MultiDiscrete policy, or a
  SAC actor whose std is state-dependent) — message points at the SAC out-of-scope note.
- **Extraction-correctness check:** assert the written `std` equals `torch.exp(log_std)` elementwise
  (trivially exact — this guards the extraction, not sampling; sampling correctness is the GDScript
  histogram test).
- Pure helpers (`std_from_policy`, `write_action_dist_json`) kept import-light so the Python unittest can
  exercise them without a full SB3 train; heavy imports lazy inside `main()` (repo convention).

### 2. `addons/godot_native_rl/controllers/action_dist.gd` (new — mirrors `obs_normalize.gd`)
- `validate(stats: Dictionary) -> bool` — `std` present as a non-empty numeric array; if `action_dim`
  present it must equal `std.size()`.
- `to_typed(stats: Dictionary) -> Dictionary` — `{"std": PackedFloat32Array}` (coerce once; the per-frame
  hot path never re-coerces). Invalid → `push_error` + `{}`.
- No `normalize`-style transform here: the actual Gaussian draw lives in `decode_actions` (it needs the
  per-segment mean + the rng, which already flow there).

### 3. `addons/godot_native_rl/controllers/action_decode.gd` (extend — backward-compatible)
New trailing param (the plumbing choice: a **separate** dict, mirroring `obs_norm_stats` as its own core
dict — keeps action-space *shape* separate from sampling *params*):
```
decode_actions(output, action_space, deterministic := true, rng := null, action_dist := {}) -> Dictionary
```
- Continuous branch, maintain a continuous-only counter `c` (advances `+1` per continuous value consumed):
  - `deterministic` **or** `action_dist` empty/no `std` → **unchanged**: `mean` (optional `tanh`).
  - else → `sample = mean + std[c] * _draw_normal(rng)`, then optional `tanh`. `_draw_normal(rng)` =
    `rng.randfn(0.0, 1.0)` if `rng != null` else `randfn(0.0, 1.0)` (matches the discrete branch's
    rng-or-global fallback).
  - Defensive: if `c >= std.size()` (shouldn't happen — validated at load), fall back to mean for that dim
    rather than indexing out of range.
- Discrete branch and all existing validation/error sentinels: **untouched**. Every existing caller (which
  passes no `action_dist`) is unaffected — still mean/`argmax`.

### 4. `addons/godot_native_rl/controllers/ncnn_controller_core.gd` (extend)
- New state: `var action_dist_stats: Dictionary = {}`.
- `choose_and_apply_action` passes `action_dist_stats` as the new `decode_actions` arg (one-line change
  alongside the existing `deterministic_inference`, `rng`).

### 5. `ncnn_ai_controller_2d.gd` / `ncnn_ai_controller_3d.gd` (extend — mirror obs-norm wiring)
- `@export_file("*.json") var action_dist_stats_path: String = ""`.
- `_load_action_dist_stats()` — open/parse/validate → `_core.action_dist_stats = ActionDist.to_typed(...)`
  (clone of `_load_obs_norm_stats`); called in `_ready()`'s NCNN_INFERENCE branch.
- `set_action_dist_for_test(stats)` — test seam (clone of `set_obs_norm_stats_for_test`).

## Data flow

```
obs ─► ncnn ─► output (mean for continuous dims)
                     │
       ActionDecode.decode_actions(deterministic, rng, action_dist)
         └─ continuous: deterministic|no-std ? mean (opt tanh)
                        : (mean + std[c]·randfn(rng)) (opt tanh)
                     │
                 set_action
```

## Error handling
- All existing `ActionDecode` validation runs **before** the sampling branch (unchanged): non-positive
  size, output too short, length mismatch, unknown `action_type` → `push_error` + `{}`.
- `ActionDist.validate` rejects a malformed/empty/mismatched-length sidecar at **load** time (loud), so a
  bad fixture never reaches the per-frame path.
- The sampling branch cannot produce a structurally invalid action: it emits exactly `size` floats per
  continuous key, identical in shape to the deterministic path; the only difference is value.

## Testing (TDD) — PR 1
- **`test/python/test_export_action_dist.py`** (new): a tiny synthetic policy-like object with a known
  `log_std` → assert `std == exp(log_std)` and JSON shape `{std, action_dim}`; non-Box / missing-`log_std`
  → `ValueError`.
- **`test/unit/test_action_dist.gd`** (new): `validate` (good/empty-std/length-mismatch/missing-key);
  `to_typed` coercion + invalid → `{}`.
- **`test/unit/test_action_decode.gd`** (extend):
  - deterministic continuous (default, no `action_dist`) → **byte-identical to current** (regression guard).
  - `deterministic = false` + seeded `RandomNumberGenerator` + `action_dist = {std:[σ…]}`: many draws →
    empirical per-dim mean ≈ `mean` and std ≈ `σ` within loose tolerance; same seed → identical sequence
    (reproducibility); `tanh` applied **after** the Gaussian draw when `squash` set.
  - `action_dist = {}` while `deterministic = false` → falls back to mean (continuous unchanged; only
    discrete keys sample).
  - multi-continuous-key (two size-1 keys) → each uses its own `std[c]` (positional mapping).
- **`test/unit/test_stochastic_inference.gd`** (extend, optional): stub-runner controller with
  `action_dist_stats` set + fixed seed → reproducible continuous action end-to-end.
- Full `./test/run_tests.sh` green; existing golden/argmax/mean regressions unchanged (defaults
  deterministic, no `action_dist`).

## Docs — PR 1
- `docs/guide/deploying.md` — new "Continuous action sampling (DiagGaussian std sidecar)" section right
  after **VecNormalize obs stats** (~line 156): `export_action_dist.py` → JSON → `action_dist_stats_path`.
- `docs/guide/building-your-agent.md` — extend the `deterministic_inference` bullet (line 33) to note it
  drives **continuous** sampling too when `action_dist_stats_path` is set; add an `action_dist_stats_path`
  bullet beside `obs_norm_stats_path` (line 36).
- `README.md` — controller-exports table: add `action_dist_stats_path`.
- `CLAUDE.md` — Key commands: add the `export_action_dist.py` line; note `decode_actions` is now
  `action_dist`-aware.
- `docs/godot-rl-gap-analysis-2026-06-02.md` — continuous-sampling row → **exceeds** godot_rl.
- `docs/BACKLOG.md` — **no new entry** (#64 is a GitHub-only follow-up to item 43, already ✅; per
  CLAUDE.md, BACKLOG.md isn't extended with new items). Optionally add a one-line "continuous follow-up
  done" note under item 43. PR body `Closes #64` (PR 1 ships the capability; PR 2 references it).

---

# PR 2 — FlyBy example (stacked on PR 1, the out-of-the-box payoff)

The capability is only proven if a game dev can **run a continuous-PPO net natively, out of the box**
(no Python, no training). PR 2 ports **FlyBy** (`edbeeching/godot_rl_agents_examples/examples/FlyBy`) —
a cartoon plane with two continuous actions (`pitch`, `turn`) — and ships its trained net.

### Decisions (2026-06-09, post-brainstorm)
- **Faithful asset port.** Vendor upstream's `cartoon_plane/` glTF (`scene.gltf` + `scene.bin` +
  `.material`s + `texture_07.png`) and the HDR sky into `examples/fly_by/`, with an
  `examples/fly_by/ATTRIBUTION.md` crediting `edbeeching/godot_rl_agents_examples` (MIT, © Edward
  Beeching 2022) and carrying the MIT notice. Visuals don't affect headless runs (Godot headless skips
  rendering), so the play/eval scenes stay headless-compatible.
- **Faithful 8-dim obs over a goal sequence.** A ring/sequence of goal nodes flown through in order.
  Obs mirrors upstream exactly: current-goal local-frame direction (3) + `dist/50` (1), next-goal
  local-frame direction (3) + `next_dist/50` (1) = **8 dims**. Goal vectors are expressed in the
  plane-local frame (`plane.global_transform.basis.inverse() * (goal_pos - plane_pos)`), which encodes
  heading. `compute_obs` is a pure helper (unit-tested).
- **Single-arena training only.** Ship `fly_by_train.tscn` (one world) like `train_chase`. No
  `_train_parallel.tscn` in this PR; add later as a follow-up if single-arena throughput is too slow.
- **Action application:** `pitch`/`turn` ∈ [-1,1] rotate the plane basis; constant forward speed.
  `set_action` clamps to [-1,1] (PPO mean is unbounded, and the #64 DiagGaussian sample can exceed the
  range) — no `"squash"` key, clamp game-side. Reward via `RewardBuilder`: progress shaping toward the
  current goal + `goal_reached` event bonus + step penalty + arena-exit penalty; done on arena exit.
- **File split** mirrors the `rover_3d` precedent (the closest 3D example): two files —
  `fly_by_game.gd` (Node3D that owns the plane body + goal ring + arena, integrates motion manually
  on a `Node3D` like `RoverGame.move_agent` for headless determinism + pure-helper testability, no
  `CharacterBody3D` physics) and `fly_by_agent.gd` (NcnnAIController3D: obs/action/reward wiring). The
  vendored cartoon_plane glTF is a visual child of the plane body and doesn't affect the sim.

## Components — PR 2
- **`examples/fly_by/`**: port the plane env to our framework — `fly_by_agent.gd` (`get_obs`,
  `get_action_space` → `{pitch:{size 1,continuous}, turn:{size 1,continuous}}`, `set_action`),
  `fly_by_world.tscn`, `fly_by_train.tscn` (+ a `_train_parallel.tscn` only if cheap), and a
  **standalone headless-compatible `fly_by.tscn` play scene** driven by `NcnnAIController3D`. Reuse the
  godot_rl scene's geometry/feel; swap their `AIController` for `NcnnSync`.
- **`scripts/train_fly_by.sh`**: SB3 PPO continuous over the train scene (port 11008), mirroring
  `train_chase.sh`. Documents `TIMESTEPS`/`SCENE` overrides.
- **Export**: policy mean → ncnn (`export_to_ncnn.py`, ONNX or TorchScript per the usual path) **plus**
  `export_action_dist.py` for the std sidecar.
- **Committed runnable artifacts**: `models/fly_by_policy.ncnn.{param,bin}` + `models/fly_by_action_dist.json`
  (+ obs-norm sidecar if trained with VecNormalize). The play scene wires `action_dist_stats_path` and
  ships **`deterministic_inference = true` by default** (predictable demo), with stochastic flight as a
  documented toggle that demonstrates #64.

## Testing — PR 2
- **Behavioral regression** on the *real* model in a headless eval scene: deterministic run asserts stable
  expected behavior (plane stays controlled / makes progress) — golden-style, like the trained-chase and
  BallChase regressions. Plus a seeded-stochastic **reproducibility** assertion (same seed → same first
  actions) so the continuous-sampling path is guarded end-to-end on a real net.
- **`test/run_tests.sh`** integration: add the FlyBy smoke/regression alongside the existing trained-example
  checks.

## Docs — PR 2
- `docs/guide/running-examples.md` — add a FlyBy section (continuous-PPO), mirroring the BallChase section
  (~line 72), including how to toggle stochastic flight.
- `docs/guide/training.md` — add the `train_fly_by.sh` command.
- `README.md` — examples list: add FlyBy (continuous-control PPO).
- `CLAUDE.md` — "Current state" examples line + a `train_fly_by.sh` Key-command entry.
- `docs/BACKLOG.md` — no new entry (see PR 1 note); PR body references #64.
- (No dedicated `docs/examples/*_tutorial.md` — only `chase_the_target` has one; out of scope unless
  requested.)

## Open risk — PR 2
The training run is the only real unknown (time, hyperparameters, whether FlyBy converges cleanly under
our wire bridge). Stacking it behind PR 1 means the **capability still ships** even if the example needs
iteration. FlyBy is the simplest continuous env in the source repo (2 actions), chosen to minimize this.
