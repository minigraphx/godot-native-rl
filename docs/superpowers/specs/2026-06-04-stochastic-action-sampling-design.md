# Stochastic Action Sampling (`deterministic_inference`) — Design

**Date:** 2026-06-04
**Backlog item:** 43 (`docs/BACKLOG.md`) — GitHub issue **#16**
**Builds on:** item 21 (continuous + multi-key action deploy, `ActionDecode`) and the
`NcnnControllerCore` / `NcnnAIController2D/3D` deploy path
**Upstream reference:** `edbeeching/godot_rl_agents` →
`godot_rl/wrappers/onnx/stable_baselines_export.py` (export emits **logits/mean**, never a sampled
action) and the plugin's `AIController` / `Sync` `deterministic_inference` export
**Roadmap reference:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md`
**Status:** Approved (brainstorm → spec)

## Problem & motivation

Upstream `Sync`/`AIController` expose a `deterministic_inference` flag (default `true`). When `false`,
**discrete** actions are drawn from `softmax(logits)` instead of `argmax`, enabling exploration during
eval or human-in-the-loop play without retraining. Our deploy path is always deterministic: `ActionDecode`
hardcodes `argmax` for discrete keys. This issue adds the flag.

### Why this is discrete-only (and still full parity)

Verified from godot_rl source: the ONNX/ncnn export (`OnnxablePolicy.forward_ppo`) returns
`action_net(action_hidden)` — **raw logits** for discrete, the **mean** for continuous. The argmax /
sampling / squashing all happen on the **Godot deploy side**, which is exactly what `ActionDecode` does.
The exported policy contains **no `std`/`log_std` head**, so neither godot_rl nor this repo can sample
continuous actions from the model output. Therefore **discrete-only sampling is full deploy-time parity
with godot_rl** — there is no std in the output to sample continuous from.

### Forward compatibility (continuous is a designed-in fast-follow, not lost)

For **PPO continuous (Box)** action spaces, sb3's std is a **state-independent** learned parameter
(`policy.log_std`) — a fixed vector per action dim, never part of the network output by nature. It is
trivially exportable as a **sidecar JSON** (mirroring the existing `obs_norm_stats` / VecNormalize
pattern), after which the game side can sample a DiagGaussian (`mean + std·N(0,1)`, then optional `tanh`).
That **exceeds** godot_rl (game-side continuous sampling with nothing extra in the ncnn output) and fits
the project moat. It is **deferred to a follow-up issue** (see "Out of scope"), but this design lays the
plumbing so it drops in with no rework: the `deterministic` flag + injected `rng` already flow through
`decode_actions` to **both** action branches; the continuous follow-up only adds a stats loader and a
Gaussian draw. **SAC** continuous uses a *state-dependent* std that the export hardcodes to `tanh(mean)`;
sampling it would change the SAC export path and is out of scope for both this issue and the immediate
follow-up.

## Components

### 1. `InferenceMath` (pure, extend — `argmax` unchanged)
- `softmax(logits: PackedFloat32Array) -> PackedFloat32Array` — numerically stable (subtract `max`
  before `exp`); returns an empty array for empty input.
- `sample_categorical(probs: PackedFloat32Array, u: float) -> int` — inverse-CDF: walk the cumulative
  sum, return the first index where `cumsum > u`; clamp to the last index on floating-point drift
  (`u` at/above the total). Empty input → `-1` (same error sentinel as `argmax`). `u` is expected in
  `[0, 1)`; values outside are handled by the clamp (no assert, deploy path stays robust).

Both are pure and deterministic given their inputs — the RNG lives outside, so the math is unit-tested
with fixed `u` values and needs no seeding.

### 2. `ActionDecode.decode_actions` — backward-compatible signature bump
```
decode_actions(output, action_space, deterministic := true, rng: RandomNumberGenerator = null) -> Dictionary
```
- Discrete branch: `deterministic` → `InferenceMath.argmax(segment)` (**unchanged path**); else →
  `InferenceMath.sample_categorical(InferenceMath.softmax(segment), _draw(rng))`, where
  `_draw(rng)` returns `rng.randf()` if `rng != null` else the global `randf()`.
- Continuous branch: **unchanged** (mean, optional `tanh` via the per-key `squash` flag). The
  `deterministic`/`rng` params already reach this branch for the continuous follow-up; they are simply
  unused here.
- Multi-discrete / multiple discrete keys each sample **independently** — the existing per-segment loop
  already gives this; no special-casing.
- Default arguments mean **every existing caller and test is unaffected** (still `argmax`). The empty-dict
  error sentinel and all length/`action_type` validation are untouched.

### 3. `NcnnControllerCore`
- New state: `var deterministic_inference: bool = true`, `var rng := RandomNumberGenerator.new()`.
- Helper `setup_rng(seed: int) -> void`: `seed < 0` → `rng.randomize()`; else `rng.seed = seed`
  (reproducible stochastic eval).
- `choose_and_apply_action` passes `deterministic_inference` and `rng` to `decode_actions`. No other
  change; the float and image paths both flow through the same decode call.

### 4. `NcnnAIController2D` / `NcnnAIController3D`
- `@export var deterministic_inference: bool = true`
- `@export var inference_seed: int = -1`  (`-1` = randomize each run; `>= 0` = fixed seed)
- In `_ready()`, NCNN_INFERENCE branch (alongside the existing runner/obs-norm setup):
  `_core.deterministic_inference = deterministic_inference` and `_core.setup_rng(inference_seed)`.
- Test seam: tests target the pure math + `decode_actions` with an injected `RandomNumberGenerator`;
  a stub-runner controller test sets `_core.deterministic_inference`/`_core.rng` directly (the core
  fields are plain `var`s, already reachable like `_core.obs_norm_stats` in existing tests).

## Data flow

```
obs ─► ncnn ─► output (logits | mean)
                      │
            ActionDecode.decode_actions(deterministic, rng)
              ├─ discrete:  deterministic ? argmax            : sample_categorical(softmax(logits), rng.randf())
              └─ continuous: mean (optional tanh)             [unchanged; sampling = follow-up]
                      │
                  set_action
```

## Error handling

- All existing `ActionDecode` validation (non-positive size, output too short, length mismatch, unknown
  `action_type` → `push_error` + `{}`) is preserved and runs **before** the deterministic/stochastic
  branch, so a malformed output fails identically in both modes.
- `softmax`/`sample_categorical` on an empty segment return empty/`-1`; in practice `size >= 1` is already
  guaranteed by the size guard, so this is defensive only.
- No new failure surface: the stochastic path cannot produce an out-of-range action (`sample_categorical`
  always returns an index in `[0, size)` or `-1`, matching `argmax`).

## Testing (TDD)

Pure math (`test/unit/test_inference_math.gd`, extend):
- `softmax`: known small vectors (hand-computed), sums to 1.0 within tolerance, **numerically stable** for
  large logits (e.g. `[1000, 1001]` → no inf/nan), uniform logits → uniform probs.
- `sample_categorical`: `u = 0` → first index with non-zero prob; boundary `u`s just below/above each
  cumulative edge → expected index; `u → 1` → last index; one-hot probs → that index regardless of `u`;
  empty → `-1`.

Decode (`test/unit/test_action_decode.gd`, extend):
- `deterministic = true` (default) → identical to current `argmax` results (regression guard).
- `deterministic = false` with a **seeded** `RandomNumberGenerator`: peaked logits return the peak;
  a many-draw histogram for a known logit vector ≈ `softmax` within a loose tolerance; two seeded runs
  with the same seed produce identical sequences (reproducibility).
- Multi-discrete / two discrete keys: each key sampled independently (seeded, deterministic assertion).
- Continuous keys: unchanged (mean / `tanh`) in both modes — guard that the new params don't perturb them.

Controller wiring (`test/unit/test_stochastic_inference.gd`, new):
- Stub runner returning fixed logits; controller with `deterministic_inference = false` and a fixed
  `inference_seed` → two controllers produce the **same** sampled action (end-to-end reproducibility).
- `setup_rng(-1)` vs `setup_rng(42)` behavior (randomize vs fixed) at the core level.

Full `./test/run_tests.sh` green — existing golden / argmax regressions unchanged (default is
deterministic).

## Out of scope (tracked as follow-up)

- **Continuous DiagGaussian sampling via `log_std` sidecar (PPO).** New export step pulling
  `policy.log_std → exp()` into an `action_dist.json` sidecar, a controller `action_dist_stats_path`
  loader, and a Gaussian draw in `decode_actions`' continuous branch. Designed-for here (flag + `rng`
  already plumbed to that branch); filed as its own issue, organized under #16.
- **SAC continuous sampling** — requires changing the SAC export path (currently deterministic
  `tanh(mean)`); separate spec.

## Docs to update in the closing PR

- `README.md` — controller exports table / deploy section (`deterministic_inference`, `inference_seed`).
- `CLAUDE.md` — controller `choose_and_apply_action` note (now `deterministic`-aware) + the new exports.
- `docs/godot-rl-gap-analysis-2026-06-02.md` — `deterministic_inference` row → parity (discrete; note
  continuous follow-up).
- `docs/BACKLOG.md` — item 43 ✅.
- PR `Closes #16`.
