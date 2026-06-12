# Intrinsic reward — RND (phase 1 of #27)

## Problem

Most real games are **sparse-reward**: the agent rarely stumbles onto reward, so pure PPO explores
poorly. A curiosity bonus added to the environment reward fixes this. #27 asks for a pluggable
intrinsic-reward signal "addable to any training script … composes with the existing reward path,"
shipping **RND first, then ICM**. This is phase 1 (RND); ICM is the phase-2 sub-issue (#201).

## Why RND first

RND (Random Network Distillation) is the simplest robust intrinsic signal: a *fixed* randomly-init
target network and a *trained* predictor both map an observation to a feature vector; the predictor's
error is high on rarely-seen states (it has only been trained on visited states) and fades as the
agent revisits them. It is **state-only** (no action/next-state needed), dependency-light, and well
understood. ICM (forward/inverse dynamics) needs the action and two more learned models — phased out.

## Design

Intrinsic reward is **training-only** — it shapes the reward the trainer optimizes; the exported ncnn
policy is unchanged. Split `scripts/intrinsic.py` by dependency so the mixing/normalization logic is
unit-testable with no ML stack:

### Pure stdlib helpers (no numpy/torch)

- `RunningMeanStd` — Welford running mean/variance over scalar batches (Chan parallel update). First
  batch initializes mean/M2 directly (no pseudo-count at mean 0), so a constant stream reports
  variance 0 and the consumer's div-by-std guard engages instead of dividing by a spurious tiny std.
- `combine_rewards(extrinsic, intrinsic, coef)` — elementwise `extrinsic + coef*intrinsic`.
- `normalize_intrinsic(intrinsic, rms)` — divide by the running std (RND convention: divide-only, no
  mean subtraction, so the bonus stays non-negative); near-zero std falls back to a unit denominator.

### RND network (lazy torch)

- `RNDModel`: a frozen-random `target` MLP + a trainable `predictor` MLP (obs_dim → feature_dim).
  `intrinsic_reward(obs)` = per-sample mean squared predictor-vs-target error (detached);
  `update(obs, opt)` trains the predictor toward the target. Built via `_rnd_model_base()` so the
  `nn.Module` base resolves only when torch is present; `make_rnd(obs_dim, device=...)` returns
  `(model, optimizer)` with the model moved to device **before** the optimizer is built.

### Wiring (train_cleanrl.py)

`--intrinsic {none,rnd}` + `--intrinsic_coef` on `PPOConfig`. When `rnd`: build the model/optimizer/
running-std once; in the rollout loop compute the arrived-in state's novelty, normalize, and mix into
that step's reward (so `compute_gae` sees the combined reward); after each rollout, train the
predictor on the visited states. `scripts/train_cleanrl.sh` gains `INTRINSIC`/`INTRINSIC_COEF` +
`SAVE_MODEL_PATH`/`ONNX_EXPORT_PATH` env passthrough.

The signal is deliberately wired into the single-file CleanRL trainer (where the rollout loop is
explicit and we control the reward at each step); the same `intrinsic.py` module is reusable from any
other trainer that exposes its per-step reward.

## Tests

- `test/python/test_intrinsic.py`: pure helpers (RunningMeanStd batched==incremental / empty no-op /
  constant→0 var; combine_rewards mix / zero-coef / length mismatch; normalize_intrinsic divide /
  near-zero fallback) **run everywhere**; torch-guarded RND tests (intrinsic_reward shape ≥ 0; target
  frozen + predictor trainable; update reduces novelty + loss on repeated obs) run in CI's `.venv-train`.
- `test/python/test_train_cleanrl.py`: `--intrinsic` default `none`/coef 0.5, `rnd` opt-in, unknown
  signal rejected.
- `test/run_tests.sh`: a guarded **CleanRL+RND CI smoke** (skipped if godot_rl absent) trains a tiny
  chase run with `INTRINSIC=rnd` and asserts the `.pt` is produced — end-to-end coverage of the
  rollout integration.

## Out of scope (→ #201)

ICM (forward/inverse dynamics, action-dependent intrinsic reward); wiring intrinsic reward into the
SB3 trainers (chase/rover — callback-based injection is a separate shape).
