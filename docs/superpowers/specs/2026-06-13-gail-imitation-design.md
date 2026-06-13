# GAIL — Adversarial Imitation Learning (#61)

**Date:** 2026-06-13 · **Issue:** #61 (AMP/GAIL) · **Status:** design (autonomous batch — decided +
documented). Scope: **GAIL**, not AMP. AMP needs reference-motion (mocap/animation) data this repo
doesn't have; GAIL learns from the **expert demonstrations we already record** (#10), so it's the
tractable, in-infrastructure half of the issue.

## Goal

Train a policy to reproduce an expert's behaviour **with no environment reward** — purely by
imitating committed expert-demo trajectories. Concretely: a chase policy that catches the target,
learned only from `examples/chase_the_target/demos/chase_expert_demos.json` via a discriminator.

## How (mirrors the `scripts/intrinsic.py` / ICM pattern)

A new `scripts/gail.py` with a torch discriminator behind the same lazy-import factory pattern:

- **Discriminator** D(obs, one_hot(action)) → a real logit; `sigmoid` = P(this (s,a) came from the
  EXPERT). MLP, obs_dim + n_actions → 1.
- **GAIL reward** per step = `-log(1 - sigmoid(D(s,a)))` (softplus-stable form) — high when the
  policy's (s,a) looks expert-like. This *replaces* the env reward (pure imitation; an
  `--imitation-coef` could later mix, but #61 ships pure imitation to prove it).
- **Adversarial update**, once per PPO update: binary cross-entropy training D to output 1 on a
  sampled batch of expert (s,a) pairs and 0 on the policy's rollout (s,a) pairs. The policy (PPO)
  maximises the GAIL reward → the two co-adapt.

Pure, torch-free helpers (unit-tested without torch): the reward transform and the expert-batch
sampler index math.

## Wiring (`train_cleanrl.py`)

A new `--imitation gail --demos <path>` (orthogonal to `--intrinsic`). When set: load the expert
(s,a) pairs via `load_expert_demos`, build the discriminator, and in the rollout replace the env
reward with the GAIL reward; after each rollout, train D on policy-vs-expert. The exported policy is
unchanged (deploys like any chase net). Discrete single-head only (chase) for M1 — documented.

## Validation

- Pure-helper unit tests (reward transform monotonic + bounded; expert sampler).
- Torch-guarded discriminator tests: reward shape/sign; D update separates expert from policy on a
  toy set (accuracy rises).
- Guarded **CleanRL+GAIL CI smoke** (tiny chase run loading the committed demos → policy .pt).
- **Trained behavioral regression**: a GAIL-trained chase net (env reward OFF) catches ≥ K targets
  under ncnn — proving the policy learned to chase *by imitation alone*. Committed net + golden.

## Files

- New: `scripts/gail.py`, `test/python/test_gail.py`, behavioral/golden test scenes + committed net.
- Modified: `scripts/train_cleanrl.py` (`--imitation gail`), `scripts/train_cleanrl.sh` (`IMITATION`
  passthrough), `run_tests.sh` (guarded smoke + trained check), docs.

## Honest scope

Pure imitation on a discrete single-head task (chase). Continuous/multi-head GAIL and the AMP
reference-motion variant are follow-ups; AMP specifically is blocked on having motion-clip data.
