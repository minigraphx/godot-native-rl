# Rover Reward Tuning — Design

**Date:** 2026-05-31
**Relates to:** backlog item 6 (3D rover) — training quality
**Status:** Approved (brainstorm → direct implementation, empirically validated)

## Problem

The first full training attempt (stopped at 261k steps) did not learn: `ep_rew_mean ≈ −4.97` and
flat, with `approx_kl ≈ 0`, `clip_fraction = 0`, `policy_gradient_loss ≈ 1e-8`. Diagnosis: reward
terms `evaluate()` once per **physics frame** (≈1000/episode at `reset_after=1000`), so the
`step_penalty = 0.005` alone sums to **−5.0/episode** — exactly the observed mean. That constant
swamps the per-frame progress signal (`(prev−cur)/max_distance ≈ ±0.002`) and the rarely-fired
`goal_bonus = +1.0`, leaving PPO no usable gradient. Not a delivery bug (progress sign correct,
bumps net ~0) — purely a weight-balance problem.

## Change

Rebalance the three `RoverAgent` exported reward weights (only change):

| Weight | Old | New | Rationale |
|---|---|---|---|
| `step_penalty` (per frame) | 0.005 | **0.001** | Episode total −1.0 (was −5.0); per-frame progress toward goal (~+0.002) now beats it, giving a directional gradient. |
| `goal_bonus` (per goal reached) | 1.0 | **3.0** | Reaching a goal is a strong, learnable event; goals relocate so several/episode accumulate. |
| `collision_penalty` (per bumped frame) | 0.25 | **0.05** | `bumped` fires every blocked frame; 0.25/frame over-punishes wall contact and can drown the goal signal. |

Progress-shaping scale stays `max_distance`; `reset_after` stays 1000; no new term types, no
structural/controller changes. Unit tests (action/obs helpers) are unaffected.

## Validation methodology (rebalance + short runs)

1. Clear `models/rover_checkpoints/` — old checkpoints learned the old reward, so retraining starts
   `FRESH` (the value function is objective-specific).
2. Short run: `FRESH=1 TIMESTEPS=50000 ./scripts/train_rover.sh` (~20 min); watch `ep_rew_mean`.
3. **Success bar:** `ep_rew_mean` trends clearly upward and turns **positive** (goal bonuses
   accumulating ⇒ the rover is actually reaching goals). If met → commit to the full run.
   If still flat → iterate: strengthen progress shaping (add a weight) and/or raise `goal_bonus`,
   re-validate on another short run.

## Out of scope

No new reward terms, no episode-length change, no curriculum/curiosity. Those are later levers if
short-run iteration stalls.
