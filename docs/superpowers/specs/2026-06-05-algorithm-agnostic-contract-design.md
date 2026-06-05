# Algorithm-agnostic training/deploy contract — guard with synthetic non-PPO regressions

**Issue:** #45 (`backlog`, `area:parity`/`area:training`, `priority:2`)
**Date:** 2026-06-05
**Status:** design approved

## Problem

PPO is the only RL algorithm we have *proven* end-to-end, but the runtime is — by design — not
coupled to it. The deploy path is a pure forward pass: `obs → ncnn → output → ActionDecode`, keyed
only off the output *shape* and the `action_space` `action_type`, never the training algorithm. We
want this to be an **explicit, guarded** contract so it can't silently rot as the project adopts
SAC / DQN / TD3 / A2C / off-policy methods.

## Already done (audit, 2026-06-05)

Most of issue #45 is satisfied — it landed alongside the recurrent/contract docs work:

| #45 checklist item | Status |
|---|---|
| Document the algorithm-agnostic deploy contract in `DEVELOPMENT.md` | ✅ `## The deploy contract (algorithm-agnostic)` exists |
| `action_decode` / `obs_normalize` stay pure + note in code | ✅ both pure `RefCounted` helpers; `action_decode.gd` documents it (see gap below for `obs_normalize`) |
| Note PPO/godot_rl vestiges (`state_ins`) + confirm inert | ✅ documented: pnnx prunes `state_ins` at conversion → inert at deploy |
| Decode-side non-PPO guard test | ✅ `test/unit/test_algorithm_agnostic_decode.gd` (DQN/SAC/TD3/hybrid through one path) |
| **≥1 non-PPO *trained* regression** (train → export → ncnn → behavioral check) | ❌ the only real gap |

### Coverage already present by composition

- `synthetic_continuous` (TinyMLP → real ncnn → continuous decode) proves the **continuous**
  pipeline round-trip.
- `test_chase_golden_inference` (PPO) proves the **discrete** pipeline round-trip + argmax.
- `test_algorithm_agnostic_decode` proves DQN Q-value / SAC-tanh decode **numerics** (no pipeline).

## Decision

Satisfy the remaining box with **synthetic non-PPO fixtures now**, and **file the live-trained SB3
SAC run as a separate `needs-training-run` follow-up issue**. This closes #45's contract robustly in
deterministic CI without a multi-hour training run, matching how the docs/test already frame the
trained run as "a separate slice."

The synthetic regressions target the two combinations *not yet guarded end-to-end*:

- **DQN:** deliberately **unbounded Q-values** surviving fp32 conversion, argmax preserved through
  the real ncnn pipeline (today's discrete round-trip is PPO logits, small range).
- **SAC:** the **`squash: true` (tanh)** continuous decode over a *real* ncnn round-trip (today
  squash is only unit-tested, never through actual pnnx→ncnn).

## Components

### 1. `scripts/make_synthetic_dqn.py`
Mirrors `make_synthetic_continuous.py`. A tiny **seeded** MLP Q-net (`OBS_DIM → hidden → N` actions)
with weights/biases scaled so outputs are clearly **unbounded Q-values** (magnitude ~tens), distinct
from small PPO logits. Runs under `.venv-train`; exports through the **real** `export_to_ncnn.py`
(pnnx → ncnn). Writes:
- `models/synthetic_dqn.ncnn.{param,bin}`
- `models/synthetic_dqn_golden.json` — `{ "obs": [...], "output": [...torch Q-values...],
  "argmax": <int> }`

### 2. `scripts/make_synthetic_sac.py`
Same pattern. A tiny seeded MLP continuous actor (`OBS_DIM → hidden → ACT_DIM` **raw means**,
pre-tanh — SAC squashes at deploy, not in the network). Real-pipeline export. Writes:
- `models/synthetic_sac.ncnn.{param,bin}`
- `models/synthetic_sac_golden.json` — `{ "obs": [...], "output": [...raw means...],
  "squashed": [...tanh(mean)...] }`

### 3. `test/unit/test_algorithm_agnostic_golden_inference.gd`
One test file, both fixtures, mirrors `test_recurrent_golden_inference.gd`. Loads each fixture via
`NcnnRunner`, runs the fixed obs:
- **DQN:** `ActionDecode.decode_actions(out, {"move": {"size": N, "action_type": "discrete"}})` ==
  golden argmax (asserted **exactly**) → unbounded-Q argmax survives fp32 end-to-end. Raw-value
  parity held to a **relative** tolerance (`|out[i] - golden[i]| <= rtol * |golden[i]| + atol_floor`,
  `rtol = 1e-2`, small `atol_floor = 1e-3` for near-zero entries) rather than a blanket loosened
  absolute atol — so large-magnitude Q-values stay tightly (proportionally) checked.
- **SAC:** ncnn output ≈ golden raw means (atol = 1e-2) **and**
  `ActionDecode.decode_actions(out, {"steer": {"size": D, "action_type": "continuous",
  "squash": true}})` ≈ golden `tanh(mean)` (atol = 1e-2) → squash path over a real ncnn round-trip.

### 4. Wire-up + housekeeping
- **Commit** the generated `models/synthetic_dqn.*`, `models/synthetic_sac.*` fixtures + golden
  JSONs (same convention as existing `synthetic_*`; CI runs the golden test, not the make scripts).
- Add `test_algorithm_agnostic_golden_inference.gd` to `test/run_tests.sh`.
- Add a one-line explicit "algorithm-agnostic" note to `obs_normalize.gd` to fully close the
  "audit + note in code" box (`action_decode.gd` already has its note).
- Docs: update the **"Guarded by"** line in `DEVELOPMENT.md` to cite the new end-to-end test; flip
  item 45 in `BACKLOG.md`; add item 45 to `CLAUDE.md` Done list; remove the #45 subsection from
  `docs/TESTING_OPEN_ISSUES.md` §4 (per its own maintenance rule); check
  `docs/godot-rl-gap-analysis-2026-06-02.md` for an algorithm-agnostic claim to update.
- **File the follow-up issue:** "Trained SB3 SAC non-PPO regression (live train → export → ncnn →
  behavioral check)", labelled `needs-training-run`, referenced from #45's closing note and the
  `DEVELOPMENT.md` contract section.

## Out of scope (YAGNI)

- No live training, no new train scripts/scenes — that is the follow-up `needs-training-run` issue.
- No DiagGaussian / `log_std` sidecar (that is #64) — the SAC fixture deploys the **deterministic**
  `tanh(mean)`.
- No new C++ — pure GDScript test + Python fixture makers reusing the existing export pipeline.

## Testing strategy (TDD)

1. Write `test_algorithm_agnostic_golden_inference.gd` first, asserting against the (not-yet-existing)
   committed fixtures + golden JSONs → **RED** (missing fixtures).
2. Implement `make_synthetic_dqn.py` / `make_synthetic_sac.py`; generate + commit fixtures.
3. Run the single test → **GREEN**.
4. `./test/run_tests.sh` green (no regressions) before merge.

## Risks / notes

- **fp32 conversion drift on unbounded Q-values:** large-magnitude outputs make a flat absolute
  atol meaningless. Assert **argmax equality exactly** (the behaviorally-meaningful invariant) and
  hold raw-value closeness to a **relative** tolerance (`rtol = 1e-2` + a small `atol_floor` for
  near-zero entries) — more precise than loosening atol, since each entry is checked proportionally
  to its own magnitude. SAC means are tanh-bounded and small, so they keep the standard atol = 1e-2.
- **Fixture determinism:** seed torch + use a fixed obs vector (as `make_synthetic_*` already do) so
  the golden JSON is reproducible across machines.
- **Branch/doc ordering:** the `TESTING_OPEN_ISSUES.md` §4 #45 removal depends on PR #72 (which
  first adds the file) landing; rebase onto updated `main` before that edit.
