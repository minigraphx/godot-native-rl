# Continuous + Multi-Key Action Deployment (Backlog Item 21)

**Date:** 2026-06-01
**Status:** Approved — ready for planning
**Branch:** `feat/backlog-21-continuous-multikey-actions`

## Problem

The deploy-side inference path only supports a single discrete action key. Both
`NcnnControllerCore.choose_and_apply_action` and the C++ `NcnnRunner.run_discrete_action`
argmax the entire model output and apply it to `get_action_space().keys()[0]`. This means a
policy trained with godot_rl can be *trained* with continuous (PPO-continuous / SAC),
multi-discrete, or multiple simultaneous action keys, but cannot be *deployed* natively via
ncnn. This blocks the SAC path for the hide & seek example (item 12) and is a documented
deploy-side gap (`docs/ncnn_vs_onnx.md`).

## Key insight

`NcnnRunner.run_inference()` already returns the raw output vector as a `PackedFloat32Array`.
All of continuous, multi-discrete, and multi-key decoding is therefore a pure-GDScript
transformation of that vector against the action space — **no C++ change and no rebuild**.
This mirrors the item-36 deploy-side image work (which also added only GDScript glue) and
preserves the "no rebuild" property.

The output→action mapping mirrors a standard policy head: walk the `action_space` keys in
insertion order and consume one contiguous segment of the output per key.

## Scope

In scope:
- Continuous action deployment (mean output, optional per-key tanh squash).
- Multi-discrete and multiple simultaneous action keys.
- Pure-GDScript decoder + controller refactor to use it.
- Headless unit tests + a committed synthetic-model end-to-end golden (numerical closeness).

Explicitly out of scope (separate backlog items):
- Observation normalization parity (item 24).
- Recurrent / LSTM hidden state (item 22).
- Batched multi-agent inference (item 23).

## Architecture

### 1. New pure helper: `addons/godot_native_rl/controllers/action_decode.gd`

```
decode_actions(output: PackedFloat32Array, action_space: Dictionary) -> Dictionary
```

Walks `action_space` keys in insertion order, consuming one segment of `output` per key:

- **discrete**: segment length = `size` (the number of discrete choices); argmax over that
  segment → an `int` index in `[0, size)`. Reuses `InferenceMath.argmax` on the sliced
  sub-array. First index wins on ties (matches the existing C++ and `InferenceMath` behavior).
- **continuous**: segment length = `size`; each value optionally passed through `tanh()` iff
  that key's action-space entry has `"squash": true`. Default (absent) = no squash. Result is
  an `Array` of floats.

Returns `{key: int | Array}`. The total consumed length must equal `output.size()`; a mismatch
is a train/deploy shape error → `push_error` and return `{}` (the error sentinel the controller
checks). An unknown `action_type` likewise → `push_error` + `{}`.

Single-key discrete reduces to exactly today's behavior (argmax over the whole output), so the
chase/rover/hide-and-seek examples are unchanged.

Per-key `squash` (rather than a controller-level export) was chosen so mixed action spaces can
have some continuous keys squashed and others not, and so the agent declares it once in the same
place it declares the action shape. godot_rl's continuous convention is actions in `[-1, 1]`, so
tanh alone (no additional scaling) is sufficient at deploy time; this is documented.

### 2. Refactor `NcnnControllerCore.choose_and_apply_action`

- **Float path**: `run_inference(obs)` → `ActionDecode.decode_actions(output, agent.get_action_space())`
  → `agent.set_action(decoded)`. Replaces the `run_discrete_action` + `keys()[0]` argmax.
- **Image path**: `run_inference_image()` → same decode, so visual continuous policies also work.
- An empty decode (`{}`) → `push_error` + skip the action, mirroring today's `-1` sentinel handling.
- The C++ `run_discrete_action` is left intact and bound (backward compat / external callers) but
  is no longer called by the controller. **No C++ change, no rebuild.**

### 3. Test story (synthetic models, headless, committed)

- **Unit tests** for `action_decode.gd`: discrete single-key, multi-discrete, continuous
  with/without squash, mixed (discrete + continuous) space, shape-mismatch sentinel, unknown
  action_type sentinel, and tie-break parity with the current argmax.
- **End-to-end golden** (mirrors item 36): `scripts/make_synthetic_continuous.py` generates a
  seeded synthetic continuous-action model; committed `models/synthetic_continuous.ncnn.*` +
  `synthetic_continuous_golden.json`. A test asserts `run_inference` → `decode_actions` matches an
  onnxruntime reference within **`atol=1e-2`** (numerical closeness, not argmax — per the backlog
  note for continuous verification).
- **Controller test**: drive `choose_and_apply_action` against the synthetic model with a fake
  multi-key + continuous agent; assert `set_action` receives the correctly-shaped dict.
- All wired into `test/run_tests.sh`; must pass from a clean cache.

### 4. Docs

- `CLAUDE.md`: note the controller now decodes all godot_rl action types (discrete, continuous,
  multi-discrete, multi-key) deploy-side via the pure `action_decode.gd`.
- `docs/BACKLOG.md`: mark item 21 done; note the SAC path for item 12 is unblocked.
- `docs/ncnn_vs_onnx.md`: update the deploy-side gaps section (continuous/multi-key no longer a gap).

## Data flow

```
get_obs() ──▶ run_inference(obs) ──▶ raw PackedFloat32Array
                                          │
                       action_space ──▶ decode_actions ──▶ {key: int | [floats]}
                                          │                        │
                              (shape mismatch → {} → skip)   set_action(decoded)
```

## Error handling

- Shape mismatch (consumed ≠ output length) → `push_error`, return `{}`, controller skips action.
- Unknown `action_type` → `push_error`, return `{}`.
- Runner missing / model not loaded → unchanged early-return in `choose_and_apply_action`.
- Empty output → `argmax` returns `-1` per segment; an all-empty output yields a mismatch → `{}`.

## Testing

Per the project's TDD convention and `testing.md`: write the decoder unit tests first (RED),
implement, then add the synthetic-model golden and controller integration test. All headless via
`test/harness.gd`, wired into `run_tests.sh`, green from a clean script-class cache.
