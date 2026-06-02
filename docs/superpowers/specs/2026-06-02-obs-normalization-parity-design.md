# Observation-normalization parity helper (backlog item 24) — design

**Date:** 2026-06-02
**Status:** approved (brainstorm)
**Backlog:** item 24 — *Observation-normalization parity helper*

## Problem

Policies trained with SB3 `VecNormalize` learn against **normalized** observations:

```
normalized = clip((obs - running_mean) / sqrt(running_var + epsilon), -clip_obs, +clip_obs)
```

`VecNormalize` keeps its running statistics in a **separate `vec_normalize.pkl`**, never in the
policy network. The ONNX → ncnn conversion carries *only the network*, so at deploy the running
mean/std are gone. If you feed raw observations to a `VecNormalize`-trained policy, it silently
receives garbage and acts nonsensically **with no error**. `docs/ncnn_vs_onnx.md` calls this out as
the **#1 silent-failure risk** for native deployment.

Today the only safe path is hand-normalizing *inside* `get_obs()` and running that same code at train
and deploy (the chase example does this; its trainer uses `VecMonitor`, **not** `VecNormalize`). That
does not help anyone who trained with `VecNormalize`.

## Key correctness insight (shapes the whole design)

There are two mutually exclusive normalization styles, and item 24 is specifically the second:

| Style | Who normalizes | `get_obs()` returns | Deploy replay |
|---|---|---|---|
| Hand-normalize (existing) | `get_obs()` itself | **normalized** | none — same code runs both sides |
| `VecNormalize` (item 24) | Python wrapper, *outside* `get_obs()` | **raw** | **must replay the stats deploy-side** |

With `VecNormalize`, `get_obs()` returns **raw** observations at *both* train and deploy (the trainer's
`VecNormalize` wrapper does the normalization in Python). Therefore the deploy-side replay must happen
**between `get_obs()` and `run_inference()`** — i.e. in the controller's `choose_and_apply_action`,
which training never calls. Putting normalization inside `get_obs()` would **double-normalize** during
training. This is why the integration point is essentially forced to the controller, not `get_obs()`.

## Architecture

Four pieces, mirroring the item-21 (continuous-action) shape — a pure GDScript transform, a Python
export step, a seeded synthetic fixture generator, and minimal controller wiring.

### 1. Pure replay helper — `addons/godot_native_rl/controllers/obs_normalize.gd`

`ObsNormalize` (RefCounted, static functions only). The deploy-side analogue of `action_decode.gd`:
`action_decode` is the *post*-inference transform, `obs_normalize` is the *pre*-inference transform.

```gdscript
class_name ObsNormalize
extends RefCounted

# normalized[i] = clamp((obs[i] - mean[i]) / sqrt(var[i] + epsilon), -clip_obs, +clip_obs)
static func normalize(obs: PackedFloat32Array, mean: PackedFloat32Array,
		var_: PackedFloat32Array, epsilon: float, clip_obs: float) -> PackedFloat32Array

# True iff stats is well-formed: has mean+var arrays of equal, positive length and numeric
# epsilon/clip_obs. Used at load time so a malformed JSON fails loudly, not at first inference.
static func validate(stats: Dictionary) -> bool

# Parse a stats Dictionary (already JSON-decoded) into typed PackedFloat32Arrays once, so the
# hot path does not re-coerce Array->PackedFloat32Array every frame. Returns {} if invalid.
static func to_typed(stats: Dictionary) -> Dictionary
```

`normalize` requires `obs.size() == mean.size() == var_.size()`; on mismatch it returns an **empty**
`PackedFloat32Array` (the controller treats empty as "skip the action", same contract as a failed
action decode). No silent fallthrough.

### 2. Stats export script — `scripts/export_vecnormalize.py`

Reads an SB3 `vec_normalize.pkl` and writes the committed JSON stats fixture. Runs under `.venv-train`
(has `stable_baselines3`). Loaded via `pickle` directly (no env needed) so it works headless.

- Pull `obs_rms.mean`, `obs_rms.var`, `vn.epsilon`, `vn.clip_obs` off the unpickled `VecNormalize`.
- **Fail fast** (clear message, non-zero exit) when:
  - the object is not a `VecNormalize` (no `obs_rms` / `clip_obs` attributes),
  - `norm_obs` is `False` (the policy was trained on raw obs — replay would be wrong),
  - `obs_rms` is a **dict** (multi-key Dict obs → out of scope for this item; tell the user).
- Pure helpers (`stats_from_vecnormalize(vn) -> dict`, `write_stats_json(stats, path)`) kept import-light
  and unit-testable; heavy SB3 import stays lazy inside `main()`.
- CLI: `export_vecnormalize.py <vec_normalize.pkl> [--out PATH]` (default: alongside the pkl).

### 3. Synthetic fixture generator — `scripts/make_vecnormalize_stats.py`

Builds a **real** `VecNormalize` over a tiny dummy vec-env, updates it with **seeded** random
observations (NumPy `default_rng(seed)`), saves the `.pkl` to a temp path, then:

- runs `export_vecnormalize.stats_from_vecnormalize` → writes `models/synthetic_vecnormalize.json`,
- computes `vn.normalize_obs(raw)` for a handful of seeded raw obs vectors and writes
  `models/synthetic_vecnormalize_golden.json` = `{"stats_path": ..., "cases": [{"raw": [...],
  "normalized": [...]}, ...]}`.

This makes the GDScript golden assert the pure helper reproduces **SB3's own** `normalize_obs` output,
not a re-implementation of it. The `.pkl` itself is **not** committed (the generator recreates it
deterministically); only the derived JSON fixtures are committed, alongside the existing
`models/synthetic_continuous.*` / `models/synthetic_cnn.*` fixtures.

### 4. Controller wiring

`NcnnControllerCore`:
- new field `var obs_norm_stats: Dictionary = {}` (empty ⇒ no-op; core stays node-agnostic, holds a
  plain typed-stats dict produced by `ObsNormalize.to_typed`).
- in `choose_and_apply_action`, **float path only** (the image path is skipped — `VecNormalize` on
  pixels is not done): after building the obs vector and before `run_inference`, if
  `obs_norm_stats` is non-empty, replace the vector with `ObsNormalize.normalize(...)`. If that returns
  empty (size mismatch), `push_error` and skip the action — no garbage forward pass.

`NcnnAIController2D` / `NcnnAIController3D` (thin wrappers):
- new `@export_file("*.json") var obs_norm_stats_path: String = ""`.
- in `_setup_ncnn_runner()` (NCNN_INFERENCE only), if the path is set: read + `JSON.parse`, then
  `ObsNormalize.validate` → `to_typed` → assign to `_core.obs_norm_stats`. A missing/invalid file is a
  loud `push_error` (and leaves stats empty), never a silent skip.
- a `set_obs_norm_stats_for_test(stats)` seam mirrors `set_ncnn_runner_for_test`.

## Data flow

```
TRAIN (unchanged):
  get_obs() -> raw obs -> NcnnSync -> trainer -> VecNormalize.normalize_obs (Python)

DEPLOY (NCNN_INFERENCE):
  get_obs() -> raw obs
            -> choose_and_apply_action:
                 if obs_norm_stats: ObsNormalize.normalize(raw, ...)   # replay
                 run_inference(normalized) -> ActionDecode -> set_action
```

## JSON schema

```json
{
  "obs_size": 8,
  "mean":  [ ... obs_size floats ... ],
  "var":   [ ... obs_size floats ... ],
  "epsilon": 1e-08,
  "clip_obs": 10.0
}
```

`obs_size` is informational/validation (must equal `mean.size()` and `var.size()`). `epsilon` and
`clip_obs` come from the `.pkl` (SB3 defaults `1e-8` / `10.0`) — never hardcoded GDScript-side.

## Error handling (loud, never silent)

This feature exists to *eliminate* a silent failure, so every boundary is loud:

- **Export (Python):** non-`VecNormalize`, `norm_obs=False`, or Dict `obs_rms` → clear stderr message +
  non-zero exit. No partial JSON written.
- **Load (GDScript):** file missing, unparseable JSON, or `validate()` false → `push_error`, stats stay
  empty (controller behaves exactly as if no normalization was configured — i.e. the *pre-existing*
  raw-obs behavior, which a `VecNormalize` user will notice as a broken policy, with an error in the log
  pointing at the cause).
- **Runtime (GDScript):** `obs.size() != mean.size()` → `normalize` returns empty →
  `choose_and_apply_action` `push_error`s and skips the action (no forward pass on mismatched data).

## Testing

- **`test/unit/test_obs_normalize.gd`** — pure math:
  - basic normalize (hand-computed expected), clipping at `±clip_obs`, `epsilon` effect on zero-variance,
    length-mismatch → empty, `validate` accept/reject cases;
  - **golden parity:** load `models/synthetic_vecnormalize.json` + `..._golden.json`, run
    `ObsNormalize.normalize` on each `raw` case, assert it matches SB3's `normalized` at `atol 1e-6`.
- **`test/unit/test_controller_inference.gd`** (extend) — a stub runner records the vector it was given:
  - with `obs_norm_stats` set, assert the runner received the **normalized** vector;
  - with empty stats, assert it received the **raw** vector (no behavior change / backward compatible);
  - size-mismatch stats ⇒ action skipped (no `set_action` call).
- **`test/python/test_export_vecnormalize.py`** — construct a `VecNormalize`, exercise
  `stats_from_vecnormalize`: assert mean/var equal `obs_rms.mean/var`, `epsilon`/`clip_obs` carried;
  assert the fail-fast guards (norm_obs off, dict obs_rms) raise; round-trip `write_stats_json` →
  re-read matches.
- **`run_tests.sh`** — register the new GDScript unit test and the Python test (auto-discovered under
  `test/python/`). Fixtures are regenerated by `make_vecnormalize_stats.py` and committed.

## Docs to update (same change, not later)

- `docs/ncnn_vs_onnx.md` — the silent-failure callout: note `VecNormalize` policies are now supported
  via `export_vecnormalize.py` + `obs_norm_stats_path` replay (was "you have to … replay them yourself").
- `README.md` — short note under the deploy/inference section + the new script.
- `CLAUDE.md` — key-commands entry for `export_vecnormalize.py`; gotcha if any surfaces.
- `docs/BACKLOG.md` — mark item 24 ✅ with spec/plan links and a short summary.

## Scope / YAGNI

- **In:** single numeric `"obs"` vector (the common SB3 `Box` case). Image (String/hex) obs are skipped.
- **Out (documented follow-ups):** multi-key Dict obs (per-key stats); reward normalization (training-only,
  irrelevant to deploy); a real `VecNormalize`-trained example + behavioral regression (heavy; the
  synthetic seeded golden covers the math end-to-end, matching item 21's approach).
- **No C++ change / no rebuild** — pure GDScript + Python, like item 36 (deploy-side image glue).
