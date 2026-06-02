# Observation-normalization parity helper — Design

**Date:** 2026-06-02
**Backlog item:** 24 (`docs/BACKLOG.md`)
**Roadmap reference:** `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md` — Track A (Sensors / DX)
**Motivation doc:** `docs/ncnn_vs_onnx.md` — "Observation preprocessing parity is on you — this is the #1 silent failure."
**Status:** Approved (brainstorm → spec)

## Problem & motivation

Neither ncnn nor ONNX Runtime normalizes observations — that transform is the caller's
responsibility, and a train/deploy mismatch fails **silently** (the policy receives garbage and
acts nonsensically with no error). The repo's safe pattern is "normalize inside `get_obs()` and
run that same code at train and deploy." But policies trained with SB3 `VecNormalize` learn
against a **running mean/std** that lives in the `VecNormalize` wrapper, not in the network and
not in `get_obs()`. At deploy these statistics are frozen and must be replayed game-side, or the
policy is fed un-normalized inputs. We ship nothing for this today; it is the top silent-failure
risk called out in `ncnn_vs_onnx.md`.

This item delivers a small, optional helper that replays SB3 `VecNormalize`'s **frozen** observation
normalization game-side, plus a Python exporter that dumps the training statistics to JSON.

**Our deployment advantage:** like the sensors, this feeds `NcnnRunner`, which runs on
mobile/web/console/desktop/edge with zero runtime. The normalization is pure float math, so it
ports everywhere ncnn does.

## The formula (pinned — train and deploy MUST match exactly)

SB3 `VecNormalize.normalize_obs` (frozen, `training=False`) computes, per observation element `i`:

```
norm[i] = clip( (obs[i] - mean[i]) / sqrt(var[i] + epsilon), -clip_obs, +clip_obs )
```

with `mean[i]`, `var[i]` the frozen `obs_rms` running statistics from training and SB3 defaults
`epsilon = 1e-8`, `clip_obs = 10.0`. This is the single source of truth; both the GDScript pure
math (`obs_normalize.gd`) and the Python exporter's parity test pin to this exact arithmetic so
the two sides cannot drift.

> Note: `VecNormalize` also (optionally) normalizes *rewards*, but reward normalization only
> affects **training**, never inference — it is intentionally out of scope here.

## Decisions (from brainstorming)

1. **Scope = observation normalization only**, frozen (deploy-time) — no online stat updating, no
   reward normalization (training-only).
2. **Location:** a new `addons/godot_native_rl/obs/` folder (the addon already groups by domain:
   `sensors/`, `reward/`, `net/`, `training/`).
3. **Pure math core + thin loader node**, mirroring the sensors (`relative_position_math.gd` +
   `relative_position_sensor_2d.gd`). All bug-prone arithmetic + edge-case guards live in the pure
   core, fully headless-unit-testable.
4. **Stats transport = JSON** `{"mean":[...], "var":[...], "epsilon":..., "clip_obs":...}` — human
   readable, `FileAccess`-loadable headless, matches the project's other artifacts. A Python helper
   (`scripts/export_vecnormalize_stats.py`) produces it from a saved `VecNormalize` pickle.
5. **Composition is MANUAL** — like the sensors, no controller change. The user calls
   `normalizer.normalize(obs)` in `get_obs()` *before* inference. Documented, not wired.
6. **Immutable** — `normalize()` returns a new `Array`; never mutates the input or the stats.

## Architecture

Two new GDScript files under `addons/godot_native_rl/obs/` + one Python exporter.

### `obs/obs_normalize.gd` — pure static math

`extends RefCounted`; all functions `static`. No node/file/tree state. Holds the formula + guards.

- `normalize(obs: Array, mean: Array, var_arr: Array, epsilon: float, clip: float) -> Array`
  - Returns a new `Array` of the same length as `obs`.
  - Per element: `clampf((obs[i] - mean[i]) / sqrt(var[i] + epsilon), -clip, clip)`.
  - **Guards (fail fast, never silently swallow — project convention):**
    - `mean`/`var_arr` length must equal `obs` length → else `push_error` and return a stable
      copy of `obs` unchanged (so the caller still gets a correctly-shaped obs; a misconfigured
      normalizer must not crash the inference loop, but it is loudly flagged).
    - `var[i] + epsilon <= 0.0` (degenerate/negative variance) → `push_error` and skip the divide
      for that element (pass the centered value through, still clipped) — avoids NaN/inf.
    - `clip <= 0.0` → treated as "no clipping" (`INF`) so a 0 default doesn't zero every obs.

  Rationale for the unchanged-on-mismatch fallback: this is the *anti-silent-failure* helper, so a
  length mismatch is exactly the bug we want surfaced — `push_error` makes it visible in logs while
  keeping the obs shape stable so the surrounding loop is debuggable.

### `obs/obs_normalizer.gd` — thin loader node

`extends Node` (not 2D/3D — it is frame-agnostic). Loads frozen stats from a JSON file and applies
the pure math. References the pure core via `preload` (path-based, per the headless `class_name`
gotcha).

Exports:
- `stats_path: String` — `res://`/`user://` path to the JSON stats file.

State (frozen after load):
- `_mean: Array`, `_var: Array`, `_epsilon: float = 1e-8`, `_clip: float = 10.0`, `_loaded: bool`.

API:
- `load_stats() -> bool`: read + parse the JSON at `stats_path`, validate (`mean`/`var` present,
  same length, numeric `epsilon`/`clip_obs` with SB3 defaults if absent), populate state. Returns
  `false` + `push_error` on any failure (missing file, bad JSON, length mismatch); never throws.
- `set_stats_for_test(mean: Array, var_arr: Array, epsilon: float, clip: float) -> void`: test seam
  to inject stats without a file (mirrors the sensors' `set_target_for_test`).
- `normalize(obs: Array) -> Array`: if not loaded → one-time `push_error` and return `obs`
  unchanged (stable shape); else delegate to `ObsNormalize.normalize(obs, _mean, _var, _epsilon, _clip)`.
- `is_loaded() -> bool`, `obs_size() -> int` (the stats length, for obs-space declaration).

### `scripts/export_vecnormalize_stats.py` — Python exporter

Runs under `.venv-train` (has `stable_baselines3`/`numpy`). Reads a saved `VecNormalize` pickle and
dumps `{"mean","var","epsilon","clip_obs"}` to JSON for the GDScript loader.

- Pure, dependency-injectable core (heavy imports lazy inside `main()`, per repo convention):
  - `stats_dict(obs_rms, epsilon, clip_obs) -> dict`: reads `obs_rms.mean` / `obs_rms.var`
    (numpy arrays) → Python `list[float]`, assembles the dict. No SB3/numpy import at module load
    (works on a duck-typed fake rms in tests).
  - `dump_stats(stats: dict, out_path) -> None`: `json.dump` with stable key order.
- `main()` (lazy `VecNormalize` import): `VecNormalize.load(path, venv=None)` *or*, when no venv is
  available, unpickle and read `obs_rms` directly; pull `epsilon`/`clip_obs` off the loaded wrapper
  (fall back to SB3 defaults `1e-8` / `10.0`); write JSON. CLI: `--vecnormalize PATH --out PATH`.

## Data flow

```
TRAIN (Python, .venv-train):
    VecNormalize wrapper accumulates obs_rms.{mean,var} during training, saved as a pickle.
    scripts/export_vecnormalize_stats.py  ->  vecnormalize_stats.json  {mean,var,epsilon,clip_obs}

DEPLOY (Godot):
    var normalizer := ObsNormalizer.new(); normalizer.stats_path = "res://.../vecnormalize_stats.json"
    normalizer.load_stats()
    # in get_obs():
    var raw := build_raw_obs()                 # Array[float]
    var obs := normalizer.normalize(raw)       # frozen VecNormalize transform, identical to train
    # -> NcnnRunner.run_inference(obs)
```

Manual composition only (same as the sensors). The controller is untouched, so the existing
trained-chase/trained-rover inference and golden regressions are unaffected.

## Error handling

Validate at boundaries; never silently swallow (project convention):
- Stats file missing / unreadable / bad JSON → `load_stats()` returns `false` + `push_error`.
- `mean`/`var` length mismatch (vs each other, or vs `obs`) → `push_error`, obs returned unchanged.
- `var[i] + epsilon <= 0` → `push_error`, element passes through centered (no divide), still clipped.
- `clip <= 0` → no clipping (avoids a default-0 zeroing every obs).
- `normalize()` before `load_stats()`/`set_stats_for_test()` → one-time `push_error`, obs unchanged.

## Testing strategy

TDD, headless `extends SceneTree` harness (`test/harness.gd`), all references via `preload`
(no bare `class_name`). New `test/unit/test_*.gd` are auto-discovered by `test/run_tests.sh`.

**`test_obs_normalize.gd` (pure math, no node/file):**
- Known `mean`/`var`/`clip` → hand-computed expected (mirroring the pinned formula exactly).
- Clipping: a large centered value clips to `±clip`.
- `epsilon` guard: tiny/zero var + epsilon is finite (no NaN/inf), result matches the formula.
- Length-mismatch guard: `mean`/`var` shorter than `obs` → obs returned unchanged (and not crash).
- `clip <= 0` → no clipping.
- Immutability: the input `obs` Array is not mutated.

**`test_obs_normalizer.gd` (loader node via `set_stats_for_test`):**
- `set_stats_for_test` then `normalize` matches the pure-math result.
- `obs_size()` equals the stats length; `is_loaded()` reflects state.
- `normalize` before loading → obs returned unchanged (no crash).

**`test/python/test_export_vecnormalize_stats.py` (pure dump/format helper, stdlib `unittest`):**
- `stats_dict` on a duck-typed fake rms (`.mean`/`.var` as lists or fake-numpy) → correct dict with
  `epsilon`/`clip_obs`.
- `dump_stats` round-trips through JSON.
- **Parity check (the whole point):** compute SB3's expected `normalize_obs` for a sample obs with
  plain numpy arithmetic (`clip((obs-mean)/sqrt(var+eps), -clip, clip)`) and assert the exported
  stats fed through the *same* formula give the same numbers — and pin those same expected numbers
  in the GDScript `test_obs_normalize.gd`, so train and deploy are anchored to one set of values.

**Run (this item's tests only; parent runs the full suite):**
- `godot --headless --path . --script res://test/unit/test_obs_normalize.gd`
- `godot --headless --path . --script res://test/unit/test_obs_normalizer.gd`
- `.venv-train/bin/python -m unittest discover -s test/python -p 'test_export_vecnormalize_stats.py'`

## Scope boundaries (YAGNI / explicit deferrals)

- **No** online stat updating (deploy is frozen).
- **No** reward normalization (training-only; never affects inference).
- **No** controller auto-wiring — manual composition, documented.
- **No** image/CNN obs normalization (`VecNormalize` is for vector obs; image obs use a fixed scale).
- **No** `.npz`/binary transport — JSON only.

## Follow-ups to record on completion

1. An end-to-end example/regression that trains a tiny policy *with* `VecNormalize`, exports stats,
   and asserts game-side parity against the SB3 wrapper on real data (pins the full loop, not just
   the formula).
2. Optional controller integration (auto-apply a configured normalizer before inference) — fold
   into the broader "sensor/normalizer auto-discovery" item-5 follow-up.
