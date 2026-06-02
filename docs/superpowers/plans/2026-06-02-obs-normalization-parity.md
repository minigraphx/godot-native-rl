# Observation-normalization Parity Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let policies trained with SB3 `VecNormalize` deploy natively by replaying the saved obs mean/var game-side before `run_inference`, closing the #1 silent-failure gap in `ncnn_vs_onnx.md`.

**Architecture:** A pure GDScript helper (`ObsNormalize`) replays `clip((obs-mean)/sqrt(var+eps), ±clip_obs)` between `get_obs()` and `run_inference()` (deploy-only — training keeps returning raw obs). A Python script exports the stats from `vec_normalize.pkl` to committed JSON; a seeded generator produces a golden fixture asserting the GDScript math matches SB3's own `normalize_obs`. The controller core gains an optional `obs_norm_stats` dict; the thin 2D/3D wrappers load it from an exported JSON path.

**Tech Stack:** GDScript (Godot 4.6, headless `SceneTree` tests), Python 3.13 under `.venv-train` (stable_baselines3 + numpy + gymnasium), stdlib `unittest`.

---

## File structure

- **Create** `addons/godot_native_rl/controllers/obs_normalize.gd` — pure `ObsNormalize` (normalize / validate / to_typed).
- **Create** `test/unit/test_obs_normalize.gd` — pure-math unit tests + committed-golden parity.
- **Create** `scripts/export_vecnormalize.py` — `vec_normalize.pkl` → stats JSON (+ pure helpers).
- **Create** `test/python/test_export_vecnormalize.py` — tests for the export helpers.
- **Create** `scripts/make_vecnormalize_stats.py` — seeded generator for the committed fixtures.
- **Create** `models/synthetic_vecnormalize.json`, `models/synthetic_vecnormalize_golden.json` — committed fixtures (generated).
- **Modify** `addons/godot_native_rl/controllers/ncnn_controller_core.gd` — add `obs_norm_stats` + apply in float path.
- **Modify** `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` and `..._3d.gd` — `obs_norm_stats_path` export + loader + test seam.
- **Modify** `test/unit/test_controller_inference.gd` — capture runner input; assert normalized/raw/skip behavior.
- **Modify** `docs/ncnn_vs_onnx.md`, `README.md`, `CLAUDE.md`, `docs/BACKLOG.md` — docs.

`test/run_tests.sh` needs **no change** — it auto-discovers `test/unit/test_*.gd` and `test/python/test_*.py`.

---

## Task 1: Pure `ObsNormalize` helper + math tests

**Files:**
- Create: `addons/godot_native_rl/controllers/obs_normalize.gd`
- Test: `test/unit/test_obs_normalize.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_obs_normalize.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ObsNormalize = preload("res://addons/godot_native_rl/controllers/obs_normalize.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Basic: (obs - mean)/sqrt(var + eps), in-range so no clipping.
	var out := ObsNormalize.normalize(
		PackedFloat32Array([2.0, 0.0]), PackedFloat32Array([1.0, 0.0]),
		PackedFloat32Array([4.0, 1.0]), 0.0, 10.0)
	h.assert_true(absf(out[0] - 0.5) < 1e-6 and absf(out[1]) < 1e-6, "basic normalize")

	# Clipping at +clip_obs and -clip_obs.
	var hi := ObsNormalize.normalize(
		PackedFloat32Array([100.0]), PackedFloat32Array([0.0]),
		PackedFloat32Array([1.0]), 0.0, 10.0)
	h.assert_true(absf(hi[0] - 10.0) < 1e-6, "clips to +clip_obs")
	var lo := ObsNormalize.normalize(
		PackedFloat32Array([-100.0]), PackedFloat32Array([0.0]),
		PackedFloat32Array([1.0]), 0.0, 10.0)
	h.assert_true(absf(lo[0] + 10.0) < 1e-6, "clips to -clip_obs")

	# Epsilon avoids div-by-zero on zero variance (clip high so it isn't clamped).
	var eps_out := ObsNormalize.normalize(
		PackedFloat32Array([1.0]), PackedFloat32Array([0.0]),
		PackedFloat32Array([0.0]), 1e-8, 1e9)
	h.assert_true(eps_out[0] > 100.0, "epsilon avoids div-by-zero (large finite value)")

	# Size mismatch -> empty.
	var bad := ObsNormalize.normalize(
		PackedFloat32Array([1.0, 2.0]), PackedFloat32Array([0.0]),
		PackedFloat32Array([1.0]), 0.0, 10.0)
	h.assert_eq(bad.size(), 0, "size mismatch returns empty")

	# validate accept/reject.
	h.assert_true(ObsNormalize.validate({"mean": [0.0], "var": [1.0], "epsilon": 1e-8, "clip_obs": 10.0}),
		"validate accepts well-formed")
	h.assert_true(not ObsNormalize.validate({"mean": [0.0], "var": [1.0, 2.0], "epsilon": 1e-8, "clip_obs": 10.0}),
		"validate rejects unequal lengths")
	h.assert_true(not ObsNormalize.validate({"mean": [], "var": [], "epsilon": 1e-8, "clip_obs": 10.0}),
		"validate rejects empty")
	h.assert_true(not ObsNormalize.validate({"var": [1.0], "epsilon": 1e-8, "clip_obs": 10.0}),
		"validate rejects missing key")

	# to_typed coerces JSON arrays into PackedFloat32Array + floats.
	var typed := ObsNormalize.to_typed({"mean": [0.5], "var": [2.0], "epsilon": 1e-8, "clip_obs": 5.0})
	h.assert_true(typed["mean"] is PackedFloat32Array and absf(typed["clip_obs"] - 5.0) < 1e-6,
		"to_typed coerces types")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_obs_normalize.gd`
Expected: FAIL/error — `obs_normalize.gd` does not exist yet (parse/preload error).

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/controllers/obs_normalize.gd`:

```gdscript
class_name ObsNormalize
extends RefCounted

# Pure deploy-side replay of SB3 VecNormalize observation normalization. The pre-inference analogue
# of action_decode.gd (the post-inference transform). VecNormalize keeps its running mean/var in a
# separate .pkl (not in the policy network), so a converted ncnn model has lost them; this replays
# the exact transform between get_obs() and run_inference():
#   normalized[i] = clamp((obs[i] - mean[i]) / sqrt(var[i] + epsilon), -clip_obs, +clip_obs)
# Size mismatch (a train/deploy shape error) -> empty array (controller skips the action; no silent
# garbage forward pass). Mirrors stable_baselines3.common.vec_env.VecNormalize._normalize_obs.

static func normalize(obs: PackedFloat32Array, mean: PackedFloat32Array,
		var_: PackedFloat32Array, epsilon: float, clip_obs: float) -> PackedFloat32Array:
	if obs.size() != mean.size() or obs.size() != var_.size():
		push_error("ObsNormalize.normalize: size mismatch (obs %d, mean %d, var %d)." % [
			obs.size(), mean.size(), var_.size()])
		return PackedFloat32Array()
	var out := PackedFloat32Array()
	out.resize(obs.size())
	for i in obs.size():
		var z: float = (obs[i] - mean[i]) / sqrt(var_[i] + epsilon)
		out[i] = clampf(z, -clip_obs, clip_obs)
	return out

# True iff a JSON-decoded stats dict is well-formed for normalize(): mean+var present as equal,
# non-empty numeric arrays, plus epsilon and clip_obs keys. Checked at load so a bad fixture fails
# loudly up front, not at the first inference frame.
static func validate(stats: Dictionary) -> bool:
	if not (stats.has("mean") and stats.has("var") and stats.has("epsilon") and stats.has("clip_obs")):
		return false
	var mean = stats["mean"]
	var var_ = stats["var"]
	if not (mean is Array or mean is PackedFloat32Array):
		return false
	if not (var_ is Array or var_ is PackedFloat32Array):
		return false
	if mean.size() == 0 or mean.size() != var_.size():
		return false
	return true

# Coerce a validated JSON stats dict into typed PackedFloat32Arrays + floats once, so the per-frame
# hot path doesn't re-coerce. Returns {} (and push_error) if invalid.
static func to_typed(stats: Dictionary) -> Dictionary:
	if not validate(stats):
		push_error("ObsNormalize.to_typed: invalid stats dictionary.")
		return {}
	return {
		"mean": PackedFloat32Array(stats["mean"]),
		"var": PackedFloat32Array(stats["var"]),
		"epsilon": float(stats["epsilon"]),
		"clip_obs": float(stats["clip_obs"]),
	}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_obs_normalize.gd`
Expected: PASS (all assertions OK, harness exit 0).

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/obs_normalize.gd test/unit/test_obs_normalize.gd
git commit -m "feat: pure ObsNormalize VecNormalize replay helper"
```

---

## Task 2: Python `export_vecnormalize.py` + tests

**Files:**
- Create: `scripts/export_vecnormalize.py`
- Test: `test/python/test_export_vecnormalize.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_export_vecnormalize.py`:

```python
"""Tests for scripts/export_vecnormalize.py (stats extraction from SB3 VecNormalize)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export_vecnormalize as ev  # noqa: E402


def _make_vecnormalize(seed: int = 0, obs_dim: int = 4):
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _Dummy(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros(obs_dim, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(obs_dim, dtype=np.float32), 0.0, False, False, {}

    venv = DummyVecEnv([lambda: _Dummy()])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=False)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        vn.obs_rms.update(rng.normal(size=(8, obs_dim)).astype(np.float32))
    return vn


class TestStatsFromVecNormalize(unittest.TestCase):
    def test_extracts_mean_var_epsilon_clip(self):
        vn = _make_vecnormalize()
        stats = ev.stats_from_vecnormalize(vn)
        self.assertEqual(stats["obs_size"], 4)
        self.assertEqual(len(stats["mean"]), 4)
        self.assertEqual(len(stats["var"]), 4)
        for got, exp in zip(stats["mean"], vn.obs_rms.mean):
            self.assertAlmostEqual(got, float(exp), places=6)
        for got, exp in zip(stats["var"], vn.obs_rms.var):
            self.assertAlmostEqual(got, float(exp), places=6)
        self.assertAlmostEqual(stats["epsilon"], float(vn.epsilon))
        self.assertAlmostEqual(stats["clip_obs"], float(vn.clip_obs))

    def test_rejects_norm_obs_disabled(self):
        vn = _make_vecnormalize()
        vn.norm_obs = False
        with self.assertRaises(ValueError):
            ev.stats_from_vecnormalize(vn)

    def test_rejects_non_vecnormalize(self):
        with self.assertRaises(ValueError):
            ev.stats_from_vecnormalize(object())

    def test_rejects_dict_obs_rms(self):
        vn = _make_vecnormalize()
        vn.obs_rms = {"a": vn.obs_rms}
        with self.assertRaises(ValueError):
            ev.stats_from_vecnormalize(vn)

    def test_write_and_reread(self):
        vn = _make_vecnormalize()
        stats = ev.stats_from_vecnormalize(vn)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "stats.json"
            ev.write_stats_json(stats, path)
            reread = json.loads(path.read_text())
        self.assertEqual(reread, stats)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_export_vecnormalize -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_vecnormalize'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/export_vecnormalize.py`:

```python
#!/usr/bin/env python3
"""Export SB3 VecNormalize observation statistics to a committed JSON fixture.

A policy trained with stable_baselines3 VecNormalize learns on normalized observations, but the
running mean/var live in a separate vec_normalize.pkl (never in the policy network). After
ONNX -> ncnn conversion those stats are gone, so they must be replayed game-side before
run_inference. This reads the .pkl and writes {obs_size, mean, var, epsilon, clip_obs} as JSON for
the GDScript ObsNormalize replay helper.

Run under .venv-train (has stable_baselines3).

Usage:
    .venv-train/bin/python scripts/export_vecnormalize.py path/to/vec_normalize.pkl [--out stats.json]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any


def stats_from_vecnormalize(vn: Any) -> dict:
    """Extract a JSON-serializable obs-normalization stats dict from a VecNormalize object.

    Fails fast (ValueError) when the object can't be replayed for a single obs vector: not a
    VecNormalize, obs normalization disabled, multi-key (Dict) obs, or a non-1-D obs_rms.
    """
    if not (hasattr(vn, "obs_rms") and hasattr(vn, "clip_obs") and hasattr(vn, "epsilon")):
        raise ValueError("not a VecNormalize object (missing obs_rms/clip_obs/epsilon)")
    if hasattr(vn, "norm_obs") and not vn.norm_obs:
        raise ValueError("VecNormalize has norm_obs=False; policy trained on raw obs, nothing to replay")
    obs_rms = vn.obs_rms
    if isinstance(obs_rms, dict):
        raise ValueError("multi-key (Dict) observations are out of scope; only a single obs vector is supported")
    mean = obs_rms.mean
    var = obs_rms.var
    if getattr(mean, "ndim", None) != 1 or mean.shape != var.shape:
        raise ValueError(f"unexpected obs_rms shape (mean {mean.shape}, var {var.shape}); expected a 1-D vector")
    return {
        "obs_size": int(mean.shape[0]),
        "mean": [float(x) for x in mean],
        "var": [float(x) for x in var],
        "epsilon": float(vn.epsilon),
        "clip_obs": float(vn.clip_obs),
    }


def write_stats_json(stats: dict, path: Path) -> None:
    path.write_text(json.dumps(stats, indent=2) + "\n")


def load_vecnormalize(pkl_path: Path) -> Any:
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export SB3 VecNormalize obs stats to JSON.")
    p.add_argument("pkl", help="path to vec_normalize.pkl")
    p.add_argument("--out", default=None, help="output JSON path (default: <pkl-stem>.json beside the pkl)")
    a = p.parse_args(argv)

    pkl_path = Path(a.pkl)
    if not pkl_path.is_file():
        print(f"ERROR: pkl not found: {a.pkl}", file=sys.stderr)
        return 1
    out_path = Path(a.out) if a.out else pkl_path.with_suffix(".json")

    try:
        vn = load_vecnormalize(pkl_path)
        stats = stats_from_vecnormalize(vn)
    except (ValueError, pickle.UnpicklingError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    write_stats_json(stats, out_path)
    print(f"OK: {out_path} (obs_size={stats['obs_size']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_export_vecnormalize -v`
Expected: PASS (5 tests OK).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_vecnormalize.py test/python/test_export_vecnormalize.py
git commit -m "feat: export_vecnormalize.py (VecNormalize .pkl -> stats JSON)"
```

---

## Task 3: Seeded fixture generator + committed golden + GDScript parity

**Files:**
- Create: `scripts/make_vecnormalize_stats.py`
- Create (generated, committed): `models/synthetic_vecnormalize.json`, `models/synthetic_vecnormalize_golden.json`
- Modify: `test/unit/test_obs_normalize.gd`

- [ ] **Step 1: Write the generator**

Create `scripts/make_vecnormalize_stats.py`:

```python
#!/usr/bin/env python3
"""Generate seeded VecNormalize stats + a golden fixture for the obs-normalization replay test.

Builds a real SB3 VecNormalize over a dummy Box-obs env, updates it with seeded random
observations, then writes:
  models/synthetic_vecnormalize.json         - exported stats (via export_vecnormalize)
  models/synthetic_vecnormalize_golden.json  - {stats_path, cases:[{raw, normalized}]} where
                                               `normalized` is SB3's own vn.normalize_obs(raw).

So test/unit/test_obs_normalize.gd asserts the GDScript ObsNormalize.normalize reproduces SB3's
actual output (atol 1e-6), not a re-implementation of it. The .pkl is not committed (this script
recreates it deterministically); only the derived JSON fixtures are committed.

Run under .venv-train.  Regenerate:  .venv-train/bin/python scripts/make_vecnormalize_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "scripts"))

import export_vecnormalize as ev  # noqa: E402

OBS_DIM = 6
SEED = 11


def build_vecnormalize():
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _Dummy(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros(OBS_DIM, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(OBS_DIM, dtype=np.float32), 0.0, False, False, {}

    venv = DummyVecEnv([lambda: _Dummy()])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=False)
    rng = np.random.default_rng(SEED)
    for _ in range(100):
        vn.obs_rms.update(rng.normal(loc=0.5, scale=2.0, size=(16, OBS_DIM)).astype(np.float32))
    return vn


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    vn = build_vecnormalize()

    stats = ev.stats_from_vecnormalize(vn)
    stats_path = MODELS / "synthetic_vecnormalize.json"
    ev.write_stats_json(stats, stats_path)

    rng = np.random.default_rng(SEED + 1)
    cases = []
    for _ in range(5):
        raw = rng.normal(loc=0.5, scale=2.0, size=(OBS_DIM,)).astype(np.float32)
        normalized = np.asarray(vn.normalize_obs(raw), dtype=np.float32).ravel()
        cases.append({
            "raw": [float(x) for x in raw],
            "normalized": [float(x) for x in normalized],
        })
    golden = {"stats_path": "res://models/synthetic_vecnormalize.json", "cases": cases}
    golden_path = MODELS / "synthetic_vecnormalize_golden.json"
    golden_path.write_text(json.dumps(golden, indent=2) + "\n")

    print(f"OK: {stats_path}")
    print(f"OK: {golden_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the committed fixtures**

Run: `.venv-train/bin/python scripts/make_vecnormalize_stats.py`
Expected: prints two `OK:` lines; `models/synthetic_vecnormalize.json` and
`models/synthetic_vecnormalize_golden.json` now exist. Sanity-check:
`python -c "import json;d=json.load(open('models/synthetic_vecnormalize.json'));print(d['obs_size'], len(d['mean']))"`
Expected: `6 6`.

- [ ] **Step 3: Add the golden-parity block to the GDScript test (failing first)**

In `test/unit/test_obs_normalize.gd`, add a JSON loader method after `_initialize`:

```gdscript
func _load_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	assert(f != null, "cannot open %s" % path)
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	assert(parsed is Dictionary, "cannot parse %s" % path)
	return parsed
```

Then insert this block in `_initialize()` immediately **before** `h.finish(self)`:

```gdscript
	# --- Golden parity: GDScript helper must reproduce SB3 vn.normalize_obs (atol 1e-6) ---
	var stats := _load_json("res://models/synthetic_vecnormalize.json")
	var golden := _load_json("res://models/synthetic_vecnormalize_golden.json")
	h.assert_true(ObsNormalize.validate(stats), "committed stats fixture validates")
	var typed_stats := ObsNormalize.to_typed(stats)
	for gcase in golden["cases"]:
		var raw := PackedFloat32Array(gcase["raw"])
		var expected: Array = gcase["normalized"]
		var got := ObsNormalize.normalize(raw, typed_stats["mean"], typed_stats["var"],
			typed_stats["epsilon"], typed_stats["clip_obs"])
		h.assert_eq(got.size(), expected.size(), "golden case length matches")
		var max_diff := 0.0
		for i in got.size():
			max_diff = maxf(max_diff, absf(got[i] - float(expected[i])))
		h.assert_true(max_diff < 1e-6, "golden case parity (max diff %f < 1e-6)" % max_diff)
```

Note: this step depends on Step 2's fixtures existing. (If run before Step 2 it fails on the
`assert f != null` — that is the expected RED.)

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_obs_normalize.gd`
Expected: PASS — all earlier math assertions plus the golden-parity cases (max diff < 1e-6).

- [ ] **Step 5: Commit**

```bash
git add scripts/make_vecnormalize_stats.py models/synthetic_vecnormalize.json \
  models/synthetic_vecnormalize_golden.json test/unit/test_obs_normalize.gd
git commit -m "test: seeded VecNormalize golden + GDScript parity (atol 1e-6)"
```

---

## Task 4: Controller wiring (core + 2D/3D wrappers) + integration test

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`
- Test: `test/unit/test_controller_inference.gd`

- [ ] **Step 1: Write the failing integration test**

In `test/unit/test_controller_inference.gd`, replace the `FakeRunner` class so it records its input:

```gdscript
# Fake that mimics NcnnRunner.run_inference (float path) -> raw output vector.
class FakeRunner:
	var loaded := true
	var output := PackedFloat32Array([0.0, 0.0, 0.0, 0.9, 0.0])  # argmax == 3 over size-5
	var last_input := PackedFloat32Array()
	func is_model_loaded() -> bool:
		return loaded
	func run_inference(input) -> PackedFloat32Array:
		last_input = input
		return output
```

Then add this block in `_initialize()` immediately **before** `h.finish(self)` (Stub.get_obs()
returns `[0.0, 0.0, 1.0, 0.0, 0.5]`, size 5):

```gdscript
	# Obs normalization: identity stats (mean 0, var 1) -> runner sees raw obs unchanged.
	var na = Stub.new()
	var nr := FakeRunner.new()
	na.set_ncnn_runner_for_test(nr)
	na.set_obs_norm_stats_for_test({
		"mean": PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0]),
		"var": PackedFloat32Array([1.0, 1.0, 1.0, 1.0, 1.0]),
		"epsilon": 0.0, "clip_obs": 10.0})
	na.infer_and_act()
	h.assert_true(absf(nr.last_input[2] - 1.0) < 1e-6 and absf(nr.last_input[4] - 0.5) < 1e-6,
		"identity stats feed obs unchanged to runner")
	na.free()

	# Non-identity stats actually transform: obs[2]=1, mean=1, var=4 -> (1-1)/sqrt(4)=0.
	var na2 = Stub.new()
	var nr2 := FakeRunner.new()
	na2.set_ncnn_runner_for_test(nr2)
	na2.set_obs_norm_stats_for_test({
		"mean": PackedFloat32Array([0.0, 0.0, 1.0, 0.0, 0.0]),
		"var": PackedFloat32Array([1.0, 1.0, 4.0, 1.0, 1.0]),
		"epsilon": 0.0, "clip_obs": 10.0})
	na2.infer_and_act()
	h.assert_true(absf(nr2.last_input[2]) < 1e-6, "non-identity stats transform obs[2] -> 0")
	na2.free()

	# Empty stats (default) -> raw obs (backward compatible).
	var nb = Stub.new()
	var nbr := FakeRunner.new()
	nb.set_ncnn_runner_for_test(nbr)
	nb.infer_and_act()
	h.assert_true(absf(nbr.last_input[2] - 1.0) < 1e-6, "empty stats feeds raw obs (backward compatible)")
	nb.free()

	# Size-mismatch stats -> normalize returns empty -> action skipped (no set_action).
	var nc = Stub.new()
	var ncr := FakeRunner.new()
	nc.set_ncnn_runner_for_test(ncr)
	nc.set_obs_norm_stats_for_test({
		"mean": PackedFloat32Array([0.0]), "var": PackedFloat32Array([1.0]),
		"epsilon": 0.0, "clip_obs": 10.0})
	nc.infer_and_act()
	h.assert_eq(nc.last_action, null, "size-mismatch stats skips action")
	nc.free()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_controller_inference.gd`
Expected: FAIL/error — `set_obs_norm_stats_for_test` is not defined and the core does not normalize.

- [ ] **Step 3: Wire the core**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, add the preload near the existing
`ActionDecode` const:

```gdscript
const ObsNormalize = preload("res://addons/godot_native_rl/controllers/obs_normalize.gd")
```

Add the field next to the other `var` declarations (after `var reward_source = null`):

```gdscript
var obs_norm_stats: Dictionary = {}
```

Replace the `else` (float) branch inside `choose_and_apply_action` with the normalizing version:

```gdscript
	else:
		var obs_dict: Dictionary = agent.get_obs()
		assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
		var obs_vec := PackedFloat32Array(obs_dict["obs"])
		if not obs_norm_stats.is_empty():
			obs_vec = ObsNormalize.normalize(obs_vec, obs_norm_stats["mean"], obs_norm_stats["var"],
				obs_norm_stats["epsilon"], obs_norm_stats["clip_obs"])
			if obs_vec.is_empty():
				push_error("NcnnControllerCore.choose_and_apply_action: obs normalization failed (size mismatch); skipping action.")
				return
		output = runner.run_inference(obs_vec)
```

- [ ] **Step 4: Wire the 2D wrapper**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`:

Add the preload after the existing `NcnnControllerCore` const:

```gdscript
const ObsNormalize = preload("res://addons/godot_native_rl/controllers/obs_normalize.gd")
```

Add the export var after `@export var output_blob_name: String = "out0"`:

```gdscript
@export_file("*.json") var obs_norm_stats_path: String = ""
```

In `_ready()`, load stats after setting up the runner:

```gdscript
func _ready() -> void:
	add_to_group("AGENT")
	collect_reward_adapters()
	if control_mode == ControlModes.NCNN_INFERENCE:
		_setup_ncnn_runner()
		_load_obs_norm_stats()
```

Add these two methods after `set_ncnn_runner_for_test`:

```gdscript
func _load_obs_norm_stats() -> void:
	if obs_norm_stats_path.is_empty():
		return
	var f := FileAccess.open(obs_norm_stats_path, FileAccess.READ)
	if f == null:
		push_error("NcnnAIController2D: cannot open obs_norm_stats_path '%s'." % obs_norm_stats_path)
		return
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary) or not ObsNormalize.validate(parsed):
		push_error("NcnnAIController2D: invalid obs-norm stats JSON at '%s'." % obs_norm_stats_path)
		return
	_core.obs_norm_stats = ObsNormalize.to_typed(parsed)

func set_obs_norm_stats_for_test(stats: Dictionary) -> void:
	_core.obs_norm_stats = stats
```

- [ ] **Step 5: Wire the 3D wrapper identically**

Apply the **same four edits** to `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`,
changing the `push_error` prefixes to `NcnnAIController3D:` to match the file's existing messages:
(1) add the `ObsNormalize` preload after its `NcnnControllerCore` const; (2) add the
`@export_file("*.json") var obs_norm_stats_path: String = ""` export after `output_blob_name`;
(3) call `_load_obs_norm_stats()` after `_setup_ncnn_runner()` in `_ready()`; (4) add the
`_load_obs_norm_stats()` and `set_obs_norm_stats_for_test()` methods after `set_ncnn_runner_for_test`,
with `NcnnAIController3D:` in the two `push_error` strings.

- [ ] **Step 6: Run the controller + obs-normalize tests to verify they pass**

Run: `godot --headless --path . --script res://test/unit/test_controller_inference.gd`
Expected: PASS — existing float/image/mixed/no-runner cases **and** the four new normalization cases.

Run: `godot --headless --path . --script res://test/unit/test_obs_normalize.gd`
Expected: PASS (unchanged — core edits don't touch the pure helper).

- [ ] **Step 7: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd \
  addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd \
  addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd \
  test/unit/test_controller_inference.gd
git commit -m "feat: deploy-side VecNormalize obs replay in controller (2D/3D)"
```

---

## Task 5: Full suite + docs

**Files:**
- Modify: `docs/ncnn_vs_onnx.md`, `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Run the full test suite (clean cache)**

Run: `rm -f .godot/global_script_class_cache.cfg && ./test/run_tests.sh`
Expected: ends with `All tests passed.` (new `test_obs_normalize.gd` and
`test_export_vecnormalize.py` are auto-discovered).

- [ ] **Step 2: Update `docs/ncnn_vs_onnx.md`**

Replace the last sentence of the "Observation preprocessing parity is on you" bullet (around line 204-206,
"If you instead train with SB3 `VecNormalize`…the converted network does not carry them.") with:

```markdown
  If you instead train with SB3 `VecNormalize` (running mean/std), export those statistics with
  `scripts/export_vecnormalize.py vec_normalize.pkl` and point the controller at the resulting JSON via
  its `obs_norm_stats_path` — the addon replays the exact `clip((obs-mean)/sqrt(var+eps), ±clip_obs)`
  transform game-side before inference (pure `ObsNormalize`, verified against SB3 at `atol 1e-6`).
```

Then update the "Current limitations" list (around line 230) so observation-normalization is no longer
listed as a remaining gap — change the "Remaining deploy-side gaps" sentence to read
`recurrent/LSTM state (item 22) and batched multi-agent inference (item 23)` (drop item 24).

- [ ] **Step 3: Update `README.md`**

In the deploy/inference section (where `NCNN_INFERENCE` / `model_param_path` are described), add:

```markdown
**VecNormalize policies:** if you trained with SB3 `VecNormalize`, export its stats with
`.venv-train/bin/python scripts/export_vecnormalize.py path/to/vec_normalize.pkl` and set the
controller's `obs_norm_stats_path` to the generated JSON. The addon replays the running mean/std
game-side before inference (the network itself does not carry them).
```

- [ ] **Step 4: Update `CLAUDE.md`**

Under "Key commands", add after the convert/verify entry:

```markdown
- **Export VecNormalize stats (deploy):** `.venv-train/bin/python scripts/export_vecnormalize.py
  vec_normalize.pkl` → JSON; set the controller's `obs_norm_stats_path` so `ObsNormalize` replays
  the obs mean/std game-side before inference (policies trained with SB3 `VecNormalize`).
```

In the addon-description paragraph (the `controllers/` list), add `obs_normalize.gd` alongside
`action_decode.gd` as the pre-inference transform helper.

- [ ] **Step 5: Update `docs/BACKLOG.md`**

Change item 24's status from ⬜ to ✅ and append a completion note mirroring item 21's style:

```markdown
24. ✅ **Observation-normalization parity helper** — replay SB3 `VecNormalize` obs stats game-side.
    Added pure `addons/godot_native_rl/controllers/obs_normalize.gd` (`normalize`/`validate`/`to_typed`),
    `scripts/export_vecnormalize.py` (`vec_normalize.pkl` → committed JSON), and an
    `obs_norm_stats_path` export on `NcnnAIController2D/3D` that loads the stats into
    `NcnnControllerCore`, which normalizes obs in the float inference path (deploy-only — `get_obs()`
    stays raw so training never double-normalizes).
    **Done 2026-06-02** — spec `docs/superpowers/specs/2026-06-02-obs-normalization-parity-design.md`,
    plan `docs/superpowers/plans/2026-06-02-obs-normalization-parity.md`. Verified by GDScript unit
    tests + a seeded synthetic golden (`scripts/make_vecnormalize_stats.py` →
    `models/synthetic_vecnormalize*.json`) asserting the GDScript replay reproduces SB3's own
    `normalize_obs` at **atol 1e-6**, plus controller integration tests (normalized/raw/skip) and
    Python export tests. No C++ change/rebuild. Full suite green from a clean cache.
```

Also update the item-9/limitations cross-references if any list item 24 as open. Then in the
"Done:" summary line near the top of the backlog, add `24` to the completed list.

- [ ] **Step 6: Commit**

```bash
git add docs/ncnn_vs_onnx.md README.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: VecNormalize obs-normalization parity (item 24 done)"
```

---

## Self-review notes

- **Spec coverage:** pure helper (Task 1), export script (Task 2), synthetic golden generator +
  fixtures (Task 3), controller core + 2D/3D wiring + integration tests (Task 4), docs + full suite
  (Task 5). JSON schema, deploy-only data flow, loud error handling, and atol-1e-6 golden all covered.
- **Placeholder scan:** none — every code/edit step shows concrete content.
- **Type consistency:** `normalize`/`validate`/`to_typed` signatures identical across helper, tests,
  core, and wrappers; stats dict keys (`mean`/`var`/`epsilon`/`clip_obs`/`obs_size`) consistent across
  Python JSON, GDScript validate/to_typed, and controller use; `obs_norm_stats` (core field) and
  `obs_norm_stats_path` (export) named consistently; `set_obs_norm_stats_for_test` used in Task 4 test
  and defined in the wrappers.
- **Note on Task 3 ordering:** the GDScript golden block (Task 3 Step 3) depends on the committed
  fixtures from Step 2; that is intentional and called out in the step.
```
