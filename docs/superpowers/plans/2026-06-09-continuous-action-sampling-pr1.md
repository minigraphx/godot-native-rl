# Continuous Action Sampling — PR 1 (capability) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship deploy-side continuous DiagGaussian action sampling for PPO (Box) policies — export `std = exp(policy.log_std)` to a flat JSON sidecar and sample `mean + std·N(0,1)` (then optional `tanh`) game-side, mirroring the existing VecNormalize obs-stats pattern.

**Architecture:** A flat sidecar `{"std":[...], "action_dim":N}` (SB3's `log_std` is a single flat vector with no godot_rl key names). A pure Python exporter writes it; a pure GDScript `ActionDist` loader/validator coerces it; `ActionDecode.decode_actions` gains a fifth `action_dist` param and a continuous-only counter that applies `std` positionally; the controller core + 2D/3D wrappers load and pass it, exactly like `obs_norm_stats`. The `deterministic`/`rng` plumbing already reaches the continuous branch (designed-in by #16), so no new wiring.

**Tech Stack:** GDScript (Godot 4.5+, TAB indent), Python 3.13 (`.venv-train`: SB3/torch), stdlib `unittest`, dependency-free GDScript harness at `test/harness.gd`.

**Spec:** `docs/superpowers/specs/2026-06-09-continuous-action-sampling-design.md`. Branch already created: `feat/continuous-action-sampling-64`.

---

## File Structure

- **Create** `scripts/export_action_dist.py` — extract `std=exp(log_std)` from a PPO `.zip` → flat JSON. Pure helpers + thin `main()`.
- **Create** `test/python/test_export_action_dist.py` — unit tests for the exporter (pure helpers + a torch-tensor extraction case).
- **Create** `addons/godot_native_rl/controllers/action_dist.gd` — pure `validate()`/`to_typed()` (clone of `obs_normalize.gd`'s loader half).
- **Create** `test/unit/test_action_dist.gd` — unit tests for the loader/validator.
- **Create** `test/unit/stub_continuous_agent.gd` — a continuous-action concrete controller for the wiring test (clone of `stub_agent.gd`).
- **Modify** `addons/godot_native_rl/controllers/action_decode.gd` — add `action_dist` param + continuous-sampling branch.
- **Modify** `test/unit/test_action_decode.gd` — add continuous-sampling assertions.
- **Modify** `addons/godot_native_rl/controllers/ncnn_controller_core.gd` — add `action_dist_stats` field, pass to decode.
- **Modify** `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` and `ncnn_ai_controller_3d.gd` — `@export action_dist_stats_path`, loader, `_ready()` call, `set_action_dist_for_test()`.
- **Modify** `test/unit/test_stochastic_inference.gd` — add continuous wiring assertions.
- **Modify** docs: `docs/guide/deploying.md`, `docs/guide/building-your-agent.md`, `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`.

Tests auto-discover: `run_tests.sh` loops `test/unit/test_*.gd` (line 30) and runs `python -m unittest discover -s test/python -p 'test_*.py'` (line 95). The two stub files are not `test_*` so they are helpers, not test runs (like `stub_agent.gd`).

---

### Task 1: Python exporter `scripts/export_action_dist.py`

**Files:**
- Create: `scripts/export_action_dist.py`
- Test: `test/python/test_export_action_dist.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_export_action_dist.py`:

```python
"""Tests for scripts/export_action_dist.py (std extraction from SB3 PPO log_std)."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import export_action_dist as ead  # noqa: E402


class TestStdFromLogStd(unittest.TestCase):
    def test_exp_of_log_std(self):
        stats = ead.std_from_log_std([0.0, math.log(2.0), math.log(0.5)])
        self.assertEqual(stats["action_dim"], 3)
        self.assertAlmostEqual(stats["std"][0], 1.0, places=6)
        self.assertAlmostEqual(stats["std"][1], 2.0, places=6)
        self.assertAlmostEqual(stats["std"][2], 0.5, places=6)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ead.std_from_log_std([])


class TestStdFromModel(unittest.TestCase):
    def test_extracts_from_policy_log_std(self):
        import torch
        model = SimpleNamespace(policy=SimpleNamespace(log_std=torch.tensor([0.0, math.log(3.0)])))
        stats = ead.std_from_model(model)
        self.assertEqual(stats["action_dim"], 2)
        self.assertAlmostEqual(stats["std"][1], 3.0, places=5)

    def test_no_log_std_raises(self):
        model = SimpleNamespace(policy=SimpleNamespace())
        with self.assertRaises(ValueError):
            ead.std_from_model(model)


class TestWriteJson(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "action_dist.json"
            ead.write_action_dist_json({"std": [1.0, 2.0], "action_dim": 2}, out)
            loaded = json.loads(out.read_text())
            self.assertEqual(loaded["std"], [1.0, 2.0])
            self.assertEqual(loaded["action_dim"], 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_export_action_dist -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_action_dist'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/export_action_dist.py`:

```python
#!/usr/bin/env python3
"""Export SB3 PPO continuous (Box) action std to a committed JSON sidecar.

PPO's continuous std is a state-independent learned parameter (policy.log_std) — a fixed per-dim
vector that is never part of the network output, so an ncnn-converted policy can't sample continuous
actions. This extracts std = exp(log_std) into a flat JSON sidecar for the GDScript ActionDist
deploy-side DiagGaussian sampler (the post-inference analogue of export_vecnormalize.py).

Run under .venv-train (has stable_baselines3 / torch).

Usage:
    .venv-train/bin/python scripts/export_action_dist.py path/to/ppo_model.zip [--out action_dist.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def std_from_log_std(log_std: list) -> dict:
    """Build the flat JSON-serializable action-dist dict from a 1-D log_std vector.

    std = exp(log_std), positional over the continuous action dims. Pure (no torch/SB3) so it is
    unit-testable directly. Empty -> ValueError (not a continuous Box policy).
    """
    if len(log_std) == 0:
        raise ValueError("log_std is empty; not a continuous (Box) action policy")
    std = [float(math.exp(float(x))) for x in log_std]
    return {"std": std, "action_dim": len(std)}


def std_from_model(model: Any) -> dict:
    """Extract std from a loaded SB3 model's policy.log_std parameter.

    Fails fast (ValueError) when there is no log_std (discrete/MultiDiscrete policy, or a SAC actor
    whose std is state-dependent — SAC continuous sampling is out of scope; see the design doc).
    """
    policy = getattr(model, "policy", None)
    log_std_param = getattr(policy, "log_std", None)
    if log_std_param is None:
        raise ValueError(
            "policy has no log_std (not a PPO/A2C continuous Box policy; "
            "SAC's state-dependent std is out of scope)")
    log_std = [float(x) for x in log_std_param.detach().cpu().numpy().reshape(-1)]
    return std_from_log_std(log_std)


def write_action_dist_json(stats: dict, path: Path) -> None:
    path.write_text(json.dumps(stats, indent=2) + "\n")


def load_model(zip_path: Path) -> Any:
    from stable_baselines3 import PPO  # lazy: keep import cost out of the pure-helper tests
    return PPO.load(str(zip_path), device="cpu")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export SB3 PPO continuous action std to JSON.")
    p.add_argument("model", help="path to the SB3 PPO .zip checkpoint")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: <model-stem>_action_dist.json beside the model)")
    a = p.parse_args(argv)

    model_path = Path(a.model)
    if not model_path.is_file():
        print(f"ERROR: model not found: {a.model}", file=sys.stderr)
        return 1
    out_path = Path(a.out) if a.out else model_path.with_name(model_path.stem + "_action_dist.json")

    try:
        model = load_model(model_path)
        stats = std_from_model(model)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    write_action_dist_json(stats, out_path)
    print(f"OK: {out_path} (action_dim={stats['action_dim']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_export_action_dist -v`
Expected: PASS (4 tests OK).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_action_dist.py test/python/test_export_action_dist.py
git commit -m "feat: export_action_dist.py — PPO log_std -> std sidecar (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: GDScript loader `action_dist.gd`

**Files:**
- Create: `addons/godot_native_rl/controllers/action_dist.gd`
- Test: `test/unit/test_action_dist.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_action_dist.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ActionDist = preload("res://addons/godot_native_rl/controllers/action_dist.gd")

func _initialize() -> void:
	var h := Harness.new()

	# validate: accepts a well-formed std array.
	h.assert_true(ActionDist.validate({"std": [0.3, 0.5]}), "validate accepts std array")
	# validate: accepts matching action_dim.
	h.assert_true(ActionDist.validate({"std": [0.3, 0.5], "action_dim": 2}),
		"validate accepts matching action_dim")
	# validate: rejects mismatched action_dim.
	h.assert_true(not ActionDist.validate({"std": [0.3, 0.5], "action_dim": 3}),
		"validate rejects action_dim mismatch")
	# validate: rejects empty std.
	h.assert_true(not ActionDist.validate({"std": []}), "validate rejects empty std")
	# validate: rejects missing std key.
	h.assert_true(not ActionDist.validate({"action_dim": 2}), "validate rejects missing std")
	# validate: rejects non-numeric std element.
	h.assert_true(not ActionDist.validate({"std": [0.3, "x"]}), "validate rejects non-numeric std")

	# to_typed: coerces JSON array into PackedFloat32Array.
	var typed := ActionDist.to_typed({"std": [0.25, 0.75]})
	h.assert_true(typed["std"] is PackedFloat32Array, "to_typed coerces to PackedFloat32Array")
	h.assert_true(absf(typed["std"][0] - 0.25) < 1e-6 and absf(typed["std"][1] - 0.75) < 1e-6,
		"to_typed preserves values")

	# to_typed: invalid -> {}.
	h.assert_eq(ActionDist.to_typed({"std": []}), {}, "to_typed invalid -> {}")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `${GODOT:-/opt/homebrew/bin/godot-mono} --headless --path . --script res://test/unit/test_action_dist.gd`
Expected: FAIL — cannot load `action_dist.gd` (file does not exist).

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/controllers/action_dist.gd`:

```gdscript
class_name ActionDist
extends RefCounted

# Pure deploy-side loader for the continuous-action DiagGaussian std sidecar. The post-inference
# analogue of obs_normalize.gd: SB3 PPO's std is a state-independent learned parameter
# (policy.log_std) that is never in the network output, so an ncnn-converted policy has lost it.
# export_action_dist.py writes std = exp(log_std) as a flat JSON ({"std": [...], "action_dim": N});
# this validates + coerces it. The actual Gaussian draw lives in action_decode.gd's continuous
# branch (it already has the per-segment mean + rng). std is applied positionally across the
# continuous action dims. PPO continuous only; SAC's state-dependent std is out of scope.

# True iff a JSON-decoded dict is well-formed: a non-empty numeric `std` array, and (if present)
# an `action_dim` equal to std.size(). Checked at load so a bad fixture fails loudly up front.
static func validate(stats: Dictionary) -> bool:
	if not stats.has("std"):
		return false
	var std = stats["std"]
	if not (std is Array or std is PackedFloat32Array):
		return false
	if std.size() == 0:
		return false
	for v in std:
		if not (v is float or v is int):
			return false
	if stats.has("action_dim"):
		var action_dim = stats["action_dim"]
		if not (action_dim is int or action_dim is float) or int(action_dim) != std.size():
			return false
	return true

# Coerce a validated JSON dict into a typed PackedFloat32Array once, so the per-frame hot path
# doesn't re-coerce. Returns {} (and push_error) if invalid.
static func to_typed(stats: Dictionary) -> Dictionary:
	if not validate(stats):
		push_error("ActionDist.to_typed: invalid stats dictionary.")
		return {}
	return {"std": PackedFloat32Array(stats["std"])}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `${GODOT:-/opt/homebrew/bin/godot-mono} --headless --path . --script res://test/unit/test_action_dist.gd`
Expected: PASS (harness prints all assertions OK, exit 0).

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/action_dist.gd test/unit/test_action_dist.gd
git commit -m "feat: ActionDist loader for the continuous std sidecar (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `decode_actions` continuous-sampling branch

**Files:**
- Modify: `addons/godot_native_rl/controllers/action_decode.gd:17` (signature), `:18-19` (counter), `:39-44` (continuous branch)
- Test: `test/unit/test_action_decode.gd` (append before `h.finish(self)`)

- [ ] **Step 1: Write the failing test**

In `test/unit/test_action_decode.gd`, insert the following block immediately before the final `h.finish(self)` line (line 117):

```gdscript
	# --- Continuous DiagGaussian sampling (deterministic=false + action_dist std) ---
	# Deterministic (default) with std present is unchanged: still the mean (regression guard).
	var dist := {"std": PackedFloat32Array([0.3, 0.3])}
	var r_det := ActionDecode.decode_actions(PackedFloat32Array([0.25, -0.5]), cont, true, null, dist)
	h.assert_true(absf(r_det["steer"][0] - 0.25) < 1e-6 and absf(r_det["steer"][1] - (-0.5)) < 1e-6,
		"continuous deterministic + std -> still mean")

	# deterministic=false but empty action_dist -> falls back to mean (continuous unchanged).
	var rng_nostd := RandomNumberGenerator.new(); rng_nostd.seed = 3
	var r_nostd := ActionDecode.decode_actions(PackedFloat32Array([0.25, -0.5]), cont, false, rng_nostd, {})
	h.assert_true(absf(r_nostd["steer"][0] - 0.25) < 1e-6 and absf(r_nostd["steer"][1] - (-0.5)) < 1e-6,
		"continuous stochastic, empty action_dist -> mean")

	# Reproducibility: same seed + std -> identical samples.
	var rc_a := RandomNumberGenerator.new(); rc_a.seed = 11
	var rc_b := RandomNumberGenerator.new(); rc_b.seed = 11
	var sa := ActionDecode.decode_actions(PackedFloat32Array([1.0, -1.0]), cont, false, rc_a, dist)
	var sb := ActionDecode.decode_actions(PackedFloat32Array([1.0, -1.0]), cont, false, rc_b, dist)
	h.assert_true(absf(sa["steer"][0] - sb["steer"][0]) < 1e-9 and absf(sa["steer"][1] - sb["steer"][1]) < 1e-9,
		"continuous same seed -> identical samples")

	# Sampling actually perturbs away from the mean (not a no-op).
	var rc_p := RandomNumberGenerator.new(); rc_p.seed = 11
	var sp := ActionDecode.decode_actions(PackedFloat32Array([1.0, -1.0]), cont, false, rc_p, dist)
	h.assert_true(absf(sp["steer"][0] - 1.0) > 1e-4, "continuous sampling perturbs the mean")

	# Histogram: large-N empirical mean ~ provided mean, std ~ provided sigma (loose tolerance).
	var rc_h := RandomNumberGenerator.new(); rc_h.seed = 2024
	var sigma := 0.5
	var dist_h := {"std": PackedFloat32Array([sigma])}
	var one := {"steer": {"size": 1, "action_type": "continuous"}}
	var draws := 5000
	var sum_v := 0.0
	var sumsq := 0.0
	for i in range(draws):
		var val: float = ActionDecode.decode_actions(PackedFloat32Array([2.0]), one, false, rc_h, dist_h)["steer"][0]
		sum_v += val
		sumsq += (val - 2.0) * (val - 2.0)
	var emp_mean := sum_v / draws
	var emp_std := sqrt(sumsq / draws)
	h.assert_true(absf(emp_mean - 2.0) < 0.05, "continuous histogram: empirical mean ~ 2.0 (got %f)" % emp_mean)
	h.assert_true(absf(emp_std - sigma) < 0.05, "continuous histogram: empirical std ~ 0.5 (got %f)" % emp_std)

	# tanh applied AFTER the Gaussian draw when squash set (result stays within (-1, 1)).
	var rc_s := RandomNumberGenerator.new(); rc_s.seed = 8
	var dist_s := {"std": PackedFloat32Array([0.5, 0.5])}
	var r_sq := ActionDecode.decode_actions(PackedFloat32Array([2.0, -2.0]), cont_sq, false, rc_s, dist_s)
	h.assert_true(r_sq["steer"][0] > -1.0 and r_sq["steer"][0] < 1.0, "continuous sampled+squash in (-1,1)")

	# Multi-continuous keys: positional std mapping (zero std -> sampling == mean, proves alignment).
	var two := {"x": {"size": 1, "action_type": "continuous"}, "y": {"size": 1, "action_type": "continuous"}}
	var dist_two := {"std": PackedFloat32Array([0.0, 0.0])}
	var r_two := ActionDecode.decode_actions(PackedFloat32Array([0.3, -0.7]), two, false, RandomNumberGenerator.new(), dist_two)
	h.assert_true(absf(r_two["x"][0] - 0.3) < 1e-6 and absf(r_two["y"][0] - (-0.7)) < 1e-6,
		"multi-continuous: zero-std positional mapping -> means")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `${GODOT:-/opt/homebrew/bin/godot-mono} --headless --path . --script res://test/unit/test_action_decode.gd`
Expected: FAIL — `decode_actions` does not accept a 5th arg (or the std is ignored so the "perturbs the mean" / histogram assertions fail).

- [ ] **Step 3: Write minimal implementation**

In `addons/godot_native_rl/controllers/action_decode.gd`, change the signature (line 17) from:

```gdscript
static func decode_actions(output: PackedFloat32Array, action_space: Dictionary, deterministic: bool = true, rng: RandomNumberGenerator = null) -> Dictionary:
```

to:

```gdscript
static func decode_actions(output: PackedFloat32Array, action_space: Dictionary, deterministic: bool = true, rng: RandomNumberGenerator = null, action_dist: Dictionary = {}) -> Dictionary:
```

Add a continuous-only counter beside `var index := 0` (after line 19):

```gdscript
	var cont_index := 0  # advances per continuous value consumed; indexes action_dist["std"] positionally
```

Replace the continuous branch (lines 39-44) from:

```gdscript
		elif action_type == "continuous":
			var squash: bool = entry.get("squash", false)
			var values: Array = []
			for v in segment:
				values.append(tanh(v) if squash else v)
			result[key] = values
```

to:

```gdscript
		elif action_type == "continuous":
			var squash: bool = entry.get("squash", false)
			# Stochastic continuous (PPO DiagGaussian): sample mean + std·N(0,1), then optional tanh.
			# Only when non-deterministic AND a std sidecar is present; else the mean (unchanged).
			var has_std: bool = (not deterministic) and action_dist.has("std")
			var std: PackedFloat32Array = action_dist["std"] if has_std else PackedFloat32Array()
			var values: Array = []
			for v in segment:
				var x: float = v
				if has_std and cont_index < std.size():
					# rng=null falls back to Godot's global RNG (mirrors the discrete branch).
					var z: float = rng.randfn(0.0, 1.0) if rng != null else randfn(0.0, 1.0)
					x = v + std[cont_index] * z
				values.append(tanh(x) if squash else x)
				cont_index += 1
			result[key] = values
```

Also update the header comment block (lines 5-13) — change the continuous line to note sampling:

```gdscript
#   continuous -> the next `size` values, optionally tanh-squashed (per-key "squash": true)
#   continuous (stochastic) -> mean + std·N(0,1) via rng when deterministic=false and an
#                 action_dist {"std": [...]} sidecar is supplied (PPO DiagGaussian); std is applied
#                 positionally across the continuous dims, then the optional tanh squash.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `${GODOT:-/opt/homebrew/bin/godot-mono} --headless --path . --script res://test/unit/test_action_decode.gd`
Expected: PASS (all assertions, including the new continuous-sampling block, OK).

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/action_decode.gd test/unit/test_action_decode.gd
git commit -m "feat: continuous DiagGaussian sampling in ActionDecode (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Core wiring — pass `action_dist_stats` into decode

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd:18` (field), `:111` (decode call)

This is verified by the controller wiring test in Task 5 (the core has no standalone test for `obs_norm_stats` either). No new test here; the existing `test_action_decode.gd` and `test_stochastic_inference.gd` cover the behavior.

- [ ] **Step 1: Add the core field**

In `ncnn_controller_core.gd`, after line 18 (`var obs_norm_stats: Dictionary = {}`), add:

```gdscript
# Continuous DiagGaussian std sidecar (from ActionDist.to_typed of a <model>_action_dist.json).
# Empty -> continuous actions are the mean (current behavior). Used only when
# deterministic_inference=false; applied positionally across continuous action dims.
var action_dist_stats: Dictionary = {}
```

- [ ] **Step 2: Pass it into decode_actions**

In `ncnn_controller_core.gd`, change line 111 from:

```gdscript
	var action: Dictionary = ActionDecode.decode_actions(output, agent.get_action_space(), deterministic_inference, rng)
```

to:

```gdscript
	var action: Dictionary = ActionDecode.decode_actions(output, agent.get_action_space(), deterministic_inference, rng, action_dist_stats)
```

- [ ] **Step 3: Verify the core still parses (smoke)**

Run: `${GODOT:-/opt/homebrew/bin/godot-mono} --headless --path . --script res://test/unit/test_action_decode.gd`
Expected: PASS (no parse error introduced in the preloaded chain).

- [ ] **Step 4: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd
git commit -m "feat: core carries action_dist_stats into decode (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Controller wiring (2D + 3D) + wiring test

**Files:**
- Create: `test/unit/stub_continuous_agent.gd`
- Modify: `test/unit/test_stochastic_inference.gd`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`

- [ ] **Step 1: Create the continuous stub agent**

Create `test/unit/stub_continuous_agent.gd`:

```gdscript
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0]}

func get_reward() -> float:
	return reward

func get_action_space() -> Dictionary:
	return {"steer": {"size": 2, "action_type": "continuous"}}

var last_action = null

func set_action(action) -> void:
	last_action = action
```

- [ ] **Step 2: Write the failing wiring test**

In `test/unit/test_stochastic_inference.gd`: add a preload + a continuous fake runner at the top (after the existing `const Stub` line), and a test block before `h.finish(self)`.

Add after the `const Stub = ...` line near the top:

```gdscript
const ContStub = preload("res://test/unit/stub_continuous_agent.gd")

# Fake runner returning a fixed 2-D continuous mean over the continuous stub's "steer" space.
class FakeContRunner:
	var loaded := true
	var output := PackedFloat32Array([1.0, -1.0])
	func is_model_loaded() -> bool:
		return loaded
	func run_inference(_input) -> PackedFloat32Array:
		return output
```

Add this block immediately before the final `h.finish(self)`:

```gdscript
	# --- Continuous DiagGaussian sampling wiring (action_dist_stats) ---
	# set_action_dist_for_test populates the core field.
	var probe := ContStub.new()
	probe.set_action_dist_for_test({"std": PackedFloat32Array([0.3, 0.3])})
	h.assert_true(probe._core.action_dist_stats.has("std"), "set_action_dist_for_test sets core field")
	probe.free()

	# Same seed + action_dist std on two continuous controllers -> identical sampled action,
	# and sampling perturbs away from the fixed mean [1.0, -1.0].
	var c1 := ContStub.new()
	c1.set_ncnn_runner_for_test(FakeContRunner.new())
	c1.set_stochastic_for_test(false, 55)
	c1.set_action_dist_for_test({"std": PackedFloat32Array([0.3, 0.3])})
	var c2 := ContStub.new()
	c2.set_ncnn_runner_for_test(FakeContRunner.new())
	c2.set_stochastic_for_test(false, 55)
	c2.set_action_dist_for_test({"std": PackedFloat32Array([0.3, 0.3])})
	c1.infer_and_act()
	c2.infer_and_act()
	h.assert_eq(c1.last_action, c2.last_action, "continuous: same seed -> identical sampled action")
	h.assert_true(absf(c1.last_action["steer"][0] - 1.0) > 1e-4,
		"continuous: controller sampling perturbs the mean")
	c1.free()
	c2.free()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `${GODOT:-/opt/homebrew/bin/godot-mono} --headless --path . --script res://test/unit/test_stochastic_inference.gd`
Expected: FAIL — `set_action_dist_for_test` is not a method on the controller (and `_core.action_dist_stats` is empty / sampling is a no-op).

- [ ] **Step 4: Wire the 2D controller**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`:

Add the preload after line 6 (`const ObsNormalize = ...`):

```gdscript
const ActionDist = preload("res://addons/godot_native_rl/controllers/action_dist.gd")
```

Add the export after line 16 (`@export_file("*.json") var obs_norm_stats_path: String = ""`):

```gdscript
@export_file("*.json") var action_dist_stats_path: String = ""  # continuous DiagGaussian std sidecar
```

In `_ready()`, after the `_load_obs_norm_stats()` call (line 63), add:

```gdscript
		_load_action_dist_stats()
```

Add the loader + test seam after `set_obs_norm_stats_for_test` (after line 107):

```gdscript
func _load_action_dist_stats() -> void:
	if action_dist_stats_path.is_empty():
		return
	var f := FileAccess.open(action_dist_stats_path, FileAccess.READ)
	if f == null:
		push_error("NcnnAIController2D: cannot open action_dist_stats_path '%s'." % action_dist_stats_path)
		return
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary) or not ActionDist.validate(parsed):
		push_error("NcnnAIController2D: invalid action-dist stats JSON at '%s'." % action_dist_stats_path)
		return
	_core.action_dist_stats = ActionDist.to_typed(parsed)

func set_action_dist_for_test(stats: Dictionary) -> void:
	_core.action_dist_stats = stats
```

- [ ] **Step 5: Wire the 3D controller (identical change)**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`, make the same four edits. The file mirrors the 2D controller line-for-line (preload at line 6, export at line 16, `_load_obs_norm_stats()` call at line 63, `set_obs_norm_stats_for_test` at line 106-107).

Add the preload after `const ObsNormalize = ...`:

```gdscript
const ActionDist = preload("res://addons/godot_native_rl/controllers/action_dist.gd")
```

Add the export after `@export_file("*.json") var obs_norm_stats_path: String = ""`:

```gdscript
@export_file("*.json") var action_dist_stats_path: String = ""  # continuous DiagGaussian std sidecar
```

In `_ready()`, after the `_load_obs_norm_stats()` call:

```gdscript
		_load_action_dist_stats()
```

After `set_obs_norm_stats_for_test`:

```gdscript
func _load_action_dist_stats() -> void:
	if action_dist_stats_path.is_empty():
		return
	var f := FileAccess.open(action_dist_stats_path, FileAccess.READ)
	if f == null:
		push_error("NcnnAIController3D: cannot open action_dist_stats_path '%s'." % action_dist_stats_path)
		return
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary) or not ActionDist.validate(parsed):
		push_error("NcnnAIController3D: invalid action-dist stats JSON at '%s'." % action_dist_stats_path)
		return
	_core.action_dist_stats = ActionDist.to_typed(parsed)

func set_action_dist_for_test(stats: Dictionary) -> void:
	_core.action_dist_stats = stats
```

- [ ] **Step 6: Run test to verify it passes**

Run: `${GODOT:-/opt/homebrew/bin/godot-mono} --headless --path . --script res://test/unit/test_stochastic_inference.gd`
Expected: PASS (the continuous wiring block + all existing discrete assertions OK).

- [ ] **Step 7: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd \
        addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd \
        test/unit/stub_continuous_agent.gd test/unit/test_stochastic_inference.gd
git commit -m "feat: controllers load action_dist_stats_path sidecar (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Full test suite + docs

**Files:**
- Modify: `docs/guide/deploying.md`, `docs/guide/building-your-agent.md`, `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`

- [ ] **Step 1: Run the full suite (regression gate)**

Run: `./test/run_tests.sh`
Expected: all green — existing golden/argmax/mean regressions unchanged (defaults stay deterministic, no `action_dist`), plus the new GDScript + Python tests pass.

- [ ] **Step 2: Update `docs/guide/deploying.md`**

After the **## VecNormalize obs stats** section (ends ~line 168), add:

```markdown
## Continuous action sampling (DiagGaussian std sidecar)

A PPO policy with a continuous (Box) action space stores its std as a state-independent learned
parameter (`policy.log_std`) that is never part of the network output — so an ncnn-converted policy
emits only the action mean. To sample (explore at deploy / vary behavior) instead of always taking
the mean, export the std to a sidecar and replay the DiagGaussian draw game-side:

```bash
.venv-train/bin/python scripts/export_action_dist.py models/policy.zip   # -> models/policy_action_dist.json
```

Point the controller's `action_dist_stats_path` export at the JSON and set
`deterministic_inference = false`. The deploy side then samples `mean + std·N(0,1)` (seeded by
`inference_seed` for reproducible eval), applying the optional per-key `tanh` squash after the draw.
With `deterministic_inference = true` (default) the std is ignored and the mean is used unchanged.
This **exceeds godot_rl**, whose export drops the std entirely. (PPO/A2C continuous only; SAC's
state-dependent std is out of scope — its actor already exports `tanh(mean)`.)
```

- [ ] **Step 3: Update `docs/guide/building-your-agent.md`**

Change the `deterministic_inference` bullet (line 33-34) to note continuous sampling, and add an
`action_dist_stats_path` bullet after the `obs_norm_stats_path` bullet (line 36):

```markdown
- `deterministic_inference` — (default `true`) when `false`, discrete actions are sampled from
  `softmax(logits)`, and continuous actions are sampled from a DiagGaussian if an
  `action_dist_stats_path` sidecar is set (else the mean).
- `inference_seed` — (default `-1`) seed the sampler for reproducible stochastic eval.
- `obs_norm_stats_path` — path to a VecNormalize stats JSON (see [deploying.md](deploying.md)).
- `action_dist_stats_path` — path to a continuous-action std JSON sidecar for DiagGaussian sampling
  (see [deploying.md](deploying.md)); only used when `deterministic_inference = false`.
```

- [ ] **Step 4: Update `README.md`**

README does not enumerate controller `@export` vars — it points to the deploying guide. Change the
deploying-guide bullet (line 25) from:

```markdown
- [Deploying](docs/guide/deploying.md) — NcnnRunner, INT8, VecNormalize, platform targets
```

to:

```markdown
- [Deploying](docs/guide/deploying.md) — NcnnRunner, INT8, VecNormalize, continuous action sampling, platform targets
```

- [ ] **Step 5: Update `CLAUDE.md`**

In the **Key commands** list, add a line near the other export helpers:

```markdown
- **Export continuous action std (deploy):** `.venv-train/bin/python scripts/export_action_dist.py
  models/policy.zip` → `*_action_dist.json`; set the controller's `action_dist_stats_path` and
  `deterministic_inference=false` so `ActionDecode` samples `mean + std·N(0,1)` (PPO DiagGaussian,
  game-side) instead of the mean. PPO continuous only (SAC std is state-dependent — out of scope).
```

- [ ] **Step 6: Update `docs/godot-rl-gap-analysis-2026-06-02.md`**

Update the "Stochastic action sampling" row (line 59) — change the middle cell so the continuous
follow-up is now done/exceeds. From:

```markdown
| Stochastic action sampling | ✅ `deterministic_inference` flag (softmax vs argmax) | ✅ `deterministic_inference` + `inference_seed`; discrete softmax-sample (continuous follow-up #64) | ✅ done (#16) |
```

to:

```markdown
| Stochastic action sampling | ✅ `deterministic_inference` flag (softmax vs argmax) | ✅ `deterministic_inference` + `inference_seed`; discrete softmax-sample **+ continuous DiagGaussian sample via `log_std` sidecar (export drops std — game-side only)** | ✅ done (#16, #64) |
```

- [ ] **Step 7: Re-run the suite + commit**

```bash
./test/run_tests.sh
git add docs/guide/deploying.md docs/guide/building-your-agent.md README.md CLAUDE.md \
        docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: continuous action sampling sidecar (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **GDScript indentation is TABS.** The code blocks above use tabs; preserve them.
- **`godot-mono` is the dev binary** (`/opt/homebrew/bin/godot-mono`, Godot 4.5.1). `run_tests.sh`
  respects `GODOT=`. Single GDScript tests run via
  `godot --headless --path . --script res://test/unit/<file>.gd`.
- **Python tests run under `.venv-train`** (`.venv-train/bin/python -m unittest ...`). `torch` is
  available there; only `TestStdFromModel` imports it.
- **Do not push to `main`.** Work stays on `feat/continuous-action-sampling-64`. PR opens after the
  suite is green; PR body `Closes #64`. PR 2 (FlyBy example) stacks on this branch.
- **`randfn` is a Godot global** (`@GlobalScope.randfn(mean, deviation)`) and a `RandomNumberGenerator`
  method — both used in Task 3, mirroring the discrete branch's `randf()`/`rng.randf()` fallback.
```
